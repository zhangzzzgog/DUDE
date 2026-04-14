import json
import os
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from datasets import Dataset
from datasets import concatenate_datasets

from train.datasets import load_local_dataset
from train.formatter import add_row
from train.formatter import format_url
from train.formatter import make_conversation
from src.config import SETTINGS
from src.config import require_zhipuai_api_key
from src.model import GLM
from src.model import Local
from src.parser import extract_xml
from src.template import system_prompt


class EvalEXP:
    def __init__(
        self,
        model_id: str = SETTINGS.default_local_model,
        data_path: str = SETTINGS.data_path,
        images_dir: str = SETTINGS.images_dir,
        output_dir: str = os.path.join(SETTINGS.output_dir, "opt_exp"),
        device: str = SETTINGS.default_device,
        use_api: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model_id = model_id
        self.data_path = data_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.device = device
        self.use_api = use_api

        self.dataset: Optional[Dataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.success: Optional[Dataset] = None
        self.failure: Optional[Dataset] = None

        self.model = None
        self.sum_agent = None
        self.trained_model_path = None

        self.api_key = api_key or SETTINGS.zhipuai_api_key
        self.base_url = base_url or SETTINGS.base_url

    def load_data(self, test_size: float = 0.2, seed: int = 42, mode: str = "Train") -> None:
        """Load the dataset and prepare the split used by optimization."""

        print("Loading dataset...")
        self.dataset = load_local_dataset(
            self.data_path,
            self.images_dir,
            load_images=False,
            Train=True,
        )

        if mode == "Train":
            split_dataset = self.dataset.train_test_split(test_size=test_size, seed=seed)
            self.train_dataset = split_dataset["train"]
            self.test_dataset = split_dataset["test"]
        elif mode == "Eval":
            self.train_dataset = self.dataset
            self.test_dataset = Dataset.from_list([])
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.train_dataset = self.train_dataset.map(make_conversation)
        print(
            f"Loaded {len(self.train_dataset)} training samples and {len(self.test_dataset)} test samples"
        )

    def setup_cloud_model(self) -> None:
        """Use a remote GLM model as the evaluator."""

        self.model = GLM(
            model_name=SETTINGS.default_eval_model,
            api_key=require_zhipuai_api_key(self.api_key),
            SYSTEM_PROMPT="",
        )

    def load_trained_model(self, model_path: Optional[str] = None) -> None:
        """Load a local finetuned evaluator model."""

        model_path = model_path or self.trained_model_path
        if model_path is None:
            raise ValueError("No trained model path available. Please train the model first.")

        print(f"Loading trained model from {model_path}...")
        self.model = Local(
            model_name=self.model_id,
            SYSTEM_PROMPT="",
            tools=[],
            model_path=model_path,
        )
        print("Trained model loaded successfully for optimization")

    def load_exp_summarizer(self) -> None:
        """Load the model that summarizes repeated evaluator failures into experience text."""

        print("Loading summarizer model...")
        base_system_prompt = (
            "You are an experienced optimizer for a web browsing click evaluator. "
            "Your job is to summarize reusable experience that helps the evaluator avoid repeated mistakes. "
            "Below is the evaluator system prompt template: "
            f"{system_prompt} "
            "You will receive a batch of failed samples. Each sample includes a screenshot, the user task, "
            "and the click produced by the agent. Carefully analyze why the evaluator failed and output exactly "
            "one XML block in the form <exp>...</exp>."
        )
        self.sum_agent = GLM(
            model_name=SETTINGS.default_eval_model,
            api_key=require_zhipuai_api_key(self.api_key),
            SYSTEM_PROMPT=base_system_prompt,
        )

    @staticmethod
    def _build_failure_summary(samples: Dataset, exp: Optional[str]) -> str:
        previous_exp = exp if exp else "None"
        task_parts = []
        click_parts = []
        for sample in samples:
            miss_count = int(sample.get("again", 1))
            task_parts.append(f"<Miss eval*{miss_count}> {sample['messages'][1]['content']}")
            click_parts.append(str(sample["click"]))

        return (
            f"Your previous experience: {previous_exp}\n"
            f"User task: {'; '.join(task_parts)}\n"
            f"Agent output: {'; '.join(click_parts)}\n"
        )

    @staticmethod
    def _build_eval_message(sample: Dict[str, Any], exp: str) -> List[Dict[str, Any]]:
        sample_image_url = format_url(sample["image_path_normalized"][0])
        return [
            {"role": "system", "content": f"{exp}\n{system_prompt}"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"User task: {sample['messages'][1]['content']}\n"
                            f"Agent output: {sample['click']}\n"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": sample_image_url},
                    },
                ],
            },
        ]

    @staticmethod
    def _dataset_from_samples(samples: List[Dict[str, Any]]) -> Dataset:
        if not samples:
            return Dataset.from_list([])
        return Dataset.from_list(samples)

    def opt_exp_context(
        self,
        select_num: int = 40,
        batch_count: int = 10,
        success_batch_count: int = 3,
        max_iterations: int = 10,
    ) -> Optional[str]:
        """Iteratively summarize failure cases and use the summary as evaluator experience."""

        if self.train_dataset is None:
            raise ValueError("Training dataset is not loaded. Call load_data() first.")
        if self.model is None:
            raise ValueError("Evaluator model is not initialized.")
        if self.sum_agent is None:
            raise ValueError("Summarizer model is not initialized.")

        capped_select_num = min(select_num, len(self.train_dataset))
        self.failure = self.train_dataset.select(range(capped_select_num))
        self.failure = self.failure.map(lambda x: add_row(x, "again", 1))

        if capped_select_num < len(self.train_dataset):
            self.success = self.train_dataset.select(range(capped_select_num, len(self.train_dataset)))
            self.success = self.success.map(lambda x: add_row(x, "again", 0))
        else:
            self.success = Dataset.from_list([])

        iteration = 0
        exp = None

        while len(self.failure) != 0 and iteration < max_iterations:
            print(f"\n--- Iteration {iteration + 1} ---")

            failure_ids = self.failure["id"]
            selected_failure_ids = random.sample(failure_ids, min(batch_count, len(failure_ids)))
            f_samples = self.failure.filter(lambda x: x["id"] in selected_failure_ids)
            self.failure = self.failure.filter(lambda x: x["id"] not in selected_failure_ids)

            if len(self.success) > 0:
                success_ids = self.success["id"]
                selected_success_ids = random.sample(success_ids, min(success_batch_count, len(success_ids)))
                s_samples = self.success.filter(lambda x: x["id"] in selected_success_ids)
                self.success = self.success.filter(lambda x: x["id"] not in selected_success_ids)
            else:
                s_samples = Dataset.from_list([])

            print("Summarizing failures...")
            image_paths = [format_url(sample["image_path_normalized"][0]) for sample in f_samples]
            content = [{"type": "text", "text": self._build_failure_summary(f_samples, exp)}]
            for url in image_paths:
                content.append({"type": "image_url", "image_url": {"url": url}})

            messages = [
                {"role": "system", "content": self.sum_agent.system_prompt},
                {"role": "user", "content": content},
            ]
            output = self.sum_agent.call_model(messages)
            exp = extract_xml(output, "exp") or output
            print(exp)

            samples = concatenate_datasets([f_samples, s_samples]) if len(s_samples) > 0 else f_samples
            success_samples: List[Dict[str, Any]] = []
            failure_samples: List[Dict[str, Any]] = []

            for sample in samples:
                message = self._build_eval_message(sample, exp)
                output = self.model.call_model(message)
                judge_text = extract_xml(output, "judge")
                conf_text = extract_xml(output, "conf")

                try:
                    judge = int(judge_text)
                except (TypeError, ValueError):
                    judge = -999

                try:
                    conf = float(conf_text)
                except (TypeError, ValueError):
                    conf = 0.0
                conf = max(0.0, min(1.0, conf))

                sample_copy = dict(sample)
                sample_copy["judge"] = judge
                sample_copy["conf"] = conf

                if judge == sample_copy.get("gen_type"):
                    success_samples.append(sample_copy)
                else:
                    sample_copy["again"] = int(sample_copy.get("again", 0)) + 1
                    failure_samples.append(sample_copy)

            if success_samples:
                self.success = concatenate_datasets([self.success, self._dataset_from_samples(success_samples)])
            if failure_samples:
                self.failure = concatenate_datasets([self.failure, self._dataset_from_samples(failure_samples)])

            iteration += 1

        return exp

    def run_full_pipeline(self, train_params: Optional[Dict[str, Any]] = None, optimize_params: Optional[Dict[str, Any]] = None):
        """Run the full optimization pipeline."""

        print("Starting full optimization pipeline...")
        self.load_data(mode="Train")

        if self.use_api or self.api_key:
            self.setup_cloud_model()
        else:
            self.load_trained_model()

        self.load_exp_summarizer()
        optimize_params = optimize_params or {}
        experience = self.opt_exp_context(**optimize_params)

        print("Full pipeline completed successfully!")
        return experience


def main() -> None:
    system = EvalEXP(use_api=True, api_key=require_zhipuai_api_key())

    result = system.run_full_pipeline(
        train_params={
            "learning_rate": 1e-5,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
        },
        optimize_params={
            "select_num": 25,
            "batch_count": 5,
            "max_iterations": 10,
        },
    )

    with open("optimization_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Optimization results saved to optimization_results.json")


if __name__ == "__main__":
    main()

