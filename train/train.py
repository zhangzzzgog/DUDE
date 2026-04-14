import os
import time
import json

import torch
from datetime import datetime

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer, PeftModel
from types import MethodType

from train.reward import hybrid_label_confidence_reward
from train.datasets import load_local_dataset
from train.formatter import make_conversation
from src.model import Local
from src.config import SETTINGS


DEFAULT_QWEN3_TRAIN_MODEL = "Qwen/Qwen3-VL-2B-Thinking"


class IntegratedTrainOptimize:
    def __init__(self,
                 model_id=None,
                 data_path=SETTINGS.data_path,
                 images_dir=SETTINGS.images_dir,
                 output_dir=os.path.join(str(SETTINGS.project_root), "data", "train"),
                 device=SETTINGS.default_device,
                 verbose=False,
                 log_samples=False):

        self.model_id = model_id or DEFAULT_QWEN3_TRAIN_MODEL
        self.data_path = data_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.device = device
        self.verbose = verbose
        self.log_samples = log_samples

        # Dataset containers.
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None

        # Model and processor handles.
        self.model = None
        self.processor = None

        # Path to the trained checkpoint produced by this run.
        self.trained_model_path = None
        self.stage1_snapshot_path = None

    def _log(self, message: str) -> None:
        print(message)

    def _debug(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _log_sample(self, title: str, sample) -> None:
        if self.log_samples:
            print(title)
            print(sample)

    def _get_torch_dtype(self):
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32

    def _get_model_load_kwargs(self):
        kwargs = {
            "pretrained_model_name_or_path": self.model_id,
            "torch_dtype": self._get_torch_dtype(),
        }
        if self.device in {"auto", "cuda"}:
            kwargs["device_map"] = "auto"
        return kwargs

    def load_data(self, test_size=0.2, seed=42):
        """Load the dataset, split it, and prepare training conversations."""

        self._log("Loading dataset...")
        self.dataset = load_local_dataset(self.data_path, self.images_dir, load_images=False, Train=True)

        # An explicit shuffle is unnecessary here because train_test_split already handles the split order.
        # self.dataset = shuffle(self.dataset, seed)

        # Split the dataset into train and test partitions.
        split_dataset = self.dataset.train_test_split(test_size=test_size, seed=seed)
        self._log_sample("Train sample preview:", split_dataset["train"][0])
        self._log_sample("Test sample preview:", split_dataset["test"][0])
        self.train_dataset = split_dataset['train']
        self.test_dataset = split_dataset['test']
        self._debug(f"Train split size: {len(self.train_dataset)}")
        self._debug(f"Test split size: {len(self.test_dataset)}")

        # Convert the raw records into the conversation format expected by training.
        self.train_dataset = self.train_dataset.map(make_conversation)
        self._log_sample("Mapped training sample preview:", self.train_dataset[0])

        # Save a JSONL snapshot of the training split for reward-time bookkeeping.
        out_dir = os.path.join(self.output_dir, "stage1_result")
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        fname = f"stage1_result{ts}.jsonl"
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w", encoding="utf-8") as fw:
            for r in self.train_dataset:
                rec_copy = dict(r) if isinstance(r, dict) else {"record": r}
                rec_copy["status"] = None
                fw.write(json.dumps(rec_copy, ensure_ascii=False) + "\n")
        self.stage1_snapshot_path = out_path
        self._debug(f"Saved dataset snapshot to {out_path}")

        self._log(f"Loaded {len(self.train_dataset)} training samples and {len(self.test_dataset)} test samples")

    def setup_model(self):
        """Set up the base model, processor, and LoRA adapters."""
        self._log("Setting up model and processor...")

        # Load the processor first so it can be shared by training and reward computation.
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True, padding_side="left")

        # Load the base multimodal model. Prefer flash attention when it is available.
        model_load_kwargs = self._get_model_load_kwargs()
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                attn_implementation="flash_attention_2",
                **model_load_kwargs,
            )
        except Exception as exc:
            self._debug(f"Falling back to the default attention backend: {exc}")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(**model_load_kwargs)

        # Attach LoRA adapters to the attention projection layers.
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )

        self.model = get_peft_model(self.model, lora_config)
        if self.verbose:
            self.model.print_trainable_parameters()

        # Patch batch_decode so invalid token ids do not crash post-processing.
        def safe_batch_decode(self, sequences, **kwargs):
            pad_id = self.tokenizer.pad_token_id or 0
            vocab_size = len(self.tokenizer)

            def clean_one(seq):
                if isinstance(seq, torch.Tensor):
                    ids = seq.clone().detach().cpu().tolist()
                else:
                    ids = list(seq)

                cleaned = []
                for x in ids:
                    try:
                        v = int(x)
                    except Exception:
                        v = pad_id
                    if v < 0 or v >= vocab_size:
                        v = pad_id
                    cleaned.append(v)
                return cleaned

            if isinstance(sequences, torch.Tensor):
                cleaned_batch = [clean_one(row) for row in sequences]
            else:
                cleaned_batch = [clean_one(row) for row in sequences]

            return self.tokenizer.batch_decode(cleaned_batch, **kwargs)

        self.processor.batch_decode = MethodType(safe_batch_decode, self.processor)

    def train_model(self,
                   learning_rate=1e-5,
                   num_train_epochs=1,
                   per_device_train_batch_size=2,
                   num_generations=2,
                   max_completion_length=512,
                   max_prompt_length=8192,
                   logging_steps=10,
                   save_steps=10):
        """Train the model with GRPO and save the resulting checkpoint."""
        self._log("Starting model training...")

        # Create a timestamped JSONL file for reward-time sample logging.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        recorded_basename = f"recorded_samples_{ts}.jsonl"
        recorded_path = os.path.join(self.output_dir, recorded_basename)
        os.makedirs(self.output_dir, exist_ok=True)
        self._debug(
            f"Training config: epochs={num_train_epochs}, batch_size={per_device_train_batch_size}, lr={learning_rate}"
        )
        self._debug(f"Reward samples will be recorded to: {recorded_path}")
        # Use a named wrapper instead of functools.partial so GRPOTrainer can read __name__.
        # Capture the snapshot path in the closure so worker processes can still access it.
        snapshot = self.stage1_snapshot_path
        def _reward_wrapper(*a, **kw):
            return hybrid_label_confidence_reward(*a, recorded_samples_path=recorded_path, snapshot_path=snapshot, run_ts=ts, **kw)

        # Preserve a readable function name for trainer-side logging and serialization.
        try:
            _reward_wrapper.__name__ = getattr(hybrid_label_confidence_reward, "__name__", "hybrid_label_confidence_reward")
        except Exception:
            pass

        reward_funcs = [_reward_wrapper]

        # Configure the GRPO training run.
        training_args = GRPOConfig(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            remove_unused_columns=False,
            num_train_epochs=num_train_epochs,
            bf16=True,
            per_device_train_batch_size=per_device_train_batch_size,
            max_completion_length=max_completion_length,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            logging_steps=logging_steps,
            save_strategy="steps",
            save_steps=save_steps,
        )

        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.processor,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=self.train_dataset,
        )

        trainer.train()

        trainer.save_model(self.output_dir)
        self.trained_model_path = self.output_dir

        self._log(f"Model training completed. Model saved to {self.output_dir}")

    def load_trained_model(self, model_path=None):
        """Load a trained checkpoint for downstream optimization or evaluation."""
        if model_path is None:
            model_path = self.trained_model_path

        if model_path is None:
            raise ValueError("No trained model path available. Please train the model first.")

        self._log(f"Loading trained model from {model_path}...")

        # Build a local agent instance and then attach the finetuned weights.
        self.agent = Local(
            model_name=self.model_id,
            SYSTEM_PROMPT="",
            tools=[]
        )

        # Load the trained adapter weights on top of the base model.
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            **self._get_model_load_kwargs(),
        )
        self.agent.model = PeftModel.from_pretrained(base_model, model_path)

        self._log("Trained model loaded successfully")

    def run_full_pipeline(self,train_params=None):
        """Run the end-to-end training pipeline."""
        self._log("Starting full training pipeline...")

        # 1. Load the dataset.
        self.load_data()

        # 2. Set up the model and processor.
        self.setup_model()

        # 3. Train the model.
        train_params = train_params or {}
        self.train_model(**train_params)


def main():
    """Example entry point for the integrated training pipeline."""
    # Create the integrated training system.
    system = IntegratedTrainOptimize(verbose=False, log_samples=False)

    # Run the default training pipeline.
    system.run_full_pipeline(
        train_params={
            'learning_rate': 1e-5,
            'num_train_epochs': 1,
            'per_device_train_batch_size': 2,
        }
    )


if __name__ == "__main__":
    main()




