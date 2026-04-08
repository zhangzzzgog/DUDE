import os
import time
import json
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import numpy as np
import functools
from datetime import datetime

from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from types import MethodType

from src import hybrid_label_confidence_reward
from src import load_local_dataset
from src import make_conversation
from src import Local
from src.config import SETTINGS



class IntegratedTrainOptimize:
    def __init__(self, 
                 model_id=SETTINGS.default_local_model,
                 data_path=SETTINGS.data_path,
                 images_dir=SETTINGS.images_dir,
                 output_dir=os.path.join(SETTINGS.output_dir, "train"),
                 device=SETTINGS.default_device,
                 use_api=False,
                 api_key=None,
                 base_url=None):
        
        self.model_id = model_id
        self.data_path = data_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.device = device
        
        # 鍒濆鍖栨暟鎹泦
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None
        
        # 鍒濆鍖栨ā鍨嬪拰澶勭悊鍣?
        self.model = None
        self.processor = None
        
        # 璁粌鍚庣殑妯″瀷璺緞
        self.trained_model_path = None

        self.API_KEY = api_key or SETTINGS.zhipuai_api_key
        self.BASE_URL = base_url or SETTINGS.base_url
        
    def load_data(self, test_size=0.2, seed=42):
        """鍔犺浇鍜屽噯澶囨暟鎹泦"""
       
        print("Loading dataset...")
        self.dataset = load_local_dataset(self.data_path, self.images_dir, load_images=False,Train=True)
        # self.dataset = load_local_dataset(self.data_path, self.images_dir, load_images=False,Train=True, shuffle=shuffle, seed=seed)

        # Shuffle鏁版嵁闆嗭紙娌℃湁蹇呰锛宻plit鍑芥暟鑷繁浼氬垎锛?
        # self.dataset = shuffle(self.dataset, seed)

        # 鍒嗗壊鏁版嵁闆?
        split_dataset = self.dataset.train_test_split(test_size=test_size, seed=seed)
        print("======================")
        print(split_dataset["train"][0])
        print("======================")
        print(split_dataset["test"][0])
        print("======================")
        self.train_dataset = split_dataset['train']
        print(f"LEN {len(self.train_dataset)}")
        self.test_dataset = split_dataset['test']
        print(f"LEN {len(self.test_dataset)}")

        # 鏍煎紡鍖栬缁冩暟鎹?
        self.train_dataset = self.train_dataset.map(make_conversation)
        print("==========ATTENTION NOW============")
        print(self.train_dataset[0])
        print("======================")

        # 鍦ㄨ缁冩ā寮忎笅锛屽皢鐢熸垚鐨?records 淇濆瓨涓?jsonl锛屾枃浠跺悕涓?stage1_result + 鏃堕棿鎴?jsonl
        out_dir = os.path.join(os.getcwd(), "stage1_result")
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
        print(f"Saved dataset snapshot to {out_path}")

        print(f"Loaded {len(self.train_dataset)} training samples and {len(self.test_dataset)} test samples")
        
    def setup_model(self):
        """璁剧疆妯″瀷鍜屽鐞嗗櫒"""
        print("Setting up model and processor...")
        
        # 鍔犺浇澶勭悊鍣?
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True, padding_side="left")
        
        # 鍔犺浇妯″瀷
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        
        # 璁剧疆LoRA閰嶇疆
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # 娣诲姞瀹夊叏鐨勬壒澶勭悊瑙ｇ爜鏂规硶
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

    #鍙€夊湪绾挎ā鍨嬭交閲忓寲鍚姩
    def setup_cloud_model(self):
        self.model =  GLM(
                model_name=SETTINGS.default_eval_model,
                api_key=self.API_KEY,
                SYSTEM_PROMPT=""
            )

    def train_model(self, 
                   learning_rate=1e-5,
                   num_train_epochs=1,
                   per_device_train_batch_size=2,
                   num_generations=2,
                   max_completion_length=512,
                   max_prompt_length=8192,
                   logging_steps=10,
                   save_steps=10,
                   reward_type="combined"):
        """璁粌鍩哄骇妯″瀷"""
        print("Starting model training...")
        
        # 鐢熸垚甯︽椂闂存埑鐨勫崟娆¤缁冭褰曟枃浠讹紙绠€鍗曟ā寮忥級锛屽苟浼犲叆 run_ts
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        recorded_basename = f"recorded_samples_{ts}.jsonl"
        recorded_path = os.path.join(self.output_dir, recorded_basename)
        os.makedirs(self.output_dir, exist_ok=True)
        # 浣跨敤鏈€灏忓寲淇敼锛氫笉瑕佷紶鍏?functools.partial锛堟病鏈?__name__锛夛紝
        # 鑰屾槸浼犲叆涓€涓叿鍚?wrapper锛屼娇 GRPOTrainer 鑳借鍙?__name__銆?
        # 鎹曡幏 snapshot 璺緞鍒伴棴鍖咃紝纭繚搴忓垪鍖栧埌 worker 鏃朵粛鍙闂紙鏇寸ǔ鍋ワ級
        snapshot = getattr(self, 'stage1_snapshot_path', None)
        def _reward_wrapper(*a, **kw):
            return hybrid_label_confidence_reward(*a, recorded_samples_path=recorded_path, snapshot_path=snapshot, run_ts=ts, **kw)

        # 淇濊瘉鏈夊彲璇荤殑鍚嶅瓧
        try:
            _reward_wrapper.__name__ = getattr(hybrid_label_confidence_reward, "__name__", "hybrid_label_confidence_reward")
        except Exception:
            pass

        reward_funcs = [_reward_wrapper]
        
        # 閰嶇疆璁粌鍙傛暟
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
        
        # 鍒涘缓璁粌鍣?
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.processor,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=self.train_dataset,
        )
        
        # 寮€濮嬭缁?
        trainer.train()
        
        # 淇濆瓨妯″瀷
        trainer.save_model(self.output_dir)
        self.trained_model_path = self.output_dir
        
        print(f"Model training completed. Model saved to {self.output_dir}")
        
    def load_trained_model(self, model_path=None):
        """鍔犺浇璁粌濂界殑妯″瀷鐢ㄤ簬浼樺寲"""
        if model_path is None:
            model_path = self.trained_model_path
        
        if model_path is None:
            raise ValueError("No trained model path available. Please train the model first.")
        
        print(f"Loading trained model from {model_path}...")
        
        # 鍒涘缓鏈湴Agent瀹炰緥
        self.agent = Local(
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            SYSTEM_PROMPT="",
            tools=[]
        )
        
        # 鍔犺浇璁粌濂界殑鏉冮噸
        from peft import PeftModel
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.agent.model = PeftModel.from_pretrained(base_model, model_path)
        
        print("Trained model loaded successfully for optimization")
        
    
    def run_full_pipeline(self, 
                         train_params=None,
                         optimize_params=None):
        """杩愯瀹屾暣鐨勮缁冨拰浼樺寲娴佹按绾?""
        print("Starting full training and optimization pipeline...")
        
        # 1. 鍔犺浇鏁版嵁
        self.load_data()
        
        # 2. 璁剧疆妯″瀷
        self.setup_model()
        
        # 3. 璁粌妯″瀷
        train_params = train_params or {}
        self.train_model(**train_params)
        

def main():
    """涓诲嚱鏁扮ず渚?""
    # 鍒涘缓鏁村悎鐨勮缁冨拰浼樺寲绯荤粺
    system = IntegratedTrainOptimize()
    
    # 杩愯瀹屾暣娴佹按绾匡紝浣跨敤鏂扮殑缁勫悎濂栧姳鍑芥暟閫傞厤褰撳墠鏁版嵁闆嗙粨鏋?
    results = system.run_full_pipeline(
        train_params={
            'learning_rate': 1e-5,
            'num_train_epochs': 1,
            'per_device_train_batch_size': 2,
            'reward_type': 'combined',  # 浣跨敤缁勫悎濂栧姳锛氭纭偣鍑?+ 鏆楁ā寮忛伩鍏?
            # 鍙€夊鍔辩被鍨嬶細
            # 'combined' - 缁勫悎濂栧姳锛堟帹鑽愶紝閫傞厤褰撳墠鏁版嵁闆嗭級
            # 'correct_only' - 浠呮纭偣鍑诲鍔?
            # 'dark_avoidance' - 浠呮殫妯″紡閬垮厤濂栧姳
            # 'legacy' - 浣跨敤鍘熷click_accuracy_reward锛堝悜鍚庡吋瀹癸級
        },
        optimize_params={
            'select_num': 25,
            'batch_count': 5,
            'max_iterations': 10,
        }
    )
    
    # 淇濆瓨浼樺寲缁撴灉
    import json
    with open("optimization_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Optimization results saved to optimization_results.json")


def demo_reward_types():
    """婕旂ず涓嶅悓濂栧姳绫诲瀷鐨勪娇鐢?""
    print("婕旂ず涓嶅悓濂栧姳绫诲瀷鐨勪娇鐢?..")
    
    system = IntegratedTrainOptimize()
    system.load_data()
    system.setup_model()
    
    # 娴嬭瘯涓嶅悓濂栧姳绫诲瀷
    reward_types = ['v2_reward']
    
    for reward_type in reward_types:
        print(f"\n=== 娴嬭瘯濂栧姳绫诲瀷: {reward_type} ===")
        
        # 鍙缁?涓猠poch浣滀负婕旂ず
        try:
            system.train_model(
                num_train_epochs=1,
                per_device_train_batch_size=1,
                reward_type=reward_type,
                logging_steps=5,
                save_steps=10
            )
            print(f"鉁?濂栧姳绫诲瀷 '{reward_type}' 璁粌鎴愬姛")
        except Exception as e:
            print(f"鉂?濂栧姳绫诲瀷 '{reward_type}' 璁粌澶辫触: {e}")


if __name__ == "__main__":
    main()
    # demo_reward_types()


