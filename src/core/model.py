import re
import ast
import json
from string import Template
from typing import List, Callable, Tuple, Any, Dict
from dataclasses import dataclass
from types import MethodType

from zai import ZhipuAiClient
import torch

import inspect
import platform
import numpy as np

import os
from src.config import SETTINGS, require_zhipuai_api_key

from transformers import AutoProcessor, AutoModelForVision2Seq, Glm4vForConditionalGeneration, AutoModelForImageTextToText


class BASE:
    def __init__(self, 
                 model_name: str,
                 SYSTEM_PROMPT: str=None,
                 tools: List[Callable]=[],
                 device: str=None
                 ):
        self.tools = { func.__name__: func for func in (tools or []) }
        self.model_name = model_name
        self.system_prompt = SYSTEM_PROMPT
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Token usage tracking
        self.total_tokens = 0
        self.last_call_tokens = 0


    def update_system_prompt(self, new_system_prompt: str):
        self.system_prompt = new_system_prompt

    def run(self, user_input: str|None=None, image_paths: List[str]|None=None):
        pass

    def call_model(self, messages):
        pass

        
class GPT(BASE):
    def __init__(self, 
                 tools: List[Callable], 
                 model_name: str,
                 ):
        super().__init__(tools, model_name)

class GEMINI(BASE):
    def __init__(self, 
                 tools: List[Callable], 
                 model_name: str,
                 ):
        super().__init__(tools, model_name)

    def call_model(self, messages):
        response = self.client.chat.completions.create(
                model_name=self.model_name,
                messages=messages,
            )
        
class Local(BASE):
    def __init__(self, model_name: str = "Qwen2.5-VL-3B-Instruct", SYSTEM_PROMPT: str = None, tools: List = None, model_path: str = None):
        """
        初始化本地Agent
        
        Args:
            model_name: 基础模型名称
            SYSTEM_PROMPT: 系统提示
            tools: 工具列表
            model_path: 训练好的模型路径（可选）
        """
        super().__init__(model_name, SYSTEM_PROMPT, tools or [])
        
        # 初始化模型和处理�?
        self.model = None
        self.processor = None
        self.model_path = model_path
        
        # 加载基础模型
        self._load_base_model(model_name)
        
        # 如果提供了训练好的模型路径，加载�?
        if model_path and os.path.exists(model_path):
            self._load_finetuned_model(model_path)
        
        # 初始化上下文
        self.context = [{"role": "system", "content": self.system_prompt or ""}]
    
    def _load_base_model(self, model_name: str):
        """加载基础模型（支�?Qwen2.5 / Qwen3 / UI-TARS）�?""
        try:
            name_lower = model_name.lower()
            if "qwen3" in name_lower:
                print(f"Loading Qwen3 model: {model_name}")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    padding_side="left",
                )
            elif "qwen2.5" in name_lower or "qwen2_5" in name_lower or "qwen2.5-vl" in name_lower:
                # 向后兼容旧的 Qwen2.5 流程
                from transformers import Qwen2_5_VLForConditionalGeneration
                print(f"Loading Qwen2.5 model: {model_name}")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    use_fast=True,
                    padding_side="left",
                )
            elif "ui-tars" in name_lower or "ui_tars" in name_lower:
                # 支持 UI-TARS 系列模型作为本地基座
                print(f"Loading UI-TARS model: {model_name}")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                    trust_remote_code=True,
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    padding_side="left",
                    trust_remote_code=True,
                )
            else:
                print(f"⚠️  Unsupported model: {model_name}")
                return

            # 为本地多模态模型统一添加安全的批处理解码
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

            if self.processor is not None:
                self.processor.batch_decode = MethodType(safe_batch_decode, self.processor)

            print("�?Base model loaded successfully")
        except Exception as e:
            print(f"�?Error loading base model: {e}")
            raise
    
    def _load_finetuned_model(self, model_path: str):
        """加载微调模型（根据当前基座区�?Qwen2.5 / Qwen3）�?""
        try:
            from peft import PeftModel
            print(f"Loading finetuned model from: {model_path}")

            base_name = (
                self.model.config._name_or_path
                if hasattr(self.model, "config") and hasattr(self.model.config, "_name_or_path")
                else self.model_name
            )
            lower_name = str(base_name).lower()

            if "ui-tars" in lower_name or "ui_tars" in lower_name:
                # UI-TARS 系列：使�?AutoModelForVision2Seq 作为基座
                print(f"Loading UI-TARS base model for finetuned weights: {base_name}")
                base_model = AutoModelForVision2Seq.from_pretrained(
                    base_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                    trust_remote_code=True,
                )
            elif "qwen3" in lower_name:
                base_model = AutoModelForVision2Seq.from_pretrained(
                    base_name,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                )
            else:
                # 默认回退�?Qwen2.5
                from transformers import Qwen2_5_VLForConditionalGeneration
                base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    base_name if base_name else "Qwen/Qwen2.5-VL-3B-Instruct",
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                )

            # 加载 Peft 模型
            self.model = PeftModel.from_pretrained(base_model, model_path)
            print("�?Finetuned model loaded successfully")

        except Exception as e:
            print(f"�?Error loading finetuned model: {e}")
            print("⚠️  Using base model instead")
    
    def call_model(self, messages):
        """调用模型生成响应"""
        try:
            if self.model is None or self.processor is None:
                return "Error: Model not loaded"
            
            # 应用聊天模板
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            # 递归地将所�?tensor 移到模型所在设备，避免设备不一致导致的错误
            def move_to_device(obj, device):
                import torch
                import numpy as _np

                # Tensor -> move
                if torch.is_tensor(obj):
                    return obj.to(device)

                # numpy array -> convert then move
                if isinstance(obj, _np.ndarray):
                    try:
                        return torch.from_numpy(obj).to(device)
                    except Exception:
                        return obj

                # list of scalars -> convert to tensor
                if isinstance(obj, list) and len(obj) > 0 and all(not isinstance(x, (dict, list, tuple)) for x in obj):
                    try:
                        return torch.tensor(obj).to(device)
                    except Exception:
                        pass

                # dict/list/tuple -> recurse
                if isinstance(obj, dict):
                    return {k: move_to_device(v, device) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    cls = list if isinstance(obj, list) else tuple
                    return cls(move_to_device(v, device) for v in obj)

                return obj

            try:
                model_device = None
                try:
                    model_device = next(self.model.parameters()).device
                except Exception:
                    model_device = getattr(self.model, 'device', None)

                if model_device is not None:
                    # 优先使用 BatchEncoding/对象�?.to(device) 方法（如果有�?
                    try:
                        if hasattr(inputs, 'to'):
                            inputs = inputs.to(model_device)
                        else:
                            inputs = move_to_device(inputs, model_device)
                    except Exception:
                        # 回退到递归移动
                        inputs = move_to_device(inputs, model_device)

                    # 如果有任�?tensor 仍不�?model_device，则打印设备分布（便于调试）
                    try:
                        def collect_devices(x, prefix=""):
                            out = []
                            import torch
                            if torch.is_tensor(x):
                                out.append((prefix or "tensor", str(x.device)))
                                return out
                            if isinstance(x, dict):
                                for k, v in x.items():
                                    out.extend(collect_devices(v, prefix + "." + str(k)))
                                return out
                            if isinstance(x, (list, tuple)):
                                for i, v in enumerate(x):
                                    out.extend(collect_devices(v, prefix + f"[{i}]"))
                                return out
                            return out

                        devs = collect_devices(inputs)
                        bad = [d for d in devs if model_device is not None and d[1] != str(model_device)]
                        if bad:
                            print("⚠️  Device mismatch detected in inputs (field -> device):")
                            for field, dev in devs:
                                print(f"  {field} -> {dev}")
                    except Exception:
                        pass
            except Exception:
                # 如果迁移失败，降级为原来的浅移动策略（兼容旧行为�?
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # 生成响应
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            
            # 截断输入部分
            if 'input_ids' in inputs:
                input_ids = inputs['input_ids']
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                ]
            else:
                generated_ids_trimmed = generated_ids
            
            # 解码响应
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # Token counting
            input_tokens = inputs['input_ids'].numel() if 'input_ids' in inputs else 0
            # generated_ids contains both prompt and output in some flows, 
            # but generated_ids_trimmed is just the output part.
            output_tokens = sum(len(ids) for ids in generated_ids_trimmed)
            self.last_call_tokens = input_tokens + output_tokens
            self.total_tokens += self.last_call_tokens
            
            return output_text[0] if output_text else ""
            
        except Exception as e:
            print(f"�?Error in model inference: {e}")
            return f"Error: {str(e)}"
    
    def run(self, user_input: str = None, image_paths: List[str] = None):
        """运行推理"""
        messages = []
        
        # 添加系统提示
        system_content = [{"type": "text", "text": self.system_prompt or ""}]
        messages.append({"role": "system", "content": system_content})
        
        # 添加用户输入
        user_content = []
        
        # 处理图片
        if image_paths:
            for img_path in image_paths:
                # 检查图片路径是否存�?
                if os.path.exists(img_path):
                    user_content.append({
                        "type": "image",
                        "image": img_path
                    })
                else:
                    print(f"⚠️ Image not found: {img_path[:10]}")
        
        # 处理文本输入
        if user_input:
            user_content.append({
                "type": "text",
                "text": user_input
            })
        
        if user_content:
            messages.append({"role": "user", "content": user_content})
        
        # 调用模型
        response = self.call_model(messages)
        
        # 更新上下�?
        self.context.append({
            "role": "assistant", 
            "content": [{"type": "text", "text": response}]
        })
        
        return response
    
    def update_system_prompt(self, new_system_prompt: str):
        """更新系统提示"""
        self.system_prompt = new_system_prompt
        
        # 更新上下文的第一个消�?
        if self.context and self.context[0].get("role") == "system":
            self.context[0] = {"role": "system", "content": [{"type": "text", "text": new_system_prompt}]}
        else:
            self.context.insert(0, {"role": "system", "content": [{"type": "text", "text": new_system_prompt}]})
        
        print(f"�?System prompt updated")
    
    def load_model_from_path(self, model_path: str):
        """从路径加载新的模�?""
        if model_path and os.path.exists(model_path):
            self._load_finetuned_model(model_path)
            self.model_path = model_path
        else:
            print(f"�?Model path not found or invalid: {model_path}")
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        info = {
            "model_loaded": self.model is not None,
            "processor_loaded": self.processor is not None,
            "model_path": self.model_path,
            "device": str(self.model.device) if self.model and hasattr(self.model, 'device') else "unknown"
        }
        
        if self.model and hasattr(self.model, 'config'):
            info["model_name"] = self.model.config._name_or_path
        
        return info
        

class GLM(BASE):
    def __init__(self, model_name:str, api_key:str=None, SYSTEM_PROMPT:str=None,tools:List=None,):
        super().__init__(model_name, SYSTEM_PROMPT, tools)
        self.api_key = require_zhipuai_api_key(api_key)`r`n        self.client = ZhipuAiClient(api_key=self.api_key)  # 填写您自己的 APIKey



    def call_model(self, messages,think=False):
        if think:
            thinking_mode = "enabled"
        else:
            thinking_mode = "disabled"
        
        response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                thinking={
                    "type":thinking_mode
                }
            )

        # Token counting
        if hasattr(response, 'usage') and response.usage:
            self.last_call_tokens = getattr(response.usage, 'total_tokens', 0)
            self.total_tokens += self.last_call_tokens

        return response.choices[0].message.content


class Qwen3VLBackend(BASE):
    
    """本地 Qwen3-VL-4B-Instruct 后端封装，用于替�?GLM 作为 ReActAgent 的模型�?

    提供统一�?call_model(messages) -> str 接口�?
    - messages: �?ReActAgent 相同的数据结构（system/user，content 可以是字符串或列表）�?
    - 返回�? 单个字符串，内部包含 <thought>/<action>/<final_answer> 等标签�?
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        SYSTEM_PROMPT: str | None = None,
        tools: List[Callable] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, SYSTEM_PROMPT=SYSTEM_PROMPT, tools=tools or [], device=device or "cpu")

        # 自动选择设备：优先使用传入的 device，其次根据环境检�?
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        try:
            print(f"Loading Qwen3 VL model: {model_name} on {self.device}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("�?Qwen3 VL model loaded successfully")
        except Exception as e:
            print(f"�?Error loading Qwen3 VL model: {e}")
            raise

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """�?ReActAgent 使用�?messages 结构转换�?Qwen3 所需格式�?

        主要处理�?
        - system.content 若为字符串，改为 [{"type":"text","text":...}]
        - user.content 中的 {"type":"image_url","image_url":{"url":...}} -> {"type":"image","url":...}
        """
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 统一�?list[dict]
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    # 来自 ReActAgent.run: {"type": "image_url", "image_url": {"url": path}}
                    if item_type == "image_url" and isinstance(item.get("image_url"), dict):
                        url = item["image_url"].get("url") or item["image_url"].get("image")
                        if url is not None:
                            new_content.append({"type": "image", "url": url})
                    # 兼容已有�?{"type":"image","image": path}
                    elif item_type == "image" and "image" in item:
                        new_content.append({"type": "image", "url": item["image"]})
                    else:
                        # 其它保持原样（如 {"type":"text",...}�?
                        new_content.append(item)
            else:
                # 不支持的 content 结构，跳�?
                continue

            converted.append({"role": role, "content": new_content})

        return converted

    def call_model(self, messages: List[Dict[str, Any]], max_new_tokens: int = 512, **generate_kwargs) -> str:
        try:
            if not hasattr(self, "model") or not hasattr(self, "processor"):
                return "Error: Qwen3 VL model not loaded"

            model_messages = self._convert_messages(messages)

            inputs = self.processor.apply_chat_template(
                model_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # 将输入移到模型所在设�?
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )

            # 只解码新生成的部�?
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]

            # Token counting
            input_tokens = inputs['input_ids'].numel()
            output_tokens = generated_ids.numel()
            self.last_call_tokens = input_tokens + output_tokens
            self.total_tokens += self.last_call_tokens

            text = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return text
        except Exception as e:
            print(f"�?Error in Qwen3 VL inference: {e}")
            return f"Error: {str(e)}"


class UITARSBackend(BASE):
    """ByteDance UI-TARS-1.5-7B 后端封装�?""

    def __init__(
        self,
        model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B",
        SYSTEM_PROMPT: str | None = None,
        tools: List[Callable] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, SYSTEM_PROMPT=SYSTEM_PROMPT, tools=tools or [], device=device or "cpu")

        # 自动选择设备
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        try:
            print(f"Loading UI-TARS model: {model_name} on {self.device}")
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            print("�?UI-TARS model loaded successfully")
        except Exception as e:
            print(f"�?Error loading UI-TARS model: {e}")
            raise

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """�?ReActAgent 使用�?messages 结构转换�?UI-TARS 所需格式�?""
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 统一�?list[dict]
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    
                    # ReActAgent 格式: {"type": "image_url", "image_url": {"url": path}}
                    if item_type == "image_url" and isinstance(item.get("image_url"), dict):
                        url = item["image_url"].get("url") or item["image_url"].get("image")
                        if url is not None:
                            new_content.append({"type": "image", "url": url})
                    
                    # 兼容格式: {"type": "image", "image": path}
                    elif item_type == "image" and "image" in item:
                        new_content.append({"type": "image", "url": item["image"]})
                    
                    # 兼容格式: {"type": "image", "url": path}
                    elif item_type == "image" and "url" in item:
                        new_content.append(item)
                        
                    else:
                        # 文本或其他保持原�?
                        new_content.append(item)
            else:
                continue

            converted.append({"role": role, "content": new_content})

        return converted

    def call_model(self, messages: List[Dict[str, Any]], max_new_tokens: int = 512, **generate_kwargs) -> str:
        try:
            if not hasattr(self, "model") or not hasattr(self, "processor"):
                return "Error: UI-TARS model not loaded"

            model_messages = self._convert_messages(messages)

            inputs = self.processor.apply_chat_template(
                model_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # 将输入移到模型所在设�?
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )

            # 只解码新生成的部�?
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]

            # Token counting
            input_tokens = inputs['input_ids'].numel()
            output_tokens = generated_ids.numel()
            self.last_call_tokens = input_tokens + output_tokens
            self.total_tokens += self.last_call_tokens

            text = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return text
        except Exception as e:
            print(f"�?Error in UI-TARS inference: {e}")
            return f"Error: {str(e)}"


class GLMFlashBackend(BASE):
    """zai-org/GLM-4.6V-Flash 后端封装�?""

    def __init__(
        self,
        model_name: str = "zai-org/GLM-4.6V-Flash",
        SYSTEM_PROMPT: str | None = None,
        tools: List[Callable] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, SYSTEM_PROMPT=SYSTEM_PROMPT, tools=tools or [], device=device or "cpu")

        # 自动选择设备
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        try:
            print(f"Loading GLM-Flash model: {model_name} on {self.device}")
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = Glm4vForConditionalGeneration.from_pretrained(
                model_name, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            if not torch.cuda.is_available():
                self.model.to(self.device)
            self.model.eval()
            print("�?GLM-Flash model loaded successfully")
        except Exception as e:
            print(f"�?Error loading GLM-Flash model: {e}")
            raise

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """�?ReActAgent 使用�?messages 结构转换�?GLM-Flash 所需格式�?""
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 统一�?list[dict]
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    
                    # ReActAgent 格式: {"type": "image_url", "image_url": {"url": path}}
                    if item_type == "image_url" and isinstance(item.get("image_url"), dict):
                        url = item["image_url"].get("url") or item["image_url"].get("image")
                        if url is not None:
                            new_content.append({"type": "image", "url": url})
                    
                    # 兼容格式: {"type": "image", "image": path}
                    elif item_type == "image" and "image" in item:
                        new_content.append({"type": "image", "url": item["image"]})
                    
                    # 兼容格式: {"type": "image", "url": path}
                    elif item_type == "image" and "url" in item:
                        new_content.append(item)
                        
                    else:
                        # 文本或其他保持原�?
                        new_content.append(item)
            else:
                continue

            converted.append({"role": role, "content": new_content})

        return converted

    def call_model(self, messages: List[Dict[str, Any]], max_new_tokens: int = 512, **generate_kwargs) -> str:
        try:
            if not hasattr(self, "model") or not hasattr(self, "processor"):
                return "Error: GLM-Flash model not loaded"

            model_messages = self._convert_messages(messages)

            inputs = self.processor.apply_chat_template(
                model_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            # TODO: GLM-4V models usually don't need token_type_ids
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )

            # 只解码新生成的部�?
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]

            # Token counting
            input_tokens = inputs['input_ids'].numel()
            output_tokens = generated_ids.numel()
            self.last_call_tokens = input_tokens + output_tokens
            self.total_tokens += self.last_call_tokens

            text = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return text
        except Exception as e:
            print(f"�?Error in GLM-Flash inference: {e}")
            return f"Error: {str(e)}"
    
    
class Holo2Backend(BASE):
    """Hcompany/Holo2-4B 后端封装�?""

    def __init__(
        self,
        model_name: str = "Hcompany/Holo2-4B",
        SYSTEM_PROMPT: str | None = None,
        tools: List[Callable] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, SYSTEM_PROMPT=SYSTEM_PROMPT, tools=tools or [], device=device or "cpu")

        # 自动选择设备
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        try:
            print(f"Loading Holo2 model: {model_name} on {self.device}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("�?Holo2 model loaded successfully")
        except Exception as e:
            print(f"�?Error loading Holo2 model: {e}")
            raise

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """�?ReActAgent 使用�?messages 结构转换�?Holo2 所需格式�?""
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    
                    if item_type == "image_url" and isinstance(item.get("image_url"), dict):
                        url = item["image_url"].get("url") or item["image_url"].get("image")
                        if url is not None:
                            new_content.append({"type": "image", "url": url})
                    elif item_type == "image" and "image" in item:
                        new_content.append({"type": "image", "url": item["image"]})
                    else:
                        new_content.append(item)
            else:
                continue

            converted.append({"role": role, "content": new_content})

        return converted

    def call_model(self, messages: List[Dict[str, Any]], max_new_tokens: int = 512, **generate_kwargs) -> str:
        try:
            if not hasattr(self, "model") or not hasattr(self, "processor"):
                return "Error: Holo2 model not loaded"

            model_messages = self._convert_messages(messages)

            inputs = self.processor.apply_chat_template(
                model_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

            # 将输入移到模型所在设�?
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            elif isinstance(inputs, dict):
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **generate_kwargs,
                )

            # 只解码新生成的部�?
            input_len = inputs["input_ids"].shape[-1]
            generated_ids = outputs[0][input_len:]
            
            # Token counting
            input_tokens = inputs['input_ids'].numel()
            output_tokens = generated_ids.numel()
            self.last_call_tokens = input_tokens + output_tokens
            self.total_tokens += self.last_call_tokens
            
            text = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # --- Holo2 Format Translation (JSON to XML) ---
            # Holo2 often outputs JSON like {"click": [x, y]} instead of <action> tags.
            # We translate it here to keep ReActAgent clean.
            import re
            # Extract numbers from patterns like "click": [818, 895] or "click": ["818", "895"]
            json_click_pattern = r'["\']click["\']\s*:\s*\[\s*["\']?(?P<x>\d+)["\']?,\s*["\']?(?P<y>\d+)["\']?\s*\]'
            match = re.search(json_click_pattern, text)
            if match and "<action>" not in text:
                x, y = match.group("x"), match.group("y")
                text = f"<thought>Detected Holo2 JSON output, translating to XML.</thought>\n<action>click({x}, {y})</action>"
            # -----------------------------------------------

            return text
        except Exception as e:
            print(f"�?Error in Holo2 inference: {e}")
            return f"Error: {str(e)}"

