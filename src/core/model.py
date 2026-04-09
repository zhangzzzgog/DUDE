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

        
class Local(BASE):
    def __init__(self, model_name: str = "Qwen2.5-VL-3B-Instruct", SYSTEM_PROMPT: str = None, tools: List = None, model_path: str = None):
        """Initialize a local agent wrapper.

        Args:
            model_name: base model identifier
            SYSTEM_PROMPT: system prompt text
            tools: tool list
            model_path: optional path to finetuned weights
        """
        super().__init__(model_name, SYSTEM_PROMPT, tools or [])

        # Initialize model, processor, and optional finetuned weights.
        self.model = None
        self.processor = None
        self.model_path = model_path

        # Load the base model first.
        self._load_base_model(model_name)

        # Load finetuned weights when a valid path is provided.
        if model_path and os.path.exists(model_path):
            self._load_finetuned_model(model_path)

        # Initialize conversation context.
        self.context = [{"role": "system", "content": self.system_prompt or ""}]
    
    def _load_base_model(self, model_name: str):
        """Load the base model backend."""
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
            elif "ui-tars" in name_lower or "ui_tars" in name_lower:
                # Support the UI-TARS family as a local base model.
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
                print(f"Unsupported model: {model_name}")
                return

            # Add a safe batch decoder for local multimodal models.
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

            print("Base model loaded successfully")
        except Exception as e:
            print(f"Error loading base model: {e}")
            raise
    def _load_finetuned_model(self, model_path: str):
        """Load finetuned weights on top of the current base model."""
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
                # Use AutoModelForVision2Seq as the base for UI-TARS models.
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
                # todo: how to manage it more greatly?
                print(f"Unsupported model: {lower_name}")
                return

            # Load LoRA / PEFT weights on top of the selected base model.
            self.model = PeftModel.from_pretrained(base_model, model_path)
            print("Finetuned model loaded successfully")

        except Exception as e:
            print(f"Error loading finetuned model: {e}")
            print("Using base model instead")
    
    def call_model(self, messages):
        """Generate a response from the current model."""
        try:
            if self.model is None or self.processor is None:
                return "Error: Model not loaded"
            
            # Apply the chat template.
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            # Recursively move all tensors to the model device to avoid device mismatch.
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
                    model_device = getattr(self.model, "device", None)

                if model_device is not None:
                    # Prefer BatchEncoding.to(device) when available.
                    try:
                        if hasattr(inputs, "to"):
                            inputs = inputs.to(model_device)
                        else:
                            inputs = move_to_device(inputs, model_device)
                    except Exception:
                        # Fall back to recursive movement.
                        inputs = move_to_device(inputs, model_device)

                    # Print remaining device mismatches for debugging.
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
                            print("Device mismatch detected in inputs (field -> device):")
                            for field, dev in devs:
                                print(f"  {field} -> {dev}")
                    except Exception:
                        pass
            except Exception:
                # If movement fails, fall back to the original shallow move behavior.
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            
            # Generate the response.
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            
            # Trim the input portion from generated ids.
            if "input_ids" in inputs:
                input_ids = inputs["input_ids"]
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                ]
            else:
                generated_ids_trimmed = generated_ids
            
            # Decode the generated response.
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Token counting
            input_tokens = inputs["input_ids"].numel() if "input_ids" in inputs else 0
            output_tokens = sum(len(ids) for ids in generated_ids_trimmed)
            self.last_call_tokens = input_tokens + output_tokens
            self.total_tokens += self.last_call_tokens
            
            return output_text[0] if output_text else ""
            
        except Exception as e:
            print(f"Error in model inference: {e}")
            return f"Error: {str(e)}"
    
    def run(self, user_input: str = None, image_paths: List[str] = None):
        """Run inference with optional text and images."""
        messages = []
        
        # Add the system prompt.
        system_content = [{"type": "text", "text": self.system_prompt or ""}]
        messages.append({"role": "system", "content": system_content})
        
        # Add user input.
        user_content = []
        
        # Process image inputs.
        if image_paths:
            for img_path in image_paths:
                # Check whether the image path exists.
                if os.path.exists(img_path):
                    user_content.append({
                        "type": "image",
                        "image": img_path
                    })
                else:
                    print(f"Image not found: {img_path[:10]}")
        
        # Process text input.
        if user_input:
            user_content.append({
                "type": "text",
                "text": user_input
            })
        
        if user_content:
            messages.append({"role": "user", "content": user_content})
        
        # Call the model.
        response = self.call_model(messages)
        
        # Update conversation context.
        self.context.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })
        
        return response
    
    def update_system_prompt(self, new_system_prompt: str):
        """Update the system prompt."""
        self.system_prompt = new_system_prompt
        
        # Update the first context message when it exists.
        if self.context and self.context[0].get("role") == "system":
            self.context[0] = {"role": "system", "content": [{"type": "text", "text": new_system_prompt}]}
        else:
            self.context.insert(0, {"role": "system", "content": [{"type": "text", "text": new_system_prompt}]})
        
        print("System prompt updated")
    
    def load_model_from_path(self, model_path: str):
        """Load a finetuned model from a local path."""
        if model_path and os.path.exists(model_path):
            self._load_finetuned_model(model_path)
            self.model_path = model_path
        else:
            print(f"Model path not found or invalid: {model_path}")
    
    def get_model_info(self) -> dict:
        """Get model metadata."""
        info = {
            "model_loaded": self.model is not None,
            "processor_loaded": self.processor is not None,
            "model_path": self.model_path,
            "device": str(self.model.device) if self.model and hasattr(self.model, "device") else "unknown"
        }
        
        if self.model and hasattr(self.model, "config"):
            info["model_name"] = self.model.config._name_or_path
        
        return info
class GLM(BASE):
    """Wrapper for Zhipu chat models accessed through the official SDK."""

    def __init__(self, model_name: str, api_key: str = None, SYSTEM_PROMPT: str = None, tools: List = None):
        super().__init__(model_name, SYSTEM_PROMPT, tools)
        self.api_key = require_zhipuai_api_key(api_key)
        self.client = ZhipuAiClient(api_key=self.api_key)

    def call_model(self, messages, think: bool = False):
        """Call the remote GLM model and return plain text content."""
        thinking_mode = "enabled" if think else "disabled"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            thinking={
                "type": thinking_mode
            }
        )

        # Token counting
        if hasattr(response, "usage") and response.usage:
            self.last_call_tokens = getattr(response.usage, "total_tokens", 0)
            self.total_tokens += self.last_call_tokens

        return response.choices[0].message.content



class Qwen3VLBackend(BASE):
    """Backend wrapper for local Qwen3-VL models used by ReActAgent.

    Provides a unified call_model(messages) -> str interface.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        SYSTEM_PROMPT: str | None = None,
        tools: List[Callable] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, SYSTEM_PROMPT=SYSTEM_PROMPT, tools=tools or [], device=device or "cpu")

        # Auto-select the device unless one is explicitly provided.
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
            print("Qwen3 VL model loaded successfully")
        except Exception as e:
            print(f"Error loading Qwen3 VL model: {e}")
            raise
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert ReActAgent messages into the Qwen3-VL input format.

        Main conversions:
        - system.content string -> [{"type": "text", "text": ...}]
        - user image_url entries -> {"type": "image", "url": ...}
        """
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Normalize content to list[dict].
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    # From ReActAgent.run: {"type": "image_url", "image_url": {"url": path}}
                    if item_type == "image_url" and isinstance(item.get("image_url"), dict):
                        url = item["image_url"].get("url") or item["image_url"].get("image")
                        if url is not None:
                            new_content.append({"type": "image", "url": url})
                    # Support existing {"type": "image", "image": path} entries.
                    elif item_type == "image" and "image" in item:
                        new_content.append({"type": "image", "url": item["image"]})
                    else:
                        # Keep other content unchanged, such as text items.
                        new_content.append(item)
            else:
                # Skip unsupported content structures.
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

            # Move inputs to the model device.
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

            # Decode only the newly generated tokens.
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
            print(f"Error in Qwen3 VL inference: {e}")
            return f"Error: {str(e)}"


class UITARSBackend(BASE):
    """Backend wrapper for ByteDance UI-TARS-1.5-7B."""

    def __init__(
        self,
        model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B",
        SYSTEM_PROMPT: str | None = None,
        tools: List[Callable] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, SYSTEM_PROMPT=SYSTEM_PROMPT, tools=tools or [], device=device or "cpu")

        # Auto-select the device.
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
            print("UI-TARS model loaded successfully")
        except Exception as e:
            print(f"Error loading UI-TARS model: {e}")
            raise

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert ReActAgent messages into the UI-TARS input format."""
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Normalize content to list[dict].
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    
                    # From ReActAgent: {"type": "image_url", "image_url": {"url": path}}
                    if item_type == "image_url" and isinstance(item.get("image_url"), dict):
                        url = item["image_url"].get("url") or item["image_url"].get("image")
                        if url is not None:
                            new_content.append({"type": "image", "url": url})
                    
                    # Support {"type": "image", "image": path}.
                    elif item_type == "image" and "image" in item:
                        new_content.append({"type": "image", "url": item["image"]})
                    
                    # Support {"type": "image", "url": path}.
                    elif item_type == "image" and "url" in item:
                        new_content.append(item)
                        
                    else:
                        # Keep text and other content unchanged.
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

            # Move inputs to the model device.
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

            # Decode only the newly generated tokens.
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
            print(f"Error in UI-TARS inference: {e}")
            return f"Error: {str(e)}"


class GLMFlashBackend(BASE):
    """Backend wrapper for zai-org/GLM-4.6V-Flash."""

    def __init__(
        self,
        model_name: str = "zai-org/GLM-4.6V-Flash",
        SYSTEM_PROMPT: str | None = None,
        tools: List[Callable] | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(model_name=model_name, SYSTEM_PROMPT=SYSTEM_PROMPT, tools=tools or [], device=device or "cpu")

        # Auto-select the device.
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
            print("GLM-Flash model loaded successfully")
        except Exception as e:
            print(f"Error loading GLM-Flash model: {e}")
            raise

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert ReActAgent messages into the GLM-Flash input format."""
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # Normalize content to list[dict].
            if isinstance(content, str):
                new_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    
                    # From ReActAgent: {"type": "image_url", "image_url": {"url": path}}
                    if item_type == "image_url" and isinstance(item.get("image_url"), dict):
                        url = item["image_url"].get("url") or item["image_url"].get("image")
                        if url is not None:
                            new_content.append({"type": "image", "url": url})
                    
                    # Support {"type": "image", "image": path}.
                    elif item_type == "image" and "image" in item:
                        new_content.append({"type": "image", "url": item["image"]})
                    
                    # Support {"type": "image", "url": path}.
                    elif item_type == "image" and "url" in item:
                        new_content.append(item)
                        
                    else:
                        # Keep text and other content unchanged.
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

            # Decode only the newly generated tokens.
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
            print(f"Error in GLM-Flash inference: {e}")
            return f"Error: {str(e)}"
    
    
