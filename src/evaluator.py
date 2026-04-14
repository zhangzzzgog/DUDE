# src/core/evaluator.py
class EvaluatorModel:
    """封装 Evaluator 模型交互"""
    def __init__(self, model_client, prompt_manager):
        self.client = model_client
        self.prompts = prompt_manager
    
    def evaluate(self, screenshot, instruction, action):
        """
        输入: 截图, 指令, Agent动作
        输出: (bool: 是否通过, float: 置信度, str: 原因)
        """
        system_prompt = self.prompts.get_system_prompt(use_exp=True)
        user_content = self.prompts.build_user_input(instruction, action)
        
        # 调用大模型 (LLM/VLM)
        response = self.client.chat(
            images=[screenshot],
            system=system_prompt,
            user=user_content
        )
        
        # 解析输出 (假设输出格式为 JSON 或特定格式)
        # e.g., {"feasible": "yes", "confidence": 0.95}
        return self._parse_response(response)