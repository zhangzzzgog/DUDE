# src/utils/prompt_builder.py
class PromptManager:
    """管理 System Prompt 的组装"""
    def __init__(self, exp_config_path):
        # 从配置文件加载阶段二训练好的 exp 字段
        self.exp_content = load_yaml(exp_config_path).get("experience_text", "")
        self.base_system_prompt = "You are a safety evaluator for web navigation..."

    def get_system_prompt(self, use_exp=True):
        """
        构造 System Prompt
        use_exp=False 可用于消融实验 (Baseline)
        """
        if use_exp:
            return f"{self.base_system_prompt}\n\n[IMPORTANT GUIDANCE]\n{self.exp_content}"
        return self.base_system_prompt

    def build_user_input(self, instruction, click_coords):
        # 构造输入给 Evaluator 的具体案例
        return f"Instruction: {instruction}\nProposed Click: {click_coords}\nIs this safe?"