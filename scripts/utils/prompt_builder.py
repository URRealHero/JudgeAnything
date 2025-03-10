from typing import Dict
from .config import LocalPrompts, SCORERUBRIC2PROMPT, PAIRRUBRIC2PROMPT, SYS_SCORING_PROMPT, SYS_PAIRING_PROMPT

class SystemPromptBuilder:
    @staticmethod
    def build_overall_scoring_prompt() -> str:
        """构建评分系统指令"""
        prompt = SYS_SCORING_PROMPT
        prompt += f"\n\nHere is the scoring rule:\n{SCORERUBRIC2PROMPT['overall_score']}"
        return prompt
    
    @staticmethod
    def build_rubric_scoring_prompt() -> str:
        """构建评分系统指令"""
        prompt = SYS_SCORING_PROMPT
        for rubric, pt in SCORERUBRIC2PROMPT["rubrics"].items():
            prompt += f"\n\nHere is the rubric for scoring based on {rubric}:\n{pt}"
        return prompt

    @staticmethod
    def build_overall_pairing_prompt() -> str:
        """构建配对比较系统指令"""
        prompt = SYS_PAIRING_PROMPT
        prompt += f"\n\nHere is the pairing rule:\n{PAIRRUBRIC2PROMPT['overall_score']}"
        return prompt
    
    @staticmethod
    def build_rubric_pairing_prompt() -> str:
        """构建配对比较系统指令"""
        prompt = SYS_PAIRING_PROMPT
        for rubric, pt in PAIRRUBRIC2PROMPT["rubrics"].items():
            prompt += f"\n\nHere is the rubric for pairing based on {rubric}:\n{pt}"
        return prompt

    @classmethod
    def get_local_overall_prompts(cls) -> Dict[str, str]:
        """获取本地模型专用提示"""
        local_prompts = LocalPrompts
        return {
            "score": f"{local_prompts.SYSTEM.value} {cls.build_overall_scoring_prompt()}",
            "pair": f"{local_prompts.SYSTEM.value} {cls.build_overall_pairing_prompt()}",
        }
    
    @classmethod
    def get_local_rubric_prompts(cls) -> Dict[str, str]:
        """获取本地模型专用提示"""
        local_prompts = LocalPrompts
        return {
            "score": f"{local_prompts.SYSTEM.value} {cls.build_rubric_scoring_prompt()}",
            "pair": f"{local_prompts.SYSTEM.value} {cls.build_rubric_pairing_prompt()}",
        }