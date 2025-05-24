# scripts/utils/config.py
from pathlib import Path
from enum import Enum
from .Scoring_Prompt import SYS_PROMPT as SYS_SCORING_PROMPT, OVERALL_SCORE_PROMPT as OVERALL_SCORE_SCORING_PROMPT,RELEVANCE_PROMPT as RELEVANCE_SCORING_PROMPT,TRUSTWORTHINESS_PROMPT as TRUSTWORTHINESS_SCORING_PROMPT, CREATIVITY_PROMPT as CREATIVITY_SCORING_PROMPT, CLARITY_PROMPT as CLARITY_SCORING_PROMPT, COHERENCE_PROMPT as COHERENCE_SCORING_PROMPT, COMPLETENESS_PROMPT as COMPLETENESS_SCORING_PROMPT
from .Pairing_Prompt import SYS_PROMPT as SYS_PAIRING_PROMPT, OVERALL_SCORE_PROMPT as OVERALL_SCORE_PAIRING_PROMPT,RELEVANCE_PROMPT as RELEVANCE_PAIRING_PROMPT,TRUSTWORTHINESS_PROMPT as TRUSTWORTHINESS_PAIRING_PROMPT, CREATIVITY_PROMPT as CREATIVITY_PAIRING_PROMPT, CLARITY_PROMPT as CLARITY_PAIRING_PROMPT, COHERENCE_PROMPT as COHERENCE_PAIRING_PROMPT, COMPLETENESS_PROMPT as COMPLETENESS_PAIRING_PROMPT


# 路径配置
ROOT_DIR = Path(__file__).resolve().parents[2]

# Step 2: use ROOT_DIR to build all other paths
BASE_DIR = ROOT_DIR / "dataset" / "X2XRawBenchmark"
BENCHMARK_FILE = BASE_DIR / "X2XBenchmark.json"

RESULT_DIR = ROOT_DIR / "dataset" / "ResponseCollection"
RESULT_FILE = RESULT_DIR / "X2XBenchmarkResponse.json"

CHECKLIST_DIR = ROOT_DIR / "dataset" / "Checklist"
CHECKLIST_FILE = CHECKLIST_DIR / "checklist.json"

FILE_STORAGE = BASE_DIR.parent / ".cache" / "file_storage.json"

# API Config
GOOGLE_MODELS = {
    "pro": "gemini-1.5-pro",
    "flash": "gemini-2.0-flash",
    "lite": "gemini-2.0-flash-lite-preview-02-05"
}

# Local Config
LOCAL_MODEL_CONFIG = {
    "max_new_tokens": 1500,
    "temperature": 0.7
}

# Prompt template
class LocalPrompts(Enum):
    SYSTEM = '<|system|>'
    USER = '<|user|>'
    ASSISTANT = '<|assistant|>'
    END = '<|end|>'
    
RUBRICS = ["relevance", "trustworthiness", "creativity", "clarity", "coherence", "completeness"]

SCORERUBRIC2PROMPT = {
    "overall_score": OVERALL_SCORE_SCORING_PROMPT,
    "rubrics":{
        "relevance": RELEVANCE_SCORING_PROMPT,
        "trustworthiness": TRUSTWORTHINESS_SCORING_PROMPT,
        "creativity": CREATIVITY_SCORING_PROMPT,
        "clarity": CLARITY_SCORING_PROMPT,
        "coherence": COHERENCE_SCORING_PROMPT,
        "completeness": COMPLETENESS_SCORING_PROMPT
    }
}

PAIRRUBRIC2PROMPT = {
    "overall_score": OVERALL_SCORE_PAIRING_PROMPT,
    "rubrics":{
        "relevance": RELEVANCE_PAIRING_PROMPT,
        "trustworthiness": TRUSTWORTHINESS_PAIRING_PROMPT,
        "creativity": CREATIVITY_PAIRING_PROMPT,
        "clarity": CLARITY_PAIRING_PROMPT,
        "coherence": COHERENCE_PAIRING_PROMPT,
        "completeness": COMPLETENESS_PAIRING_PROMPT
    }
}

