from enum import Enum
from typing_extensions import TypedDict


class Choice(Enum):
    first = "0"
    second = "1"
    third = "2"

class ScoringOverall(TypedDict):
    overall_score: int
    overall_comment: str

class ScoringRubrics(TypedDict):
    relevance_score: int
    relevance_comment: str
    trustworthiness_score: int
    trustworthiness_comment: str
    creativity_score: int
    creativity_comment: str
    clarity_score: int
    clarity_comment: str
    coherence_score: int
    coherence_comment: str
    completeness_score: int
    completeness_comment: str

class PairingOverall(TypedDict):
    overall_choice: Choice
    overall_comment: str

class PairingRubrics(TypedDict):
    relevance_choice: Choice
    relevance_comment: str
    trustworthiness_choice: Choice
    trustworthiness_comment: str
    creativity_choice: Choice 
    creativity_comment: str
    clarity_choice: Choice
    clarity_comment: str
    coherence_choice: Choice
    coherence_comment: str
    completeness_choice: Choice
    completeness_comment: str