from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str
    params: dict | None = None


class IntentCategory(str, Enum):
    TOP = "top"
    GENERAL = "general"
    USAGE = "usage"
    DEFINITION = "definition"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    EXCEPTION = "exception"


@dataclass
class IntentAnalysis:
    primary_intent: IntentCategory
    confidence: float
    scores: Dict[str, float]
    enhanced: bool = True



class ClaudeStats(BaseModel):
    answer: str = ""
    time: float = 0
    tokens: int = 0
    context_tokens: int = 0
    used_context: List[str] = Field(default_factory=list)


class JunieStats(BaseModel):
    answer: str = ""
    time: float = 0
    tokens: int = 0
    context_tokens: Optional[int] = None
    used_context: Optional[List[str]] = Field(default_factory=list)


class JunieStatsStructure(BaseModel):
    with_mcp: JunieStats = Field(default_factory=JunieStats)
    without_mcp: JunieStats = Field(default_factory=JunieStats)


class TestQuestion(BaseModel):
    id: str
    question: str
    category: str
    ground_truth_contexts: List[str] = Field(default_factory=list)
    key_facts: List[str] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    claude_stats: ClaudeStats = Field(default_factory=ClaudeStats)
    junie_stats: JunieStatsStructure = Field(default_factory=JunieStatsStructure)
    comparisons: Optional[dict] = Field(default_factory=dict)


class EvaluationCriteria(BaseModel):
    answer_completeness: Optional[str] = None
    factual_accuracy: Optional[str] = None
    specific_values: Optional[str] = None
    no_truncation: Optional[str] = None


class GroundTruthTestSuite(BaseModel):
    test_suite: str
    version: str
    created: str
    description: Optional[str] = None
    questions: List[TestQuestion]
    evaluation_criteria: Optional[EvaluationCriteria] = None

    @classmethod
    def load_from_file(cls, filepath: str) -> "GroundTruthTestSuite":
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def save_to_file(self, filepath: str):
        import json

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, ensure_ascii=False, indent=2)
