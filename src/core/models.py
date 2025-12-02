import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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


class PrompRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    context: str = Field(default="", max_length=50000)
    history: List[dict] = Field(default=[])

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question is empty")
        return v.strip()

    @field_validator("history")
    @classmethod
    def validate_history(cls, v):
        if len(v) > 20:
            raise ValueError("History cannot contain more than 20 items")
        for item in v:
            if not isinstance(item, dict):
                raise ValueError("History items must be dictionaries")
            if "role" not in item or "content" not in item:
                raise ValueError("History items must have role and content fields")
        return v


class BaseRAGResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    processing_time: Optional[float] = Field(default=None, ge=0)
    timestamp: float = Field(default_factory=time.time)

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Answer is empty")
        return v


class SimpleRAGResponse(BaseRAGResponse):
    pass


class ConversationMessage(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=50000)
    timestamp: float = Field(default_factory=time.time)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Message content is empty")
        return v.strip()


class ConversationHistory(BaseModel):
    messages: List[ConversationMessage] = Field(default_factory=list)
    max_history_pairs: int = Field(default=5, ge=1, le=20)

    def add_message(self, role: str, content: str) -> None:
        message = ConversationMessage(role=role, content=content)
        self.messages.append(message)
        self._trim_history()

    def _trim_history(self) -> None:
        max_messages = self.max_history_pairs * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_conversation_context(self) -> str:
        context_parts = []
        for msg in self.messages:
            role = msg.role.capitalize()
            context_parts.append(f"{role}: {msg.content}")
        return "\n".join(context_parts)

    def clear(self) -> None:
        self.messages.clear()

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if len(v) > 100:
            raise ValueError("Messages list cannot exceed 100 items")
        return v


class PerformanceMetrics(BaseModel):
    endpoint: str = Field(...)
    total_time: float = Field(..., ge=0)
    rag_time: float = Field(default=0.0, ge=0)
    llm_time: float = Field(default=0.0, ge=0)
    history_load_time: float = Field(default=0.0, ge=0)
    history_save_time: float = Field(default=0.0, ge=0)
    context_length: int = Field(default=0, ge=0)
    response_length: int = Field(default=0, ge=0)
    timestamp: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def validate_time_consistency(self):
        total = self.total_time
        rag = self.rag_time
        llm = self.llm_time
        hist_load = self.history_load_time
        hist_save = self.history_save_time

        min_expected = rag + llm + hist_load + hist_save
        if total < min_expected:
            raise ValueError(f"Total time {total} is less than sum of components {min_expected}")
        return self


class GroundTruthSource(BaseModel):
    answer: Optional[str] = None
    source: str
    quote: str


class TestQuestion(BaseModel):
    id: str
    question: str
    category: str
    ground_truth_sources: Dict[str, GroundTruthSource] = Field(default_factory=dict)
    correct_answer: str
    expected_keywords: List[str] = Field(default_factory=list)
    expected_documents: List[str] = Field(default_factory=list)
    key_points: List[str] = Field(default_factory=list)
    ground_truth_contexts: List[str] = Field(default_factory=list)
    key_entities: List[str] = Field(default_factory=list)
    key_facts: List[str] = Field(default_factory=list)


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
