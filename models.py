from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any
from enum import Enum
import time


class QuestionCategory(str, Enum):
    GENERAL = "general"
    MEDIUM = "medium"
    SPECIFIC = "specific"
    USAGE = "usage"
    DEFINITION = "definition"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    EXCEPTION = "exception"


class NodeKind(str, Enum):
    CLASS = "CLASS"
    METHOD = "METHOD"
    FUNCTION = "FUNCTION"
    VARIABLE = "VARIABLE"
    MODULE = "MODULE"
    IMPORT = "IMPORT"
    UNKNOWN = "UNKNOWN"


class PrompRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    context: str = Field(default="", max_length=50000)
    history: List[dict] = Field(default_factory=list, max_items=20)

    @field_validator('question')
    @classmethod
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Question is empty')
        return v.strip()

    @field_validator('history')
    @classmethod
    def validate_history(cls, v):
        for item in v:
            if not isinstance(item, dict):
                raise ValueError('History items must be dictionaries')
            if 'role' not in item or 'content' not in item:
                raise ValueError('History items must have role and content fields')
        return v


class BaseRAGResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    processing_time: Optional[float] = Field(default=None, ge=0)
    timestamp: float = Field(default_factory=time.time)

    @field_validator('answer')
    @classmethod
    def answer_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Answer is empty')
        return v


class SimpleRAGResponse(BaseRAGResponse):
    pass



class NodeRAGResponse(BaseRAGResponse):
    used_context: str = Field(default="")
    question_category: Optional[QuestionCategory] = None


class ChromaDBQueryResult(BaseModel):
    ids: List[List[str]] = Field(..., description="List of document IDs for each query")
    distances: List[List[float]] = Field(..., description="Cosine distances for each result")
    metadatas: List[List[Optional[Dict[str, Any]]]] = Field(..., description="Metadata for each document")
    documents: List[List[Optional[str]]] = Field(..., description="Document contents")

    @model_validator(mode='after')
    def validate_consistent_lengths(self):
        ids = self.ids
        distances = self.distances
        metadatas = self.metadatas
        documents = self.documents
        if not ids:
            raise ValueError('ids is empty')
        outer_lengths = [len(ids), len(distances), len(metadatas), len(documents)]
        if len(set(outer_lengths)) > 1:
            raise ValueError(f'Inconsistent outer lengths: {outer_lengths}')
        for i in range(len(ids)):
            inner_lengths = [
                len(ids[i]),
                len(distances[i]),
                len(metadatas[i]),
                len(documents[i])
            ]
            if len(set(inner_lengths)) > 1:
                raise ValueError(f'Inconsistent inner lengths for query {i}: {inner_lengths}')
        return self

    @field_validator('distances')
    @classmethod
    def validate_distances(cls, v):
        for query_distances in v:
            for distance in query_distances:
                if not (0 <= distance <= 2):
                    raise ValueError(f'Invalid cosine distance: {distance}')
        return v


class ChromaDBGetResult(BaseModel):
    ids: List[str] = Field(..., description="List of document IDs")
    metadatas: List[Optional[Dict[str, Any]]] = Field(..., description="Metadata for each document")
    documents: List[Optional[str]] = Field(..., description="Document contents")

    @model_validator(mode='after')
    def validate_consistent_lengths(self):
        ids = self.ids
        metadatas = self.metadatas
        documents = self.documents
        lengths = [len(ids), len(metadatas), len(documents)]
        if len(set(lengths)) > 1:
            raise ValueError(f'Inconsistent lengths: {lengths}')
        return self



class ImportanceScores(BaseModel):
    loc: float = Field(default=0.0, description="Lines of code score")
    out_degree: float = Field(default=0.0, description="Out-degree centrality")
    in_degree: float = Field(default=0.0, description="In-degree centrality")
    pagerank: float = Field(default=0.0, description="PageRank centrality")
    eigenvector: float = Field(default=0.0, description="Eigenvector centrality")
    katz: float = Field(default=0.0, description="Katz centrality")
    combined: float = Field(default=0.0, description="Combined importance score")

    @field_validator('*')
    @classmethod
    def scores_non_negative(cls, v):
        if v < 0:
            raise ValueError('Importance scores is negative')
        return v


class NodeMetadata(BaseModel):
    node: str = Field(..., description="Unique node identifier")
    kind: NodeKind = Field(default=NodeKind.UNKNOWN)
    label: str = Field(..., min_length=1)
    related_entities: List[str] = Field(default_factory=list, description="List of related node IDs")
    importance: ImportanceScores = Field(default_factory=ImportanceScores)
    uri: Optional[str] = Field(default=None, description="File URI")
    location: Optional[str] = Field(default=None, description="Location in file (line:col)")

    @field_validator('node')
    @classmethod
    def node_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Node ID is empty')
        return v.strip()

    @field_validator('label')
    @classmethod
    def label_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Label is empty')
        return v.strip()

    @field_validator('related_entities')
    @classmethod
    def validate_related_entities(cls, v):
        return list(set(entity.strip() for entity in v if entity.strip()))




class ConversationMessage(BaseModel):
    role: str = Field(..., pattern=r'^(user|assistant|system)$')
    content: str = Field(..., min_length=1, max_length=50000)
    timestamp: float = Field(default_factory=time.time)

    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Message content is empty')
        return v.strip()


class ConversationHistory(BaseModel):
    messages: List[ConversationMessage] = Field(default_factory=list, max_items=100)
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

    @model_validator(mode='after')
    def validate_time_consistency(self):
        total = self.total_time
        rag = self.rag_time
        llm = self.llm_time
        hist_load = self.history_load_time
        hist_save = self.history_save_time

        min_expected = rag + llm + hist_load + hist_save
        if total < min_expected:
            raise ValueError(f'Total time {total} is less than sum of components {min_expected}')
        return self