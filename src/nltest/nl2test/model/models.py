from __future__ import annotations

from enum import Enum
from typing import Annotated, List, Dict, Optional, Tuple, Any, Union, Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class SnippetType(Enum):
    CLASS = "class"
    METHOD = "method"


class Snippet(BaseModel):
    qualified_class_name: str


class MethodSnippet(Snippet):
    code: str
    method_signature: str


class ClassSnippet(Snippet):
    simple_class_name: str


class AgentState(BaseModel):
    messages: Annotated[List[Union[BaseMessage, Dict[str, str]]], "List of messages (Human/AI/Tool)"] = []
    iterations: Annotated[int, "The current iteration number of the agent"] = 0
    atomic_blocks: Annotated[List[AtomicBlock], "The current state of the AtomicBlocks"] = []


class GrammaticalBlock(BaseModel):
    order: int  # For sequencing
    subjects: List[str] = []
    verbs: List[str]  # In preprocessing, we break this down into one verb per block. We include past participle here.
    past_participles: List[str] = []
    direct_objs: List[str] = []
    indirect_objs: List[str] = []
    prep_phrases: List[Dict[str, str]] = []
    polarity: Literal["positive", "negative"] = "positive"
    conditions: List[str] = []
    simplified: str = ""


class AtomicBlock(GrammaticalBlock):
    candidate_methods: List[Dict[str, str]] = []
    notes: str = ""


class AbstractionLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NL2TestInput(BaseModel):
    id: int = -1
    description: str
    project_name: str
    qualified_class_name: Annotated[str, "The qualified class name of the class containing the test"]
    method_signature: Annotated[str, "The method signature of the test case"]
    abstraction_level: AbstractionLevel | None = None
    is_bdd: bool = False
