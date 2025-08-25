from __future__ import annotations

from enum import Enum
from typing import Annotated, List, Dict, Optional, Tuple, Any, Union, Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, ConfigDict


class SnippetType(Enum):
    CLASS = "class"
    METHOD = "method"


class Snippet(BaseModel):
    implementing_class_name: str  # The class that directly implements the method


class MethodSnippet(Snippet):
    code: str
    method_signature: str
    containing_class_name: str  # The class that inherits/contains the method (class under analysis)


class ClassSnippet(Snippet):
    simple_class_name: str


class AgentState(BaseModel):
    messages: Annotated[List[Union[BaseMessage, Dict[str, str]]], "List of messages (Human/AI/Tool)"] = Field(
        default_factory=list)
    iterations: Annotated[int, "The current iteration number of the agent"] = 0
    atomic_blocks: Annotated[List[AtomicBlock], "The current state of the AtomicBlocks"] = Field(default_factory=list)
    final_comments: Annotated[Optional[str], "Final comments from the agent, if any"] = ""


class PrepPhrase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preposition: str
    object: str


class GrammaticalBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order: int  # For sequencing
    subjects: List[str]
    verbs: List[str]  # In preprocessing, we break this down into one verb per block. We include past participle here.
    past_participles: List[str]
    direct_objs: List[str]
    indirect_objs: List[str]
    prep_phrases: List[PrepPhrase]
    polarity: Literal["positive", "negative"]
    conditions: List[str]
    simplified: str


class CandidateMethod(BaseModel):
    implementing_class_name: str  # The class that directly implements the method
    containing_class_name: str  # The class that inherits/contains the method (class under analysis)
    method_signature: str


class AtomicBlock(GrammaticalBlock):
    candidate_methods: List[CandidateMethod]
    notes: str

    @classmethod
    def from_grammatical_block(
            cls,
            gb: GrammaticalBlock,
            *,
            candidate_methods: List[CandidateMethod] | None = None,
            notes: str = ""
    ) -> "AtomicBlock":
        cm = list(candidate_methods) if candidate_methods is not None else []
        return cls(**gb.model_dump(), candidate_methods=cm, notes=notes)


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

    model_config = ConfigDict(use_enum_values=True)


class QueryMethodArgs(BaseModel):
    qualified_class_name: str = Field(..., description="The qualified class name of the class implementing the method.")
    method_signature: str = Field(..., description="The method signature of the method to query.")


class QueryClassArgs(BaseModel):
    qualified_class_name: str = Field(..., description="The qualified class name of the class to query.")


class QueryVectorDataArgs(BaseModel):
    query: str = Field(..., description="The query string to search for.")
    i: int = Field(..., gt=0, description="1-based start rank, must be > 0.")
    j: int = Field(..., ge=1, description="1-based end rank (inclusive), must be >= i.")


class ReachableMethodsArgs(BaseModel):
    qualified_class_name: str = Field(..., description="The fully qualified class name to list reachable methods from.")
    visibility_mode: Literal["public", "same_package", "same_package_or_subclass"] = Field(...,
                                                                                           description="The visibility mode of the reachable methods.")


class InstructionArgs(BaseModel):
    instructions: str = Field(..., description="The instructions for the modification.")
    # NOTE: We don't need to pass in the atomic blocks here for modify_atomic_blocks because we're using the state.atomic_blocks


class FinalizeBlocksArgs(BaseModel):
    comments: str = Field(..., description="Comments about any problems with the procedure or concerns.")
    # NOTE: We don't need to pass in the atomic blocks here because we're using the state.atomic_blocks


class ModifyAtomicBlockNotesArgs(BaseModel):
    order: int = Field(..., description="The order of the atomic block to modify.")
    new_notes: str = Field(..., description="The new notes for the atomic block.")
