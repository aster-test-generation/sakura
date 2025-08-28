from __future__ import annotations

from typing import Annotated, List, Literal, Dict, Optional, Union

from pydantic import BaseModel, Field

from .decomposition import AtomicBlock
from langchain_core.messages import BaseMessage


class AgentState(BaseModel):
    messages: Annotated[
        List[Union[BaseMessage, Dict[str, str]]], "List of messages (Human/AI/Tool)"
    ] = Field(default_factory=list)
    iterations: Annotated[int, "The current iteration number of the agent"] = 0
    atomic_blocks: Annotated[
        List[AtomicBlock], "The current state of the AtomicBlocks"
    ] = Field(default_factory=list)
    final_comments: Annotated[
        Optional[str], "Final comments from the agent, if any"
    ] = ""


class QueryMethodArgs(BaseModel):
    qualified_class_name: str = Field(
        ...,
        description="The qualified class name of the class implementing the method.",
    )
    method_signature: str = Field(
        ..., description="The method signature of the method to query."
    )


class QueryClassArgs(BaseModel):
    qualified_class_name: str = Field(
        ..., description="The qualified class name of the class to query."
    )


class QueryVectorDataArgs(BaseModel):
    query: str = Field(..., description="The query string to search for.")
    i: int = Field(..., gt=0, description="1-based start rank, must be > 0.")
    j: int = Field(..., ge=1, description="1-based end rank (inclusive), must be >= i.")


class ReachableMethodsArgs(BaseModel):
    qualified_class_name: str = Field(
        ...,
        description="The fully qualified class name to list reachable methods from.",
    )
    visibility_mode: Literal["public", "same_package", "same_package_or_subclass"] = (
        Field(..., description="The visibility mode of the reachable methods.")
    )


class InstructionArgs(BaseModel):
    instructions: str = Field(..., description="The instructions for the modification.")
    current_blocks: List[AtomicBlock] = Field(
        ..., description="The current state of the AtomicBlocks."
    )
    # NOTE: We don't need to pass in the atomic blocks here for modify_atomic_blocks because we're using the state.atomic_blocks


class FinalizeBlocksArgs(BaseModel):
    comments: str = Field(
        ..., description="Comments about any problems with the procedure or concerns."
    )
    current_blocks: List[AtomicBlock] = Field(
        ..., description="The current state of the AtomicBlocks."
    )
    # NOTE: We don't need to pass in the atomic blocks here because we're using the state.atomic_blocks


class ModifyAtomicBlockNotesArgs(BaseModel):
    order: int = Field(..., description="The order of the atomic block to modify.")
    new_notes: str = Field(..., description="The new notes for the atomic block.")
