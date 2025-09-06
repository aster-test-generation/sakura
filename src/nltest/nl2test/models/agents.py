from __future__ import annotations

from typing import Annotated, List, Literal, Dict, Optional, Union

from pydantic import BaseModel, Field

from .decomposition import (
    AtomicBlock,
    Scenario,
    AtomicBlockList,
    GrammaticalBlockList,
    LocalizedScenario,
)
from langchain_core.messages import BaseMessage


class AgentState(BaseModel):
    messages: Annotated[
        List[Union[BaseMessage, Dict[str, str]]], "List of messages (Human/AI/Tool)"
    ] = Field(default_factory=list)
    iterations: Annotated[int, "The current iteration number of the agent"] = 0
    tool_calls: Annotated[
        Dict[str, Dict[str, int]],
        "Mapping: tool name -> encoded argument -> count of calls",
    ] = Field(default_factory=dict)

    # Inputs
    grammatical_blocks: Annotated[
        Optional[GrammaticalBlockList],
        "Input grammatical blocks when using grammatical mode",
    ] = None
    scenario: Annotated[
        Optional[Scenario],
        "Input Scenario when using Gherkin mode",
    ] = None

    # For test packaging
    package: Annotated[
        Optional[str], "The package selected by the agent for the test code."
    ] = None
    class_name: Annotated[
        Optional[str], "The test class name selected by the agent."
    ] = None

    # Outputs
    atomic_blocks: Annotated[
        Optional[AtomicBlockList], "The current state of the AtomicBlocks"
    ] = None
    localized_scenario: Annotated[
        Optional[LocalizedScenario],
        "Finalized LocalizedScenario produced by localization",
    ] = None
    final_comments: Annotated[
        Optional[str], "Final comments from the agent, if any"
    ] = ""


class QueryMethodArgs(BaseModel):
    qualified_class_name: Annotated[
        str,
        Field(
            description="The qualified class name of the class implementing the method."
        ),
    ]
    method_signature: Annotated[
        str, Field(description="The method signature of the method to query.")
    ]


class QueryClassArgs(BaseModel):
    qualified_class_name: Annotated[
        str, Field(description="The qualified class name of the class to query.")
    ]


class QueryVectorDataArgs(BaseModel):
    query: Annotated[str, Field(description="The query string to search for.")]
    i: Annotated[int, Field(gt=0, description="1-based start rank, must be > 0.")]
    j: Annotated[
        int,
        Field(ge=1, description="1-based end rank (inclusive), must be >= i."),
    ]


class ReachableMethodsArgs(BaseModel):
    qualified_class_name: Annotated[
        str,
        Field(
            description="The fully qualified class name to list reachable methods from."
        ),
    ]
    visibility_mode: Annotated[
        Literal["public", "same_package", "same_package_or_subclass"],
        Field(description="The visibility mode of the reachable methods."),
    ]


class InstructionArgs(BaseModel):
    instructions: Annotated[
        str, Field(description="The instructions for the modification.")
    ]
    current_blocks: Annotated[
        AtomicBlockList, Field(description="The current state of the AtomicBlocks.")
    ]
    # NOTE: We don't need to pass in the atomic blocks here for modify_atomic_blocks because we're using the state.atomic_blocks


class TestCodeArgs(BaseModel):
    """Arguments for providing raw test code directly to the generate tool."""

    test_code: Annotated[str, Field(description="")]
    qualified_class_name: Annotated[
        str, Field(description="The fully qualified class name for the test class.")
    ]


class FinalizeAtomicBlockArgs(BaseModel):
    current_blocks: Annotated[
        AtomicBlockList, Field(description="The current state of the AtomicBlocks.")
    ]
    comments: Annotated[
        str,
        Field(
            description="Comments about any problems with the procedure or concerns."
        ),
    ]
    # NOTE: We don't need to pass in the atomic blocks here because we're using the state.atomic_blocks


class ModifyAtomicBlockNotesArgs(BaseModel):
    order: Annotated[int, Field(description="The order of the atomic block to modify.")]
    new_notes: Annotated[str, Field(description="The new notes for the atomic block.")]


class FinalizeScenarioArgs(BaseModel):
    scenario: Annotated[
        LocalizedScenario,
        Field(description="The current LocalizedScenario to finalize."),
    ]
    comments: Annotated[
        str,
        Field(
            description="Comments about any problems with the procedure or concerns."
        ),
    ]


class ModifyScenarioArgs(BaseModel):
    """Arguments for modifying a localized scenario in composition tools."""

    scenario: Annotated[LocalizedScenario, Field(description="")]


class ModifyAtomicBlocksArgs(BaseModel):
    """Arguments for modifying atomic blocks in composition tools."""

    atomic_blocks: Annotated[AtomicBlockList, Field(description="")]


class ModifyScenarioCommentArgs(BaseModel):
    id: Annotated[int, Field(description="The id of the localized step to update.")]
    comment: Annotated[str, Field(description="The new comment for the step.")]


class ModifyAtomicBlockNoteArgs(BaseModel):
    order: Annotated[int, Field(description="The order of the atomic block to update.")]
    note: Annotated[str, Field(description="The new note for the atomic block.")]
