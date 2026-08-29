"""
End-to-end tests for the NL2Test multi-agent flow (Supervisor -> Localization -> Composition).

These tests drive the *real* agent machinery -- the LangGraph ReAct loop, tool
dispatch, tool-error envelopes, duplicate/parallel-call guards, message
redaction, sub-agent state hand-off, test-file lifecycle, force-finalize
fallbacks and the NL2Test pipeline -- while replacing only the parts that
require the outside world:

* **LLM calls** -- ``LLMClient.invoke_messages`` is replaced by a deterministic
  script keyed on the calling agent (and on the requested schema for structured
  calls). The patched method still builds the real LangChain runnable, so tool
  schemas and structured-output schemas are genuinely validated on every turn.
* **Embeddings** -- a deterministic bag-of-words embedder replaces the HTTP
  embedder. The real extractors, the real FAISS vector stores and the real
  searchers are all exercised on top of it.
* **Static analysis** -- a small in-memory Java "project" built from real CLDK
  models (``JType``/``JCallable``) stands in for a codeanalyzer run, so the
  tests are self-contained (the ``spring-petclinic`` submodule and generated
  ``analysis.json`` files are not available on a fresh clone).
* **Maven** -- compilation/execution is decided by scanning the generated test
  sources for sentinel markers, so the compile/repair loop is driven by what the
  agents actually wrote to disk rather than by a call counter.

Everything else is the production code path.
"""

from __future__ import annotations

import json
import re
import textwrap
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import count
from math import sqrt
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import pytest
from cldk.models.java.models import (
    JCallable,
    JCallableParameter,
    JComment,
    JField,
    JMethodDetail,
    JType,
)
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from sakura.nl2test.generation.composition.orchestrators.gherkin import (
    GherkinCompositionOrchestrator,
)
from sakura.nl2test.generation.localization.orchestrators.gherkin import (
    GherkinLocalizationOrchestrator,
)
from sakura.nl2test.generation.supervisor.orchestrators.gherkin import (
    GherkinSupervisorOrchestrator,
)
from sakura.nl2test.models import (
    AgentState,
    ArgBinding,
    CandidateMethod,
    FinalizeScenarioArgs,
    GherkinStep,
    LocalizedScenario,
    NL2TestInput,
    Scenario,
    Step,
)
from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.nl2test.pipeline import Pipeline as NL2TestPipeline
from sakura.nl2test.preprocessing.embedders.base import BaseEmbedder
from sakura.nl2test.preprocessing.indexers import ClassIndexer, MethodIndexer
from sakura.nl2test.preprocessing.indexers.base import BaseIndexer
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.compilation.maven import CompilationError, CompilationScopeResult
from sakura.utils.config import Config, init_config

# Aliased so pytest does not try to collect it as a test class.
from sakura.utils.evaluation import TestGrader as StructuralTestGrader
from sakura.utils.execution.maven import ExecutionIssue
from sakura.utils.llm import UsageTracker
from sakura.utils.llm.llm_client import LLMClient
from sakura.utils.llm.model import Provider
from sakura.utils.models import (
    NL2TestCoverageEval,
    NL2TestStructuralEval,
)

# ---------------------------------------------------------------------------
# The fake application under test
# ---------------------------------------------------------------------------

APP_PACKAGE = "com.example.inventory"
SERVICE = f"{APP_PACKAGE}.InventoryService"
ITEM = f"{APP_PACKAGE}.Item"
ABSTRACT_STORE = f"{APP_PACKAGE}.AbstractStore"
GT_TEST_CLASS = f"{APP_PACKAGE}.InventoryServiceTest"

ADD_ITEM_SIG = "addItem(com.example.inventory.Item)"
ITEM_COUNT_SIG = "getItemCount()"
FIND_BY_NAME_SIG = "findByName(java.lang.String)"
SERVICE_CTOR_SIG = "InventoryService()"
ITEM_CTOR_SIG = "Item(java.lang.String, int)"
ITEM_FACTORY_SIG = "of(java.lang.String, int)"

DRAFT_TEST_CLASS = f"{APP_PACKAGE}.scratch.InventoryDraftTest"
FINAL_TEST_CLASS = f"{APP_PACKAGE}.InventoryServiceGeneratedTest"
GENERATED_METHOD_SIG = "testAddItemIncreasesCount()"

# Sentinels understood by the fake Maven build.
BROKEN_MARKER = "/*BROKEN*/"
FAILING_MARKER = "/*FAILING*/"

NL_DESCRIPTION = (
    "Verify that adding an item to a new InventoryService increases the item "
    "count to one, then clear the inventory."
)


def _param(name: str, type_: str) -> JCallableParameter:
    return JCallableParameter(
        name=name,
        type=type_,
        annotations=[],
        modifiers=[],
        start_line=-1,
        end_line=-1,
        start_column=-1,
        end_column=-1,
    )


def _callable(
    *,
    signature: str,
    declaration: str,
    code: str,
    modifiers: Sequence[str],
    return_type: Optional[str],
    parameters: Sequence[JCallableParameter] = (),
    annotations: Sequence[str] = (),
    is_constructor: bool = False,
    comments: Sequence[str] = (),
) -> JCallable:
    """Build a CLDK ``JCallable`` with the boilerplate filled in."""
    return JCallable(
        signature=signature,
        is_implicit=False,
        is_constructor=is_constructor,
        comments=[JComment(content=c) for c in comments],
        annotations=list(annotations),
        modifiers=list(modifiers),
        declaration=declaration,
        parameters=list(parameters),
        return_type=return_type,
        code=code,
        start_line=1,
        end_line=1 + code.count("\n"),
        code_start_line=1,
        referenced_types=[],
        accessed_fields=[],
        call_sites=[],
        variable_declarations=[],
        crud_operations=[],
        crud_queries=[],
        cyclomatic_complexity=1,
    )


def _field(name: str, type_: str, modifiers: Sequence[str]) -> JField:
    return JField(
        comment=None,
        type=type_,
        start_line=1,
        end_line=1,
        variables=[name],
        modifiers=list(modifiers),
        annotations=[],
    )


def _jtype(
    *,
    modifiers: Sequence[str] = ("public",),
    extends: Sequence[str] = (),
    implements: Sequence[str] = (),
    callables: Dict[str, JCallable],
    fields: Sequence[JField] = (),
    is_interface: bool = False,
) -> JType:
    return JType(
        is_interface=is_interface,
        is_class_or_interface_declaration=True,
        is_concrete_class=not is_interface,
        comments=[],
        extends_list=list(extends),
        implements_list=list(implements),
        modifiers=list(modifiers),
        annotations=[],
        parent_type="",
        nested_type_declarations=[],
        callable_declarations=dict(callables),
        field_declarations=list(fields),
        enum_constants=[],
    )


class FakeJavaAnalysis:
    """
    Duck-typed stand-in for ``cldk.analysis.java.JavaAnalysis``.

    Only the surface the NL2Test agents (and the hamster helpers they delegate
    to) actually touch is implemented. Using real CLDK models keeps the
    downstream code -- ``CommonAnalysis``, ``Reachability``, the snippet
    extractors and every static-analysis tool -- on its production path.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self._classes: Dict[str, JType] = {}
        self._java_files: Dict[str, str] = {}
        self._imports: Dict[str, List[str]] = {}
        self._callees: Dict[tuple[str, str], List[Dict[str, Any]]] = {}

    # -- construction helpers -------------------------------------------------

    def add_class(
        self,
        qualified_class_name: str,
        jtype: JType,
        *,
        java_file: str,
        imports: Sequence[str] = (),
    ) -> None:
        self._classes[qualified_class_name] = jtype
        self._java_files[qualified_class_name] = java_file
        self._imports[java_file] = list(imports)

    def add_callees(
        self,
        qualified_class_name: str,
        method_signature: str,
        callees: List[Dict[str, Any]],
    ) -> None:
        self._callees[(qualified_class_name, method_signature)] = callees

    # -- JavaAnalysis surface -------------------------------------------------

    def get_classes(self) -> Dict[str, JType]:
        return dict(self._classes)

    def get_class(self, qualified_class_name: str) -> Optional[JType]:
        return self._classes.get(qualified_class_name)

    def get_methods_in_class(self, qualified_class_name: str) -> List[str]:
        jtype = self._classes.get(qualified_class_name)
        if not jtype:
            return []
        return [
            sig
            for sig, jcallable in jtype.callable_declarations.items()
            if not jcallable.is_constructor
        ]

    def get_constructors(self, qualified_class_name: str) -> List[str]:
        jtype = self._classes.get(qualified_class_name)
        if not jtype:
            return []
        return [
            sig
            for sig, jcallable in jtype.callable_declarations.items()
            if jcallable.is_constructor
        ]

    def get_method(
        self, qualified_class_name: str, method_signature: str
    ) -> Optional[JCallable]:
        jtype = self._classes.get(qualified_class_name)
        if not jtype:
            return None
        return jtype.callable_declarations.get(method_signature)

    def get_java_file(self, qualified_class_name: str) -> str:
        return self._java_files.get(qualified_class_name, "")

    def get_java_compilation_unit(self, file_path: str) -> Optional[SimpleNamespace]:
        if file_path not in self._imports:
            return None
        return SimpleNamespace(imports=list(self._imports[file_path]))

    def get_callees(
        self,
        source_class_name: str,
        source_method_declaration: str,
        using_symbol_table: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        key = (source_class_name, source_method_declaration)
        return {"callee_details": list(self._callees.get(key, []))}


def build_fake_analysis(project_root: Path) -> FakeJavaAnalysis:
    """Build the synthetic inventory application used by every test here."""
    analysis = FakeJavaAnalysis(project_root)

    main_dir = project_root / "src" / "main" / "java" / "com" / "example" / "inventory"
    test_dir = project_root / "src" / "test" / "java" / "com" / "example" / "inventory"

    # -- AbstractStore --------------------------------------------------------
    abstract_store = _jtype(
        modifiers=["public", "abstract"],
        implements=["java.io.Serializable"],
        callables={
            "clear()": _callable(
                signature="clear()",
                declaration="public void clear()",
                code="{\n    items.clear();\n}",
                modifiers=["public"],
                return_type="void",
                comments=["Removes every stored item from the store."],
            ),
            "auditLog(java.lang.String)": _callable(
                signature="auditLog(java.lang.String)",
                declaration="protected void auditLog(String message)",
                code="{\n    System.out.println(message);\n}",
                modifiers=["protected"],
                return_type="void",
                parameters=[_param("message", "java.lang.String")],
            ),
        },
        fields=[
            _field("items", "java.util.List<com.example.inventory.Item>", ["protected"])
        ],
    )
    analysis.add_class(
        ABSTRACT_STORE,
        abstract_store,
        java_file=str(main_dir / "AbstractStore.java"),
        imports=["java.io.Serializable", "java.util.List"],
    )

    # -- Item -----------------------------------------------------------------
    item = _jtype(
        callables={
            ITEM_CTOR_SIG: _callable(
                signature=ITEM_CTOR_SIG,
                declaration="public Item(String name, int quantity)",
                code="{\n    this.name = name;\n    this.quantity = quantity;\n}",
                modifiers=["public"],
                return_type=None,
                parameters=[
                    _param("name", "java.lang.String"),
                    _param("quantity", "int"),
                ],
                is_constructor=True,
            ),
            "getName()": _callable(
                signature="getName()",
                declaration="public String getName()",
                code="{\n    return name;\n}",
                modifiers=["public"],
                return_type="java.lang.String",
            ),
            "getQuantity()": _callable(
                signature="getQuantity()",
                declaration="public int getQuantity()",
                code="{\n    return quantity;\n}",
                modifiers=["public"],
                return_type="int",
            ),
            "setQuantity(int)": _callable(
                signature="setQuantity(int)",
                declaration="public void setQuantity(int quantity)",
                code="{\n    this.quantity = quantity;\n}",
                modifiers=["public"],
                return_type="void",
                parameters=[_param("quantity", "int")],
            ),
            ITEM_FACTORY_SIG: _callable(
                signature=ITEM_FACTORY_SIG,
                declaration="public static Item of(String name, int quantity)",
                code="{\n    return new Item(name, quantity);\n}",
                modifiers=["public", "static"],
                return_type=ITEM,
                parameters=[
                    _param("name", "java.lang.String"),
                    _param("quantity", "int"),
                ],
            ),
        },
        fields=[
            _field("name", "java.lang.String", ["private", "final"]),
            _field("quantity", "int", ["private"]),
        ],
    )
    analysis.add_class(ITEM, item, java_file=str(main_dir / "Item.java"), imports=[])

    # -- InventoryService -----------------------------------------------------
    service = _jtype(
        extends=[ABSTRACT_STORE],
        callables={
            SERVICE_CTOR_SIG: _callable(
                signature=SERVICE_CTOR_SIG,
                declaration="public InventoryService()",
                code="{\n    this.items = new ArrayList<>();\n}",
                modifiers=["public"],
                return_type=None,
                is_constructor=True,
            ),
            ADD_ITEM_SIG: _callable(
                signature=ADD_ITEM_SIG,
                declaration="public void addItem(Item item)",
                code='{\n    items.add(item);\n    auditLog("added " + item.getName());\n}',
                modifiers=["public"],
                return_type="void",
                parameters=[_param("item", ITEM)],
                comments=["Adds an item to the inventory."],
            ),
            ITEM_COUNT_SIG: _callable(
                signature=ITEM_COUNT_SIG,
                declaration="public int getItemCount()",
                code="{\n    return items.size();\n}",
                modifiers=["public"],
                return_type="int",
                comments=["Returns the number of items currently stored."],
            ),
            FIND_BY_NAME_SIG: _callable(
                signature=FIND_BY_NAME_SIG,
                declaration="public Item findByName(String name)",
                code=(
                    "{\n    for (Item item : items) {\n"
                    "        if (item.getName().equals(name)) {\n"
                    "            return item;\n        }\n    }\n    return null;\n}"
                ),
                modifiers=["public"],
                return_type=ITEM,
                parameters=[_param("name", "java.lang.String")],
            ),
        },
        fields=[
            _field("items", "java.util.List<com.example.inventory.Item>", ["private"])
        ],
    )
    analysis.add_class(
        SERVICE,
        service,
        java_file=str(main_dir / "InventoryService.java"),
        imports=["java.util.ArrayList", "java.util.List"],
    )

    # -- Ground-truth test class (excluded from indexing as a test class) -----
    gt_test = _jtype(
        callables={
            GENERATED_METHOD_SIG: _callable(
                signature=GENERATED_METHOD_SIG,
                declaration="public void testAddItemIncreasesCount()",
                code=(
                    "{\n    InventoryService service = new InventoryService();\n"
                    '    service.addItem(new Item("widget", 3));\n'
                    "    assertEquals(1, service.getItemCount());\n}"
                ),
                modifiers=["public"],
                return_type="void",
                annotations=["@Test"],
            )
        },
    )
    analysis.add_class(
        GT_TEST_CLASS,
        gt_test,
        java_file=str(test_dir / "InventoryServiceTest.java"),
        imports=[
            "org.junit.jupiter.api.Test",
            "com.example.inventory.InventoryService",
        ],
    )

    # Call-site data for get_call_site_details.
    analysis.add_callees(
        SERVICE,
        ADD_ITEM_SIG,
        [
            {
                "callee_method": JMethodDetail(
                    method_declaration="public String getName()",
                    klass=ITEM,
                    method=item.callable_declarations["getName()"],
                ),
                "calling_lines": [3],
            }
        ],
    )

    return analysis


POM_XML = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <project xmlns="http://maven.apache.org/POM/4.0.0">
      <modelVersion>4.0.0</modelVersion>
      <groupId>com.example</groupId>
      <artifactId>inventory</artifactId>
      <version>1.0.0</version>
      <dependencies>
        <dependency>
          <groupId>org.junit.jupiter</groupId>
          <artifactId>junit-jupiter</artifactId>
          <version>5.10.2</version>
          <scope>test</scope>
        </dependency>
        <dependency>
          <groupId>org.assertj</groupId>
          <artifactId>assertj-core</artifactId>
          <version>3.25.3</version>
          <scope>test</scope>
        </dependency>
      </dependencies>
    </project>
    """
)


def write_fake_project(project_root: Path) -> None:
    """Materialize just enough of a Maven project for the file/pom tooling."""
    main_dir = project_root / "src" / "main" / "java" / "com" / "example" / "inventory"
    test_dir = project_root / "src" / "test" / "java" / "com" / "example" / "inventory"
    main_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    (project_root / "pom.xml").write_text(POM_XML, encoding="utf-8")
    (main_dir / "InventoryService.java").write_text(
        "package com.example.inventory;\npublic class InventoryService {}\n",
        encoding="utf-8",
    )
    (main_dir / "Item.java").write_text(
        "package com.example.inventory;\npublic class Item {}\n", encoding="utf-8"
    )
    (main_dir / "AbstractStore.java").write_text(
        "package com.example.inventory;\npublic abstract class AbstractStore {}\n",
        encoding="utf-8",
    )
    (test_dir / "InventoryServiceTest.java").write_text(
        "package com.example.inventory;\npublic class InventoryServiceTest {}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Deterministic embedder (replaces the HTTP/Ollama embedding service)
# ---------------------------------------------------------------------------

_CAMEL_SPLIT = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


class DeterministicEmbedder(BaseEmbedder):
    """
    Hash-bucketed bag-of-words embedder.

    Deterministic across processes (``zlib.crc32`` rather than ``hash``) and
    identifier-aware, so ``addItem`` matches a query about "add item". The real
    FAISS vector stores and searchers run unchanged on top of it.
    """

    def __init__(self, dim: int = 96) -> None:
        super().__init__(dim)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok.lower() for tok in _CAMEL_SPLIT.findall(text or "")]

    def _vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in self._tokenize(text):
            vec[zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norm = sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


# ---------------------------------------------------------------------------
# Fake Maven build (compilation + execution)
# ---------------------------------------------------------------------------


class FakeMavenCompilation:
    """Reports a compilation error for any generated test carrying the marker."""

    def __init__(self, project_root: Path, module_root: Path | None = None) -> None:
        self.root = Path(module_root or project_root)

    def compile_scope(self) -> CompilationScopeResult:
        errors: List[CompilationError] = []
        test_root = self.root / "src" / "test" / "java"
        if not test_root.exists():
            return CompilationScopeResult(success=True, output="", errors=[])
        for java_file in sorted(test_root.rglob("*.java")):
            source = java_file.read_text(encoding="utf-8")
            if BROKEN_MARKER not in source:
                continue
            line = next(
                (
                    idx
                    for idx, text in enumerate(source.splitlines(), start=1)
                    if BROKEN_MARKER in text
                ),
                1,
            )
            errors.append(
                CompilationError(
                    file=str(java_file),
                    line=line,
                    column=9,
                    message="cannot find symbol",
                    details=["symbol:   method totalItems()", f"location: {SERVICE}"],
                )
            )
        return CompilationScopeResult(
            success=not errors,
            output="",
            errors=errors,
        )

    def get_compilation_errors(self) -> List[CompilationError]:
        return self.compile_scope().errors


class FakeMavenExecution:
    """Reports a test failure for any generated test carrying the marker."""

    def __init__(self, project_root: Path, module_root: Path | None = None) -> None:
        self.root = Path(module_root or project_root)

    def get_execution_errors(
        self, qualified_class_name: str, method_signature: str
    ) -> List[ExecutionIssue]:
        rel = Path(*qualified_class_name.split(".")).with_suffix(".java")
        java_file = self.root / "src" / "test" / "java" / rel
        if not java_file.exists():
            return []
        source = java_file.read_text(encoding="utf-8")
        if FAILING_MARKER not in source:
            return []
        return [
            ExecutionIssue(
                class_name=qualified_class_name,
                test_name=method_signature,
                kind="failure",
                message="expected: <1> but was: <0>",
                error_type="org.opentest4j.AssertionFailedError",
                file=str(java_file),
                line=12,
                stack_trace="org.opentest4j.AssertionFailedError: expected: <1> but was: <0>",
            )
        ]


# ---------------------------------------------------------------------------
# Scripted LLM
# ---------------------------------------------------------------------------

_tool_call_ids = count(1)


def tool_call(name: str, **args: Any) -> Dict[str, Any]:
    return {
        "name": name,
        "args": args,
        "id": f"call_{next(_tool_call_ids)}",
        "type": "tool_call",
    }


def turn(*calls: Dict[str, Any], content: str = "") -> AIMessage:
    """One scripted assistant turn requesting the given tool calls."""
    return AIMessage(content=content, tool_calls=list(calls))


def no_tool_turn(content: str) -> AIMessage:
    """An assistant turn with prose only -- drives the nudge/force-end paths."""
    return AIMessage(content=content)


@dataclass
class ScriptedLLM:
    """
    Deterministic replacement for every ``LLMClient`` invocation in a run.

    Tool-calling turns are routed by the calling agent class (supplied by
    ``ReActAgent.call_model`` via ``context["agent"]``); structured-output calls
    are routed by the requested Pydantic schema.
    """

    turns: Dict[str, List[AIMessage]] = field(default_factory=lambda: defaultdict(list))
    structured: Dict[str, Callable[[], Any]] = field(default_factory=dict)
    call_log: List[str] = field(default_factory=list)
    exhausted: List[str] = field(default_factory=list)
    protocol_violations: List[str] = field(default_factory=list)
    prompts_seen: Dict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    tools_seen: Dict[str, set] = field(default_factory=lambda: defaultdict(set))
    # Tool results snapshotted the first time a model sees them. The agents
    # redact stale payloads in place, so the final message history is not a
    # faithful record of what each turn was actually given.
    observed_results: Dict[str, List[Any]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _seen_tool_call_ids: set = field(default_factory=set)

    def script(self, agent: str, *agent_turns: AIMessage) -> "ScriptedLLM":
        self.turns[agent].extend(agent_turns)
        return self

    def on_schema(self, schema_name: str, factory: Callable[[], Any]) -> "ScriptedLLM":
        self.structured[schema_name] = factory
        return self

    @property
    def pending(self) -> Dict[str, int]:
        return {agent: len(queue) for agent, queue in self.turns.items() if queue}

    def _snapshot_results(self, agent: str, messages: Sequence[BaseMessage]) -> None:
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            call_id = message.tool_call_id
            if call_id in self._seen_tool_call_ids:
                continue
            self._seen_tool_call_ids.add(call_id)
            content = message.content
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    pass
            self.observed_results[agent].append(content)

    def ok_results(self, agent: str) -> List[Any]:
        """Successful tool payloads, in the order the agent's model saw them."""
        return [
            payload["data"]
            for payload in self.observed_results.get(agent, [])
            if isinstance(payload, dict) and payload.get("status") == "ok"
        ]

    def error_codes(self, agent: str) -> List[str]:
        return [
            payload["error"]["code"]
            for payload in self.observed_results.get(agent, [])
            if isinstance(payload, dict)
            and payload.get("status") == "error"
            and isinstance(payload.get("error"), dict)
        ]

    # -- conversation invariants ---------------------------------------------

    def _check_protocol(self, route: str, messages: Sequence[BaseMessage]) -> None:
        if not messages or not isinstance(messages[0], SystemMessage):
            self.protocol_violations.append(
                f"{route}: first message is not a SystemMessage"
            )
            return
        extra_system = [m for m in messages[1:] if isinstance(m, SystemMessage)]
        if extra_system:
            self.protocol_violations.append(
                f"{route}: {len(extra_system)} extra SystemMessages"
            )

        answered = {
            m.tool_call_id
            for m in messages
            if isinstance(m, ToolMessage) and m.tool_call_id
        }
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            for call in message.tool_calls or []:
                if call.get("id") not in answered:
                    self.protocol_violations.append(
                        f"{route}: tool call {call.get('name')} ({call.get('id')}) unanswered"
                    )

    # -- the patched entry point ---------------------------------------------

    def respond(
        self,
        client: LLMClient,
        messages: Sequence[BaseMessage],
        *,
        tools: Optional[Sequence[BaseTool]],
        schema: Any,
        context: Optional[Dict[str, Any]],
    ) -> Any:
        if schema is not None:
            route = f"schema:{getattr(schema, '__name__', schema)}"
            self.call_log.append(route)
            factory = self.structured.get(getattr(schema, "__name__", ""))
            if factory is None:
                self.exhausted.append(route)
                raise AssertionError(f"No scripted structured response for {route}")
            return factory()

        agent = (context or {}).get("agent", "<unknown>")
        self.call_log.append(agent)
        self._check_protocol(agent, messages)
        self._snapshot_results(agent, messages)
        if tools:
            self.tools_seen[agent].update(t.name for t in tools)
        last_human = next(
            (m for m in reversed(messages) if m.__class__.__name__ == "HumanMessage"),
            None,
        )
        if last_human is not None:
            self.prompts_seen[agent].append(str(last_human.content))

        queue = self.turns.get(agent) or []
        if not queue:
            self.exhausted.append(agent)
            # Fall through to prose so the graph terminates instead of hanging.
            return no_tool_turn(f"[script exhausted for {agent}]")
        return queue.pop(0)


def install_scripted_llm(monkeypatch: pytest.MonkeyPatch, script: ScriptedLLM) -> None:
    """Patch ``LLMClient.invoke_messages`` while keeping runnable construction real."""

    def fake_invoke_messages(
        self: LLMClient,
        messages,
        *,
        tools=None,
        tool_choice="auto",
        response_format=None,
        extra_model_kwargs=None,
        schema=None,
        strict=True,
        method="json_schema",
        temperature=None,
        context=None,
    ):
        # Build the real runnable so tool schemas / structured-output schemas
        # stay genuinely validated on every turn.
        self._build_runnable(
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            extra_model_kwargs=extra_model_kwargs,
            schema=schema,
            strict=strict,
            method=method,
            temperature=temperature,
        )
        self._usage_tracker.record(
            input_tokens=100 * len(list(messages)), output_tokens=64
        )
        return script.respond(
            self, list(messages), tools=tools, schema=schema, context=context
        )

    monkeypatch.setattr(LLMClient, "invoke_messages", fake_invoke_messages)


# ---------------------------------------------------------------------------
# Scenario / test-code fixtures data
# ---------------------------------------------------------------------------


def build_scenario() -> Scenario:
    return Scenario(
        setup=[
            Step(
                id=1, task="Create a new InventoryService", uses="", produces="service"
            )
        ],
        gherkin_groups=[
            GherkinStep(
                given=[
                    Step(
                        id=2,
                        task="Create an item named widget with quantity 3",
                        uses="",
                        produces="item",
                    )
                ],
                when=[
                    Step(
                        id=3,
                        task="Add the item to the inventory service",
                        uses="service,item",
                        produces="",
                    )
                ],
                then=[
                    Step(
                        id=4,
                        task="Assert the inventory holds exactly one item",
                        uses="service",
                        produces="",
                    )
                ],
            )
        ],
        teardown=[Step(id=5, task="Clear the inventory", uses="service", produces="")],
    )


def build_localized_scenario() -> LocalizedScenario:
    localized = LocalizedScenario.from_scenario(build_scenario())
    localized.setup[0].candidate_methods = [
        CandidateMethod(
            declaring_class_name=SERVICE,
            containing_class_name=SERVICE,
            method_signature=SERVICE_CTOR_SIG,
        )
    ]
    localized.setup[0].comments = "Instantiate the service under test."

    group = localized.gherkin_groups[0]
    group.given[0].candidate_methods = [
        CandidateMethod(
            declaring_class_name=ITEM,
            containing_class_name=ITEM,
            method_signature=ITEM_CTOR_SIG,
        ),
        CandidateMethod(
            declaring_class_name=ITEM,
            containing_class_name=ITEM,
            method_signature=ITEM_FACTORY_SIG,
        ),
    ]
    group.when[0].candidate_methods = [
        CandidateMethod(
            declaring_class_name=SERVICE,
            containing_class_name=SERVICE,
            method_signature=ADD_ITEM_SIG,
        )
    ]
    group.when[0].arg_bindings = [ArgBinding(arg_name="item", arg_value="${item}")]
    group.then[0].candidate_methods = [
        CandidateMethod(
            declaring_class_name=SERVICE,
            containing_class_name=SERVICE,
            method_signature=ITEM_COUNT_SIG,
        )
    ]
    localized.teardown[0].candidate_methods = [
        CandidateMethod(
            declaring_class_name=ABSTRACT_STORE,
            containing_class_name=SERVICE,
            method_signature="clear()",
        )
    ]
    return localized


DRAFT_TEST_CODE = textwrap.dedent(
    f"""\
    package com.example.inventory.scratch;

    import static org.junit.jupiter.api.Assertions.assertEquals;

    import com.example.inventory.InventoryService;
    import com.example.inventory.Item;
    import org.junit.jupiter.api.Test;

    public class InventoryDraftTest {{

        @Test
        public void testAddItemIncreasesCount() {{
            InventoryService service = new InventoryService();
            service.addItem(new Item("widget", 3));
            assertEquals(1, service.totalItems()); {BROKEN_MARKER}
        }}
    }}
    """
)

FIXED_TEST_CODE = textwrap.dedent(
    """\
    package com.example.inventory;

    import static org.junit.jupiter.api.Assertions.assertEquals;

    import org.junit.jupiter.api.Test;

    public class InventoryServiceGeneratedTest {

        @Test
        public void testAddItemIncreasesCount() {
            InventoryService service = new InventoryService();
            service.addItem(new Item("widget", 3));
            assertEquals(1, service.getItemCount());
            service.clear();
        }
    }
    """
)

POLISHED_TEST_CODE = FIXED_TEST_CODE.replace(
    "public class InventoryServiceGeneratedTest {",
    "/** Verifies addItem increases the inventory count. */\npublic class InventoryServiceGeneratedTest {",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class E2EContext:
    project_root: Path
    output_dir: Path
    analysis: FakeJavaAnalysis
    config: Config
    method_searcher: Any
    class_searcher: Any
    nl2_input: NL2TestInput
    tracker: UsageTracker

    @property
    def test_base_dir(self) -> Path:
        return self.project_root / "src" / "test" / "java"

    def generated_test_path(self, qualified_class_name: str) -> Path:
        return self.test_base_dir / Path(*qualified_class_name.split(".")).with_suffix(
            ".java"
        )


@pytest.fixture
def e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterable[E2EContext]:
    project_root = tmp_path / "inventory-app"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_fake_project(project_root)

    Config.reset()
    config = init_config(
        project_name="inventory-app",
        base_project_dir=str(tmp_path),
        project_output_dir=str(output_dir),
        use_stored_index=False,
        llm_provider=Provider.OPENAI,
        llm_model="scripted-model",
        emb_provider=Provider.OPENAI,
        emb_model="scripted-embedder",
        llm_api_key="test-llm-key",
        emb_api_key="test-emb-key",
        can_parallel_tool=True,
        localization_max_iters=15,
        composition_max_iters=20,
        supervisor_max_iters=10,
        store_code_iteration=True,
    )

    # Embeddings: deterministic, offline. Everything downstream (extractors,
    # FAISS stores, searchers) stays on the production path.
    monkeypatch.setattr(
        BaseIndexer, "_initialize_embedder", lambda self: DeterministicEmbedder()
    )

    # Maven: decided by what the agents actually wrote to disk.
    monkeypatch.setattr(
        "sakura.nl2test.generation.common.compilation_execution.JavaMavenCompilation",
        FakeMavenCompilation,
    )
    monkeypatch.setattr(
        "sakura.nl2test.generation.common.compilation_execution.JavaMavenExecution",
        FakeMavenExecution,
    )
    monkeypatch.setattr(
        "sakura.nl2test.pipeline.JavaMavenCompilation", FakeMavenCompilation
    )

    analysis = build_fake_analysis(project_root)
    method_searcher = MethodIndexer(analysis).build_index()
    class_searcher = ClassIndexer(analysis).build_index()

    nl2_input = NL2TestInput(
        id=7,
        description=NL_DESCRIPTION,
        project_name="inventory-app",
        qualified_class_name=GT_TEST_CLASS,
        method_signature=GENERATED_METHOD_SIG,
    )

    try:
        yield E2EContext(
            project_root=project_root,
            output_dir=output_dir,
            analysis=analysis,
            config=config,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            tracker=UsageTracker(),
        )
    finally:
        Config.reset()


# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------


def localization_script() -> List[AIMessage]:
    """Seven turns covering every localization tool plus two failure modes."""
    return [
        # Parallel vector-store lookups.
        turn(
            tool_call("query_class_db", query="inventory service", i=1, j=2),
            tool_call("query_method_db", query="add item to inventory", i=1, j=3),
        ),
        # Parallel static-analysis lookups.
        turn(
            tool_call(
                "get_method_details",
                qualified_class_name=SERVICE,
                method_signature=ADD_ITEM_SIG,
            ),
            tool_call(
                "extract_method_code",
                qualified_class_name=SERVICE,
                method_signature=ADD_ITEM_SIG,
                start_line=1,
                end_line=4,
            ),
        ),
        turn(
            tool_call(
                "search_reachable_methods_in_class",
                qualified_class_name=SERVICE,
                query="count the stored items",
                visibility_mode="public",
                k=5,
            ),
            tool_call("get_inherited_library_classes", qualified_class_name=SERVICE),
        ),
        turn(
            tool_call(
                "get_call_site_details",
                qualified_class_name=SERVICE,
                method_signature=ADD_ITEM_SIG,
            )
        ),
        # Failure mode 1: a hallucinated class must come back as a tool error.
        turn(
            tool_call(
                "get_method_details",
                qualified_class_name=f"{APP_PACKAGE}.Ghost",
                method_signature="nope()",
            )
        ),
        # Failure mode 2: an identical repeat must be rejected as a duplicate.
        turn(tool_call("query_method_db", query="add item to inventory", i=1, j=3)),
        turn(
            tool_call(
                "finalize",
                localized_scenario=build_localized_scenario().model_dump(),
                comments="Every step maps to InventoryService or Item APIs.",
            )
        ),
    ]


def composition_script() -> List[AIMessage]:
    """Eleven turns: explore, draft, fail to compile, repair, verify, polish."""
    return [
        turn(
            tool_call("get_class_fields", qualified_class_name=SERVICE),
            tool_call(
                "get_class_constructors_and_factories", qualified_class_name=ITEM
            ),
        ),
        turn(
            tool_call("get_getters_and_setters", qualified_class_name=ITEM),
            tool_call("get_maven_dependencies"),
        ),
        turn(
            tool_call(
                "get_method_details",
                qualified_class_name=SERVICE,
                method_signature=ITEM_COUNT_SIG,
            ),
            tool_call(
                "extract_method_code",
                qualified_class_name=ITEM,
                method_signature="getName()",
                start_line=1,
                end_line=3,
            ),
        ),
        # Draft in a scratch package, with a call that does not exist.
        turn(
            tool_call(
                "generate_test_code",
                test_code=DRAFT_TEST_CODE,
                qualified_class_name=DRAFT_TEST_CLASS,
                method_signature=GENERATED_METHOD_SIG,
            )
        ),
        turn(tool_call("compile_and_execute_test")),
        turn(tool_call("view_test_code", start_line=1, end_line=60)),
        # Repair: correct API *and* a different package -> exercises the
        # rename/delete branch of the test-file lifecycle.
        turn(
            tool_call(
                "generate_test_code",
                test_code=FIXED_TEST_CODE,
                qualified_class_name=FINAL_TEST_CLASS,
                method_signature=GENERATED_METHOD_SIG,
            )
        ),
        turn(tool_call("compile_and_execute_test")),
        # Polish: same target path -> exercises the in-place overwrite branch.
        turn(
            tool_call(
                "generate_test_code",
                test_code=POLISHED_TEST_CODE,
                qualified_class_name=FINAL_TEST_CLASS,
                method_signature=GENERATED_METHOD_SIG,
            )
        ),
        turn(
            tool_call(
                "modify_scenario_comment",
                id=4,
                comment="Asserted with getItemCount() after the repair.",
            )
        ),
        turn(tool_call("finalize", comments="Generated test compiles and passes.")),
    ]


def supervisor_script() -> List[AIMessage]:
    """Six turns: delegate, delegate, verify, inspect, drift, finalize."""
    return [
        turn(
            tool_call(
                "call_localization_agent",
                instructions="Map steps 1-5 onto InventoryService and Item APIs.",
            )
        ),
        turn(
            tool_call(
                "call_composition_agent",
                instructions="Write a JUnit 5 test from the localized mappings for steps 1-5.",
            )
        ),
        turn(tool_call("compile_and_execute_test")),
        turn(tool_call("view_test_code", start_line=1, end_line=100)),
        # No tool call -> must be nudged rather than silently ended.
        no_tool_turn("The generated test looks complete."),
        turn(tool_call("finalize", comments="Test compiles and passes.")),
    ]


def full_script() -> ScriptedLLM:
    return (
        ScriptedLLM()
        .script("SupervisorReActAgent", *supervisor_script())
        .script("LocalizationReActAgent", *localization_script())
        .script("CompositionReActAgent", *composition_script())
    )


# ---------------------------------------------------------------------------
# Helpers for assertions
# ---------------------------------------------------------------------------


def tool_payloads(state: AgentState) -> List[Dict[str, Any]]:
    """Decode every tool-result envelope in an agent's message history."""
    payloads: List[Dict[str, Any]] = []
    for message in state.messages:
        if not isinstance(message, ToolMessage):
            continue
        content = message.content
        if isinstance(content, dict):
            payloads.append(content)
            continue
        if not isinstance(content, str):
            continue
        try:
            payloads.append(json.loads(content))
        except json.JSONDecodeError:
            continue
    return payloads


def error_codes(state: AgentState) -> List[str]:
    return [
        payload["error"]["code"]
        for payload in tool_payloads(state)
        if payload.get("status") == "error" and isinstance(payload.get("error"), dict)
    ]


def final_payloads_by_tool(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    """Decode the *final* tool envelopes, grouped by the tool that produced them."""
    names: Dict[str, str] = {}
    for message in state.messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                if call.get("id") and call.get("name"):
                    names[call["id"]] = call["name"]

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for message in state.messages:
        if not isinstance(message, ToolMessage):
            continue
        tool_name = message.name or names.get(message.tool_call_id)
        if not tool_name:
            continue
        content = message.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                continue
        if isinstance(content, dict):
            grouped[tool_name].append(content)
    return grouped


def flat_trajectory(state: AgentState) -> List[str]:
    flat: List[str] = []
    for trajectory in state.tool_trajectories:
        flat.extend(trajectory)
    flat.extend(state.curr_tool_trajectory)
    return flat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiAgentEndToEnd:
    def test_supervisor_run_produces_a_compiling_test(
        self, e2e: E2EContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The headline test: run the whole Supervisor -> Localization -> Composition
        flow on a fake input and assert it converges on a saved, compiling test.
        """
        script = full_script()
        install_scripted_llm(monkeypatch, script)

        supervisor = GherkinSupervisorOrchestrator(
            analysis=e2e.analysis,
            method_searcher=e2e.method_searcher,
            class_searcher=e2e.class_searcher,
            nl2_input=e2e.nl2_input,
            base_project_dir=str(e2e.project_root),
            test_base_dir=e2e.test_base_dir,
            module_root=e2e.project_root,
            usage_tracker=e2e.tracker,
        )

        supervisor_state, localization_state, composition_state = (
            supervisor.assign_task(LocalizedScenario.from_scenario(build_scenario()))
        )

        # -- the scripted conversation was consumed exactly as written --------
        assert script.exhausted == [], f"scripts ran dry: {script.exhausted}"
        assert script.pending == {}, f"unused scripted turns: {script.pending}"
        assert script.protocol_violations == [], script.protocol_violations
        assert (
            len(script.call_log) == 24
        )  # 6 supervisor + 7 localization + 11 composition

        # -- success prediction ----------------------------------------------
        assert supervisor_state.finalize_called is True
        # The finalize tool hands back {"comments": ...}; the agent must unwrap it
        # rather than stringifying the dict.
        assert supervisor_state.final_comments == "Test compiles and passes."
        assert supervisor_state.force_end_attempts == 0
        assert supervisor_state.package == APP_PACKAGE
        assert supervisor_state.class_name == "InventoryServiceGeneratedTest"
        assert supervisor_state.method_signature == GENERATED_METHOD_SIG

        # -- the generated artifact ------------------------------------------
        final_path = e2e.generated_test_path(FINAL_TEST_CLASS)
        assert final_path.exists(), f"expected generated test at {final_path}"
        source = final_path.read_text(encoding="utf-8")
        assert "class InventoryServiceGeneratedTest" in source
        assert "service.getItemCount()" in source
        assert BROKEN_MARKER not in source
        assert "Verifies addItem increases the inventory count." in source

        # The scratch draft was moved, not left behind.
        draft_path = e2e.generated_test_path(DRAFT_TEST_CLASS)
        assert not draft_path.exists()
        assert (
            not draft_path.parent.exists()
        ), "empty scratch package should be cleaned up"

        # Three generate_test_code calls -> three archived code iterations.
        iterations = sorted((e2e.output_dir / "code_iteration").glob("*.java"))
        assert len(iterations) == 3

        # -- sub-agent outcomes ----------------------------------------------
        assert localization_state.finalize_called is True
        assert composition_state.finalize_called is True
        assert composition_state.final_comments == "Generated test compiles and passes."

        # -- localization results propagated all the way up -------------------
        scenario = supervisor_state.localized_scenario
        assert scenario is not None
        when_step = scenario.gherkin_groups[0].when[0]
        assert [c.method_signature for c in when_step.candidate_methods] == [
            ADD_ITEM_SIG
        ]
        then_step = scenario.gherkin_groups[0].then[0]
        assert then_step.candidate_methods[0].method_signature == ITEM_COUNT_SIG
        # ...and the composition agent's comment edit survived the hand-back.
        assert then_step.comments == "Asserted with getItemCount() after the repair."

        # -- tool trajectories -------------------------------------------------
        supervisor_tools = flat_trajectory(supervisor_state)
        assert supervisor_tools == [
            "call_localization_agent",
            "call_composition_agent",
            "compile_and_execute_test",
            "view_test_code",
            "finalize",
        ]
        localization_tools = flat_trajectory(localization_state)
        assert localization_tools[-1] == "finalize"
        assert {
            "query_class_db",
            "query_method_db",
            "get_method_details",
            "extract_method_code",
            "search_reachable_methods_in_class",
            "get_inherited_library_classes",
            "get_call_site_details",
        } <= set(localization_tools)
        composition_tools = flat_trajectory(composition_state)
        assert composition_tools.count("generate_test_code") == 3
        assert composition_tools.count("compile_and_execute_test") == 2
        assert composition_tools[-1] == "finalize"

        # -- token accounting --------------------------------------------------
        totals = e2e.tracker.totals()
        assert totals["calls"] == 24
        assert totals["input_tokens"] > 0
        assert totals["output_tokens"] == 24 * 64

    def test_localization_agent_exercises_search_and_error_paths(
        self, e2e: E2EContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retrieval, static analysis, duplicate suppression and tool errors."""
        script = full_script()
        install_scripted_llm(monkeypatch, script)

        supervisor = GherkinSupervisorOrchestrator(
            analysis=e2e.analysis,
            method_searcher=e2e.method_searcher,
            class_searcher=e2e.class_searcher,
            nl2_input=e2e.nl2_input,
            base_project_dir=str(e2e.project_root),
            test_base_dir=e2e.test_base_dir,
            module_root=e2e.project_root,
            usage_tracker=e2e.tracker,
        )
        _, localization_state, _ = supervisor.assign_task(
            LocalizedScenario.from_scenario(build_scenario())
        )

        codes = error_codes(localization_state)
        assert (
            "ClassNotFoundError" in codes
        ), "hallucinated class should surface as a tool error"
        assert (
            "duplicate_tool_call" in codes
        ), "identical repeat call should be suppressed"

        # Results, in the order the localization model was shown them. Turn 1
        # issued query_class_db then query_method_db as parallel calls.
        results = script.ok_results("LocalizationReActAgent")

        class_hits, method_hits = results[0], results[1]
        class_names = [hit["declaring_class_name"] for hit in class_hits]
        assert SERVICE in class_names
        # The ground-truth test class is a test class, so it must not be indexed.
        assert GT_TEST_CLASS not in class_names

        # Retrieval runs through the real extractors, FAISS store and searcher
        # on top of the deterministic embedder, and ranks the obvious match first.
        assert method_hits[0]["method_signature"] == ADD_ITEM_SIG
        assert method_hits[0]["declaring_class_name"] == SERVICE
        assert (
            len(method_hits) == 3
        ), "query_method_db(i=1, j=3) should return a 3-item window"

        # get_method_details resolves visibility through CommonAnalysis.
        details = next(
            r
            for r in results
            if isinstance(r, dict) and r.get("method_signature") == ADD_ITEM_SIG
        )
        assert details["visibility"] == "public"
        assert details["return_type"] == "void"
        assert details["parameter_types"] == [ITEM]

        dict_lists = [
            r
            for r in results
            if isinstance(r, list) and all(isinstance(h, dict) for h in r) and r
        ]

        # Reachable-method search walks the superclass chain into AbstractStore
        # and drops methods that are not visible under the requested mode.
        reachable = next(
            r
            for r in dict_lists
            if any(h.get("method_signature") == "clear()" for h in r)
        )
        by_signature = {h["method_signature"]: h for h in reachable}
        assert by_signature["clear()"]["declaring_class_name"] == ABSTRACT_STORE
        assert by_signature["clear()"]["containing_class_name"] == SERVICE
        assert ITEM_COUNT_SIG in by_signature
        assert (
            "auditLog(java.lang.String)" not in by_signature
        ), "protected members must be filtered out under visibility_mode='public'"

        # get_inherited_library_classes reports only types outside the application.
        assert ["java.io.Serializable"] in results

        # get_call_site_details resolves callees through the symbol table.
        call_sites = next(
            r for r in dict_lists if any(h.get("num_times_called") for h in r)
        )
        assert call_sites[0]["qualified_class_name"] == ITEM
        assert call_sites[0]["method_signature"] == "getName()"

    def test_composition_agent_repairs_after_compilation_failure(
        self, e2e: E2EContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The compile -> diagnose -> regenerate -> compile loop really runs."""
        script = full_script()
        install_scripted_llm(monkeypatch, script)

        supervisor = GherkinSupervisorOrchestrator(
            analysis=e2e.analysis,
            method_searcher=e2e.method_searcher,
            class_searcher=e2e.class_searcher,
            nl2_input=e2e.nl2_input,
            base_project_dir=str(e2e.project_root),
            test_base_dir=e2e.test_base_dir,
            module_root=e2e.project_root,
            usage_tracker=e2e.tracker,
        )
        supervisor_state, _, _ = supervisor.assign_task(
            LocalizedScenario.from_scenario(build_scenario())
        )

        # Assert on what the composition model was actually shown, in order.
        results = script.ok_results("CompositionReActAgent")
        compilations = [
            r["compilation"]
            for r in results
            if isinstance(r, dict) and "compilation" in r
        ]
        assert len(compilations) == 2, "expected two compile_and_execute_test results"

        first, second = compilations
        assert first["status"] == "compilation_error"
        assert first["has_errors_for_target"] is True
        assert any(
            "cannot find symbol" in detail
            for detail in first["error_details_for_target_class"]
        )
        assert first["target_class_file"] == "InventoryDraftTest.java"

        assert second["status"] == "success"
        assert second["has_errors_for_project"] is False
        assert second["has_errors_for_target"] is False
        assert second["target_class_file"] == "InventoryServiceGeneratedTest.java"

        executions = [
            r["execution"] for r in results if isinstance(r, dict) and "execution" in r
        ]
        assert executions[0]["status"] == "compilation_errors"
        assert executions[-1]["status"] == "success"
        assert executions[-1]["num_failures"] == 0
        assert executions[-1]["num_errors"] == 0

        # The supervisor independently re-verified the repaired test.
        supervisor_compiles = [
            payload["data"]
            for payload in final_payloads_by_tool(supervisor_state)[
                "compile_and_execute_test"
            ]
        ]
        assert len(supervisor_compiles) == 1
        assert supervisor_compiles[0]["compilation"]["status"] == "success"
        assert supervisor_compiles[0]["execution"]["status"] == "success"

        # Maven dependencies were read from the real pom.xml.
        artifacts = {
            entry.get("artifact_id")
            for result in results
            if isinstance(result, list)
            for entry in result
            if isinstance(entry, dict)
        }
        assert {"junit-jupiter", "assertj-core"} <= artifacts

    def test_stale_test_code_is_redacted_from_history(
        self, e2e: E2EContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Superseded code/compilation payloads are replaced with placeholders."""
        script = full_script()
        install_scripted_llm(monkeypatch, script)

        supervisor = GherkinSupervisorOrchestrator(
            analysis=e2e.analysis,
            method_searcher=e2e.method_searcher,
            class_searcher=e2e.class_searcher,
            nl2_input=e2e.nl2_input,
            base_project_dir=str(e2e.project_root),
            test_base_dir=e2e.test_base_dir,
            module_root=e2e.project_root,
            usage_tracker=e2e.tracker,
        )
        _, _, composition_state = supervisor.assign_task(
            LocalizedScenario.from_scenario(build_scenario())
        )

        redacted = "(redacted since new code generated)"

        # While it was current, view_test_code returned the real draft source...
        live_views = [
            result
            for result in script.ok_results("CompositionReActAgent")
            if isinstance(result, dict)
            and "source" in result
            and result.get("qualified_class_name") == DRAFT_TEST_CLASS
        ]
        assert live_views, "expected a live view_test_code result for the draft"
        assert BROKEN_MARKER in live_views[0]["source"]
        assert live_views[0]["total_lines"] > 0

        # ...and once new code was generated, that payload was redacted in place.
        final_views = final_payloads_by_tool(composition_state)["view_test_code"]
        assert len(final_views) == 1
        assert final_views[0]["data"]["source"] == redacted

        final_compiles = final_payloads_by_tool(composition_state)[
            "compile_and_execute_test"
        ]
        assert len(final_compiles) == 2
        assert all(
            payload["data"]["compilation"] == redacted for payload in final_compiles
        )
        assert all(
            payload["data"]["execution"] == redacted for payload in final_compiles
        )

        # The earliest generate_test_code argument payload is redacted too, so
        # obsolete drafts stop costing tokens on every later turn.
        generate_args = [
            call["args"]
            for message in composition_state.messages
            if isinstance(message, AIMessage)
            for call in (message.tool_calls or [])
            if call.get("name") == "generate_test_code"
        ]
        assert len(generate_args) == 3
        assert generate_args[0]["test_code"] == redacted
        assert generate_args[1]["test_code"] == redacted
        assert BROKEN_MARKER not in json.dumps(generate_args)

    def test_supervisor_can_re_delegate_to_both_sub_agents(
        self, e2e: E2EContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The supervisor's iterate-until-success loop: delegate, inspect, delegate
        again. Each re-delegation must start the sub-agent from a cleaned
        conversation while preserving its cumulative bookkeeping.
        """
        second_pass = build_localized_scenario()
        second_pass.gherkin_groups[0].then[
            0
        ].comments = "Second pass: use getItemCount()."

        repeat_query = tool_call("query_method_db", query="add item", i=1, j=2)
        script = (
            ScriptedLLM()
            .script(
                "SupervisorReActAgent",
                turn(
                    tool_call(
                        "call_localization_agent",
                        instructions="First pass: map steps 1-5.",
                    )
                ),
                turn(
                    tool_call("call_composition_agent", instructions="Draft the test.")
                ),
                turn(
                    tool_call(
                        "call_localization_agent",
                        instructions="Second pass: step 4 must use getItemCount().",
                    )
                ),
                turn(
                    tool_call(
                        "call_composition_agent",
                        instructions="Repair the assertion using getItemCount().",
                    )
                ),
                turn(
                    tool_call("finalize", comments="Converged after one repair round.")
                ),
            )
            .script(
                "LocalizationReActAgent",
                # First delegation.
                turn(dict(repeat_query, id="loc_query_a")),
                turn(
                    tool_call(
                        "finalize",
                        localized_scenario=build_localized_scenario().model_dump(),
                        comments="First pass complete.",
                    )
                ),
                # Second delegation repeats the identical query on purpose: the
                # duplicate guard is per-delegation, not per-run.
                turn(dict(repeat_query, id="loc_query_b")),
                turn(
                    tool_call(
                        "finalize",
                        localized_scenario=second_pass.model_dump(),
                        comments="Second pass complete.",
                    )
                ),
            )
            .script(
                "CompositionReActAgent",
                turn(
                    tool_call(
                        "generate_test_code",
                        test_code=DRAFT_TEST_CODE,
                        qualified_class_name=DRAFT_TEST_CLASS,
                        method_signature=GENERATED_METHOD_SIG,
                    )
                ),
                turn(tool_call("finalize", comments="Draft saved.")),
                turn(
                    tool_call(
                        "generate_test_code",
                        test_code=FIXED_TEST_CODE,
                        qualified_class_name=FINAL_TEST_CLASS,
                        method_signature=GENERATED_METHOD_SIG,
                    )
                ),
                turn(tool_call("compile_and_execute_test")),
                turn(tool_call("finalize", comments="Repaired and verified.")),
            )
        )
        install_scripted_llm(monkeypatch, script)

        supervisor = GherkinSupervisorOrchestrator(
            analysis=e2e.analysis,
            method_searcher=e2e.method_searcher,
            class_searcher=e2e.class_searcher,
            nl2_input=e2e.nl2_input,
            base_project_dir=str(e2e.project_root),
            test_base_dir=e2e.test_base_dir,
            module_root=e2e.project_root,
            usage_tracker=e2e.tracker,
        )
        supervisor_state, localization_state, composition_state = (
            supervisor.assign_task(LocalizedScenario.from_scenario(build_scenario()))
        )

        assert script.exhausted == []
        assert script.pending == {}
        assert script.protocol_violations == [], script.protocol_violations

        # Both sub-agents ran twice, each delegation recorded as its own trajectory.
        assert localization_state.tool_trajectories == [
            ["query_method_db", "finalize"],
            ["query_method_db", "finalize"],
        ]
        assert composition_state.tool_trajectories == [
            ["generate_test_code", "finalize"],
            ["generate_test_code", "compile_and_execute_test", "finalize"],
        ]

        # Cumulative counts survive the reset; the identical repeat query in the
        # second delegation was executed, not rejected as a duplicate.
        query_counts = localization_state.total_tool_calls["query_method_db"]
        assert list(query_counts.values()) == [2], query_counts
        assert "duplicate_tool_call" not in script.error_codes("LocalizationReActAgent")

        # ...because the per-delegation view was cleared before the second run,
        # so it now counts that delegation alone.
        assert list(localization_state.curr_tool_calls["query_method_db"].values()) == [
            1
        ]

        # The second localization pass is the one that reached the supervisor.
        assert supervisor_state.localized_scenario is not None
        then_step = supervisor_state.localized_scenario.gherkin_groups[0].then[0]
        assert then_step.comments == "Second pass: use getItemCount()."

        # The repair replaced the draft on disk.
        assert supervisor_state.class_name == "InventoryServiceGeneratedTest"
        assert e2e.generated_test_path(FINAL_TEST_CLASS).exists()
        assert not e2e.generated_test_path(DRAFT_TEST_CLASS).exists()
        assert supervisor_state.finalize_called is True


class TestAgentFallbacks:
    def test_supervisor_force_finalizes_at_the_iteration_limit(
        self, e2e: E2EContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A supervisor that never calls finalize is coerced into finishing."""
        e2e.config.set("supervisor", "max_iters", 4)

        script = ScriptedLLM().script(
            "SupervisorReActAgent",
            *[
                turn(tool_call("view_test_code", start_line=1, end_line=10 * idx))
                for idx in range(1, 5)
            ],
        )
        install_scripted_llm(monkeypatch, script)

        supervisor = GherkinSupervisorOrchestrator(
            analysis=e2e.analysis,
            method_searcher=e2e.method_searcher,
            class_searcher=e2e.class_searcher,
            nl2_input=e2e.nl2_input,
            base_project_dir=str(e2e.project_root),
            test_base_dir=e2e.test_base_dir,
            module_root=e2e.project_root,
            usage_tracker=e2e.tracker,
        )
        supervisor_state, _, _ = supervisor.assign_task(
            LocalizedScenario.from_scenario(build_scenario())
        )

        assert script.exhausted == []
        assert supervisor_state.iterations == 4
        assert supervisor_state.finalize_called is True
        assert supervisor_state.force_end_attempts == 1
        assert supervisor_state.final_comments == "No issues."
        assert flat_trajectory(supervisor_state)[-1] == "finalize"
        assert supervisor_state.total_tool_calls["finalize"]["<force_finalize>"] == 1

        # The iteration-limit warnings really were injected into the prompt.
        prompts = "\n".join(script.prompts_seen["SupervisorReActAgent"])
        assert "SYSTEM NOTICE: Second-to-last iteration." in prompts
        assert "SYSTEM NOTICE: Final iteration." in prompts

        # No test was ever produced, so no class name is reported upstream.
        assert supervisor_state.class_name is None

    def test_localization_force_finalize_recovers_scenario_via_structured_output(
        self, e2e: E2EContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When finalize is never called, the scenario is rebuilt structurally."""
        e2e.config.set("localization", "max_iters", 2)

        recovered = build_localized_scenario()
        script = (
            ScriptedLLM()
            .script(
                "LocalizationReActAgent",
                turn(tool_call("query_method_db", query="add item", i=1, j=2)),
                turn(tool_call("query_class_db", query="inventory", i=1, j=2)),
            )
            .on_schema(
                "FinalizeScenarioArgs",
                lambda: FinalizeScenarioArgs(
                    localized_scenario=recovered,
                    comments="Recovered from the conversation history.",
                ),
            )
        )
        install_scripted_llm(monkeypatch, script)

        orchestrator = GherkinLocalizationOrchestrator(
            analysis=e2e.analysis,
            method_searcher=e2e.method_searcher,
            class_searcher=e2e.class_searcher,
            nl2_input=e2e.nl2_input,
            usage_tracker=e2e.tracker,
        )
        state = orchestrator.assign_task(
            LocalizedScenario.from_scenario(build_scenario()),
            instructions="Localize every step.",
        )

        assert script.exhausted == []
        assert "schema:FinalizeScenarioArgs" in script.call_log
        assert state.finalize_called is True
        assert state.force_end_attempts == 1
        assert state.final_comments == "Recovered from the conversation history."
        assert state.localized_scenario is not None
        assert (
            state.localized_scenario.gherkin_groups[0]
            .when[0]
            .candidate_methods[0]
            .method_signature
            == ADD_ITEM_SIG
        )
        assert state.total_tool_calls["finalize"]["<force_finalize>"] == 1

    def test_parallel_tool_calls_are_rejected_when_disabled(
        self, e2e: E2EContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With parallel calling off, only the first call in a turn executes."""
        e2e.config.set("llm", "can_parallel_tool", False)

        script = ScriptedLLM().script(
            "LocalizationReActAgent",
            turn(
                tool_call("query_method_db", query="add item", i=1, j=2),
                tool_call("query_class_db", query="inventory", i=1, j=2),
            ),
            turn(
                tool_call(
                    "finalize",
                    localized_scenario=build_localized_scenario().model_dump(),
                    comments="Done.",
                )
            ),
        )
        install_scripted_llm(monkeypatch, script)

        orchestrator = GherkinLocalizationOrchestrator(
            analysis=e2e.analysis,
            method_searcher=e2e.method_searcher,
            class_searcher=e2e.class_searcher,
            nl2_input=e2e.nl2_input,
            usage_tracker=e2e.tracker,
        )
        state = orchestrator.assign_task(
            LocalizedScenario.from_scenario(build_scenario()),
            instructions="Localize every step.",
        )

        assert script.exhausted == []
        assert "parallel_call_disallowed" in error_codes(state)
        assert flat_trajectory(state) == ["query_method_db", "finalize"]
        assert state.finalize_called is True

    def test_project_compilation_error_outside_the_target_is_surfaced(
        self, e2e: E2EContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken *unrelated* source is reported as a project-level error."""
        unrelated = (
            e2e.test_base_dir
            / "com"
            / "example"
            / "inventory"
            / "PreexistingBrokenTest.java"
        )
        unrelated.parent.mkdir(parents=True, exist_ok=True)
        unrelated.write_text(
            f"package com.example.inventory;\npublic class PreexistingBrokenTest {{ {BROKEN_MARKER} }}\n",
            encoding="utf-8",
        )

        script = ScriptedLLM().script(
            "CompositionReActAgent",
            turn(
                tool_call(
                    "generate_test_code",
                    test_code=FIXED_TEST_CODE,
                    qualified_class_name=FINAL_TEST_CLASS,
                    method_signature=GENERATED_METHOD_SIG,
                )
            ),
            turn(tool_call("compile_and_execute_test")),
            turn(
                tool_call("finalize", comments="Blocked by an unrelated build failure.")
            ),
        )
        install_scripted_llm(monkeypatch, script)

        orchestrator = GherkinCompositionOrchestrator(
            analysis=e2e.analysis,
            method_searcher=e2e.method_searcher,
            class_searcher=e2e.class_searcher,
            nl2_input=e2e.nl2_input,
            project_root=str(e2e.project_root),
            test_base_dir=e2e.test_base_dir,
            module_root=e2e.project_root,
            usage_tracker=e2e.tracker,
        )
        state = orchestrator.assign_task(
            build_localized_scenario(), instructions="Write the test."
        )

        assert script.exhausted == []
        assert "project_compilation_error" in error_codes(state)
        assert state.finalize_called is True
        # The generated test itself still landed on disk.
        assert e2e.generated_test_path(FINAL_TEST_CLASS).exists()


class TestPipelineEndToEnd:
    def test_run_nl2test_reports_a_successful_generation(
        self, e2e: E2EContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The full ``Pipeline.run_nl2test`` path: NL decomposition, preprocessing,
        the three agents, compilation and result assembly.

        Grading (structural/coverage/localization scoring) is stubbed -- it is a
        separate concern with its own tests, and it needs a real codeanalyzer
        run to be meaningful.
        """
        script = full_script().on_schema("Scenario", build_scenario)
        install_scripted_llm(monkeypatch, script)

        structural = NL2TestStructuralEval(
            obj_creation_recall=1.0,
            obj_creation_precision=1.0,
            assertion_recall=1.0,
            assertion_precision=1.0,
            callable_recall=1.0,
            callable_precision=1.0,
            focal_recall=1.0,
            focal_precision=1.0,
        )
        coverage = NL2TestCoverageEval(
            class_coverage=100.0,
            method_coverage=75.0,
            line_coverage=80.0,
            branch_coverage=50.0,
        )
        monkeypatch.setattr(
            StructuralTestGrader,
            "grade",
            lambda self, *args, **kwargs: (structural, coverage),
        )
        monkeypatch.setattr(
            NL2TestPipeline, "regenerate_analysis", lambda self, **kwargs: e2e.analysis
        )
        monkeypatch.setattr(
            "sakura.nl2test.evaluation.localization_grader.LocalizationGrader.grade_from_state",
            lambda self, state, nl2_input: None,
        )

        common = CommonAnalysis(e2e.analysis)
        pipeline = NL2TestPipeline(
            e2e.analysis,
            project_root=e2e.project_root,
            analysis_dir=e2e.output_dir,
            decomposition_mode=DecompositionMode.GHERKIN,
            application_classes=[SERVICE, ITEM, ABSTRACT_STORE],
            test_utility_classes=[],
            common_analysis=common,
        )
        pipeline.run_preprocessing()

        result = pipeline.run_nl2test(e2e.nl2_input)

        assert script.exhausted == []
        assert script.pending == {}
        assert script.call_log[0] == "schema:Scenario", "decomposition runs first"

        evaluation = result.eval
        assert evaluation.compiles is True
        assert evaluation.nl2test_metadata.qualified_test_class_name == FINAL_TEST_CLASS
        assert evaluation.nl2test_metadata.method_signature == GENERATED_METHOD_SIG
        assert "service.getItemCount()" in evaluation.nl2test_metadata.code
        assert evaluation.structured_eval == structural
        assert evaluation.coverage_eval == coverage

        # Token accounting covers decomposition plus all three agents.
        assert evaluation.llm_calls == 25
        assert evaluation.input_tokens > 0
        assert evaluation.output_tokens > 0

        # Per-agent tool logs are wired through to the result.
        tool_log = evaluation.tool_log
        assert tool_log is not None
        assert tool_log.supervisor_tool_log.tool_counts["call_localization_agent"] == 1
        assert tool_log.supervisor_tool_log.tool_counts["call_composition_agent"] == 1
        assert tool_log.composition_tool_log.tool_counts["generate_test_code"] == 3
        assert tool_log.localization_tool_log.tool_counts["finalize"] == 1

        # The localized scenario is returned for downstream analysis...
        assert result.localized_scenario is not None
        assert (
            result.localized_scenario.gherkin_groups[0]
            .when[0]
            .candidate_methods[0]
            .method_signature
            == ADD_ITEM_SIG
        )

        # ...and the pipeline cleans the generated test out of the project.
        assert not e2e.generated_test_path(FINAL_TEST_CLASS).exists()
