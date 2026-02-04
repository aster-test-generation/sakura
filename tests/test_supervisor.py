from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from cldk import CLDK
from cldk.analysis import AnalysisLevel

from sakura.nl2test.generation.supervisor.orchestrators.gherkin import (
    GherkinSupervisorOrchestrator,
)
from sakura.nl2test.generation.supervisor.tools.base import BaseSupervisorTools
from sakura.nl2test.generation.supervisor.tools.gherkin import GherkinSupervisorTools
from sakura.nl2test.generation.supervisor.tools.grammatical import (
    GrammaticalSupervisorTools,
)
from sakura.nl2test.models import AgentState
from sakura.nl2test.models.decomposition import DecompositionMode, LocalizedScenario
from sakura.nl2test.preprocessing.indexers import MethodIndexer, ClassIndexer
from sakura.nl2test.preprocessing.nl_decomposer import NLDecomposer
from sakura.test2nl.model.models import Test2NLEntry
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.compilation.maven import JavaMavenCompilation
from sakura.utils.evaluation.test_grader import TestGrader
from sakura.utils.file_io.structured_data_manager import StructuredDataManager
from sakura.utils.file_io.test_file_manager import TestFileManager, TestFileInfo
from sakura.utils.llm import UsageTracker
from sakura.utils.models import NL2TestMetadata
from sakura.utils.pretty.prints import pretty_print
from sakura.utils.utilities import test2nl_entry_to_nl2test_input


class TestSupervisorAgent:
    @pytest.fixture(autouse=True)
    def _inject(self, petclinic_analysis, petclinic_config, petclinic_paths):
        self.analysis = petclinic_analysis
        self.config = petclinic_config
        self.project_root = petclinic_paths.project_root
        self.output_dir = petclinic_paths.project_output_dir

    def test_supervisor_end_to_end(self):
        tracker = UsageTracker()
        # Load dataset entries from CSV relative to this test file
        test_dir = Path(__file__).resolve().parent
        data_dir = test_dir / "output" / "resources" / "test2nl"
        sdm = StructuredDataManager(data_dir)
        entries = sdm.load("test2nl.csv", Test2NLEntry, format="csv")
        assert len(entries) > 0, "No Test2NL entries loaded from CSV"

        # Tighten iteration limits for this test
        self.config.set("localization", "max_iters", 40)
        self.config.set("composition", "max_iters", 40)
        self.config.set("supervisor", "max_iters", 10)

        # Pick a random entry and convert to NL2TestInput
        entry = random.choice(entries)
        nl2_input = test2nl_entry_to_nl2test_input(entry)

        # Build project root from config (resources/{project_name})
        project_name = nl2_input.project_name
        base_project_dir = Path(self.config.get("project", "base_project_dir"))
        project_root = base_project_dir / project_name
        assert project_root.exists(), "Project root does not exist"

        # Decompose NL into Gherkin Scenario and wrap as LocalizedScenario
        decomposer = NLDecomposer(mode=DecompositionMode.GHERKIN, usage_tracker=tracker)
        scenario = decomposer.decompose(nl2_input.description)
        localized = LocalizedScenario.from_scenario(scenario)

        # Index database
        method_indexer = MethodIndexer(self.analysis)
        class_indexer = ClassIndexer(self.analysis)
        method_searcher = method_indexer.build_index()
        class_searcher = class_indexer.build_index()

        # Instantiate Supervisor (GHERKIN mode) and run
        supervisor = GherkinSupervisorOrchestrator(
            analysis=self.analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            base_project_dir=str(project_root),
            usage_tracker=tracker,
        )

        supervisor_state, localization_state, composition_state = (
            supervisor.assign_task(localized)
        )

        # Validate return types are AgentState
        assert isinstance(supervisor_state, AgentState)
        assert isinstance(localization_state, AgentState)
        assert isinstance(composition_state, AgentState)

        pretty_print("Supervisor state", supervisor_state)
        pretty_print("Localization state", localization_state)
        pretty_print("Composition state", composition_state)

        # Validate updated AgentState includes selected package/class
        assert supervisor_state.package is not None
        assert supervisor_state.class_name is not None

        prices = tracker.totals()
        pretty_print("Token usage", prices)

        # Build qualified test class name
        package = supervisor_state.package
        class_name = supervisor_state.class_name
        qualified_test_class_name = f"{package}.{class_name}" if package else class_name
        method_signature = supervisor_state.method_signature or ""

        # Regenerate analysis to pick up the new test class
        new_analysis = CLDK(language="java").analysis(
            project_path=project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=self.output_dir,
            eager=True,
        )

        # Get application classes for grading
        cmn = CommonAnalysis(new_analysis)
        _, application_classes, test_utility_classes = cmn.categorize_classes()

        # Gather compilation errors
        compilation_errors = JavaMavenCompilation(project_root).get_compilation_errors()
        erroneous_files = [err.file for err in compilation_errors]

        # Create grader and grade the test
        test_grader = TestGrader(
            analysis=new_analysis,
            project_root=project_root,
            project_erroneous_files=erroneous_files,
            application_classes=application_classes,
            test_utility_classes=test_utility_classes,
        )

        nl2_metadata = NL2TestMetadata(
            qualified_test_class_name=qualified_test_class_name,
            code="",
            method_signature=method_signature or None,
        )

        eval_result = test_grader.grade(nl2_input, nl2_metadata)

        # Load and attach test code
        fm = TestFileManager(project_root)
        info = TestFileInfo(qualified_class_name=qualified_test_class_name)
        try:
            code = fm.load(info, encode_class_name=False)
            eval_result.nl2test_metadata.code = code
        except FileNotFoundError:
            pass

        # Print evaluation results
        pretty_print("Evaluation result", eval_result)
        pretty_print("Compiles", eval_result.compiles)
        pretty_print("Structural eval", eval_result.structured_eval)
        pretty_print("Coverage eval", eval_result.coverage_eval)

        # Assert that the generated test compiles
        assert eval_result.compiles, "Generated test should compile"

        # Clean up generated test file
        try:
            fm.delete_single(info, encode_class_name=False)
        except Exception:
            pass


class TestSupervisorToolInjection:
    """Tests verifying tool injection after refactoring to shared utilities."""

    def test_base_supervisor_tools_contains_expected_tools(self):
        """Verify BaseSupervisorTools creates expected base tools."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = BaseSupervisorTools(llm=mock_llm, project_root="/tmp/test")
        tools, allow_duplicates = tool_builder.all()

        tool_names = {t.name for t in tools}
        expected_base_tools = {"view_test_code", "compile_and_execute_test", "finalize"}

        assert expected_base_tools.issubset(tool_names), (
            f"Missing base tools: {expected_base_tools - tool_names}"
        )

    def test_gherkin_supervisor_tools_contains_all_tools(self):
        """Verify GherkinSupervisorTools includes base + Gherkin-specific tools."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = GherkinSupervisorTools(llm=mock_llm, project_root="/tmp/test")
        tools, allow_duplicates = tool_builder.all()

        tool_names = {t.name for t in tools}
        expected_tools = {
            "view_test_code",
            "compile_and_execute_test",
            "finalize",
            "call_localization_agent",
            "call_composition_agent",
        }

        assert expected_tools == tool_names, (
            f"Tool mismatch. Expected: {expected_tools}, Got: {tool_names}"
        )

    def test_grammatical_supervisor_tools_contains_all_tools(self):
        """Verify GrammaticalSupervisorTools includes base + Grammatical-specific tools."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = GrammaticalSupervisorTools(
            llm=mock_llm, project_root="/tmp/test"
        )
        tools, allow_duplicates = tool_builder.all()

        tool_names = {t.name for t in tools}
        expected_tools = {
            "view_test_code",
            "compile_and_execute_test",
            "finalize",
            "call_localization_agent",
            "call_composition_agent",
        }

        assert expected_tools == tool_names, (
            f"Tool mismatch. Expected: {expected_tools}, Got: {tool_names}"
        )

    def test_supervisor_allow_duplicate_tools_correct(self):
        """Verify allow_duplicate_tools list contains expected tools."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = GherkinSupervisorTools(llm=mock_llm, project_root="/tmp/test")
        _, allow_duplicates = tool_builder.all()

        allow_dup_names = {t.name for t in allow_duplicates}
        expected_duplicates = {"view_test_code", "compile_and_execute_test"}

        assert expected_duplicates == allow_dup_names, (
            f"Allow duplicates mismatch. Expected: {expected_duplicates}, Got: {allow_dup_names}"
        )

    def test_deferred_tool_call_localization_returns_instructions(self):
        """Verify call_localization_agent returns instructions as expected."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = GherkinSupervisorTools(llm=mock_llm, project_root="/tmp/test")
        tools, _ = tool_builder.all()

        call_loc_tool = next(t for t in tools if t.name == "call_localization_agent")
        result = call_loc_tool.func(instructions="Find relevant methods")

        assert result == {"instructions": "Find relevant methods"}

    def test_deferred_tool_call_composition_returns_instructions(self):
        """Verify call_composition_agent returns instructions as expected."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = GherkinSupervisorTools(llm=mock_llm, project_root="/tmp/test")
        tools, _ = tool_builder.all()

        call_comp_tool = next(t for t in tools if t.name == "call_composition_agent")
        result = call_comp_tool.func(instructions="Generate test code")

        assert result == {"instructions": "Generate test code"}

    def test_deferred_tool_finalize_returns_status(self):
        """Verify finalize tool returns expected status dict."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = BaseSupervisorTools(llm=mock_llm, project_root="/tmp/test")
        tools, _ = tool_builder.all()

        finalize_tool = next(t for t in tools if t.name == "finalize")
        result = finalize_tool.func()

        assert result == {"status": "finalize"}

    def test_deferred_tool_view_test_code_returns_inputs(self):
        """Verify view_test_code returns all input arguments."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = BaseSupervisorTools(llm=mock_llm, project_root="/tmp/test")
        tools, _ = tool_builder.all()

        view_tool = next(t for t in tools if t.name == "view_test_code")
        result = view_tool.func(
            qualified_class_name="org.example.TestClass",
            method_signature="testMethod()",
        )

        assert result["qualified_class_name"] == "org.example.TestClass"
        assert result["method_signature"] == "testMethod()"

    def test_deferred_tool_compile_and_execute_returns_empty(self):
        """Verify compile_and_execute_test returns empty dict."""
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tool_builder = BaseSupervisorTools(llm=mock_llm, project_root="/tmp/test")
        tools, _ = tool_builder.all()

        compile_tool = next(t for t in tools if t.name == "compile_and_execute_test")
        result = compile_tool.func()

        assert result == {}


class TestSupervisorForceEnd:
    """Tests for supervisor agent force_end behavior (NoArgs optimization)."""

    def test_execute_force_end_sets_finalize_called(self):
        """Verify _execute_force_end sets finalize_called=True without LLM call."""
        from sakura.nl2test.generation.supervisor.agent import SupervisorReActAgent

        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tools, _ = BaseSupervisorTools(llm=mock_llm, project_root="/tmp/test").all()
        agent = SupervisorReActAgent(
            llm=mock_llm,
            tools=tools,
            system_message="Test system message",
            nl_description="Test description",
            project_root="/tmp/test",
        )

        state = AgentState()
        assert state.finalize_called is False
        assert state.force_end_attempts == 0

        result_state = agent._execute_force_end(state)

        assert result_state.finalize_called is True
        assert result_state.final_comments == ""
        assert result_state.force_end_attempts == 1

    def test_execute_force_end_increments_attempts(self):
        """Verify _execute_force_end increments force_end_attempts each call."""
        from sakura.nl2test.generation.supervisor.agent import SupervisorReActAgent

        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tools, _ = BaseSupervisorTools(llm=mock_llm, project_root="/tmp/test").all()
        agent = SupervisorReActAgent(
            llm=mock_llm,
            tools=tools,
            system_message="Test system message",
            nl_description="Test description",
            project_root="/tmp/test",
        )

        state = AgentState()
        state.force_end_attempts = 2

        result_state = agent._execute_force_end(state)

        assert result_state.force_end_attempts == 3

    def test_force_finalize_prompt_methods_raise_not_implemented(self):
        """Verify prompt methods raise NotImplementedError since they're unused."""
        from sakura.nl2test.generation.supervisor.agent import SupervisorReActAgent

        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tools, _ = BaseSupervisorTools(llm=mock_llm, project_root="/tmp/test").all()
        agent = SupervisorReActAgent(
            llm=mock_llm,
            tools=tools,
            system_message="Test system message",
            nl_description="Test description",
            project_root="/tmp/test",
        )

        with pytest.raises(NotImplementedError):
            agent._get_force_finalize_system_prompt()

        with pytest.raises(NotImplementedError):
            agent._get_force_finalize_chat_prompt()

        with pytest.raises(NotImplementedError):
            agent._process_force_finalize_result(MagicMock(), AgentState())

    def test_get_finalize_schema_returns_no_args(self):
        """Verify _get_finalize_schema returns NoArgs."""
        from sakura.nl2test.generation.supervisor.agent import SupervisorReActAgent
        from sakura.nl2test.models.agents import NoArgs

        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        tools, _ = BaseSupervisorTools(llm=mock_llm, project_root="/tmp/test").all()
        agent = SupervisorReActAgent(
            llm=mock_llm,
            tools=tools,
            system_message="Test system message",
            nl_description="Test description",
            project_root="/tmp/test",
        )

        assert agent._get_finalize_schema() is NoArgs
