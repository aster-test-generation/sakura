from pathlib import Path
from typing import List, Optional, Tuple, Union

from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis

from nltest.nl2test.evaluation.localization_grader import LocalizationGrader
from nltest.nl2test.generation.localization.orchestrators import (
    GherkinLocalizationOrchestrator,
    GrammaticalLocalizationOrchestrator,
)
from nltest.nl2test.generation.supervisor.orchestrators.gherkin import (
    GherkinSupervisorOrchestrator,
)
from nltest.nl2test.generation.supervisor.orchestrators.grammatical import (
    GrammaticalSupervisorOrchestrator,
)
from nltest.nl2test.models import (
    AgentState,
    AtomicBlock,
    AtomicBlockList,
    GrammaticalBlockList,
    LocalizationEval,
    LocalizedScenario,
    NL2LocalizationOutput,
    NL2TestInput,
    NL2TestMetadata,
    Scenario,
)
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.preprocessing.indexers import ClassIndexer, MethodIndexer
from nltest.nl2test.preprocessing.nl_decomposer import NLDecomposer
from nltest.nl2test.preprocessing.searchers import ClassSearcher, MethodSearcher
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.compilation.maven import CompilationError, JavaMavenCompilation
from nltest.utils.evaluation import TestGrader
from nltest.utils.exceptions import ProjectCompilationError
from nltest.utils.file_io.test_file_manager import TestFileInfo, TestFileManager
from nltest.utils.llm import UsageTracker
from nltest.utils.models import (
    AgentToolLog,
    NL2TestCoverageEval,
    NL2TestEval,
    NL2TestStructuralEval,
    ToolLog,
)
from nltest.utils.pretty.color_logger import RichLog


class Pipeline:
    def __init__(
        self,
        analysis: JavaAnalysis,
        *,
        project_root: Path,
        analysis_dir: Path,
        grading_analysis_dir: Path | None = None,
        decomposition_mode: DecompositionMode = DecompositionMode.GHERKIN,
    ):
        self.analysis = analysis
        # Expect absolute paths for project and analysis directories.
        self.project_root = Path(project_root)
        self.decomposition_mode = decomposition_mode
        self.analysis_dir = Path(analysis_dir)
        self.grading_analysis_dir = (
            Path(grading_analysis_dir) if grading_analysis_dir else self.analysis_dir
        )

        self.method_indexer = MethodIndexer(analysis)
        self.class_indexer = ClassIndexer(analysis)

        # Initialized during preprocessing
        self.method_searcher: Optional[MethodSearcher] = None
        self.class_searcher: Optional[ClassSearcher] = None

        # Initialize TestGrader with current analysis and application classes
        self.common = CommonAnalysis(self.analysis)
        _, _application_classes, _test_utility_classes = (
            self.common.categorize_classes()
        )
        self.application_classes = _application_classes
        self.test_utility_classes = _test_utility_classes
        self.test_grader = TestGrader(
            analysis=self.analysis,
            project_root=self.project_root,
            application_classes=_application_classes,
            test_utility_classes=_test_utility_classes,
        )
        self.localization_grader = LocalizationGrader(
            analysis=self.analysis,
            project_root=self.project_root,
            decomposition_mode=self.decomposition_mode,
            application_classes=_application_classes,
            test_utility_classes=_test_utility_classes,
        )

    def run_project_compilation(self) -> List[CompilationError]:
        """Compile the project before test generation."""
        return JavaMavenCompilation(self.project_root).get_compilation_errors()

    def run_preprocessing(
        self,
        *,
        exclude_test_dirs: bool = False,
    ) -> Tuple[MethodSearcher, ClassSearcher]:
        # Build search indices with optional filtering of test sources
        self.method_searcher = self.method_indexer.build_index(
            exclude_test_dirs=exclude_test_dirs
        )
        self.class_searcher = self.class_indexer.build_index(
            exclude_test_dirs=exclude_test_dirs
        )

        return self.method_searcher, self.class_searcher

    def decompose_natural_language(
        self,
        nl_description: str,
        usage_tracker: UsageTracker | None = None,
    ) -> Union[GrammaticalBlockList, Scenario]:
        """Decompose natural language based on pipeline decomposition mode.

        Returns GrammaticalBlockList if GRAMMATICAL, or Scenario if GHERKIN.
        """
        nl_decomposer = NLDecomposer(
            mode=self.decomposition_mode,
            usage_tracker=usage_tracker,
        )
        return nl_decomposer.decompose(nl_description)

    def run_localization_agent(
        self,
        nl2_input: NL2TestInput,
        blocks: Union[
            GrammaticalBlockList, Scenario, AtomicBlockList, LocalizedScenario
        ],
    ) -> Tuple[Union[AtomicBlockList, LocalizedScenario], str]:
        if not self.method_searcher or not self.class_searcher:
            raise RuntimeError("Preprocessing must be run before localization agent")

        instructions = "Localize to relevant code with candidate methods and comments for each block."

        if self.decomposition_mode == DecompositionMode.GHERKIN:
            localization_orchestrator = GherkinLocalizationOrchestrator(
                analysis=self.analysis,
                method_searcher=self.method_searcher,
                class_searcher=self.class_searcher,
                nl2_input=nl2_input,
            )
            # Ensure we pass a LocalizedScenario to the orchestrator
            if isinstance(blocks, Scenario):
                blocks = LocalizedScenario.from_scenario(blocks)
            # If already LocalizedScenario, pass through
        else:
            localization_orchestrator = GrammaticalLocalizationOrchestrator(
                analysis=self.analysis,
                method_searcher=self.method_searcher,
                class_searcher=self.class_searcher,
                nl2_input=nl2_input,
            )
            # Ensure we pass an AtomicBlockList to the orchestrator
            if isinstance(blocks, GrammaticalBlockList):
                blocks = AtomicBlockList(
                    atomic_blocks=[
                        AtomicBlock.from_grammatical_block(gb)
                        for gb in blocks.grammatical_blocks
                    ]
                )

        # New behavior: assign_task returns an AgentState; extract outputs.
        updated_state: AgentState = localization_orchestrator.assign_task(
            blocks, instructions=instructions
        )
        if self.decomposition_mode == DecompositionMode.GHERKIN:
            assert updated_state.localized_scenario is not None, (
                "Localization agent did not return LocalizedScenario"
            )
            return (
                updated_state.localized_scenario,
                updated_state.final_comments or "No comments.",
            )
        else:
            assert updated_state.atomic_blocks is not None, (
                "Localization agent did not return AtomicBlockList"
            )
            return (
                updated_state.atomic_blocks,
                updated_state.final_comments or "No comments.",
            )

    def run_localization_evaluation_pipeline(
        self, nl2_input: NL2TestInput
    ) -> NL2LocalizationOutput:
        # Ensure preprocessing was run
        if not self.method_searcher or not self.class_searcher:
            raise RuntimeError(
                "Preprocessing has not been run. Call run_preprocessing() before evaluating."
            )

        # Decompose natural language (type depends on mode)
        blocks = self.decompose_natural_language(nl2_input.description)

        # Run localization agent
        result, comments = self.run_localization_agent(nl2_input, blocks)

        # Run evaluation using LocalizationGrader with detailed output
        grader = LocalizationGrader(
            analysis=self.analysis,
            project_root=self.project_root,
            decomposition_mode=self.decomposition_mode,
            application_classes=self.application_classes,
            test_utility_classes=self.test_utility_classes,
        )
        evaluation_results = grader.grade(result, nl2_input)

        # Create and return NL2LocalizationOutput
        return NL2LocalizationOutput(
            nl2_input=nl2_input,
            localized_blocks=result,
            evaluation_results=evaluation_results,
        )

    def _empty_localization_eval(self, nl2_input: NL2TestInput) -> LocalizationEval:
        return LocalizationEval(
            qualified_class_name=nl2_input.qualified_class_name,
            method_signature=nl2_input.method_signature,
            all_focal_methods=[],
            covered_focal_methods=[],
            uncovered_focal_methods=[],
            tp=0,
            fn=0,
            localization_recall=0.0,
        )

    def _agent_tool_log_from_state(self, state: AgentState | None) -> AgentToolLog:
        if not state:
            return AgentToolLog(tool_counts={}, tool_trajectories=[])

        tool_counts = {
            tool_name: sum(arg_counts.values())
            for tool_name, arg_counts in state.total_tool_calls.items()
        }
        tool_trajectories = [
            trajectory.copy() for trajectory in state.tool_trajectories
        ]
        return AgentToolLog(
            tool_counts=tool_counts,
            tool_trajectories=tool_trajectories,
        )

    def _localization_eval_from_state(
        self, supervisor_state: AgentState | None, nl2_input: NL2TestInput
    ) -> LocalizationEval:
        localization_eval = self._empty_localization_eval(nl2_input)
        if not supervisor_state:
            return localization_eval

        localization_target = (
            supervisor_state.localized_scenario
            if self.decomposition_mode == DecompositionMode.GHERKIN
            else supervisor_state.atomic_blocks
        )
        if localization_target is None:
            return localization_eval

        try:
            return self.localization_grader.grade(localization_target, nl2_input)
        except Exception as exc:
            RichLog.warn(
                "Localization evaluation failed for "
                f"{nl2_input.qualified_class_name}::{nl2_input.method_signature} "
                f"(id={nl2_input.id}): {exc}"
            )
            return localization_eval

    def _build_tool_log(
        self,
        supervisor_state: AgentState | None,
        localization_state: AgentState | None,
        composition_state: AgentState | None,
    ) -> ToolLog:
        return ToolLog(
            supervisor_tool_log=self._agent_tool_log_from_state(supervisor_state),
            localization_tool_log=self._agent_tool_log_from_state(localization_state),
            composition_tool_log=self._agent_tool_log_from_state(composition_state),
        )

    def _empty_nl2test_eval(self, nl2_input: NL2TestInput) -> NL2TestEval:
        """Create a default NL2TestEval with empty grading results."""
        return NL2TestEval(
            compiles=False,
            nl2test_input=nl2_input,
            nl2test_metadata=NL2TestMetadata(qualified_test_class_name="", code=""),
            structured_eval=None,
            coverage_eval=None,
            localization_eval=None,
            tool_log=None,
            input_tokens=0,
            output_tokens=0,
            llm_calls=0,
        )

    @staticmethod
    def _zero_structural_eval() -> NL2TestStructuralEval:
        return NL2TestStructuralEval(
            obj_creation_recall=0.0,
            obj_creation_precision=0.0,
            assertion_recall=0.0,
            assertion_precision=0.0,
            callable_recall=0.0,
            callable_precision=0.0,
            focal_recall=0.0,
            focal_precision=0.0,
        )

    @staticmethod
    def _zero_coverage_eval() -> NL2TestCoverageEval:
        return NL2TestCoverageEval(
            class_coverage=0.0,
            method_coverage=0.0,
            line_coverage=0.0,
            branch_coverage=0.0,
        )

    def regenerate_analysis(self, *, eager: bool = True) -> JavaAnalysis:
        """Regenerate analysis for grading newly created test classes."""
        self.grading_analysis_dir.mkdir(parents=True, exist_ok=True)
        analysis = CLDK(language="java").analysis(
            project_path=self.project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=self.grading_analysis_dir,
            eager=eager,
        )
        return analysis

    def run_nl2test(self, nl2_input: NL2TestInput) -> NL2TestEval:
        if not self.method_searcher or not self.class_searcher:
            raise Exception("Preprocessing not completed...")

        run_usage_tracker = UsageTracker()

        self.test_grader.set_analysis(self.analysis)
        self.localization_grader.set_analysis(self.analysis)

        # Decompose into initial blocks
        blocks = self.decompose_natural_language(
            nl2_input.description,
            usage_tracker=run_usage_tracker,
        )

        # module_root is absolute (or None) from CommonAnalysis.
        # test_base_dir is absolute, anchored to module_root or project_root.
        # resolved_module_root is absolute (module_root when present, else project_root).
        module_root = self.common.resolve_module_root(nl2_input.qualified_class_name)
        test_base_dir = self.common.resolve_test_base_dir(
            module_root, project_root=self.project_root
        )
        resolved_module_root = module_root or self.project_root

        # Prepare blocks for supervisor orchestrator
        if self.decomposition_mode == DecompositionMode.GHERKIN:
            if isinstance(blocks, Scenario):
                sup_blocks = LocalizedScenario.from_scenario(blocks)
            else:
                raise TypeError("Unexpected blocks type for GHERKIN mode.")
            supervisor = GherkinSupervisorOrchestrator(
                analysis=self.analysis,
                method_searcher=self.method_searcher,
                class_searcher=self.class_searcher,
                nl2_input=nl2_input,
                base_project_dir=str(self.project_root),
                test_base_dir=test_base_dir,
                module_root=resolved_module_root,
                usage_tracker=run_usage_tracker,
            )
        else:
            if isinstance(blocks, GrammaticalBlockList):
                sup_blocks = AtomicBlockList(
                    atomic_blocks=[
                        AtomicBlock.from_grammatical_block(gb)
                        for gb in blocks.grammatical_blocks
                    ]
                )
            else:
                raise TypeError("Unexpected blocks type for GRAMMATICAL mode.")
            supervisor = GrammaticalSupervisorOrchestrator(
                analysis=self.analysis,
                method_searcher=self.method_searcher,
                class_searcher=self.class_searcher,
                nl2_input=nl2_input,
                base_project_dir=str(self.project_root),
                test_base_dir=test_base_dir,
                module_root=resolved_module_root,
                usage_tracker=run_usage_tracker,
            )

        try:
            supervisor_state, localization_state, composition_state = (
                supervisor.assign_task(sup_blocks)
            )
        except ProjectCompilationError as exc:
            raise ProjectCompilationError(
                f"Project compilation failed outside the generated test for input {nl2_input.id}.",
                extra_info=getattr(exc, "extra_info", {}),
            ) from exc

        tool_log = self._build_tool_log(
            supervisor_state, localization_state, composition_state
        )

        has_class_name = bool(
            supervisor_state and (supervisor_state.class_name or "").strip()
        )

        final_result: NL2TestEval = self._empty_nl2test_eval(nl2_input)
        final_result.tool_log = tool_log

        if not has_class_name:
            # If no class name from composition agent, use localization agent to determine localization effectiveness and return
            final_result.localization_eval = self._localization_eval_from_state(
                supervisor_state, nl2_input
            )
            final_result.structured_eval = self._zero_structural_eval()
            final_result.coverage_eval = self._zero_coverage_eval()
        else:
            class_name = supervisor_state.class_name
            simple_class_name = class_name.strip() if class_name else ""
            package = (supervisor_state.package or "").strip()
            method_signature = (
                (supervisor_state.method_signature or "").strip()
                if supervisor_state
                else ""
            )
            qualified_test_class_name = (
                f"{package}.{simple_class_name}" if package else simple_class_name
            )

            # Build NL2TestMetadata for the predicted class; code filled after grading
            nl2_metadata = NL2TestMetadata(
                qualified_test_class_name=qualified_test_class_name,
                code="",
                method_signature=method_signature or None,
            )
            final_result.nl2test_metadata = nl2_metadata

            pred_simple_file = qualified_test_class_name.rsplit(".", 1)[-1] + ".java"
            pred_rel_path = qualified_test_class_name.replace(".", "/") + ".java"

            def _matches_error_path(err: str) -> bool:
                normalized = err.replace("\\", "/")
                if "/" in normalized:
                    return normalized.endswith(pred_rel_path)
                return normalized.endswith(pred_simple_file)

            compilation_errors: List[CompilationError] = JavaMavenCompilation(
                self.project_root, module_root=resolved_module_root
            ).get_compilation_errors()
            erroneous_files = [
                compilation_error.file for compilation_error in compilation_errors
            ]
            final_result.compiles = not any(
                _matches_error_path(err) for err in erroneous_files
            )

            try:
                # Regenerate analysis to pick up new test class and update grader
                new_analysis = self.regenerate_analysis(eager=True)
            except Exception as exc:
                RichLog.warn(
                    "Regenerating analysis failed for "
                    f"{qualified_test_class_name}::{method_signature} "
                    f"(id={nl2_input.id}); skipping grading: {exc}"
                )
            else:
                self.test_grader.set_analysis(new_analysis)
                self.localization_grader.set_analysis(new_analysis)

                structured_eval, coverage_eval = self.test_grader.grade(
                    nl2_input, nl2_metadata, compiles=final_result.compiles
                )
                final_result.structured_eval = structured_eval
                final_result.coverage_eval = coverage_eval
                final_result.localization_eval = self._localization_eval_from_state(
                    supervisor_state, nl2_input
                )

            fm = TestFileManager(self.project_root, test_base_dir=test_base_dir)
            info = TestFileInfo(qualified_class_name=qualified_test_class_name)
            try:
                code = fm.load(info, encode_class_name=False)
                final_result.nl2test_metadata.code = code
            except FileNotFoundError:
                pass
            finally:
                try:
                    fm.delete_single(info, encode_class_name=False)
                except Exception:
                    pass

        totals = run_usage_tracker.totals()
        final_result.input_tokens = totals["input_tokens"]
        final_result.output_tokens = totals["output_tokens"]
        final_result.llm_calls = totals["calls"]

        return final_result
