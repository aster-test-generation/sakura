from typing import Tuple, List, Union
from pathlib import Path

from cldk.analysis.java import JavaAnalysis
from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.nl2test.generation.localization.orchestrators import (
    GrammaticalLocalizationOrchestrator,
    GherkinLocalizationOrchestrator,
)
from nltest.nl2test.generation.supervisor.agent import SupervisorReActAgent
from nltest.nl2test.generation.supervisor.orchestrators.gherkin import (
    GherkinSupervisorOrchestrator,
)
from nltest.nl2test.generation.supervisor.orchestrators.grammatical import (
    GrammaticalSupervisorOrchestrator,
)
from nltest.nl2test.models import (
    AgentState,
    NL2TestInput,
    AtomicBlock,
    GrammaticalBlock,
    GrammaticalBlockList,
    AtomicBlockList,
    NL2LocalizationOutput,
    NL2EvaluationResults,
    LocalizationEvaluationResults,
    Scenario,
    LocalizedScenario,
)
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.nl2test.preprocessing.indexers import ClassIndexer
from nltest.nl2test.preprocessing.indexers import MethodIndexer
from nltest.nl2test.preprocessing.searchers import ClassSearcher
from nltest.nl2test.preprocessing.searchers import MethodSearcher
from nltest.nl2test.preprocessing.nl_decomposer import NLDecomposer
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.llm.llm_client import LLMClient, ClientType
from nltest.nl2test.evaluation.localization_grader import LocalizationGrader
from nltest.utils.pretty.prints import pretty_print
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.evaluation.test_grader import TestGrader
from nltest.utils.execution import JavaCompilation
from nltest.utils.file_io.test_file_manager import TestFileManager, TestFileInfo


class Pipeline:
    def __init__(
        self,
        analysis: JavaAnalysis,
        project_root: Path,
        *,
        decomposition_mode: DecompositionMode = DecompositionMode.GHERKIN,
        analysis_dir: Path | None = None,
    ):
        self.analysis = analysis
        self.project_root = Path(project_root)
        self.decomposition_mode = decomposition_mode
        self.analysis_dir = (
            Path(analysis_dir) if analysis_dir is not None else self.project_root
        )

        self.nl_decomposer = NLDecomposer(mode=self.decomposition_mode)
        self.method_indexer = MethodIndexer(analysis)
        self.class_indexer = ClassIndexer(analysis)

        self.decision_llm = LLMClient(ClientType.DECISION)
        self.structured_llm = LLMClient(ClientType.STRUCTURED)

        # Initialized during preprocessing
        self.method_searcher: MethodSearcher = None
        self.class_searcher: ClassSearcher = None

    def run_preprocessing(self) -> Tuple[MethodSearcher, ClassSearcher]:
        # Build search indices
        self.method_searcher = self.method_indexer.build_index()
        self.class_searcher = self.class_indexer.build_index()

        return self.method_searcher, self.class_searcher

    def decompose_natural_language(
        self, nl_description: str
    ) -> Union[GrammaticalBlockList, Scenario]:
        """Decompose natural language based on pipeline decomposition mode.

        Returns GrammaticalBlockList if GRAMMATICAL, or Scenario if GHERKIN.
        """
        return self.nl_decomposer.decompose(nl_description)

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
            assert (
                updated_state.localized_scenario is not None
            ), "Localization agent did not return LocalizedScenario"
            return (
                updated_state.localized_scenario,
                updated_state.final_comments or "No comments.",
            )
        else:
            assert (
                updated_state.atomic_blocks is not None
            ), "Localization agent did not return AtomicBlockList"
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
            nl2_input,
            self.analysis,
            self.project_root,
            decomposition_mode=self.decomposition_mode,
        )
        coverage_score, detailed_results = grader.grade(result, detailed_output=True)

        # Convert detailed_results dict to LocalizationEvaluationResults model
        evaluation_results = None
        if detailed_results:
            evaluation_results = LocalizationEvaluationResults(**detailed_results)

        # Create and return NL2LocalizationOutput
        return NL2LocalizationOutput(
            nl2_input=nl2_input,
            localized_blocks=result,
            evaluation_results=evaluation_results,
        )

    def _empty_nl2_evaluation(self, nl2_input: NL2TestInput) -> NL2EvaluationResults:
        """Create an empty NL2EvaluationResults object with zeroed metrics."""
        return NL2EvaluationResults(
            nl2_input=nl2_input,
            pred_class_name="",
            pred_method_signature="",
            gt_class_name=nl2_input.qualified_class_name,
            gt_method_signature=nl2_input.method_signature,
            structural_score=0.0,
            structural_metrics={
                "objects_created": 0.0,
                "constructors": 0.0,
                "application_calls": 0.0,
                "library_calls": 0.0,
                "assertions": 0.0,
                "order_aware_assertion_coverage": 0.0,
                "ground_truth_assertion_coverage": 0.0,
                "longest_callable_subsequence": 0.0,
                "ground_truth_callable_coverage": 0.0,
                "compilation_score": 0.0,
                "focal_method_coverage": 0.0,
            },
            test_code="",
        )

    def regenerate_analysis(self, *, eager: bool = True) -> JavaAnalysis:
        """Regenerate the Java analysis to include newly created test classes."""
        self.analysis = CLDK(language="java").analysis(
            project_path=self.project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=self.analysis_dir,
            eager=eager,
        )
        return self.analysis

    def run_nl2test(self, nl2_input: NL2TestInput) -> NL2EvaluationResults:
        # Decompose into initial blocks
        blocks = self.decompose_natural_language(nl2_input.description)

        # Prepare blocks for supervisor orchestrator
        if self.decomposition_mode == DecompositionMode.GHERKIN:
            if isinstance(blocks, Scenario):
                sup_blocks = LocalizedScenario.from_scenario(blocks)
            elif isinstance(blocks, LocalizedScenario):
                sup_blocks = blocks
            else:
                raise TypeError("Unexpected blocks type for GHERKIN mode.")
            orchestrator = GherkinSupervisorOrchestrator(
                analysis=self.analysis,
                method_searcher=self.method_searcher,
                class_searcher=self.class_searcher,
                nl2_input=nl2_input,
                base_project_dir=str(self.project_root),
            )
        else:
            if isinstance(blocks, GrammaticalBlockList):
                sup_blocks = AtomicBlockList(
                    atomic_blocks=[
                        AtomicBlock.from_grammatical_block(gb)
                        for gb in blocks.grammatical_blocks
                    ]
                )
            elif isinstance(blocks, AtomicBlockList):
                sup_blocks = blocks
            else:
                raise TypeError("Unexpected blocks type for GRAMMATICAL mode.")
            orchestrator = GrammaticalSupervisorOrchestrator(
                analysis=self.analysis,
                method_searcher=self.method_searcher,
                class_searcher=self.class_searcher,
                nl2_input=nl2_input,
                base_project_dir=str(self.project_root),
            )

        agent_state: AgentState = orchestrator.assign_task(sup_blocks)

        if not agent_state or not agent_state.class_name:
            return self._empty_nl2_evaluation(nl2_input)

        simple_class_name = agent_state.class_name
        package = agent_state.package or ""
        qualified_class_name = (
            f"{package}.{simple_class_name}" if package else simple_class_name
        )

        # Check compilation; if predicted class has errors, return empty
        erroneous_classes = JavaCompilation.get_erroneous_classes(self.project_root)
        predicted_file = f"{simple_class_name}.java"
        if any(ec == predicted_file for ec in erroneous_classes):
            return self._empty_nl2_evaluation(nl2_input)

        # Regenerate analysis to pick up new test class
        self.regenerate_analysis(eager=True)

        # Identify predicted test method (first test method found)
        common = CommonAnalysis(self.analysis)
        test_methods = common.get_test_methods_in_class(qualified_class_name)
        if not test_methods:
            return self._empty_nl2_evaluation(nl2_input)
        _, pred_method_sig = test_methods[0]

        # Grade structural similarity
        grader = TestGrader(
            self.analysis,
            self.project_root,
            project_erroneous_classes=set(erroneous_classes),
        )
        structural_score, structural_metrics = grader.grade_structural(
            pred_method_sig=pred_method_sig,
            pred_class_name=qualified_class_name,
            gt_method_sig=nl2_input.method_signature,
            gt_class_name=nl2_input.qualified_class_name,
        )

        # Prepare evaluation results
        eval_result = NL2EvaluationResults(
            nl2_input=nl2_input,
            pred_class_name=qualified_class_name,
            pred_method_signature=pred_method_sig,
            gt_class_name=nl2_input.qualified_class_name,
            gt_method_signature=nl2_input.method_signature,
            structural_score=structural_score,
            structural_metrics=structural_metrics,
        )

        # Load and attach test code, then delete the file
        fm = TestFileManager(self.project_root)
        info = TestFileInfo(qualified_class_name=qualified_class_name)
        try:
            code = fm.load(info, encode_class_name=False)
            eval_result.test_code = code
        except FileNotFoundError:
            eval_result.test_code = ""
        finally:
            try:
                fm.delete_single(info, encode_class_name=False)
            except Exception:
                pass

        return eval_result
