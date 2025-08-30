from typing import Tuple, List, Union
from pathlib import Path

from cldk.analysis.java import JavaAnalysis

from nltest.nl2test.generation.localization.orchestrator import LocalizationOrchestrator
from nltest.nl2test.generation.composition.orchestrator import CompositionOrchestrator
from nltest.nl2test.generation.supervisor.agent import SupervisorReActAgent
from nltest.nl2test.generation.supervisor.tools import SupervisorTools
from nltest.nl2test.models import (
    AgentState,
    NL2TestInput,
    AtomicBlock,
    GrammaticalBlock,
    GrammaticalBlockList,
    AtomicBlockList,
    NL2LocalizationOutput,
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
from nltest.utils.file_io.structured_data_manager import StructuredDataManager
from nltest.utils.config.config import Config
from nltest.utils.pretty.prints import pretty_print


class Pipeline:
    def __init__(self, analysis: JavaAnalysis, project_root: Path, *, decomposition_mode: DecompositionMode = DecompositionMode.GRAMMATICAL):
        self.analysis = analysis
        self.project_root = Path(project_root)
        self.decomposition_mode = decomposition_mode
        
        # Get output directory from config
        config = Config()
        output_dir = Path(config.get("project", "output_dir"))
        self.data_manager = StructuredDataManager(output_dir)
        
        self.nl_decomposer = NLDecomposer(mode=self.decomposition_mode)
        self.method_indexer = MethodIndexer(analysis)
        self.class_indexer = ClassIndexer(analysis)
        
        self.decision_llm = LLMClient(ClientType.DECISION)
        self.structured_llm = LLMClient(ClientType.STRUCTURED)

        # Initialized during preprocessing
        self.method_searcher: MethodSearcher = None
        self.class_searcher: ClassSearcher = None
        
        # Initialized during respective methods
        self.localization_orchestrator: LocalizationOrchestrator = None
        self.composition_orchestrator: CompositionOrchestrator = None
        self.supervisor_agent: SupervisorReActAgent = None

    def run_preprocessing(self) -> Tuple[MethodSearcher, ClassSearcher]:
        # Build search indices
        self.method_searcher = self.method_indexer.build_index()
        self.class_searcher = self.class_indexer.build_index()
        
        return self.method_searcher, self.class_searcher

    def decompose_natural_language(self, nl_description: str) -> Union[GrammaticalBlockList, Scenario]:
        """Decompose natural language based on pipeline decomposition mode.

        Returns GrammaticalBlockList if GRAMMATICAL, or Scenario if GHERKIN.
        """
        return self.nl_decomposer.decompose(nl_description)

    def run_localization_agent(
        self,
        nl2_input: NL2TestInput,
        blocks: Union[GrammaticalBlockList, Scenario],
    ) -> Tuple[Union[AtomicBlockList, LocalizedScenario], str]:
        if not self.method_searcher or not self.class_searcher:
            raise RuntimeError("Preprocessing must be run before localization agent")
            
        self.localization_orchestrator = LocalizationOrchestrator(
            analysis=self.analysis,
            method_searcher=self.method_searcher,
            class_searcher=self.class_searcher,
            nl2_input=nl2_input,
            decomposition_mode=self.decomposition_mode,
        )
        
        instructions = (
            "Localize to relevant code: For grammatical blocks, convert them into atomic blocks"
            " with candidate methods and notes. For Gherkin scenarios, produce a LocalizedScenario"
            " that annotates each block with candidate methods and comments."
        )
        if self.decomposition_mode == DecompositionMode.GHERKIN:
            return self.localization_orchestrator.assign_task(instructions, scenario=blocks)  # type: ignore[arg-type]
        else:
            return self.localization_orchestrator.assign_task(instructions, grammatical_blocks=blocks)  # type: ignore[arg-type]

    def run_localization_evaluation_pipeline(self, nl2_input: NL2TestInput) -> NL2LocalizationOutput:
        # Run preprocessing
        self.run_preprocessing()
        
        # Decompose natural language (type depends on mode)
        blocks = self.decompose_natural_language(nl2_input.description)

        # Run localization agent
        result, comments = self.run_localization_agent(nl2_input, blocks)

        # Only grammatical mode is currently supported for evaluation
        if self.decomposition_mode != DecompositionMode.GRAMMATICAL:
            raise RuntimeError("run_localization_evaluation_pipeline currently supports GRAMMATICAL mode only.")

        if not isinstance(result, AtomicBlockList):
            raise TypeError("Expected AtomicBlockList from localization in GRAMMATICAL mode.")

        localized_blocks = result

        # Run evaluation using LocalizationGrader with detailed output
        grader = LocalizationGrader(nl2_input, self.analysis, self.project_root)
        coverage_score, detailed_results = grader.grade(localized_blocks, detailed_output=True)
        
        # Convert detailed_results dict to LocalizationEvaluationResults model
        evaluation_results = None
        if detailed_results:
            evaluation_results = LocalizationEvaluationResults(**detailed_results)
        
        # Create and return NL2LocalizationOutput
        return NL2LocalizationOutput(
            nl2_input=nl2_input,
            localized_blocks=localized_blocks,
            evaluation_results=evaluation_results,
            coverage_score=coverage_score
        )
