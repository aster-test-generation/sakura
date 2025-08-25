from typing import Tuple, List
from pathlib import Path

from cldk.analysis.java import JavaAnalysis

from nltest.nl2test.generation.localization.orchestrator import LocalizationOrchestrator
from nltest.nl2test.generation.composition.orchestrator import CompositionOrchestrator
from nltest.nl2test.generation.supervisor.agent import SupervisorReActAgent
from nltest.nl2test.generation.supervisor.tools import SupervisorTools
from nltest.nl2test.model.models import AgentState, NL2TestInput, AtomicBlock, GrammaticalBlock
from nltest.nl2test.preprocessing.indexers.class_indexer import ClassIndexer
from nltest.nl2test.preprocessing.indexers.method_indexer import MethodIndexer
from nltest.nl2test.preprocessing.searchers.class_searcher import ClassSearcher
from nltest.nl2test.preprocessing.searchers.method_searcher import MethodSearcher
from nltest.nl2test.preprocessing.nl_decomposer import NLDecomposer
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.llm.llm_client import LLMClient, ClientType
from nltest.nl2test.evaluation.localization_grader import LocalizationGrader
from nltest.utils.file_io.structured_data_manager import StructuredDataManager
from nltest.utils.config.config import Config


class Pipeline:
    def __init__(self, analysis: JavaAnalysis, project_root: Path):
        self.analysis = analysis
        self.project_root = Path(project_root)
        
        # Get output directory from config
        config = Config()
        output_dir = Path(config.get("project", "output_dir"))
        self.data_manager = StructuredDataManager(output_dir)
        
        self.nl_decomposer = NLDecomposer()
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

    def decompose_natural_language(self, nl_description: str) -> List[GrammaticalBlock]:
        return self.nl_decomposer.decompose(nl_description)

    def run_localization_agent(self, nl2_input: NL2TestInput, atomic_blocks: List[AtomicBlock]) -> Tuple[List[AtomicBlock], str]:
        if not self.method_searcher or not self.class_searcher:
            raise RuntimeError("Preprocessing must be run before localization agent")
            
        self.localization_orchestrator = LocalizationOrchestrator(
            analysis=self.analysis,
            method_searcher=self.method_searcher,
            class_searcher=self.class_searcher,
            nl2_input=nl2_input
        )
        
        instructions = "Localize each atomic block to relevant methods in the code base, looking for both application and inherited library methods."
        return self.localization_orchestrator.assign_task(instructions, atomic_blocks)

    def run_localization_evaluation_pipeline(self, nl2_input: NL2TestInput) -> Tuple[List[AtomicBlock], float]:
        # Run preprocessing
        self.run_preprocessing()
        
        # Decompose natural language
        grammatical_blocks = self.decompose_natural_language(nl2_input.description)
        
        # Convert to atomic blocks
        atomic_blocks = [
            AtomicBlock.from_grammatical_block(block) for block in grammatical_blocks
        ]
        
        # Run localization agent
        localized_blocks, _ = self.run_localization_agent(nl2_input, atomic_blocks)
        
        # Run evaluation using LocalizationGrader with detailed output
        grader = LocalizationGrader(nl2_input, self.analysis, self.project_root)
        coverage_score, detailed_results = grader.grade(localized_blocks, detailed_output=True)
        
        # Save detailed results to output directory
        if detailed_results:
            # Create a filename based on test class and method
            safe_class_name = nl2_input.qualified_class_name.replace(".", "_")
            safe_method_name = nl2_input.method_signature.replace("(", "_").replace(")", "_").replace(" ", "_")
            filename = f"localization_evaluation_{safe_class_name}_{safe_method_name}.json"
            
            self.data_manager.save(filename, detailed_results, format="json")
        
        return localized_blocks, coverage_score