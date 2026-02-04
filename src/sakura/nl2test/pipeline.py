_D=True
_C=False
_B=.0
_A=None
from pathlib import Path
from typing import List,Optional,Tuple,Union
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis
from sakura.nl2test.evaluation.localization_grader import LocalizationGrader
from sakura.nl2test.generation.localization.orchestrators import GherkinLocalizationOrchestrator,GrammaticalLocalizationOrchestrator
from sakura.nl2test.generation.supervisor.orchestrators.gherkin import GherkinSupervisorOrchestrator
from sakura.nl2test.generation.supervisor.orchestrators.grammatical import GrammaticalSupervisorOrchestrator
from sakura.nl2test.models import AgentState,AtomicBlock,AtomicBlockList,GrammaticalBlockList,LocalizedScenario,NL2LocalizationOutput,NL2TestInput,NL2TestMetadata,Scenario
from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.nl2test.preprocessing.indexers import ClassIndexer,MethodIndexer
from sakura.nl2test.preprocessing.nl_decomposer import NLDecomposer
from sakura.nl2test.preprocessing.searchers import ClassSearcher,MethodSearcher
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.compilation.maven import CompilationError,JavaMavenCompilation
from sakura.utils.evaluation import TestGrader
from sakura.utils.exceptions import ProjectCompilationError
from sakura.utils.file_io.test_file_manager import TestFileInfo,TestFileManager
from sakura.utils.llm import UsageTracker
from sakura.utils.models import AgentToolLog,NL2TestCoverageEval,NL2TestEval,NL2TestPipelineResult,NL2TestStructuralEval,ToolLog
from sakura.utils.pretty.color_logger import RichLog
class Pipeline:
	def __init__(A,analysis:JavaAnalysis,*,project_root:Path,analysis_dir:Path,grading_analysis_dir:Path|_A=_A,decomposition_mode:DecompositionMode=DecompositionMode.GHERKIN,application_classes:list[str],test_utility_classes:list[str],common_analysis:CommonAnalysis,grading_analysis:JavaAnalysis|_A=_A):C=grading_analysis_dir;B=analysis;A.analysis=B;A.grading_analysis=grading_analysis or B;A.project_root=Path(project_root);A.decomposition_mode=decomposition_mode;A.analysis_dir=Path(analysis_dir);A.grading_analysis_dir=Path(C)if C else A.analysis_dir;A.method_indexer=MethodIndexer(B);A.class_indexer=ClassIndexer(B);(A.method_searcher):Optional[MethodSearcher]=_A;(A.class_searcher):Optional[ClassSearcher]=_A;A.common=common_analysis;A.application_classes=list(application_classes);A.test_utility_classes=list(test_utility_classes);A.test_grader=TestGrader(analysis=A.grading_analysis,project_root=A.project_root,application_classes=A.application_classes,test_utility_classes=A.test_utility_classes);A.localization_grader=LocalizationGrader(analysis=A.grading_analysis,project_root=A.project_root,decomposition_mode=A.decomposition_mode,application_classes=A.application_classes,test_utility_classes=A.test_utility_classes)
	def run_project_compilation(A)->List[CompilationError]:'Compile the project before test generation.';return JavaMavenCompilation(A.project_root).get_compilation_errors()
	def run_preprocessing(A,*,exclude_test_dirs:bool=_C)->Tuple[MethodSearcher,ClassSearcher]:B=exclude_test_dirs;A.method_searcher=A.method_indexer.build_index(exclude_test_dirs=B);A.class_searcher=A.class_indexer.build_index(exclude_test_dirs=B);return A.method_searcher,A.class_searcher
	def decompose_natural_language(A,nl_description:str,usage_tracker:UsageTracker|_A=_A)->Union[GrammaticalBlockList,Scenario]:'Decompose natural language based on pipeline decomposition mode.\n\n        Returns GrammaticalBlockList if GRAMMATICAL, or Scenario if GHERKIN.\n        ';B=NLDecomposer(mode=A.decomposition_mode,usage_tracker=usage_tracker);return B.decompose(nl_description)
	def run_localization_agent(A,nl2_input:NL2TestInput,blocks:Union[GrammaticalBlockList,Scenario,AtomicBlockList,LocalizedScenario])->Tuple[Union[AtomicBlockList,LocalizedScenario],str]:
		F='No comments.';D=nl2_input;B=blocks
		if not A.method_searcher or not A.class_searcher:raise RuntimeError('Preprocessing must be run before localization agent')
		G='Localize to relevant code with candidate methods and comments for each block.'
		if A.decomposition_mode==DecompositionMode.GHERKIN:
			E=GherkinLocalizationOrchestrator(analysis=A.analysis,method_searcher=A.method_searcher,class_searcher=A.class_searcher,nl2_input=D)
			if isinstance(B,Scenario):B=LocalizedScenario.from_scenario(B)
		else:
			E=GrammaticalLocalizationOrchestrator(analysis=A.analysis,method_searcher=A.method_searcher,class_searcher=A.class_searcher,nl2_input=D)
			if isinstance(B,GrammaticalBlockList):B=AtomicBlockList(atomic_blocks=[AtomicBlock.from_grammatical_block(A)for A in B.grammatical_blocks])
		C:AgentState=E.assign_task(B,instructions=G)
		if A.decomposition_mode==DecompositionMode.GHERKIN:assert C.localized_scenario is not _A,'Localization agent did not return LocalizedScenario';return C.localized_scenario,C.final_comments or F
		else:assert C.atomic_blocks is not _A,'Localization agent did not return AtomicBlockList';return C.atomic_blocks,C.final_comments or F
	def run_localization_evaluation_pipeline(A,nl2_input:NL2TestInput)->NL2LocalizationOutput:
		B=nl2_input
		if not A.method_searcher or not A.class_searcher:raise RuntimeError('Preprocessing has not been run. Call run_preprocessing() before evaluating.')
		D=A.decompose_natural_language(B.description);C,G=A.run_localization_agent(B,D);E=LocalizationGrader(analysis=A.grading_analysis,project_root=A.project_root,decomposition_mode=A.decomposition_mode,application_classes=A.application_classes,test_utility_classes=A.test_utility_classes);F=E.grade(C,B);return NL2LocalizationOutput(nl2_input=B,localized_blocks=C,evaluation_results=F)
	def _agent_tool_log_from_state(D,state:AgentState|_A)->AgentToolLog:
		A=state
		if not A:return AgentToolLog(tool_counts={},tool_trajectories=[])
		B={A:sum(B.values())for(A,B)in A.total_tool_calls.items()};C=[A.copy()for A in A.tool_trajectories];return AgentToolLog(tool_counts=B,tool_trajectories=C)
	def _build_tool_log(A,supervisor_state:AgentState|_A,localization_state:AgentState|_A,composition_state:AgentState|_A)->ToolLog:return ToolLog(supervisor_tool_log=A._agent_tool_log_from_state(supervisor_state),localization_tool_log=A._agent_tool_log_from_state(localization_state),composition_tool_log=A._agent_tool_log_from_state(composition_state))
	def _empty_nl2test_eval(A,nl2_input:NL2TestInput)->NL2TestEval:'Create a default NL2TestEval with empty grading results.';return NL2TestEval(compiles=_C,nl2test_input=nl2_input,nl2test_metadata=NL2TestMetadata(qualified_test_class_name='',code=''),structured_eval=_A,coverage_eval=_A,localization_eval=_A,tool_log=_A,input_tokens=0,output_tokens=0,llm_calls=0)
	@staticmethod
	def _zero_structural_eval()->NL2TestStructuralEval:return NL2TestStructuralEval(obj_creation_recall=_B,obj_creation_precision=_B,assertion_recall=_B,assertion_precision=_B,callable_recall=_B,callable_precision=_B,focal_recall=_B,focal_precision=_B)
	@staticmethod
	def _zero_coverage_eval()->NL2TestCoverageEval:return NL2TestCoverageEval(class_coverage=_B,method_coverage=_B,line_coverage=_B,branch_coverage=_B)
	def regenerate_analysis(A,*,eager:bool=_D)->JavaAnalysis:'Regenerate analysis for grading newly created test classes.';A.grading_analysis_dir.mkdir(parents=_D,exist_ok=_D);B=CLDK(language='java').analysis(project_path=A.project_root,analysis_backend_path=_A,analysis_level=AnalysisLevel.symbol_table,analysis_json_path=A.grading_analysis_dir,eager=eager);return B
	def run_nl2test(A,nl2_input:NL2TestInput)->NL2TestPipelineResult:
		Y='.java';L='/';D=nl2_input
		if not A.method_searcher or not A.class_searcher:raise Exception('Preprocessing not completed...')
		F=UsageTracker();A.test_grader.set_analysis(A.grading_analysis);A.localization_grader.set_analysis(A.grading_analysis);G=A.decompose_natural_language(D.description,usage_tracker=F);M=A.common.resolve_module_root(D.qualified_class_name);I=A.common.resolve_test_base_dir(M,project_root=A.project_root);J=M or A.project_root
		if A.decomposition_mode==DecompositionMode.GHERKIN:
			if isinstance(G,Scenario):N=LocalizedScenario.from_scenario(G)
			else:raise TypeError('Unexpected blocks type for GHERKIN mode.')
			O=GherkinSupervisorOrchestrator(analysis=A.analysis,method_searcher=A.method_searcher,class_searcher=A.class_searcher,nl2_input=D,base_project_dir=str(A.project_root),test_base_dir=I,module_root=J,usage_tracker=F)
		else:
			if isinstance(G,GrammaticalBlockList):N=AtomicBlockList(atomic_blocks=[AtomicBlock.from_grammatical_block(A)for A in G.grammatical_blocks])
			else:raise TypeError('Unexpected blocks type for GRAMMATICAL mode.')
			O=GrammaticalSupervisorOrchestrator(analysis=A.analysis,method_searcher=A.method_searcher,class_searcher=A.class_searcher,nl2_input=D,base_project_dir=str(A.project_root),test_base_dir=I,module_root=J,usage_tracker=F)
		try:C,Z,a=O.assign_task(N)
		except ProjectCompilationError as H:raise ProjectCompilationError(f"Project compilation failed outside the generated test for input {D.id}.",extra_info=getattr(H,'extra_info',{}))from H
		b=A._build_tool_log(C,Z,a);c=bool(C and(C.class_name or'').strip());B:NL2TestEval=A._empty_nl2test_eval(D);B.tool_log=b
		if not c:B.localization_eval=A.localization_grader.grade_from_state(C,D);B.structured_eval=A._zero_structural_eval();B.coverage_eval=A._zero_coverage_eval()
		else:
			P=C.class_name;Q=P.strip()if P else'';R=(C.package or'').strip();S=(C.method_signature or'').strip()if C else'';E=f"{R}.{Q}"if R else Q;T=NL2TestMetadata(qualified_test_class_name=E,code='',method_signature=S or _A);B.nl2test_metadata=T;d=E.rsplit('.',1)[-1]+Y;e=E.replace('.',L)+Y
			def f(err:str)->bool:
				A=err.replace('\\',L)
				if L in A:return A.endswith(e)
				return A.endswith(d)
			g:List[CompilationError]=JavaMavenCompilation(A.project_root,module_root=J).get_compilation_errors();h=[A.file for A in g];B.compiles=not any(f(A)for A in h)
			try:U=A.regenerate_analysis(eager=_D)
			except Exception as H:RichLog.warn(f"Regenerating analysis failed for {E}::{S} (id={D.id}); skipping grading: {H}")
			else:A.test_grader.set_analysis(U);A.localization_grader.set_analysis(U);i,j=A.test_grader.grade(D,T,compiles=B.compiles);B.structured_eval=i;B.coverage_eval=j;B.localization_eval=A.localization_grader.grade_from_state(C,D)
			V=TestFileManager(A.project_root,test_base_dir=I);W=TestFileInfo(qualified_class_name=E)
			try:k=V.load(W,encode_class_name=_C);B.nl2test_metadata.code=k
			except FileNotFoundError:pass
			finally:
				try:V.delete_single(W,encode_class_name=_C)
				except Exception:pass
		K=F.totals();B.input_tokens=K['input_tokens'];B.output_tokens=K['output_tokens'];B.llm_calls=K['calls'];X=_A
		if A.decomposition_mode==DecompositionMode.GHERKIN and C:X=C.localized_scenario
		return NL2TestPipelineResult(eval=B,localized_scenario=X)