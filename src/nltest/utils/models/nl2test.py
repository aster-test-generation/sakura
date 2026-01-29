from __future__ import annotations
_B=False
_A=None
from enum import Enum
from typing import Annotated,Any,Dict,List,Optional
from pydantic import BaseModel,ConfigDict,Field
class AbstractionLevel(Enum):HIGH='high';MEDIUM='medium';LOW='low'
class NL2TestInput(BaseModel):id:int=-1;description:str;project_name:str;qualified_class_name:Annotated[str,'The qualified class name of the class containing the test'];method_signature:Annotated[str,'The method signature of the test case'];abstraction_level:AbstractionLevel|_A=_A;is_bdd:bool=_B;model_config=ConfigDict(use_enum_values=True)
class AgentToolLog(BaseModel):tool_counts:Dict[str,int];tool_trajectories:List[List[str]]
class ToolLog(BaseModel):supervisor_tool_log:AgentToolLog;localization_tool_log:AgentToolLog;composition_tool_log:AgentToolLog
class NL2EvaluationResults(BaseModel):'Evaluation results for a single NL2Test generation run.';nl2_input:NL2TestInput;pred_class_name:str='';pred_method_signature:str='';gt_class_name:str='';gt_method_signature:str='';structural_score:float=.0;structural_metrics:Dict[str,float]=Field(default_factory=dict);test_code:str=''
class NL2TestMetadata(BaseModel):qualified_test_class_name:str;code:str;method_signature:Optional[str]=_A
class NL2TestStructuralEval(BaseModel):'Structural precision/recall metrics for the generated test.';obj_creation_recall:float;obj_creation_precision:float;assertion_recall:float;assertion_precision:float;callable_recall:float;callable_precision:float;focal_recall:float;focal_precision:float
class NL2TestCoverageEval(BaseModel):class_coverage:float;method_coverage:float;line_coverage:float;branch_coverage:float
class NL2TestEval(BaseModel):compiles:bool;nl2test_input:NL2TestInput;nl2test_metadata:NL2TestMetadata;structured_eval:Optional[NL2TestStructuralEval];coverage_eval:Optional[NL2TestCoverageEval];localization_eval:Optional[Any]=_A;tool_log:Optional[ToolLog]=_A;input_tokens:int=0;output_tokens:int=0;llm_calls:int=0
class NL2TestPipelineResult(BaseModel):'Result from Pipeline.run_nl2test().\n\n    Contains the evaluation result and an optional localized scenario\n    (populated only in GHERKIN decomposition mode).\n    ';eval:NL2TestEval;localized_scenario:Optional[Any]=_A
class NL2TestFailure(BaseModel):'Captures a failed NL2Test generation attempt.';nl2test_input:NL2TestInput;error:str;error_type:str;traceback:Optional[str]=_A
class OutOfBoxAgentEval(NL2TestEval):'Evaluation for out-of-box agent outputs (e.g., Gemini CLI, Claude Code).\n\n    Extends NL2TestEval with flags to track generation failures:\n    - failed_test_file_generation: Code was generated but not saved to a file\n    - failed_code_generation: No code was generated at all\n    These flags are mutually exclusive.\n    ';failed_test_file_generation:bool=_B;failed_code_generation:bool=_B