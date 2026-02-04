from __future__ import annotations
_A=None
from enum import Enum
from typing import List
from pydantic import BaseModel
from sakura.utils.constants import ABSTRACTION_TEMPERATURES
class AbstractionLevel(Enum):
	HIGH='high';MEDIUM='medium';LOW='low'
	def get_temperature(A)->float:'Return the temperature for this abstraction level.';return ABSTRACTION_TEMPERATURES[A.value]
class TrialType(Enum):DESCRIPTION='description';TEST_CASE='test_case'
class TestDescriptionInfo(BaseModel):description:str;prompt:str;abstraction_level:AbstractionLevel;temperature:float;method_signature:str;qualified_class_name:str;trial_number:int=1;id:int=-1
class RoundTripTest(BaseModel):prompt:str;temperature:float;generated_test:str;method_signature:str;qualified_class_name:str;generated_description:TestDescriptionInfo;score:float|_A=_A
class ReferencedClasses(BaseModel):'DEPRECATED';referenced_classes:List[ClassContext]
class ClassContext(BaseModel):simple_class_name:str;qualified_class_name:str;annotations:List[str]|_A=_A;extends:List[str]|_A=_A;modifiers:List[str]|_A=_A;field_declarations:List[FieldDeclaration]|_A=_A;relevant_class_methods:List[MethodContext]|_A=_A;javadoc:List[str]|_A=_A
class FieldDeclaration(BaseModel):variables:List[str]|_A=_A;type:str|_A=_A;modifiers:List[str]|_A=_A;annotations:List[str]|_A=_A;type_is_helper_class:bool|_A=_A
class CallSiteInfo(BaseModel):method_name:str;receiver_type:str;return_type:str;line_number:int;is_assertion:bool;is_helper:bool|_A=_A
class VariableInfo(BaseModel):name:str;type:str;initializer:str|_A=_A;line_number:int;type_is_helper_class:bool|_A=_A
class MethodContext(BaseModel):method_signature:str;qualified_class_name:str|_A=_A;is_getter_or_setter:bool|_A=_A;code:str|_A=_A;call_sites:List[CallSiteInfo]=[];variable_declarations:List[VariableInfo]=[];thrown_exceptions:List[str]=[];javadoc:str|_A=_A
class Test2NLEntry(BaseModel):
	id:int=-1;description:str;project_name:str;qualified_class_name:str;method_signature:str;abstraction_level:AbstractionLevel|_A=_A;is_bdd:bool=False
	@classmethod
	def from_test_description_info(B,test_description_info:TestDescriptionInfo,project_name:str)->Test2NLEntry:A=test_description_info;return B(id=A.id,description=A.description,project_name=project_name,qualified_class_name=A.qualified_class_name,method_signature=A.method_signature,abstraction_level=A.abstraction_level,is_bdd=False)