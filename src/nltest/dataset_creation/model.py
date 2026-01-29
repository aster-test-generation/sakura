from typing import List
from hamster.code_analysis.model.models import FocalClassInfo
from pydantic import BaseModel
class Test(BaseModel):qualified_class_name:str;method_signature:str;focal_details:List[FocalClassInfo]|None=None
class NL2TestDataset(BaseModel):dataset_name:str;tests_with_one_focal_methods:List[Test];tests_with_two_focal_methods:List[Test];tests_with_more_than_two_to_five_focal_methods:List[Test];tests_with_more_than_five_to_ten_focal_methods:List[Test];tests_with_more_than_ten_focal_methods:List[Test]