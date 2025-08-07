from cldk.analysis.java import JavaAnalysis
from cldk.models.java import JCallable

from nltest.test2nl.model.models import MethodContext
from nltest.utils.analysis import CommonAnalysis


class MethodExtractor:
    def __init__(self, analysis: JavaAnalysis):
        self.analysis = analysis

    @staticmethod
    def extract(method_details: JCallable, complete_methods: bool) -> MethodContext:
        if CommonAnalysis.is_getter_or_setter(method_details):
            code = None
            is_getter_or_setter = True
        elif complete_methods:
            raw_code = CommonAnalysis.get_complete_method_code(method_details.declaration, method_details.code)
            code = f"```java\n{raw_code.strip()}\n```"
            is_getter_or_setter = False
        else:
            code = None
            is_getter_or_setter = False

        return MethodContext(
            method_signature=method_details.signature,
            is_getter_or_setter=is_getter_or_setter,
            code=code,
        )
