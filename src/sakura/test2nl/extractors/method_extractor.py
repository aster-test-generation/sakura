from typing import List, Set

from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.test_statistics import CallAndAssertionSequenceDetailsInfo

from sakura.test2nl.model.models import CallSiteInfo, MethodContext, VariableInfo
from sakura.utils.analysis import CommonAnalysis


class MethodExtractor:
    def __init__(self, analysis: JavaAnalysis, application_classes: List[str]):
        self.analysis = analysis
        self.application_classes = application_classes

    def _is_helper_class(self, type_name: str | None) -> bool | None:
        """Check if type exists in repo but is not an application class."""
        if not type_name:
            return None
        non_param_types = CommonAnalysis.extract_non_parameterized_types(type_name)
        for type_ in non_param_types:
            if self.analysis.get_class(type_) and type_ not in self.application_classes:
                return True
        return False

    def extract(
        self,
        qualified_class_name: str,
        method_signature: str,
        complete_methods: bool,
        include_class_name: bool = False,
    ) -> MethodContext:
        method_details = self.analysis.get_method(
            qualified_class_name, method_signature
        )
        if not method_details:
            raise ValueError(
                f"Method {method_signature} not found in class {qualified_class_name}"
            )

        # Process if method is a getter or setter
        if CommonAnalysis.is_getter_or_setter(method_details):
            code = None
            is_getter_or_setter = True
        elif complete_methods:
            raw_code = CommonAnalysis.get_complete_method_code(
                method_details.declaration, method_details.code
            )
            code = raw_code.strip()
            is_getter_or_setter = False
        else:
            code = None
            is_getter_or_setter = False

        # Collect assertion names using CallAndAssertionSequenceDetailsInfo
        assertion_names: Set[str] = set()
        try:
            common = CommonAnalysis(self.analysis)
            testing_frameworks = common.get_testing_frameworks_for_class(
                qualified_class_name
            )
            assertion_info = CallAndAssertionSequenceDetailsInfo(self.analysis)
            sequence_details_list = (
                assertion_info.get_call_and_assertion_sequence_details_info(
                    qualified_class_name, method_signature, testing_frameworks
                )
            )
            for sequence_details in sequence_details_list:
                for assertion_detail in sequence_details.assertion_details:
                    assertion_names.add(assertion_detail.assertion_name)
        except Exception:
            # Fallback: if assertion info extraction fails, assertion_names remains empty
            pass

        # Collect call site info
        call_sites: list[CallSiteInfo] = []
        for cs in method_details.call_sites or []:
            is_assertion = cs.method_name in assertion_names
            is_helper = self._is_helper_class(cs.receiver_type)
            call_sites.append(
                CallSiteInfo(
                    method_name=cs.method_name,
                    receiver_type=cs.receiver_type,
                    return_type=cs.return_type,
                    line_number=cs.start_line,
                    is_assertion=is_assertion,
                    is_helper=is_helper,
                )
            )

        # Collect variable declarations
        variable_declarations: list[VariableInfo] = []
        for vd in method_details.variable_declarations or []:
            type_is_helper = self._is_helper_class(vd.type)
            variable_declarations.append(
                VariableInfo(
                    name=vd.name,
                    type=vd.type,
                    initializer=vd.initializer if vd.initializer else None,
                    line_number=vd.start_line,
                    type_is_helper_class=type_is_helper,
                )
            )

        # Collect Java doc comments
        javadoc: str | None = None
        for comment in method_details.comments or []:
            if comment.is_javadoc and comment.content:
                javadoc = comment.content
                break

        return MethodContext(
            method_signature=method_details.signature,
            qualified_class_name=qualified_class_name if include_class_name else None,
            is_getter_or_setter=is_getter_or_setter,
            code=code,
            call_sites=call_sites,
            variable_declarations=variable_declarations,
            thrown_exceptions=method_details.thrown_exceptions or [],
            javadoc=javadoc,
        )
