import re
from collections import deque
from typing import List, Set, Tuple, Dict, Literal, Optional

from cldk.analysis.java import JavaAnalysis
from cldk.models.java import JCallable
from hamster.code_analysis.model.models import TestingFramework
from hamster.code_analysis.utils.constants import TEST_ANNOTATIONS, SORTED_FRAMEWORK_PREFIXES
from hamster.code_analysis.focal_class_method.focal_class_method import FocalClassMethod

from nltest.utils.constants import SETUP_ANNOTATIONS, TEARDOWN_ANNOTATIONS
from nltest.utils.exceptions import ClassFileNotFound, CompilationUnitNotFound, MethodNotFoundError, ClassNotFoundError
from nltest.utils.pretty.prompt_formatting import pretty_indent


class CommonAnalysis:
    def __init__(self, analysis: JavaAnalysis):
        self.analysis = analysis

    def __get_project_root(self) -> str:
        classes = list(self.analysis.get_classes().keys())
        split_class_names = [s.split('.') for s in classes]

        if not split_class_names:
            return ""

        # Find the shortest length of the split strings
        min_length = min(len(s) for s in split_class_names)

        project_root = []

        # Iterate through the indices up to min_length
        for i in range(min_length):
            # Get the set of elements at index i
            elements = set(s[i] for s in split_class_names)

            # If all elements are the same at this position, add to common_parts
            if len(elements) == 1:
                project_root.append(elements.pop())
            else:
                break

        # Join back with '.' to return the common prefix
        return '.'.join(project_root)

    def is_test_class(self, qualified_class_name: str, testing_frameworks: List[TestingFramework]):
        """
        Determines whether a class is a test class, meaning it contains at least one test method alongside testing
        frameworks.
        Args:
            qualified_class_name: The qualified class name of the class being analyzed.
            testing_frameworks: The testing frameworks imported in the compilation unit containing the class.

        Returns:
            bool: True if the class is a test class, containing a test method, or False otherwise.

        """
        for method_signature in self.analysis.get_methods_in_class(qualified_class_name=qualified_class_name):
            if self.is_test_method(method_signature, qualified_class_name, testing_frameworks):
                return True
        return False

    def is_test_method(self, method_signature: str, qualified_class_name: str,
                       testing_frameworks: List[TestingFramework]) -> bool:
        """
        Determines whether a method, uniquely determined by its signature and qualified class name, is a test method.
        Args:
            method_signature: The signature of the method analyzed.
            qualified_class_name: The qualified class name containing the method.
            testing_frameworks: The testing frameworks imported in the compilation unit containing the class.

        Returns:
            bool: True if the method is a test method, False otherwise.

        """

        method_details = self.analysis.get_method(
            qualified_class_name=qualified_class_name,
            qualified_method_name=method_signature
        )

        if not method_details.code.isascii():  # NOTE: Do not consider non-ASCII methods (methods containing non-English characters)
            return False

        class_details = self.analysis.get_class(qualified_class_name=qualified_class_name)

        is_public = ("public" in method_details.modifiers)

        # JUnits 4 and 5 and some TestNG methods use method annotations
        has_test_annot = any(annot.split("(")[0] in TEST_ANNOTATIONS for annot in method_details.annotations)

        # JUnit 3 uses naming conventions (i.e., method must begin with "test") for testing
        is_junit3_test = (
                TestingFramework.JUNIT3 in testing_frameworks
                and any(ext.endswith("TestCase") for ext in class_details.extends_list)
                and method_signature.startswith("test")
                and is_public
                and method_details.return_type == "void"
                and len(method_details.parameters) == 0
        )

        # TestNG has class-level @Test annotations where every public method is a test case
        is_testng_test = (
                TestingFramework.TESTNG in testing_frameworks
                and any(annot.split("(")[0] == "@Test" for annot in class_details.annotations)
                and is_public
        )

        return has_test_annot or is_junit3_test or is_testng_test

    def is_setup_method(self, method_signature, qualified_class_name: str,
                        testing_frameworks: List[TestingFramework]) -> bool:
        method_details = self.analysis.get_method(qualified_class_name, method_signature)

        if not method_details.code.isascii():
            return False

        if TestingFramework.JUNIT3 in testing_frameworks and method_details.signature == "setUp()":
            return True

        for annotation in method_details.annotations:
            if annotation in SETUP_ANNOTATIONS:
                return True

        return False

    def is_teardown_method(self, method_signature, qualified_class_name: str,
                           testing_frameworks: List[TestingFramework]) -> bool:
        """
        Determines whether a method is a teardown method.
        Args:
            method_signature: The signature of the method analyzed.
            qualified_class_name: The qualified class name containing the method.
            testing_frameworks: The testing frameworks imported in the compilation unit containing the class.

        Returns:
            bool: True if the method is a teardown method, False otherwise.
        """
        method_details = self.analysis.get_method(qualified_class_name, method_signature)

        if not method_details.code.isascii():
            return False

        if TestingFramework.JUNIT3 in testing_frameworks and method_details.signature == "tearDown()":
            return True

        for annotation in method_details.annotations:
            if annotation in TEARDOWN_ANNOTATIONS:
                return True

        return False

    def get_testing_frameworks_for_class(self, qualified_class_name: str) -> List[TestingFramework]:
        """
        Gets a list of the testing frameworks available for a class by looking at its
        associated compilation unit and its imports.
        Args:
            qualified_class_name: The qualified class name of the class being analyzed.

        Returns:
            List: A list of TestingFramework objects for the class's compilation unit.

        """
        testing_frameworks = set()
        imports = self.get_imports_for_class(qualified_class_name)

        for imp in imports:
            for prefix, name in SORTED_FRAMEWORK_PREFIXES:
                if imp.startswith(prefix):
                    testing_frameworks.add(name)
                    break

        return sorted(testing_frameworks, key=lambda x: len(x.value), reverse=True)

    def get_imports_for_class(self, qualified_class_name: str) -> List[str]:
        imports: Set[str] = set()
        java_file = self.analysis.get_java_file(qualified_class_name=qualified_class_name)

        if not java_file:
            raise ClassFileNotFound(f"Java file for {qualified_class_name} not found",
                                    extra_info={"qualified_class_name": qualified_class_name})

        compilation_unit = self.analysis.get_java_compilation_unit(file_path=java_file)
        if not compilation_unit:
            raise CompilationUnitNotFound(f"Compilation unit for {qualified_class_name} not found",
                                          extra_info={"qualified_class_name": qualified_class_name})

        for imp in compilation_unit.imports:
            imports.add(imp)

        return sorted(imports, key=len, reverse=True)

    def get_referenced_app_classes(self, method_details: JCallable):
        referenced_classes = set()
        for referenced_type in method_details.referenced_types:
            referenced_classes.update(self.extract_non_parameterized_types(referenced_type))

        verified_classes = []
        for referenced_class in referenced_classes:
            if self.analysis.get_class(referenced_class) is not None:
                verified_classes.append(referenced_class)

        return sorted(verified_classes, key=len)

    def get_setup_methods(self, qualified_class_name: str) -> List[JCallable]:
        potential_methods = self.get_ascii_methods(qualified_class_name)
        testing_frameworks = self.get_testing_frameworks_for_class(qualified_class_name)
        setup_methods = []

        for method in potential_methods:
            if self.is_setup_method(method.signature, qualified_class_name, testing_frameworks):
                setup_methods.append(method)

        return setup_methods

    def get_teardown_methods(self, qualified_class_name: str) -> List[JCallable]:
        """
        Gets a list of teardown methods for a given class.
        Args:
            qualified_class_name: The qualified class name of the class being analyzed.

        Returns:
            List[JCallable]: A list of teardown methods for the class.
        """
        potential_methods = self.get_ascii_methods(qualified_class_name)
        testing_frameworks = self.get_testing_frameworks_for_class(qualified_class_name)
        teardown_methods = []

        for method in potential_methods:
            if self.is_teardown_method(method.signature, qualified_class_name, testing_frameworks):
                teardown_methods.append(method)

        return teardown_methods

    def get_test_methods_in_class(self, qualified_class_name: str) -> List[Tuple[str, str]]:
        """
        Returns a list of (qualified_class_name, method_signature) for all test methods in the class.
        A method is considered a test method if is_test_method(...) evaluates to True.
        """
        testing_frameworks = self.get_testing_frameworks_for_class(qualified_class_name)
        results: List[Tuple[str, str]] = []
        for method_sig in self.analysis.get_methods_in_class(qualified_class_name):
            if self.is_test_method(method_sig, qualified_class_name, testing_frameworks):
                results.append((qualified_class_name, method_sig))
        return results

    def get_ascii_methods(self, qualified_class_name: str) -> List[JCallable]:
        """Returns all methods in class that is ASCII"""
        valid_methods: List[JCallable] = []
        for method_signature in self.analysis.get_methods_in_class(qualified_class_name):
            method_details = self.analysis.get_method(qualified_class_name, method_signature)
            if method_details.code.isascii():
                valid_methods.append(method_details)
        return sorted(valid_methods, key=lambda x: len(x.signature))

    def get_test_methods_classes_and_application_classes(self) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Get test methods, classes, and application classes.
        Returns:
            Tuple[Dict[str, List[str]], List[str]]: Dictionary of test classes and test methods, and list of application classes.
        """
        test_classes_methods = {}
        application_classes = []
        common_analysis = CommonAnalysis(self.analysis)

        for q_class in self.analysis.get_classes():
            testing_frameworks = self.get_testing_frameworks_for_class(q_class)
            if not testing_frameworks:
                application_classes.append(q_class)
                continue

            test_methods = []
            for method_sig in self.analysis.get_methods_in_class(q_class):
                if common_analysis.is_test_method(method_sig, q_class, testing_frameworks):
                    test_methods.append(method_sig)

            if test_methods:
                test_classes_methods[q_class] = test_methods
            else:
                application_classes.append(q_class)

        return test_classes_methods, application_classes

    def is_subclass_of(self, sub_class: str, super_class: str) -> bool:
        if not sub_class or not super_class or sub_class == super_class:
            return False

        sub_info = self.analysis.get_class(sub_class)
        if not sub_info:
            return False

        stack = sub_info.extends_list
        seen = set()

        while stack:
            curr = stack.pop()
            if curr in seen:
                continue
            if curr == super_class:
                return True
            seen.add(curr)

            curr_info = self.analysis.get_class(curr)
            if curr_info:
                parents = curr_info.extends_list
                stack.extend(parents)

        return False

    def implements_interface(self, class_name: str, interface_name: str) -> bool:
        if not class_name or not interface_name:
            return False

        cls_info = self.analysis.get_class(class_name)
        if not cls_info:
            return False

        stack = []
        seen = set()

        # Consider both implemented interfaces and superclass chain
        stack.extend(cls_info.extends_list)
        stack.extend(cls_info.implements_list)

        while stack:
            curr = stack.pop()
            if curr in seen:
                continue
            if curr == interface_name:
                return True
            seen.add(curr)

            curr_info = self.analysis.get_class(curr)
            if not curr_info:
                continue

            if curr_info.is_interface:
                # Interfaces can't extend class or abstract class
                stack.extend(curr_info.implements_list)
            else:
                stack.extend(curr_info.extends_list)
                stack.extend(curr_info.implements_list)

        return False

    def is_accessible_from(
            self,
            owner_class: str,
            method_signature: str,
            *,
            accessor_class: Optional[str] = None,
            mode: Literal["public", "same_package", "same_package_or_subclass"] = "public",
    ) -> bool:
        class_details = self.analysis.get_class(owner_class)
        if not class_details:
            raise ClassNotFoundError(
                f"Class {owner_class} not found.",
                extra_info={"qualified_class_name": owner_class},
            )

        method_details = self.analysis.get_method(owner_class, method_signature)
        if not method_details:
            raise MethodNotFoundError(
                f"Method {method_signature} not found in class {owner_class}.",
                extra_info={"qualified_class_name": owner_class, "method_signature": method_signature},
            )

        mods = set(method_details.modifiers)
        owner_pkg = self.package_of(owner_class)

        # Public methods and interface/annotation non-private methods are always visible
        if "public" in mods:
            return True
        if class_details.is_interface or class_details.is_annotation_declaration:
            if "private" not in mods:
                return True

        # Implicit public constructor for public class
        if method_details.is_constructor and method_details.is_implicit and "public" in class_details.modifiers:
            return True

        # Determined all public accessibility options
        if mode == "public":
            return False

        # Determine accessor package
        acc_pkg = self.package_of(accessor_class) if accessor_class else ""

        # Same package rules
        if owner_pkg == acc_pkg:
            if "private" in mods:
                return False
            return True
        # Protected and package-private allowed in same package

        # If different package and not public, it is not accessible
        if mode == "same_package":
            return False

        # Check for subclass inheritance of protected method
        if "protected" in mods and accessor_class and self.is_subclass_of(accessor_class, owner_class):
            return True

        return False

    def is_public(self, qualified_class_name: str, method_signature: str) -> bool:
        return self.is_accessible_from(qualified_class_name, method_signature, mode="public")

    def get_method_visibility(self, qualified_class_name: str, method_signature: str) -> Literal[
        "public", "same_package", "same_package_or_subclass"]:
        """
        Determines the visibility level of a method.
        
        Returns:
            "public" if the method is accessible from anywhere
            "same_package_or_subclass" if the method is accessible from same package or subclasses
            "same_package" if the method is only accessible from the same package
        """
        if self.is_accessible_from(qualified_class_name, method_signature, mode="public"):
            return "public"
        elif self.is_accessible_from(qualified_class_name, method_signature, mode="same_package_or_subclass"):
            return "same_package_or_subclass"
        else:
            return "same_package"

    def get_complicated_focal_tests(self) -> Dict[str, List[str]]:
        test_class_map, application_classes = self.get_test_methods_classes_and_application_classes()
        complicated_tests = {}

        for test_class in test_class_map:
            testing_frameworks = self.get_testing_frameworks_for_class(test_class)
            setup_methods = self.get_setup_methods(test_class)
            setup_method_signatures = [method.signature for method in setup_methods]

            complicated_methods = []

            for method_signature in test_class_map[test_class]:
                try:
                    focal_class_method = FocalClassMethod(self.analysis, application_classes)
                    focal_classes, _, _, _ = focal_class_method.identify_focal_class_and_ui_api_test(
                        test_class, method_signature, setup_method_signatures
                    )

                    is_complicated = (
                            len(focal_classes) > 1 or
                            (len(focal_classes) == 1 and len(focal_classes[0].focal_method_names) > 1)
                    )

                    if is_complicated:
                        complicated_methods.append(method_signature)

                except Exception as e:
                    continue

            if complicated_methods:
                complicated_tests[test_class] = complicated_methods

        return complicated_tests

    def get_complicated_focal_tests_count(self) -> int:
        complicated_tests = self.get_complicated_focal_tests()
        return sum(len(methods) for methods in complicated_tests.values())

    @staticmethod
    def is_getter_or_setter(method_details: JCallable) -> bool:
        if (
                (method_details.signature.startswith("get") or method_details.signature.startswith("set"))
                and len(method_details.code.split("\n")) <= 3
        ):
            return True
        return False

    @staticmethod
    def get_complete_method_code(method_declaration: str, method_code: str) -> str:
        code = method_declaration + " " + method_code
        return pretty_indent(code)

    @staticmethod
    def extract_non_parameterized_types(parameterized_type: str) -> List[str]:
        pattern = re.compile(r"[\w\.]+\.[A-Z]\w*")  # Extracts all types ending with a capital
        non_parameterized_types = pattern.findall(parameterized_type)
        return non_parameterized_types

    @staticmethod
    def package_of(qualified_class_name: str) -> str:
        i = qualified_class_name.rfind(".")
        return qualified_class_name[:i] if i != -1 else ""

    @staticmethod
    def process_callee_signature(callee_signature: str) -> str:
        """
        Processes callee signature
        Args:
            callee_signature:

        Returns:

        """
        pattern = r"\b(?:[a-zA-Z_][\w\.]*\.)+([a-zA-Z_][\w]*)\b|<[^>]*>"

        # Find the part within the parentheses
        start = callee_signature.find("(") + 1
        end = callee_signature.rfind(")")

        # Extract the elements inside the parentheses
        elements = callee_signature[start:end].split(",")

        # Apply the regex to each element
        simplified_elements = [re.sub(pattern, r"\1", element.strip()) for element in elements]

        # Reconstruct the string with simplified elements
        return f"{callee_signature[:start]}{', '.join(simplified_elements)}{callee_signature[end:]}"

    @staticmethod
    def get_cldk_class_name(qualified_class_name: str) -> str:
        """
        Normalize inner class separators for CLDK lookups.
        """
        if "$" not in qualified_class_name:
            return qualified_class_name
        return qualified_class_name.replace("$", ".")

    @staticmethod
    def get_cldk_method_sig(qualified_class_name: str, method_signature: str) -> str:
        """
        Normalize constructor signatures for CLDK lookups, since CLDK expects constructors with name '<init>'.
        """
        simple_class_name = qualified_class_name.split(".")[-1]
        constructor_prefix = f"{simple_class_name}("
        if constructor_prefix not in method_signature:
            return method_signature
        return method_signature.replace(constructor_prefix, "<init>(", 1)
