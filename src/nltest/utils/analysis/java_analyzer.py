import os
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

from cldk.analysis.java import JavaAnalysis
from cldk.models.java import JCallable
from hamster.code_analysis.focal_class_method.focal_class_method import FocalClassMethod
from hamster.code_analysis.model.models import TestingFramework
from hamster.code_analysis.utils import constants
from hamster.code_analysis.utils.constants import (
    SORTED_FRAMEWORK_PREFIXES,
    TEST_ANNOTATIONS,
)

from nltest.utils.constants import SETUP_ANNOTATIONS, TEARDOWN_ANNOTATIONS
from nltest.utils.exceptions import (
    ClassFileNotFound,
    ClassNotFoundError,
    CompilationUnitNotFound,
    MethodNotFoundError,
)
from nltest.utils.pretty.prompt_formatting import pretty_indent


@dataclass
class ReachabilityConfig:
    allow_repetition: bool = False  # On same level
    only_helpers: bool = False
    add_extended_class: bool = False


class CommonAnalysis:
    def __init__(self, analysis: JavaAnalysis):
        self.analysis = analysis

    def __get_project_root(self) -> str:
        classes = list(self.analysis.get_classes().keys())
        split_class_names = [s.split(".") for s in classes]

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
        return ".".join(project_root)

    def is_test_class(
        self, qualified_class_name: str, testing_frameworks: List[TestingFramework]
    ):
        """
        Determines whether a class is a test class, meaning it contains at least one test method alongside testing
        frameworks.
        Args:
            qualified_class_name: The qualified class name of the class being analyzed.
            testing_frameworks: The testing frameworks imported in the compilation unit containing the class.

        Returns:
            bool: True if the class is a test class, containing a test method, or False otherwise.

        """
        for method_signature in self.analysis.get_methods_in_class(
            qualified_class_name=qualified_class_name
        ):
            if self.is_test_method(
                method_signature, qualified_class_name, testing_frameworks
            ):
                return True
        return False

    def is_test_method(
        self,
        method_signature: str,
        qualified_class_name: str,
        testing_frameworks: List[TestingFramework],
    ) -> bool:
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
            qualified_method_name=method_signature,
        )

        if not method_details.code.isascii():  # NOTE: Do not consider non-ASCII methods (methods containing non-English characters)
            return False

        class_details = self.analysis.get_class(
            qualified_class_name=qualified_class_name
        )

        is_public = "public" in method_details.modifiers

        # JUnits 4 and 5 and some TestNG methods use method annotations
        has_test_annot = any(
            annot.split("(")[0] in TEST_ANNOTATIONS
            for annot in method_details.annotations
        )

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
            and any(
                annot.split("(")[0] == "@Test" for annot in class_details.annotations
            )
            and is_public
        )

        return has_test_annot or is_junit3_test or is_testng_test

    def is_setup_method(
        self,
        method_signature,
        qualified_class_name: str,
        testing_frameworks: List[TestingFramework],
    ) -> bool:
        method_details = self.analysis.get_method(
            qualified_class_name, method_signature
        )

        if not method_details.code.isascii():
            return False

        if (
            TestingFramework.JUNIT3 in testing_frameworks
            and method_details.signature == "setUp()"
        ):
            return True

        for annotation in method_details.annotations:
            if annotation in SETUP_ANNOTATIONS:
                return True

        return False

    def is_teardown_method(
        self,
        method_signature,
        qualified_class_name: str,
        testing_frameworks: List[TestingFramework],
    ) -> bool:
        """
        Determines whether a method is a teardown method.
        Args:
            method_signature: The signature of the method analyzed.
            qualified_class_name: The qualified class name containing the method.
            testing_frameworks: The testing frameworks imported in the compilation unit containing the class.

        Returns:
            bool: True if the method is a teardown method, False otherwise.
        """
        method_details = self.analysis.get_method(
            qualified_class_name, method_signature
        )

        if not method_details.code.isascii():
            return False

        if (
            TestingFramework.JUNIT3 in testing_frameworks
            and method_details.signature == "tearDown()"
        ):
            return True

        for annotation in method_details.annotations:
            if annotation in TEARDOWN_ANNOTATIONS:
                return True

        return False

    def get_testing_frameworks_for_class(
        self, qualified_class_name: str
    ) -> List[TestingFramework]:
        """
        Gets a list of the testing frameworks available for a class by looking at its
        associated compilation unit and its imports.
        Args:
            qualified_class_name: The qualified class name of the class being analyzed.

        Returns:
            List: A list of TestingFramework objects for the class's compilation unit.

        """
        if not self.analysis.get_class(qualified_class_name):
            return []

        testing_frameworks = set()
        imports = self.get_imports_for_class(qualified_class_name)

        for imp in imports:
            for prefix, name in SORTED_FRAMEWORK_PREFIXES:
                if imp.startswith(prefix):
                    testing_frameworks.add(name)
                    break

        return sorted(testing_frameworks, key=lambda x: len(x.value), reverse=True)

    def get_imports_for_class(self, qualified_class_name: str) -> List[str]:
        if not self.analysis.get_class(qualified_class_name):
            return []

        imports: Set[str] = set()
        java_file = self.analysis.get_java_file(
            qualified_class_name=qualified_class_name
        )

        if not java_file:
            raise ClassFileNotFound(
                f"Java file for {qualified_class_name} not found",
                extra_info={"qualified_class_name": qualified_class_name},
            )

        compilation_unit = self.analysis.get_java_compilation_unit(file_path=java_file)
        if not compilation_unit:
            raise CompilationUnitNotFound(
                f"Compilation unit for {qualified_class_name} not found",
                extra_info={"qualified_class_name": qualified_class_name},
            )

        for imp in compilation_unit.imports:
            imports.add(imp)

        return sorted(imports, key=len, reverse=True)

    def get_referenced_app_classes(self, method_details: JCallable):
        referenced_classes = set()
        for referenced_type in method_details.referenced_types:
            referenced_classes.update(
                self.extract_non_parameterized_types(referenced_type)
            )

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
            if self.is_setup_method(
                method.signature, qualified_class_name, testing_frameworks
            ):
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
            if self.is_teardown_method(
                method.signature, qualified_class_name, testing_frameworks
            ):
                teardown_methods.append(method)

        return teardown_methods

    def get_test_methods_in_class(
        self, qualified_class_name: str
    ) -> List[Tuple[str, str]]:
        """
        Returns a list of (qualified_class_name, method_signature) for all test methods in the class.
        A method is considered a test method if is_test_method(...) evaluates to True.
        """
        testing_frameworks = self.get_testing_frameworks_for_class(qualified_class_name)
        results: List[Tuple[str, str]] = []
        for method_sig in self.analysis.get_methods_in_class(qualified_class_name):
            if self.is_test_method(
                method_sig, qualified_class_name, testing_frameworks
            ):
                results.append((qualified_class_name, method_sig))
        return results

    def get_ascii_methods(self, qualified_class_name: str) -> List[JCallable]:
        """Returns all methods in class that is ASCII"""
        valid_methods: List[JCallable] = []
        for method_signature in self.analysis.get_methods_in_class(
            qualified_class_name
        ):
            method_details = self.analysis.get_method(
                qualified_class_name, method_signature
            )
            if method_details.code.isascii():
                valid_methods.append(method_details)
        return sorted(valid_methods, key=lambda x: len(x.signature))

    def get_test_methods_classes_and_application_classes(
        self,
    ) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Get test methods, classes, and application classes.
        Returns:
            Tuple[Dict[str, List[str]], List[str]]: Dictionary of test classes and test methods, and list of application classes.
        """
        test_classes_methods = {}
        application_classes = []

        for q_class in self.analysis.get_classes():
            class_details = self.analysis.get_class(q_class)

            # Skip abstract classes
            if (
                class_details
                and class_details.modifiers
                and "abstract" in class_details.modifiers
            ):
                continue

            testing_frameworks = self.get_testing_frameworks_for_class(q_class)
            if not testing_frameworks:
                application_classes.append(q_class)
                continue

            # Get all reachable test methods (direct + inherited)
            try:
                reachable_test_methods = Reachability(
                    self.analysis
                ).get_reachable_test_methods(q_class, testing_frameworks)
            except ClassNotFoundError:
                application_classes.append(q_class)
                continue

            # Flatten to list of method signatures
            test_methods = []
            for declaring_class, method_sigs in reachable_test_methods.items():
                test_methods.extend(method_sigs)

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
                extra_info={
                    "qualified_class_name": owner_class,
                    "method_signature": method_signature,
                },
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
        if (
            method_details.is_constructor
            and method_details.is_implicit
            and "public" in class_details.modifiers
        ):
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
        if (
            "protected" in mods
            and accessor_class
            and self.is_subclass_of(accessor_class, owner_class)
        ):
            return True

        return False

    def is_public(self, qualified_class_name: str, method_signature: str) -> bool:
        return self.is_accessible_from(
            qualified_class_name, method_signature, mode="public"
        )

    def is_abstract_class(self, qualified_class_name: str) -> bool:
        """Returns True if the class is abstract."""
        class_details = self.analysis.get_class(qualified_class_name)
        if not class_details or not class_details.modifiers:
            return False
        return "abstract" in class_details.modifiers

    def get_method_visibility(
        self, qualified_class_name: str, method_signature: str
    ) -> Literal["public", "same_package", "same_package_or_subclass"]:
        """
        Determines the visibility level of a method.

        Returns:
            "public" if the method is accessible from anywhere
            "same_package_or_subclass" if the method is accessible from same package or subclasses
            "same_package" if the method is only accessible from the same package
        """
        if self.is_accessible_from(
            qualified_class_name, method_signature, mode="public"
        ):
            return "public"
        elif self.is_accessible_from(
            qualified_class_name, method_signature, mode="same_package_or_subclass"
        ):
            return "same_package_or_subclass"
        else:
            return "same_package"

    def get_complicated_focal_tests(self) -> Dict[str, List[str]]:
        test_class_map, application_classes = (
            self.get_test_methods_classes_and_application_classes()
        )
        complicated_tests = {}

        for test_class in test_class_map:
            testing_frameworks = self.get_testing_frameworks_for_class(test_class)
            setup_methods = self.get_setup_methods(test_class)
            setup_method_signatures = [method.signature for method in setup_methods]

            complicated_methods = []

            for method_signature in test_class_map[test_class]:
                try:
                    focal_class_method = FocalClassMethod(
                        self.analysis, application_classes
                    )
                    focal_classes, _, _, _ = (
                        focal_class_method.identify_focal_class_and_ui_api_test(
                            test_class, method_signature, setup_method_signatures
                        )
                    )

                    is_complicated = len(focal_classes) > 1 or (
                        len(focal_classes) == 1
                        and len(focal_classes[0].focal_method_names) > 1
                    )

                    if is_complicated:
                        complicated_methods.append(method_signature)

                except Exception:
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
            method_details.signature.startswith("get")
            or method_details.signature.startswith("set")
        ) and len(method_details.code.split("\n")) <= 3:
            return True
        return False

    @staticmethod
    def get_complete_method_code(method_declaration: str, method_code: str) -> str:
        code = method_declaration + " " + method_code
        return pretty_indent(code)

    @staticmethod
    def extract_non_parameterized_types(parameterized_type: str) -> List[str]:
        pattern = re.compile(
            r"[\w\.]+\.[A-Z]\w*"
        )  # Extracts all types ending with a capital
        non_parameterized_types = pattern.findall(parameterized_type)
        return non_parameterized_types

    @staticmethod
    def package_of(qualified_class_name: str) -> str:
        i = qualified_class_name.rfind(".")
        return qualified_class_name[:i] if i != -1 else ""

    @staticmethod
    def get_simple_class_name(qualified_class_name: str) -> str:
        i = qualified_class_name.rfind(".")
        return qualified_class_name[i + 1 :] if i != -1 else qualified_class_name

    @staticmethod
    def get_simple_method_name(method_signature: str) -> str:
        paren_index = method_signature.find("(")
        name_part = (
            method_signature[:paren_index] if paren_index != -1 else method_signature
        )
        dot_index = name_part.rfind(".")
        return name_part[dot_index + 1 :] if dot_index != -1 else name_part

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
        simplified_elements = [
            re.sub(pattern, r"\1", element.strip()) for element in elements
        ]

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

    @staticmethod
    def normalize_path_in_project(filepath: str, project_root: Optional[str]) -> str:
        """
        Normalize absolute compiler paths so that reports focus on project-relative locations.
        """
        if not filepath:
            return filepath
        normalized = filepath.replace("\\", "/")
        if project_root:
            root = str(Path(project_root).expanduser().resolve()).replace("\\", "/")
            root_with_sep = f"{root}/"
            if normalized.lower().startswith(root_with_sep.lower()):
                return normalized[len(root_with_sep) :]
        src_idx = normalized.find("/src/")
        if src_idx != -1:
            return normalized[src_idx + 1 :]
        return os.path.basename(normalized)


class Reachability:
    def __init__(self, analysis: JavaAnalysis):
        self.analysis = analysis
        self._reachability_cache: Dict[Tuple, Dict[str, List[str]]] = {}

    def get_helper_methods(
        self,
        qualified_class_name: str,
        method_signature: str,
        depth: int = constants.CONTEXT_SEARCH_DEPTH,
        add_extended_class: bool = False,
        allow_repetition: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Retrieves the helper methods reachable from the given method within the specified depth.

        Helper methods are methods called (directly or transitively) by the given method that are
        defined in the same class (or an extended class if add_extended_class=True).

        Args:
            qualified_class_name: The qualified name of the class.
            method_signature: The method signature.
            depth: The depth for search in call hierarchy.
            add_extended_class: If set to True, include methods from classes extended by the given class.
            allow_repetition: If set to True, allow visiting the same method multiple times in the same depth level.

        Returns:
            Dict[str, List[str]]: A map from class names to method signatures of helper methods.
        """
        method_details = self.analysis.get_method(
            qualified_class_name, method_signature
        )
        visited: Set[Tuple[str, str]] = set()
        reachability_config = ReachabilityConfig(
            allow_repetition=allow_repetition,
            add_extended_class=add_extended_class,
            only_helpers=True,
        )
        reachability_key = self._get_reachability_key(
            qualified_class_name, method_signature, reachability_config
        )

        if reachability_key in self._reachability_cache:
            reachable_methods_by_class = self._reachability_cache[reachability_key]
        else:
            reachable_methods_by_class: Dict[str, List[str]] = (
                self._collect_reachable_methods(
                    qualified_class_name,
                    method_signature,
                    depth,
                    reachability_config,
                    visited,
                )
            )
            self._reachability_cache[reachability_key] = reachable_methods_by_class

        final_reachable_methods: Dict[str, List[str]] = {}
        for class_name in reachable_methods_by_class:
            for method_signature in reachable_methods_by_class[class_name]:
                method = self.analysis.get_method(class_name, method_signature)
                if (
                    method
                    and (class_name != qualified_class_name or method != method_details)
                    and method.code.isascii()
                ):
                    final_reachable_methods.setdefault(class_name, []).append(
                        method_signature
                    )
        return final_reachable_methods

    def _collect_reachable_methods(
        self,
        qualified_class_name: str,
        method_signature: str,
        depth: int,
        reachability_config: ReachabilityConfig,
        visited: Set[Tuple[str, str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Collects reachable methods starting from the given method within a given depth.

        Args:
            qualified_class_name: The qualified name of the class.
            method_signature: The method signature.
            depth: The depth for search in call hierarchy.
            reachability_config: The configurations for reachability computation.
            visited: The set of tuples that have already been visited.

        Returns:
            Dict[str, List[str]]: A map from class names to method signatures of reachable methods.
        """
        if depth < 0:
            return {}

        if visited is None:
            visited: Set[Tuple[str, str]] = set()

        # Normalize constructors
        simple_class_name = qualified_class_name.split(".")[-1]
        if method_signature.startswith(f"{simple_class_name}("):
            method_signature = method_signature.replace(
                f"{simple_class_name}(", "<init>("
            )

        basic_key = (qualified_class_name, method_signature)

        # Check for an existing depth-level duplicate
        if basic_key in visited:
            return {}
        visited.add(basic_key)

        reachability_key = self._get_reachability_key(
            qualified_class_name, method_signature, reachability_config
        )

        # Check if already expanded in cache
        if reachability_key in self._reachability_cache:
            return self._reachability_cache[reachability_key]

        method_details = self.analysis.get_method(
            qualified_class_name, method_signature
        )

        # Check for ensuring valid method
        if not method_details:
            return {}

        # Seed result dictionary with the current method to start
        reachable_methods: Dict[str, List[str]] = {
            qualified_class_name: [method_signature]
        }

        # Determine extended classes if needed
        extend_list = []
        if reachability_config.add_extended_class:
            class_details = self.analysis.get_class(qualified_class_name)
            extend_list = class_details.extends_list if class_details else []

        child_counter: Counter[Tuple[str, str]] = Counter()

        # Handle interface-based call sites
        interface_map: Dict[str, List[str]] = {}
        for site in method_details.call_sites:
            receiver = site.receiver_type
            receiver_class = self.analysis.get_class(receiver)
            if receiver_class and receiver_class.is_interface:
                processed_sig = site.callee_signature
                interface_map.setdefault(receiver, []).append(processed_sig)

        # For all call sites with interface receiver types, collect the method signature
        for interface, callee_sigs in interface_map.items():
            for concrete_class in self.get_concrete_classes(interface_class=interface):
                # All concrete classes that implement interface
                if (
                    not reachability_config.only_helpers
                    or concrete_class == qualified_class_name
                ) or concrete_class in extend_list:
                    for callee_sig in callee_sigs:
                        child_counter[(concrete_class, callee_sig)] += 1

        # Handle direct symbol-table callees
        callees = self.analysis.get_callees(
            source_class_name=qualified_class_name,
            source_method_declaration=method_signature,
            using_symbol_table=True,
        ).get("callee_details", [])
        for callee_details in callees:
            callee_class = callee_details["callee_method"].klass
            if (
                not reachability_config.only_helpers
                or callee_class == qualified_class_name
            ) or callee_class in extend_list:
                callee_sig = callee_details["callee_method"].method.signature
                num_calls = max(len(callee_details.get("calling_lines", [])), 1)
                child_counter[(callee_class, callee_sig)] += num_calls

        # Now process unique children
        for child_key, num_calls in child_counter.items():
            child_class, child_sig = child_key
            child_reachable_methods = self._collect_reachable_methods(
                child_class, child_sig, depth - 1, reachability_config, visited
            )
            if reachability_config.allow_repetition:
                add_times = num_calls
            else:
                add_times = 1
            for _ in range(add_times):
                for child_c_class, methods_list in child_reachable_methods.items():
                    reachable_methods.setdefault(child_c_class, []).extend(methods_list)

        # This will allow a parent to revisit at the same level
        if reachability_config.allow_repetition:
            visited.remove(basic_key)

        return reachable_methods

    def get_concrete_classes(self, interface_class: str) -> List[str]:
        """
        Returns a list of concrete classes that implement the given interface class.

        Args:
            interface_class: The interface class.

        Returns:
            List[str]: List of concrete classes that implement the given interface class.
        """
        all_classes_in_application = self.analysis.get_classes()
        concrete_classes = []
        for qualified_class, class_details in all_classes_in_application.items():
            if (
                not class_details.is_interface
                and "abstract" not in class_details.modifiers
            ):
                if interface_class in class_details.implements_list:
                    concrete_classes.append(qualified_class)
        return concrete_classes

    def _get_reachability_key(
        self,
        qualified_class_name: str,
        method_signature: str,
        reachability_config: ReachabilityConfig,
    ) -> Tuple:
        """
        Generates a unique key for the reachability computation based on the input parameters.

        Args:
            qualified_class_name: The qualified name of the class.
            method_signature: The method signature.
            reachability_config: The configurations for reachability computation.

        Returns:
            Tuple: A unique reachability key.
        """
        reachability_key = (
            qualified_class_name,
            method_signature,
            reachability_config.allow_repetition,
            reachability_config.add_extended_class,
            reachability_config.only_helpers,
        )
        return reachability_key

    def get_visible_class_methods(
        self,
        qualified_class_name: str,
        *,
        visibility_mode: Literal[
            "public", "same_package", "same_package_or_subclass"
        ] = "public",
        test_package: Optional[str] = None,
        include_metadata: bool = False,
    ) -> Dict[str, List[Union[str, Dict[str, Any]]]]:
        """
        Retrieves methods reachable from qualified class along its inheritance graph. Precedence looks at the class itself,
        then superclasses, then interfaces (level-order).

        Args:
            qualified_class_name: The qualified name of the class.
            visibility_mode: The visibility mode. Either "public", "same_package", or "same_package_or_subclass".
            test_package: The package of the test class (for deciding whether a subclass is required).
            include_metadata: Include metadata in the output.

        Returns:
            Dict[str, List[str]] mapping owner (class or interface) -> list of method signatures

        Raises:
            ClassNotFoundError: If the qualified_class_name cannot be found.
        """
        common = CommonAnalysis(self.analysis)

        root_details = self.analysis.get_class(qualified_class_name)
        if not root_details:
            raise ClassNotFoundError(
                f"Class {qualified_class_name} not found.",
                extra_info={"qualified_class_name": qualified_class_name},
            )

        def _accept(owner: str, method_sig: str) -> bool:
            return common.is_accessible_from(
                owner,
                method_sig,
                accessor_class=qualified_class_name,
                mode=visibility_mode,
            )

        def _meta(owner: str, method_sig: str) -> Dict[str, Any]:
            method_details = self.analysis.get_method(owner, method_sig)
            owner_pkg = common.package_of(owner)
            mods = list(method_details.modifiers) if method_details else []
            visibility = (
                "private"
                if "private" in mods
                else "public"
                if "public" in mods
                else "protected"
                if "protected" in mods
                else "package-private"
            )
            # To call the method, a small subclass must be created that calls the method using the subclass type (this)
            requires_subclass = visibility == "protected" and owner_pkg != test_package
            return {
                "method_signature": method_sig,
                "declaring_qualified_class_name": owner,
                "modifiers": mods,
                "visibility": visibility,
                "requires_subclass": requires_subclass,
            }

        result: Dict[str, List[Union[str, Dict[str, Any]]]] = {}
        seen_sigs: set[str] = set()

        def _add_methods(owner: str) -> None:
            for method_sig in self.analysis.get_methods_in_class(owner):
                if method_sig in seen_sigs:
                    continue
                if _accept(owner, method_sig):
                    seen_sigs.add(method_sig)
                    if include_metadata:
                        result.setdefault(owner, []).append(_meta(owner, method_sig))
                    else:
                        result.setdefault(owner, []).append(method_sig)

        # Methods on the class itself
        _add_methods(qualified_class_name)

        # Superclasses in BFS order
        super_queue: deque[str] = deque(root_details.extends_list or [])
        visited_supers: set[str] = set(root_details.extends_list or [])
        super_bfs_order: List[str] = []

        while super_queue:
            sup_cls = super_queue.popleft()
            super_bfs_order.append(sup_cls)
            _add_methods(sup_cls)

            sup_details = self.analysis.get_class(sup_cls)
            if sup_details and sup_details.extends_list:
                for next_sup in sup_details.extends_list:
                    if next_sup not in visited_supers:
                        visited_supers.add(next_sup)
                        super_queue.append(next_sup)

        # Interfaces in BFS order
        iface_queue: deque[str] = deque()
        visited_ifaces: set[str] = set()

        def _enqueue_interfaces(owner: str) -> None:
            owner_details = self.analysis.get_class(owner)
            if owner_details and owner_details.implements_list:
                for iface in owner_details.implements_list:
                    if iface not in visited_ifaces:
                        visited_ifaces.add(iface)
                        iface_queue.append(iface)

        _enqueue_interfaces(qualified_class_name)
        for sup in super_bfs_order:
            _enqueue_interfaces(sup)

        while iface_queue:
            iface = iface_queue.popleft()
            _add_methods(iface)

            iface_details = self.analysis.get_class(iface)
            if iface_details and iface_details.extends_list:
                for parent_iface in iface_details.extends_list:
                    if parent_iface not in visited_ifaces:
                        visited_ifaces.add(parent_iface)
                        iface_queue.append(parent_iface)

        return result

    def get_inherited_classes_and_interfaces(
        self, qualified_class_name: str
    ) -> List[str]:
        """
        Returns all inherited types for the given class, first looking at superclasses then interfaces.
        """
        root_details = self.analysis.get_class(qualified_class_name)
        if not root_details:
            raise ClassNotFoundError(
                f"Class {qualified_class_name} not found.",
                extra_info={"qualified_class_name": qualified_class_name},
            )

        # Superclasses in level order
        super_queue: deque[str] = deque(root_details.extends_list or [])
        visited_supers: set[str] = set(root_details.extends_list or [])
        super_bfs_order: List[str] = []

        while super_queue:
            sup_cls = super_queue.popleft()
            super_bfs_order.append(sup_cls)

            sup_details = self.analysis.get_class(sup_cls)
            if sup_details and sup_details.extends_list:
                for next_sup in sup_details.extends_list:
                    if next_sup not in visited_supers:
                        visited_supers.add(next_sup)
                        super_queue.append(next_sup)

        # Interfaces in level order (from class and all discovered supers)
        iface_queue: deque[str] = deque()
        visited_ifaces: set[str] = set()
        iface_bfs_order: List[str] = []

        def _enqueue_interfaces(owner: str) -> None:
            owner_details = self.analysis.get_class(owner)
            if owner_details and owner_details.implements_list:
                for iface in owner_details.implements_list:
                    if iface not in visited_ifaces:
                        visited_ifaces.add(iface)
                        iface_queue.append(iface)

        _enqueue_interfaces(qualified_class_name)
        for sup in super_bfs_order:
            _enqueue_interfaces(sup)

        while iface_queue:
            iface = iface_queue.popleft()
            iface_bfs_order.append(iface)

            iface_details = self.analysis.get_class(iface)
            if iface_details and iface_details.extends_list:
                for parent_iface in iface_details.extends_list:
                    if parent_iface not in visited_ifaces:
                        visited_ifaces.add(parent_iface)
                        iface_queue.append(parent_iface)

        return super_bfs_order + iface_bfs_order

    def get_reachable_test_methods(
        self,
        qualified_class_name: str,
        testing_frameworks: List[TestingFramework],
    ) -> Dict[str, List[str]]:
        """
        Retrieves test methods reachable from the qualified class via inheritance.
        Traverses superclasses first (BFS), then interfaces.

        Args:
            qualified_class_name: The qualified name of the class.
            testing_frameworks: Testing frameworks available in this class's compilation unit.

        Returns:
            Dict[str, List[str]] mapping declaring class -> list of test method signatures

        Raises:
            ClassNotFoundError: If the qualified_class_name cannot be found.
        """
        common = CommonAnalysis(self.analysis)

        root_details = self.analysis.get_class(qualified_class_name)
        if not root_details:
            raise ClassNotFoundError(
                f"Class {qualified_class_name} not found.",
                extra_info={"qualified_class_name": qualified_class_name},
            )

        def _is_test(owner: str, method_sig: str) -> bool:
            owner_frameworks = common.get_testing_frameworks_for_class(owner)
            if not owner_frameworks:
                return False
            return common.is_test_method(method_sig, owner, owner_frameworks)

        result: Dict[str, List[str]] = {}
        seen_sigs: set[str] = set()

        def _add_methods(owner: str) -> None:
            for method_sig in self.analysis.get_methods_in_class(owner):
                if method_sig in seen_sigs:
                    continue
                if _is_test(owner, method_sig):
                    seen_sigs.add(method_sig)
                    result.setdefault(owner, []).append(method_sig)

        # Methods on the class itself
        _add_methods(qualified_class_name)

        # Superclasses in BFS order
        super_queue: deque[str] = deque(root_details.extends_list or [])
        visited_supers: set[str] = set(root_details.extends_list or [])

        while super_queue:
            sup_cls = super_queue.popleft()
            _add_methods(sup_cls)

            sup_details = self.analysis.get_class(sup_cls)
            if sup_details and sup_details.extends_list:
                for next_sup in sup_details.extends_list:
                    if next_sup not in visited_supers:
                        visited_supers.add(next_sup)
                        super_queue.append(next_sup)

        return result
