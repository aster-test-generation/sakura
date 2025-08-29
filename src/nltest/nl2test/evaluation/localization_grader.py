from pathlib import Path
from typing import List, Set, Tuple, Dict, Any, Optional

from cldk.analysis.java import JavaAnalysis
from hamster.code_analysis.focal_class_method.focal_class_method import FocalClassMethod
from hamster.code_analysis.model.models import TestingFramework

from nltest.nl2test.models import NL2TestInput, AtomicBlock, CandidateMethod, AtomicBlockList
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.pretty.prints import pretty_print


class LocalizationGrader:
    def __init__(self, nl2_input: NL2TestInput, analysis: JavaAnalysis, project_root: Path):
        self.nl2_input = nl2_input
        self.analysis = analysis
        self.project_root = project_root
        self.common_analysis = CommonAnalysis(analysis)

    def grade(self, atomic_blocks: AtomicBlockList, detailed_output: bool = False) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Grade the localization of atomic blocks to focal methods.
        
        Args:
            atomic_blocks: List of atomic blocks with candidate methods
            detailed_output: If True, returns comprehensive evaluation results
            
        Returns:
            Tuple[float, Optional[Dict[str, Any]]]: 
                - Percentage of focal methods covered (0.0 to 1.0)
                - Detailed results dict if detailed_output=True, None otherwise
        """
        focal_methods = self._get_focal_methods()
        
        if not focal_methods:
            pretty_print("No focal methods found", {"test_class": self.nl2_input.qualified_class_name, 
                                                   "test_method": self.nl2_input.method_signature})
            return 0.0, None if not detailed_output else self._create_detailed_results(set(), set(), 0.0, atomic_blocks)
        
        # Find optimal coverage by going through atomic blocks one-by-one
        optimal_coverage = self._find_optimal_coverage(focal_methods, atomic_blocks)
        
        # Calculate score
        coverage_score = len(optimal_coverage) / len(focal_methods) if focal_methods else 0.0
        
        # Print basic results
        self._print_evaluation_results(focal_methods, optimal_coverage, coverage_score)
        
        # Return detailed results if requested
        detailed_results = None
        if detailed_output:
            detailed_results = self._create_detailed_results(
                focal_methods, optimal_coverage, coverage_score, atomic_blocks
            )
        
        return coverage_score, detailed_results

    def _find_optimal_coverage(self, focal_methods: Set[Tuple[str, str]], atomic_blocks: AtomicBlockList) -> Set[Tuple[str, str]]:
        """
        Find optimal coverage by going through atomic blocks one-by-one.
        Each focal method can only be matched once across all atomic blocks.
        """
        covered_focal_methods = set()
        available_focal_methods = focal_methods.copy()
        
        # Go through each atomic block in order
        for block in atomic_blocks.atomic_blocks:
            block_candidates = set()
            
            # Extract candidate methods from this block
            for candidate in block.candidate_methods:
                method_key = (candidate.containing_class_name, candidate.method_signature)
                block_candidates.add(method_key)
            
            # Find focal methods that can be covered by this block
            for focal_method in available_focal_methods:
                if focal_method in block_candidates:
                    covered_focal_methods.add(focal_method)
                    available_focal_methods.remove(focal_method)
                    # Once a focal method is covered, it cannot be used again
                    break
        
        return covered_focal_methods

    def _create_detailed_results(self, focal_methods: Set[Tuple[str, str]], 
                                covered_focal_methods: Set[Tuple[str, str]], 
                                coverage_score: float, 
                                atomic_blocks: AtomicBlockList) -> Dict[str, Any]:
        """Create comprehensive evaluation results for detailed output."""
        focal_methods_str = [f"{class_name}.{method_sig}" for class_name, method_sig in focal_methods]
        covered_methods_str = [f"{class_name}.{method_sig}" for class_name, method_sig in covered_focal_methods]
        uncovered_methods_str = [f"{class_name}.{method_sig}" for class_name, method_sig in (focal_methods - covered_focal_methods)]
        
        # Analyze atomic blocks and their coverage
        block_analysis = []
        for i, block in enumerate(atomic_blocks.atomic_blocks):
            block_candidates = []
            for candidate in block.candidate_methods:
                method_key = (candidate.containing_class_name, candidate.method_signature)
                is_covered = method_key in covered_focal_methods
                block_candidates.append({
                    "containing_class": candidate.containing_class_name,
                    "method_signature": candidate.method_signature,
                    "is_focal_method": method_key in focal_methods,
                    "is_covered": is_covered
                })
            
            block_analysis.append({
                "block_index": i,
                "simplified": block.simplified,
                "candidate_methods": block_candidates,
                "notes": block.notes
            })
        
        return {
            "test_class": self.nl2_input.qualified_class_name,
            "test_method": self.nl2_input.method_signature,
            "total_focal_methods": len(focal_methods),
            "covered_focal_methods": len(covered_focal_methods),
            "uncovered_focal_methods": len(uncovered_methods_str),
            "coverage_score": coverage_score,
            "focal_methods": focal_methods_str,
            "covered_methods": covered_methods_str,
            "uncovered_methods": uncovered_methods_str,
            "atomic_blocks_analysis": block_analysis,
            "evaluation_algorithm": "optimal_coverage_one_to_one"
        }

    def _get_focal_methods(self) -> Set[Tuple[str, str]]:
        """Use Hamster to get the focal methods for the test method."""
        try:
            testing_frameworks = self.common_analysis.get_testing_frameworks_for_class(self.nl2_input.qualified_class_name)
            setup_methods = self.common_analysis.get_setup_methods(self.nl2_input.qualified_class_name)
            setup_method_signatures = [method.signature for method in setup_methods]
            
            _, application_classes = self.common_analysis.get_test_methods_classes_and_application_classes()
            
            focal_class_method = FocalClassMethod(self.analysis, testing_frameworks, application_classes)
            focal_classes, _, _, _ = focal_class_method.identify_focal_class_and_ui_api_test(
                self.nl2_input.qualified_class_name, 
                self.nl2_input.method_signature, 
                setup_method_signatures
            )
            
            focal_methods = set()
            for focal_class in focal_classes:
                for method_name in focal_class.focal_method_names:
                    focal_methods.add((focal_class.focal_class, method_name))
            
            return focal_methods
            
        except Exception as e:
            pretty_print("Error getting focal methods", {"error": str(e)})
            return set()

    def _print_evaluation_results(self, focal_methods: Set[Tuple[str, str]], 
                                 covered_focal_methods: Set[Tuple[str, str]], 
                                 coverage_score: float):
        """Print basic evaluation results."""
        focal_methods_str = [f"{class_name}.{method_sig}" for class_name, method_sig in focal_methods]
        covered_methods_str = [f"{class_name}.{method_sig}" for class_name, method_sig in covered_focal_methods]
        uncovered_methods_str = [f"{class_name}.{method_sig}" for class_name, method_sig in (focal_methods - covered_focal_methods)]
        
        results = {
            "test_class": self.nl2_input.qualified_class_name,
            "test_method": self.nl2_input.method_signature,
            "total_focal_methods": len(focal_methods),
            "covered_focal_methods": len(covered_focal_methods),
            "coverage_score": coverage_score,
            "focal_methods": focal_methods_str,
            "covered_methods": covered_methods_str,
            "uncovered_methods": uncovered_methods_str
        }
        
        pretty_print("Localization Evaluation Results", results)
