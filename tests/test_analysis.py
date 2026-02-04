from typing import Dict, List

import pytest
from hamster.code_analysis.test_statistics.setup_analysis_info import SetupAnalysisInfo

from sakura.utils.analysis import CommonAnalysis
from sakura.utils.pretty.prints import pretty_print


def test_multiple_focal(petclinic_analysis):
    analysis = petclinic_analysis
    complicated_tests = CommonAnalysis(analysis).get_complicated_focal_tests()
    total_count = CommonAnalysis(analysis).get_complicated_focal_tests_count()

    pretty_print("Complicated focal tests by class", complicated_tests)
    pretty_print("Total number of complicated focal test methods", total_count)

    assert isinstance(complicated_tests, dict), "Should return a dictionary"
    assert total_count >= 0, "Count should be non-negative"

    for test_class, methods in complicated_tests.items():
        pretty_print(
            f"Class {test_class} has {len(methods)} complicated tests", methods
        )


def test_focal_classes_and_methods_for_specific_test(petclinic_analysis):
    """Test that pretty prints focal classes and methods for a specific test method."""
    analysis = petclinic_analysis
    qualified_class_name = (
        "org.springframework.samples.petclinic.owner.PetTypeFormatterTests"
    )
    method_signature = "shouldParse()"

    setup_methods: Dict[str, List[str]] = SetupAnalysisInfo(analysis).get_setup_methods(
        qualified_class_name
    )

    _, application_classes, _ = CommonAnalysis(analysis).categorize_classes()

    from hamster.code_analysis.focal_class_method.focal_class_method import (
        FocalClassMethod,
    )

    focal_class_method = FocalClassMethod(analysis, application_classes)

    focal_classes, _, _, _ = focal_class_method.extract_test_scope(
        qualified_class_name, method_signature, setup_methods
    )

    focal_methods = set()
    for focal_class in focal_classes:
        for method_name in focal_class.focal_method_names:
            focal_methods.add((focal_class.focal_class, method_name))

    pretty_print(
        f"Focal Analysis for {qualified_class_name}.{method_signature}",
        {
            "test_class": qualified_class_name,
            "test_method": method_signature,
            "total_focal_classes": len(focal_classes),
            "total_focal_methods": len(focal_methods),
            "focal_classes": [
                {
                    "focal_class": focal_class.focal_class,
                    "focal_method_names": focal_class.focal_method_names,
                    "num_focal_methods": len(focal_class.focal_method_names),
                }
                for focal_class in focal_classes
            ],
            "focal_methods": [
                f"{class_name}.{method_sig}" for class_name, method_sig in focal_methods
            ],
        },
    )

    assert isinstance(focal_classes, list), "Focal classes should be a list"
    assert isinstance(focal_methods, set), "Focal methods should be a set"
    assert len(focal_classes) >= 0
    assert len(focal_methods) >= 0

    for i, focal_class in enumerate(focal_classes):
        pretty_print(
            f"Focal Class {i + 1}: {focal_class.focal_class}",
            {
                "focal_class": focal_class.focal_class,
                "focal_method_names": focal_class.focal_method_names,
                "num_methods": len(focal_class.focal_method_names),
            },
        )


@pytest.mark.skip(reason="Test is just for comparison")
def test_compare_focal_classes_and_methods(petclinic_analysis):
    """Test that compares focal classes and methods between ground truth and prediction."""

    gt_class = "org.apache.commons.cli.CommandLineTest"
    gt_method = "testGetOptionPropertiesWithOption()"
    pred_class = "org.apache.commons.cli.CommandLineParserTest"
    pred_method = "testOptionPropertiesExtraction()"

    analysis = petclinic_analysis

    # Helper function to get focal classes and methods for a test
    def get_focal_info(qualified_class_name: str, method_signature: str):
        setup_methods: Dict[str, List[str]] = SetupAnalysisInfo(
            analysis
        ).get_setup_methods(qualified_class_name)

        _, application_classes, _ = CommonAnalysis(analysis).categorize_classes()

        from hamster.code_analysis.focal_class_method.focal_class_method import (
            FocalClassMethod,
        )

        focal_class_method = FocalClassMethod(analysis, application_classes)

        focal_classes, _, _, _ = focal_class_method.extract_test_scope(
            qualified_class_name, method_signature, setup_methods
        )

        focal_methods = set()
        for focal_class in focal_classes:
            for method_name in focal_class.focal_method_names:
                focal_methods.add((focal_class.focal_class, method_name))

        return focal_classes, focal_methods

    # Get focal information for ground truth
    gt_focal_classes, gt_focal_methods = get_focal_info(gt_class, gt_method)

    # Get focal information for prediction
    pred_focal_classes, pred_focal_methods = get_focal_info(pred_class, pred_method)

    # Pretty print comparison
    pretty_print(
        "Focal Classes and Methods Comparison",
        {
            "ground_truth": {
                "test_class": gt_class,
                "test_method": gt_method,
                "total_focal_classes": len(gt_focal_classes),
                "total_focal_methods": len(gt_focal_methods),
                "focal_classes": [
                    {
                        "focal_class": fc.focal_class,
                        "focal_method_names": fc.focal_method_names,
                        "num_focal_methods": len(fc.focal_method_names),
                    }
                    for fc in gt_focal_classes
                ],
                "focal_methods": sorted(
                    [
                        f"{class_name}.{method_sig}"
                        for class_name, method_sig in gt_focal_methods
                    ]
                ),
            },
            "prediction": {
                "test_class": pred_class,
                "test_method": pred_method,
                "total_focal_classes": len(pred_focal_classes),
                "total_focal_methods": len(pred_focal_methods),
                "focal_classes": [
                    {
                        "focal_class": fc.focal_class,
                        "focal_method_names": fc.focal_method_names,
                        "num_focal_methods": len(fc.focal_method_names),
                    }
                    for fc in pred_focal_classes
                ],
                "focal_methods": sorted(
                    [
                        f"{class_name}.{method_sig}"
                        for class_name, method_sig in pred_focal_methods
                    ]
                ),
            },
        },
    )

    # Compute overlap scores
    gt_focal_class_names = {fc.focal_class for fc in gt_focal_classes}
    pred_focal_class_names = {fc.focal_class for fc in pred_focal_classes}

    class_intersection = gt_focal_class_names & pred_focal_class_names
    class_union = gt_focal_class_names | pred_focal_class_names
    class_overlap = len(class_intersection) / len(class_union) if class_union else 1.0

    method_intersection = gt_focal_methods & pred_focal_methods
    method_union = gt_focal_methods | pred_focal_methods
    method_overlap = (
        len(method_intersection) / len(method_union) if method_union else 1.0
    )

    pretty_print(
        "Overlap Scores",
        {
            "focal_classes": {
                "intersection": sorted(class_intersection),
                "gt_only": sorted(gt_focal_class_names - pred_focal_class_names),
                "pred_only": sorted(pred_focal_class_names - gt_focal_class_names),
                "overlap_score (IoU)": f"{class_overlap:.2%}",
                "count": f"{len(class_intersection)}/{len(class_union)}",
            },
            "focal_methods": {
                "intersection": sorted([f"{c}.{m}" for c, m in method_intersection]),
                "gt_only": sorted(
                    [f"{c}.{m}" for c, m in (gt_focal_methods - pred_focal_methods)]
                ),
                "pred_only": sorted(
                    [f"{c}.{m}" for c, m in (pred_focal_methods - gt_focal_methods)]
                ),
                "overlap_score (IoU)": f"{method_overlap:.2%}",
                "count": f"{len(method_intersection)}/{len(method_union)}",
            },
        },
    )

    # Assert equality
    assert len(gt_focal_classes) == len(pred_focal_classes), (
        f"Number of focal classes differs: GT={len(gt_focal_classes)}, "
        f"Pred={len(pred_focal_classes)}"
    )

    assert gt_focal_methods == pred_focal_methods, (
        f"Focal methods differ:\n"
        f"GT only: {gt_focal_methods - pred_focal_methods}\n"
        f"Pred only: {pred_focal_methods - gt_focal_methods}"
    )

    # Compare focal classes in detail
    assert gt_focal_class_names == pred_focal_class_names, (
        f"Focal class names differ:\n"
        f"GT only: {gt_focal_class_names - pred_focal_class_names}\n"
        f"Pred only: {pred_focal_class_names - gt_focal_class_names}"
    )


class TestCategorizeClasses:
    """Tests for CommonAnalysis.categorize_classes() using Spring PetClinic."""

    def test_categorize_classes_returns_tuple(self, petclinic_analysis):
        """Test that categorize_classes returns a 3-tuple with correct types."""
        test_class_map, application_classes, test_utility_classes = CommonAnalysis(
            petclinic_analysis
        ).categorize_classes()

        assert isinstance(test_class_map, dict)
        assert isinstance(application_classes, list)
        assert isinstance(test_utility_classes, list)

    def test_application_classes_contains_domain_classes(self, petclinic_analysis):
        """Test that application_classes includes known domain classes from src/main."""
        _, application_classes, _ = CommonAnalysis(
            petclinic_analysis
        ).categorize_classes()

        expected_app_classes = [
            "org.springframework.samples.petclinic.owner.Owner",
            "org.springframework.samples.petclinic.owner.Pet",
            "org.springframework.samples.petclinic.owner.PetType",
            "org.springframework.samples.petclinic.owner.Visit",
            "org.springframework.samples.petclinic.vet.Vet",
            "org.springframework.samples.petclinic.vet.Specialty",
            "org.springframework.samples.petclinic.model.BaseEntity",
            "org.springframework.samples.petclinic.model.Person",
        ]
        for expected in expected_app_classes:
            assert expected in application_classes, (
                f"Expected application class {expected} not found"
            )

    def test_application_classes_excludes_test_classes(self, petclinic_analysis):
        """Test that application_classes does not include test classes."""
        _, application_classes, _ = CommonAnalysis(
            petclinic_analysis
        ).categorize_classes()

        test_class_names = [
            "org.springframework.samples.petclinic.service.ClinicServiceTests",
            "org.springframework.samples.petclinic.owner.OwnerControllerTests",
            "org.springframework.samples.petclinic.vet.VetControllerTests",
        ]
        for test_class in test_class_names:
            assert test_class not in application_classes, (
                f"Test class {test_class} should not be in application_classes"
            )

    def test_test_utility_classes_contains_helper_classes(self, petclinic_analysis):
        """Test that test_utility_classes includes classes in test dirs without test methods."""
        _, _, test_utility_classes = CommonAnalysis(
            petclinic_analysis
        ).categorize_classes()

        expected_utility_classes = [
            "org.springframework.samples.petclinic.service.EntityUtils",
            "org.springframework.samples.petclinic.MysqlTestApplication",
        ]
        for expected in expected_utility_classes:
            assert expected in test_utility_classes, (
                f"Expected test utility class {expected} not found"
            )

    def test_test_utility_classes_excludes_test_classes(self, petclinic_analysis):
        """Test that test_utility_classes does not include actual test classes."""
        _, _, test_utility_classes = CommonAnalysis(
            petclinic_analysis
        ).categorize_classes()

        test_class_names = [
            "org.springframework.samples.petclinic.service.ClinicServiceTests",
            "org.springframework.samples.petclinic.owner.OwnerControllerTests",
        ]
        for test_class in test_class_names:
            assert test_class not in test_utility_classes, (
                f"Test class {test_class} should not be in test_utility_classes"
            )

    def test_test_class_map_contains_test_classes(self, petclinic_analysis):
        """Test that test_class_map includes test classes with their test methods."""
        test_class_map, _, _ = CommonAnalysis(petclinic_analysis).categorize_classes()

        expected_test_class = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        assert expected_test_class in test_class_map, (
            f"Expected test class {expected_test_class} not found in test_class_map"
        )

        test_methods = test_class_map[expected_test_class]
        assert isinstance(test_methods, list)
        assert len(test_methods) > 0, "Test class should have test methods"
        assert "shouldInsertPetIntoDatabaseAndGenerateId()" in test_methods

    def test_categorized_classes_fixture(self, petclinic_categorized_classes):
        """Test that the petclinic_categorized_classes fixture returns valid data."""
        application_classes, test_utility_classes = petclinic_categorized_classes

        assert isinstance(application_classes, list)
        assert isinstance(test_utility_classes, list)
        assert len(application_classes) > 0, "Should have application classes"

        assert (
            "org.springframework.samples.petclinic.owner.Owner" in application_classes
        )
        assert (
            "org.springframework.samples.petclinic.service.EntityUtils"
            in test_utility_classes
        )


class TestReachability:
    """Tests for Reachability helper method detection."""

    def test_get_helper_methods_includes_entity_utils(
        self, petclinic_analysis, petclinic_categorized_classes
    ):
        """Test that get_helper_methods for ClinicServiceTests returns EntityUtils."""
        from sakura.utils.analysis import Reachability

        _, test_utility_classes = petclinic_categorized_classes
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        reachability = Reachability(petclinic_analysis)
        helper_methods = reachability.get_helper_methods(
            qualified_class_name,
            method_signature,
            test_utility_classes=test_utility_classes,
        )

        entity_utils_class = "org.springframework.samples.petclinic.service.EntityUtils"
        assert entity_utils_class in helper_methods, (
            f"Expected {entity_utils_class} in helper methods, got: {list(helper_methods.keys())}"
        )


class TestSimplifyMethodSignature:
    """Tests for CommonAnalysis.simplify_method_signature()."""

    def test_no_parameters_unchanged(self):
        sig = "testMethod()"
        result = CommonAnalysis.simplify_method_signature(sig)
        assert result == "testMethod()"

    def test_single_qualified_type(self):
        sig = "addSpecialty(org.springframework.samples.petclinic.vet.Specialty)"
        result = CommonAnalysis.simplify_method_signature(sig)
        assert result == "addSpecialty(Specialty)"

    def test_array_type_preserved(self):
        sig = "main(java.lang.String[])"
        result = CommonAnalysis.simplify_method_signature(sig)
        assert result == "main(String[])"

    def test_nested_generics_removed(self):
        # Note: This might not be optimal behavior
        sig = (
            "method(java.util.Map<java.lang.String, java.util.List<java.lang.Integer>>)"
        )
        result = CommonAnalysis.simplify_method_signature(sig)
        assert result == "method(Map)"

    def test_real_cldk_signature_from_analysis(self, petclinic_analysis):
        """Test with actual method signature from CLDK analysis."""
        entity_utils_class = "org.springframework.samples.petclinic.service.EntityUtils"
        original_sig = "getById(java.util.Collection, java.lang.Class, int)"

        method = petclinic_analysis.get_method(entity_utils_class, original_sig)
        assert method is not None, f"Method {original_sig} should exist in analysis"

        simplified = CommonAnalysis.simplify_method_signature(original_sig)
        assert simplified == "getById(Collection, Class, int)"
