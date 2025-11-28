from typing import Dict, List
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.pretty.prints import pretty_print

from hamster.code_analysis.test_statistics.setup_analysis_info import SetupAnalysisInfo


def test_multiple_focal(test2nl_context):
    analysis = test2nl_context.analysis
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


def test_focal_classes_and_methods_for_specific_test(nl2test_context):
    """Test that pretty prints focal classes and methods for a specific test method."""
    analysis = nl2test_context.analysis
    qualified_class_name = (
        "org.springframework.samples.petclinic.owner.PetTypeFormatterTests"
    )
    method_signature = "shouldParse()"

    setup_methods: Dict[str, List[str]] = SetupAnalysisInfo(analysis).get_setup_methods(qualified_class_name)

    _, application_classes = CommonAnalysis(
        analysis
    ).get_test_methods_classes_and_application_classes()

    from hamster.code_analysis.focal_class_method.focal_class_method import (
        FocalClassMethod,
    )

    focal_class_method = FocalClassMethod(analysis, application_classes)

    focal_classes, _, _, _ = focal_class_method.identify_focal_class_and_ui_api_test(
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
                f"{class_name}.{method_sig}"
                for class_name, method_sig in focal_methods
            ],
        },
    )

    assert isinstance(focal_classes, list), "Focal classes should be a list"
    assert isinstance(focal_methods, set), "Focal methods should be a set"
    assert len(focal_classes) >= 0
    assert len(focal_methods) >= 0

    for i, focal_class in enumerate(focal_classes):
        pretty_print(
            f"Focal Class {i+1}: {focal_class.focal_class}",
            {
                "focal_class": focal_class.focal_class,
                "focal_method_names": focal_class.focal_method_names,
                "num_methods": len(focal_class.focal_method_names),
            },
        )


def test_compare_focal_classes_and_methods(nl2test_context):
    """Test that compares focal classes and methods between ground truth and prediction."""

    # Easily adjustable parameters
    gt_class = "org.springframework.samples.petclinic.owner.PetTypeFormatterTests"
    gt_method = "shouldParse()"
    pred_class = "org.springframework.samples.petclinic.owner.PetTypeFormatterTests"
    pred_method = "shouldParse()"

    analysis = nl2test_context.analysis

    # Helper function to get focal classes and methods for a test
    def get_focal_info(qualified_class_name: str, method_signature: str):
        setup_methods: Dict[str, List[str]] = SetupAnalysisInfo(analysis).get_setup_methods(qualified_class_name)

        _, application_classes = CommonAnalysis(
            analysis
        ).get_test_methods_classes_and_application_classes()

        from hamster.code_analysis.focal_class_method.focal_class_method import (
            FocalClassMethod,
        )

        focal_class_method = FocalClassMethod(analysis, application_classes)

        focal_classes, _, _, _ = focal_class_method.identify_focal_class_and_ui_api_test(
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
                "focal_methods": sorted([
                    f"{class_name}.{method_sig}"
                    for class_name, method_sig in gt_focal_methods
                ]),
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
                "focal_methods": sorted([
                    f"{class_name}.{method_sig}"
                    for class_name, method_sig in pred_focal_methods
                ]),
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
    gt_focal_class_names = {fc.focal_class for fc in gt_focal_classes}
    pred_focal_class_names = {fc.focal_class for fc in pred_focal_classes}

    assert gt_focal_class_names == pred_focal_class_names, (
        f"Focal class names differ:\n"
        f"GT only: {gt_focal_class_names - pred_focal_class_names}\n"
        f"Pred only: {pred_focal_class_names - gt_focal_class_names}"
    )
