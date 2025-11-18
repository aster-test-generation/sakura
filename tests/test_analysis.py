from nltest.utils.analysis import CommonAnalysis
from nltest.utils.pretty.prints import pretty_print


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


def test_focal_classes_and_methods_for_specific_test(test2nl_context):
    """Test that pretty prints focal classes and methods for a specific test method."""
    analysis = test2nl_context.analysis
    qualified_class_name = (
        "org.springframework.samples.petclinic.owner.PetTypeFormatterTests"
    )
    method_signature = "shouldParse()"

    testing_frameworks = CommonAnalysis(
        analysis
    ).get_testing_frameworks_for_class(qualified_class_name)
    setup_methods = CommonAnalysis(analysis).get_setup_methods(qualified_class_name)
    setup_method_signatures = [method.signature for method in setup_methods]

    _, application_classes = CommonAnalysis(
        analysis
    ).get_test_methods_classes_and_application_classes()

    from hamster.code_analysis.focal_class_method.focal_class_method import (
        FocalClassMethod,
    )

    focal_class_method = FocalClassMethod(analysis, application_classes)

    focal_classes, _, _, _ = focal_class_method.identify_focal_class_and_ui_api_test(
        qualified_class_name, method_signature, setup_method_signatures
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
