from nltest.utils.analysis import CommonAnalysis
from nltest.utils.pretty.prints import pretty_print

from tests._base_test2nl import BaseTest2NL


class TestCommonAnalysis(BaseTest2NL):
    def test_multiple_focal(self):
        complicated_tests = CommonAnalysis(self.analysis).get_complicated_focal_tests()
        total_count = CommonAnalysis(self.analysis).get_complicated_focal_tests_count()

        pretty_print("Complicated focal tests by class", complicated_tests)
        pretty_print("Total number of complicated focal test methods", total_count)

        self.assertIsInstance(complicated_tests, dict, "Should return a dictionary")
        self.assertGreaterEqual(total_count, 0, "Count should be non-negative")

        for test_class, methods in complicated_tests.items():
            pretty_print(
                f"Class {test_class} has {len(methods)} complicated tests", methods
            )


class TestFocalAnalysis(BaseTest2NL):
    def test_focal_classes_and_methods_for_specific_test(self):
        """Test that pretty prints focal classes and methods for a specific test method."""
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        # method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"
        method_signature = "shouldFindVets()"
        method_signature = "shouldInsertOwner()"

        qualified_class_name = (
            "org.springframework.samples.petclinic.vet.VetControllerTests"
        )
        method_signature = "testShowResourcesVetList()"

        qualified_class_name = (
            "org.springframework.samples.petclinic.owner.PetTypeFormatterTests"
        )
        method_signature = "shouldParse()"

        # Get testing frameworks and setup methods
        testing_frameworks = CommonAnalysis(
            self.analysis
        ).get_testing_frameworks_for_class(qualified_class_name)
        setup_methods = CommonAnalysis(self.analysis).get_setup_methods(
            qualified_class_name
        )
        setup_method_signatures = [method.signature for method in setup_methods]

        # Get application classes
        _, application_classes = CommonAnalysis(
            self.analysis
        ).get_test_methods_classes_and_application_classes()

        # Create focal class method analyzer
        from hamster.code_analysis.focal_class_method.focal_class_method import (
            FocalClassMethod,
        )

        focal_class_method = FocalClassMethod(
            self.analysis, application_classes
        )

        # Get focal classes and methods
        focal_classes, _, _, _ = (
            focal_class_method.identify_focal_class_and_ui_api_test(
                qualified_class_name, method_signature, setup_method_signatures
            )
        )

        # Extract focal methods
        focal_methods = set()
        for focal_class in focal_classes:
            for method_name in focal_class.focal_method_names:
                focal_methods.add((focal_class.focal_class, method_name))

        # Pretty print results
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

        # Assertions
        self.assertIsInstance(focal_classes, list, "Focal classes should be a list")
        self.assertIsInstance(focal_methods, set, "Focal methods should be a set")
        self.assertGreaterEqual(
            len(focal_classes), 0, "Should have at least 0 focal classes"
        )
        self.assertGreaterEqual(
            len(focal_methods), 0, "Should have at least 0 focal methods"
        )

        # Print detailed breakdown
        for i, focal_class in enumerate(focal_classes):
            pretty_print(
                f"Focal Class {i+1}: {focal_class.focal_class}",
                {
                    "focal_class": focal_class.focal_class,
                    "focal_method_names": focal_class.focal_method_names,
                    "num_methods": len(focal_class.focal_method_names),
                },
            )
