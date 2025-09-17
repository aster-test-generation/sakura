import os
import sys
import types

from pathlib import Path
from typing import List

from nltest.test2nl.model.models import (
    AbstractionLevel,
    TestDescriptionInfo,
    Test2NLEntry,
)
from nltest.cli import generate_descriptions
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.pretty.prints import pretty_print

from tests._base_test2nl import BaseTest2NL


class TestDescriptionGeneration(BaseTest2NL):

    def test_desc_one_abs_one_method(self):
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        abstraction_level = AbstractionLevel.HIGH
        test_description_info: TestDescriptionInfo = (
            self.desc_generator.generate_for_method(
                method_signature, qualified_class_name, abstraction_level
            )
        )
        self.assertIsNotNone(test_description_info, "LLM generation failed...")

        test_descriptions = [test_description_info]
        _ = self.data_manager.save("descriptions.json", test_descriptions)

    def test_desc_all_abs_one_method(self):
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        test_descriptions = []
        for abs_level in AbstractionLevel:
            test_description_info = self.desc_generator.generate_for_method(
                method_signature, qualified_class_name, abs_level
            )
            if test_description_info:
                test_descriptions.append(test_description_info)
        self.assertTrue(test_descriptions, "Descriptions could not be generated...")

        test2nl_entries: List[Test2NLEntry] = []
        entry_id = 1
        for test_description_info in test_descriptions:
            test_description_info.id = entry_id
            entry = Test2NLEntry.from_test_description_info(
                test_description_info, self.project_name
            )
            test2nl_entries.append(entry)
            entry_id += 1

        # Persist outputs
        self.data_manager.save("descriptions.json", test_descriptions, format="json")
        self.data_manager.save("test2nl.csv", test2nl_entries, format="csv")
        # Verify files were written
        self.assertTrue(
            (self.data_manager.base_dir / "descriptions.json").exists(),
            "Descriptions file was not created",
        )
        self.assertTrue(
            (self.data_manager.base_dir / "test2nl.csv").exists(),
            "Test2NL CSV file was not created",
        )

    def test_test2nl_load(self):
        test2nl: List[Test2NLEntry] = self.data_manager.load(
            "test2nl.csv", Test2NLEntry, format="csv"
        )

        self.assertGreater(len(test2nl), 0, "Test2NL is empty...")

        single_entry = test2nl[0]
        pretty_print("Test2NL Entry", single_entry)

        single_description = single_entry.description
        pretty_print("Test2NL Description", single_description)

    def test_description_multiple_focal(self):
        complicated_tests = CommonAnalysis(self.analysis).get_complicated_focal_tests()
        total_count = CommonAnalysis(self.analysis).get_complicated_focal_tests_count()

        pretty_print("Complicated focal tests by class", complicated_tests)
        pretty_print("Total number of complicated focal test methods", total_count)

        self.assertIsInstance(complicated_tests, dict, "Should return a dictionary")
        self.assertGreaterEqual(total_count, 0, "Count should be non-negative")

        test_descriptions = []
        entry_id = 1

        for test_class, methods in complicated_tests.items():
            pretty_print(
                f"Processing class {test_class} with {len(methods)} methods", methods
            )

            for method_signature in methods:
                for abs_level in AbstractionLevel:
                    test_description_info = self.desc_generator.generate_for_method(
                        method_signature, test_class, abs_level
                    )
                    if test_description_info:
                        test_description_info.id = entry_id
                        test_descriptions.append(test_description_info)
                        entry_id += 1

            if entry_id > 15:
                break

        self.assertTrue(test_descriptions, "No descriptions could be generated...")
        pretty_print(
            f"Generated {len(test_descriptions)} test descriptions",
            len(test_descriptions),
        )

        test2nl_entries: List[Test2NLEntry] = []
        for test_description_info in test_descriptions:
            entry = Test2NLEntry.from_test_description_info(
                test_description_info, self.project_name
            )
            test2nl_entries.append(entry)

        self.data_manager.save(
            "descriptions.json", test_descriptions, format="json", mode="append"
        )

        self.data_manager.save(
            "test2nl.csv", test2nl_entries, format="csv", mode="append"
        )

        pretty_print(
            f"Successfully saved {len(test_descriptions)} descriptions and {len(test2nl_entries)} Test2NL entries"
        )
