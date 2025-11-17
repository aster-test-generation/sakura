from typing import List

from nltest.test2nl.model.models import (
    AbstractionLevel,
    TestDescriptionInfo,
    Test2NLEntry,
)
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.pretty.prints import pretty_print


def test_desc_one_abs_one_method(test2nl_context):
    qualified_class_name = "org.springframework.samples.petclinic.service.ClinicServiceTests"
    method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

    abstraction_level = AbstractionLevel.HIGH
    desc_generator = test2nl_context.desc_generator
    test_description_info: TestDescriptionInfo = desc_generator.generate_for_method(
        method_signature, qualified_class_name, abstraction_level
    )
    assert test_description_info is not None, "LLM generation failed..."

    test_descriptions = [test_description_info]
    test2nl_context.data_manager.save("descriptions.json", test_descriptions)


def test_desc_all_abs_one_method(test2nl_context):
    qualified_class_name = "org.springframework.samples.petclinic.service.ClinicServiceTests"
    method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

    desc_generator = test2nl_context.desc_generator
    test_descriptions = []
    for abs_level in AbstractionLevel:
        test_description_info = desc_generator.generate_for_method(
            method_signature, qualified_class_name, abs_level
        )
        if test_description_info:
            test_descriptions.append(test_description_info)
    assert test_descriptions, "Descriptions could not be generated..."

    data_manager = test2nl_context.data_manager
    test2nl_entries: List[Test2NLEntry] = []
    entry_id = 1
    for test_description_info in test_descriptions:
        test_description_info.id = entry_id
        entry = Test2NLEntry.from_test_description_info(
            test_description_info, test2nl_context.project_name
        )
        test2nl_entries.append(entry)
        entry_id += 1

    data_manager.save("descriptions.json", test_descriptions, format="json")
    data_manager.save("test2nl.csv", test2nl_entries, format="csv")
    descriptions_path = data_manager.base_dir / "descriptions.json"
    csv_path = data_manager.base_dir / "test2nl.csv"
    assert descriptions_path.exists(), "Descriptions file was not created"
    assert csv_path.exists(), "Test2NL CSV file was not created"


def test_test2nl_load(test2nl_context):
    test2nl: List[Test2NLEntry] = test2nl_context.data_manager.load(
        "test2nl.csv", Test2NLEntry, format="csv"
    )
    assert len(test2nl) > 0, "Test2NL is empty..."

    single_entry = test2nl[0]
    pretty_print("Test2NL Entry", single_entry)

    single_description = single_entry.description
    pretty_print("Test2NL Description", single_description)


def test_description_multiple_focal(test2nl_context):
    analysis = test2nl_context.analysis
    complicated_tests = CommonAnalysis(analysis).get_complicated_focal_tests()
    total_count = CommonAnalysis(analysis).get_complicated_focal_tests_count()

    pretty_print("Complicated focal tests by class", complicated_tests)
    pretty_print("Total number of complicated focal test methods", total_count)

    assert isinstance(complicated_tests, dict), "Should return a dictionary"
    assert total_count >= 0, "Count should be non-negative"

    desc_generator = test2nl_context.desc_generator
    test_descriptions: List[TestDescriptionInfo] = []
    entry_id = 1

    for test_class, methods in complicated_tests.items():
        pretty_print(
            f"Processing class {test_class} with {len(methods)} methods", methods
        )

        for method_signature in methods:
            for abs_level in AbstractionLevel:
                test_description_info = desc_generator.generate_for_method(
                    method_signature, test_class, abs_level
                )
                if test_description_info:
                    test_description_info.id = entry_id
                    test_descriptions.append(test_description_info)
                    entry_id += 1

        if entry_id > 15:
            break

    assert test_descriptions, "No descriptions could be generated..."
    pretty_print(
        f"Generated {len(test_descriptions)} test descriptions",
        len(test_descriptions),
    )

    entries: List[Test2NLEntry] = []
    for test_description_info in test_descriptions:
        entry = Test2NLEntry.from_test_description_info(
            test_description_info, test2nl_context.project_name
        )
        entries.append(entry)

    data_manager = test2nl_context.data_manager
    data_manager.save("descriptions.json", test_descriptions, format="json", mode="append")
    data_manager.save("test2nl.csv", entries, format="csv", mode="append")

    print(
        f"Successfully saved {len(test_descriptions)} descriptions and {len(entries)} Test2NL entries"
    )
