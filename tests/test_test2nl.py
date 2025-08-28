import os

from dotenv import load_dotenv
from typing import List
from unittest import TestCase
from pathlib import Path

from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.test2nl.model.models import AbstractionLevel, TestDescriptionInfo, Test2NLEntry
from nltest.test2nl import Pipeline
from nltest.test2nl.prompts import RoundTripPrompt, Test2NLPrompt
from nltest.test2nl.generation import DescriptionGenerator
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.config import init_config, Config
from nltest.utils.llm.model import Provider
from nltest.utils.pretty.prints import pretty_print
from nltest.utils.file_io.structured_data_manager import StructuredDataManager


class TestTest2NL(TestCase):
    def setUp(self) -> None:
        load_dotenv()

        # === User-Defined ===
        self.project_name = "spring-petclinic"

        llm_model = "qwen/qwen3-235b-a22b-thinking-2507"

        emb_model = "NOMIC-AI-EMB-7B"

        project_root = Path(f"./resources/{self.project_name}")
        if not (project_root.exists() and project_root.is_dir()):
            raise Exception(f"Project root directory {project_root} does not exist.")
        output_dir = Path(f"./output/{self.project_name}")

        self.assertTrue(os.getenv("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY environment variable is not set.")

        self.config = init_config(
            project_name=self.project_name,
            base_project_dir=str(project_root),
            output_dir=str(output_dir),
            llm_provider=Provider.OPENROUTER,
            llm_model=llm_model,
            emb_provider=Provider.VLLM,
            emb_model=emb_model,
            llm_api_key=os.getenv("OPENROUTER_API_KEY"),  # Assign from env
            emb_api_key=None,
        )

        # Generate analysis of the current project
        self.analysis = CLDK(language="java").analysis(
            project_path=project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=output_dir,
            eager=False,
        )

        # Create orchestration
        self.pipeline = Pipeline(self.analysis, project_root, output_dir)

        # Create single generators
        self.desc_generator = DescriptionGenerator(self.analysis)
        self.data_manager = StructuredDataManager(output_dir)

        # Create prompts generators
        self.test2nl_prompt = Test2NLPrompt(self.analysis)
        self.roundtrip_prompt = RoundTripPrompt(self.analysis)

    def test_desc_one_abs_one_method(self):
        qualified_class_name = "org.springframework.samples.petclinic.service.ClinicServiceTests"
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        abstraction_level = AbstractionLevel.HIGH
        test_description_info: TestDescriptionInfo = self.desc_generator.generate_for_method(
            method_signature,
            qualified_class_name,
            abstraction_level
        )
        self.assertIsNotNone(test_description_info, "LLM generation failed...")

        test_descriptions = [test_description_info]
        save_success = self.data_manager.save("descriptions.json", test_descriptions)

    def test_desc_all_abs_one_method(self):
        qualified_class_name = "org.springframework.samples.petclinic.service.ClinicServiceTests"
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        test_descriptions = []
        for abs_level in AbstractionLevel:
            test_description_info = self.desc_generator.generate_for_method(
                method_signature,
                qualified_class_name,
                abs_level
            )
            if test_description_info:
                test_descriptions.append(test_description_info)
        self.assertTrue(test_descriptions, "Descriptions could not be generated...")

        test2nl_entries: List[Test2NLEntry] = []
        entry_id = 1
        for test_description_info in test_descriptions:
            test_description_info.id = entry_id
            entry = Test2NLEntry.from_test_description_info(test_description_info, self.project_name)
            test2nl_entries.append(entry)
            entry_id += 1

        save_success = self.data_manager.save("descriptions.json", test_descriptions, format="json")
        self.assertTrue(save_success, "Descriptions could not be saved...")
        save_success = self.data_manager.save("test2nl.csv", test2nl_entries, format="csv")
        self.assertTrue(save_success, "Test2NL could not be saved...")

    def test_test2nl_load(self):
        test2nl: List[Test2NLEntry] = self.data_manager.load("test2nl.csv", Test2NLEntry, format="csv")

        self.assertGreater(len(test2nl), 0, "Test2NL is empty...")

        single_entry = test2nl[0]
        pretty_print("Test2NL Entry", single_entry)

        single_description = single_entry.description
        pretty_print("Test2NL Description", single_description)


    def test_high_desc_prompt(self):
        qualified_class_name = "org.springframework.samples.petclinic.service.ClinicServiceTests"
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        test_desc, prompt, is_successful = self.test2nl_prompt.generate(method_signature, qualified_class_name,
                                                                        AbstractionLevel.HIGH)
        self.assertTrue(prompt, "Prompt was unsuccessfully rendered...")

        pretty_print("prompts", prompt)

        self.assertTrue(is_successful, "LLM generated was unsuccessful with prompts...")

    def test_all_low_abs(self):
        self.pipeline.reset_dataset()
        self.pipeline.run_descriptions(AbstractionLevel.LOW)

    def test_roundtrip_prompt(self):
        qualified_class_name = "org.springframework.samples.petclinic.service.ClinicServiceTests"
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        test_descriptions = self.data_manager.load("descriptions.json", TestDescriptionInfo)
        selected_description = test_descriptions[0]

        test_case, prompt, is_successful = self.roundtrip_prompt.generate(method_signature, qualified_class_name,
                                                                          selected_description.description)
        self.assertTrue(prompt, "Prompt was unsuccessfully rendered...")

        pretty_print("prompts", prompt)

        self.assertTrue(is_successful, "LLM generated was unsuccessful with prompts...")

    def test_multiple_focal(self):
        complicated_tests = CommonAnalysis(self.analysis).get_complicated_focal_tests()
        total_count = CommonAnalysis(self.analysis).get_complicated_focal_tests_count()
        
        pretty_print("Complicated focal tests by class", complicated_tests)
        pretty_print("Total number of complicated focal test methods", total_count)
        
        self.assertIsInstance(complicated_tests, dict, "Should return a dictionary")
        self.assertGreaterEqual(total_count, 0, "Count should be non-negative")
        
        for test_class, methods in complicated_tests.items():
            pretty_print(f"Class {test_class} has {len(methods)} complicated tests", methods)

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
            pretty_print(f"Processing class {test_class} with {len(methods)} methods", methods)
            
            for method_signature in methods:
                for abs_level in AbstractionLevel:
                    test_description_info = self.desc_generator.generate_for_method(
                        method_signature,
                        test_class,
                        abs_level
                    )
                    if test_description_info:
                        test_description_info.id = entry_id
                        test_descriptions.append(test_description_info)
                        entry_id += 1

            if entry_id > 15:
                break
        
        self.assertTrue(test_descriptions, "No descriptions could be generated...")
        pretty_print(f"Generated {len(test_descriptions)} test descriptions", len(test_descriptions))
        
        test2nl_entries: List[Test2NLEntry] = []
        for test_description_info in test_descriptions:
            entry = Test2NLEntry.from_test_description_info(test_description_info, self.project_name)
            test2nl_entries.append(entry)
        
        self.data_manager.save("descriptions.json", test_descriptions, format="json", mode="append")

        self.data_manager.save("test2nl.csv", test2nl_entries, format="csv", mode="append")

        pretty_print(f"Successfully saved {len(test_descriptions)} descriptions and {len(test2nl_entries)} Test2NL entries")
        
    def test_focal_classes_and_methods_for_specific_test(self):
        """Test that pretty prints focal classes and methods for a specific test method."""
        qualified_class_name = "org.springframework.samples.petclinic.service.ClinicServiceTests"
        # method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"
        method_signature = "shouldFindVets()"
        method_signature = "shouldInsertOwner()"

        qualified_class_name = "org.springframework.samples.petclinic.vet.VetControllerTests"
        method_signature = "testShowResourcesVetList()"

        # Get testing frameworks and setup methods
        testing_frameworks = CommonAnalysis(self.analysis).get_testing_frameworks_for_class(qualified_class_name)
        setup_methods = CommonAnalysis(self.analysis).get_setup_methods(qualified_class_name)
        setup_method_signatures = [method.signature for method in setup_methods]
        
        # Get application classes
        _, application_classes = CommonAnalysis(self.analysis).get_test_methods_classes_and_application_classes()
        
        # Create focal class method analyzer
        from hamster.code_analysis.focal_class_method.focal_class_method import FocalClassMethod
        focal_class_method = FocalClassMethod(self.analysis, testing_frameworks, application_classes)
        
        # Get focal classes and methods
        focal_classes, _, _, _ = focal_class_method.identify_focal_class_and_ui_api_test(
            qualified_class_name, 
            method_signature, 
            setup_method_signatures
        )
        
        # Extract focal methods
        focal_methods = set()
        for focal_class in focal_classes:
            for method_name in focal_class.focal_method_names:
                focal_methods.add((focal_class.focal_class, method_name))
        
        # Pretty print results
        pretty_print(f"Focal Analysis for {qualified_class_name}.{method_signature}", {
            "test_class": qualified_class_name,
            "test_method": method_signature,
            "total_focal_classes": len(focal_classes),
            "total_focal_methods": len(focal_methods),
            "focal_classes": [
                {
                    "focal_class": focal_class.focal_class,
                    "focal_method_names": focal_class.focal_method_names,
                    "num_focal_methods": len(focal_class.focal_method_names)
                }
                for focal_class in focal_classes
            ],
            "focal_methods": [f"{class_name}.{method_sig}" for class_name, method_sig in focal_methods]
        })
        
        # Assertions
        self.assertIsInstance(focal_classes, list, "Focal classes should be a list")
        self.assertIsInstance(focal_methods, set, "Focal methods should be a set")
        self.assertGreaterEqual(len(focal_classes), 0, "Should have at least 0 focal classes")
        self.assertGreaterEqual(len(focal_methods), 0, "Should have at least 0 focal methods")
        
        # Print detailed breakdown
        for i, focal_class in enumerate(focal_classes):
            pretty_print(f"Focal Class {i+1}: {focal_class.focal_class}", {
                "focal_class": focal_class.focal_class,
                "focal_method_names": focal_class.focal_method_names,
                "num_methods": len(focal_class.focal_method_names)
            })
        


