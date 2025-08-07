from typing import List
from unittest import TestCase
from pathlib import Path

from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.test2nl.model.models import AbstractionLevel, GeneratedDescription, Test2NLEntry
from nltest.test2nl import Pipeline
from nltest.test2nl.prompt import RoundTripPrompt, Test2NLPrompt
from nltest.test2nl.generation import DescriptionGenerator
from nltest.utils import Config
from nltest.utils.llm_configs import LLM_CONFIG, EMB_CONFIG
from nltest.utils.pretty.prints import pretty_print
from nltest.utils.file_io.structured_data_manager import StructuredDataManager


class TestTest2NL(TestCase):
    def setUp(self) -> None:

        # === User-Defined ===
        self.project_name = "spring-petclinic"

        # llm_model = "LLAMA-3.1-8B"
        # llm_model = "DEVSTRAL-24B"
        llm_model = "DEEPSEEK-R1"

        # emb_model = "NOMIC-AI-EMB-7B"
        emb_model = "QWEN-3-EMB-8B"  # Embedding model

        self.emb_model = "dengcao/Qwen3-Embedding-4B:Q5_K_M"

        summarization_temp = 0.4
        code_gen_temp = 0.5
        decision_temp = 0.3
        structured_temp = 0.3

        project_root = Path(f"./resources/{self.project_name}")
        if not (project_root.exists() and project_root.is_dir()):
            raise Exception(f"Project root directory {project_root} does not exist.")

        # Config default assignments - TODO: Regularize this
        config = Config(None, reuse=False)
        config.set("llm_provider", "name", "VELA")
        config.set("emb_provider", "name", "VELA")

        output_dir = Path(f"./output/{self.project_name}")

        # Assign VELA LLM model configs
        llm_provider = config.get("llm_provider", "name")
        config.set(llm_provider, "llm_model", val=LLM_CONFIG[llm_model]["identifier"])
        config.set(llm_provider, "llm_api_url", val=LLM_CONFIG[llm_model]["api_url"])
        config.set(llm_provider, "summarization_temp", summarization_temp)
        config.set(llm_provider, "code_gen_temp", code_gen_temp)
        config.set(llm_provider, "decision_temp", decision_temp)
        config.set(llm_provider, "structured_temp", structured_temp)
        config.set(llm_provider, "output_tokens",
                   val=LLM_CONFIG[llm_model]["output_tokens"] if "output_tokens" in LLM_CONFIG[llm_model] else 2048)

        # Assign VELA VectorStore model configs
        emb_provider = config.get("emb_provider", "name")
        config.set(emb_provider, "emb_model", val=EMB_CONFIG[emb_model]["identifier"])
        config.set(emb_provider, "emb_api_url", val=EMB_CONFIG[emb_model]["api_url"])

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
        generated_desc: GeneratedDescription = self.desc_generator.generate_for_method(
            method_signature,
            qualified_class_name,
            abstraction_level
        )
        self.assertIsNotNone(generated_desc, "LLM generation failed...")

        generated_descriptions = [generated_desc]
        save_success = self.data_manager.save("descriptions.json", generated_descriptions)
        self.assertTrue(save_success, "Description could not be saved...")

    def test_desc_all_abs_one_method(self):
        qualified_class_name = "org.springframework.samples.petclinic.service.ClinicServiceTests"
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        generated_descriptions = []
        for abs_level in AbstractionLevel:
            generated_description = self.desc_generator.generate_for_method(
                method_signature,
                qualified_class_name,
                abs_level
            )
            if generated_description:
                generated_descriptions.append(generated_description)
        self.assertTrue(generated_descriptions, "Descriptions could not be generated...")

        test2nl_entries: List[Test2NLEntry] = []
        entry_id = 1
        for gen_desc in generated_descriptions:
            gen_desc.entry_id = entry_id
            entry = Test2NLEntry(
                id=entry_id,
                description=gen_desc.description,
                project_name=self.project_name,
                qualified_class_name=gen_desc.qualified_class_name,
                method_signature=gen_desc.method_signature,
                abstraction_level=gen_desc.abstraction_level,
                is_bdd=False
            )
            test2nl_entries.append(entry)
            entry_id += 1

        save_success = self.data_manager.save("descriptions_trials.json", generated_descriptions, format="json")
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

    def test_roundtrip_prompt(self):
        qualified_class_name = "org.springframework.samples.petclinic.service.ClinicServiceTests"
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        gen_descriptions = self.data_manager.load("descriptions.json", GeneratedDescription)
        select_desc = gen_descriptions[0]

        test_case, prompt, is_successful = self.roundtrip_prompt.generate(method_signature, qualified_class_name,
                                                                          select_desc.description)
        self.assertTrue(prompt, "Prompt was unsuccessfully rendered...")

        pretty_print("prompts", prompt)

        self.assertTrue(is_successful, "LLM generated was unsuccessful with prompts...")

