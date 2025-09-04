import os
from pathlib import Path
from typing import List
from unittest import TestCase

from dotenv import load_dotenv

from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.test2nl import Pipeline
from nltest.test2nl.generation import DescriptionGenerator
from nltest.test2nl.prompts import RoundTripPrompt, Test2NLPrompt
from nltest.utils.config import init_config
from nltest.utils.file_io.structured_data_manager import StructuredDataManager
from nltest.utils.llm.model import Provider


class BaseTest2NL(TestCase):
    def setUp(self) -> None:
        load_dotenv()

        # === User-Defined ===
        self.project_name = "spring-petclinic"

        llm_model = "qwen/qwen3-235b-a22b-thinking-2507"
        emb_model = "NOMIC-AI-EMB-7B"

        # Make paths relative to the tests directory
        test_dir = Path(__file__).resolve().parent
        project_root = (test_dir / "resources" / self.project_name).resolve()
        if not (project_root.exists() and project_root.is_dir()):
            raise Exception(f"Project root directory {project_root} does not exist.")
        output_dir = (test_dir / "output" / self.project_name).resolve()

        if not os.getenv("OPENROUTER_API_KEY"):
            raise AssertionError("OPENROUTER_API_KEY environment variable is not set.")

        self.config = init_config(
            project_name=self.project_name,
            base_project_dir=str(project_root),
            output_dir=str(output_dir),
            llm_provider=Provider.OPENROUTER,
            llm_model=llm_model,
            emb_provider=Provider.VLLM,
            emb_model=emb_model,
            llm_api_key=os.getenv("OPENROUTER_API_KEY"),
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

        # Create orchestration and helpers
        self.pipeline = Pipeline(self.analysis, project_root, output_dir)
        self.desc_generator = DescriptionGenerator(self.analysis)
        self.data_manager = StructuredDataManager(output_dir)
        self.test2nl_prompt = Test2NLPrompt(self.analysis)
        self.roundtrip_prompt = RoundTripPrompt(self.analysis)

