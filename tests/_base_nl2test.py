import os
from pathlib import Path
from unittest import TestCase

from dotenv import load_dotenv

from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.utils.config import init_config

from nltest.utils.llm.model import Provider


class BaseNL2Test(TestCase):
    def setUp(self) -> None:
        load_dotenv()

        # === User-Defined ===
        project_name = "spring-petclinic"

        # llm_model = "deepseek/deepseek-chat-v3-0324"
        # llm_model = "openai/gpt-5-mini"
        # llm_model = "qwen/qwen3-coder"
        # llm_model = "moonshotai/kimi-k2"
        # llm_model = "mistralai/devstral-small"
        # llm_model = "mistralai/devstral-medium"
        # llm_model = "openai/gpt-4.1-mini"
        # llm_model = "z-ai/glm-4.5v" -> does not work
        # llm_model = "openai/gpt-4o-mini"
        # llm_model = "google/gemini-2.5-flash"
        # llm_model = "x-ai/grok-code-fast-1"
        llm_model = "openai/gpt-4.1-mini"
        emb_model = "nomic-embed-text:v1.5"

        # Make paths relative to the tests directory
        test_dir = Path(__file__).resolve().parent
        resources_dir = test_dir / "resources"
        output_base_dir = test_dir / "output"

        project_root = resources_dir / project_name
        if not (project_root.exists() and project_root.is_dir()):
            raise Exception(
                f"Project root directory {project_root.resolve()} does not exist."
            )
        output_dir = output_base_dir / project_name

        self.config = init_config(
            project_name=project_name,
            base_project_dir=str(resources_dir),
            output_dir=str(output_dir),
            llm_provider=Provider.OPENROUTER,
            llm_model=llm_model,
            emb_provider=Provider.OLLAMA,
            emb_model=emb_model,
            llm_api_key=os.getenv("LLM_API_KEY"),
            emb_api_key=None,
            localization_max_iters=5,
        )

        # Generate analysis of the current project
        self.analysis = CLDK(language="java").analysis(
            project_path=project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=output_dir,
            eager=False,
        )
