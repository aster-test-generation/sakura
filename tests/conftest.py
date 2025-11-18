from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis
from dotenv import load_dotenv

from nltest.test2nl import Pipeline as Test2NLPipeline
from nltest.test2nl.generation import DescriptionGenerator
from nltest.test2nl.prompts import RoundTripPrompt, Test2NLPrompt
from nltest.utils.config import Config, init_config
from nltest.utils.file_io.structured_data_manager import StructuredDataManager
from nltest.utils.llm.model import Provider


@dataclass(frozen=True)
class ProjectPaths:
    tests_dir: Path
    resources_dir: Path
    project_root: Path
    output_root: Path
    project_output_dir: Path
    project_name: str


@dataclass
class NL2TestContext:
    analysis: JavaAnalysis
    config: Config
    project_name: str
    project_root: Path
    resources_dir: Path
    output_dir: Path


@dataclass
class Test2NLContext:
    base: NL2TestContext
    pipeline: Test2NLPipeline
    desc_generator: DescriptionGenerator
    data_manager: StructuredDataManager
    test2nl_prompt: Test2NLPrompt
    roundtrip_prompt: RoundTripPrompt

    @property
    def analysis(self) -> JavaAnalysis:
        return self.base.analysis

    @property
    def config(self) -> Config:
        return self.base.config

    @property
    def project_name(self) -> str:
        return self.base.project_name

    @property
    def project_root(self) -> Path:
        return self.base.project_root

    @property
    def output_dir(self) -> Path:
        return self.base.output_dir


def _resolve_provider(env_var: str, default: Provider) -> Provider:
    raw = os.getenv(env_var)
    if not raw:
        return default
    try:
        return Provider(raw.lower())
    except ValueError:
        return default


def _env_bool(env_var: str, default: bool) -> bool:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@pytest.fixture(scope="session")
def project_paths() -> ProjectPaths:
    load_dotenv()
    tests_dir = Path(__file__).resolve().parent
    project_name = os.getenv("TEST_PROJECT_NAME", "spring-petclinic")

    resources_dir = tests_dir / "resources"
    project_root = resources_dir / project_name
    if not project_root.is_dir():
        raise RuntimeError(
            f"Project root directory {project_root} does not exist. "
            "Ensure test datasets are initialized."
        )

    output_root = tests_dir / "output"
    project_output_dir = output_root / project_name
    project_output_dir.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        tests_dir=tests_dir,
        resources_dir=resources_dir,
        project_root=project_root,
        output_root=output_root,
        project_output_dir=project_output_dir,
        project_name=project_name,
    )


@pytest.fixture(scope="session")
def analysis(project_paths: ProjectPaths) -> JavaAnalysis:
    return CLDK(language="java").analysis(
        project_path=project_paths.project_root,
        analysis_backend_path=None,
        analysis_level=AnalysisLevel.symbol_table,
        analysis_json_path=project_paths.project_output_dir,
        eager=False,
    )


def _init_test_config(project_paths: ProjectPaths) -> Config:
    llm_model = os.getenv("TEST_LLM_MODEL", "google/gemini-2.5-flash")
    emb_model = os.getenv("TEST_EMB_MODEL", "nomic-embed-text:v1.5")
    llm_provider = _resolve_provider("TEST_LLM_PROVIDER", Provider.OPENROUTER)
    emb_provider = _resolve_provider("TEST_EMB_PROVIDER", Provider.VLLM)
    llm_api_url = os.getenv("TEST_LLM_API_URL")
    emb_api_url = os.getenv("TEST_EMB_API_URL")
    llm_api_key = os.getenv("LLM_API_KEY")
    emb_api_key = os.getenv("EMB_API_KEY")

    return init_config(
        project_name=project_paths.project_name,
        base_project_dir=str(project_paths.resources_dir),
        project_output_dir=str(project_paths.project_output_dir),
        use_stored_index=False,
        llm_provider=llm_provider,
        llm_model=llm_model,
        emb_provider=emb_provider,
        emb_model=emb_model,
        llm_api_url=llm_api_url,
        emb_api_url=emb_api_url,
        llm_api_key=llm_api_key,
        emb_api_key=emb_api_key,
        can_parallel_tool=_env_bool("TEST_CAN_PARALLELIZE_TOOL", True),
    )


@pytest.fixture
def nl2test_context(project_paths: ProjectPaths, analysis: JavaAnalysis):
    Config.reset()
    config = _init_test_config(project_paths)
    ctx = NL2TestContext(
        analysis=analysis,
        config=config,
        project_name=project_paths.project_name,
        project_root=project_paths.project_root,
        resources_dir=project_paths.resources_dir,
        output_dir=project_paths.project_output_dir,
    )
    try:
        yield ctx
    finally:
        Config.reset()


@pytest.fixture
def test2nl_context(nl2test_context: NL2TestContext) -> Test2NLContext:
    pipeline = Test2NLPipeline(
        nl2test_context.analysis,
        nl2test_context.project_name,
        nl2test_context.output_dir,
        nl2test_context.project_root,
    )
    desc_generator = DescriptionGenerator(nl2test_context.analysis)
    data_manager = StructuredDataManager(nl2test_context.output_dir)
    test2nl_prompt = Test2NLPrompt(nl2test_context.analysis)
    roundtrip_prompt = RoundTripPrompt(nl2test_context.analysis)

    return Test2NLContext(
        base=nl2test_context,
        pipeline=pipeline,
        desc_generator=desc_generator,
        data_manager=data_manager,
        test2nl_prompt=test2nl_prompt,
        roundtrip_prompt=roundtrip_prompt,
    )
