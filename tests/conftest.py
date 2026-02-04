from __future__ import annotations

import os
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis
from dotenv import load_dotenv

from sakura.test2nl import Pipeline as Test2NLPipeline
from sakura.test2nl.generation import DescriptionGenerator
from sakura.test2nl.prompts import Test2NLPrompt
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.config import Config, init_config
from sakura.utils.file_io.structured_data_manager import StructuredDataManager
from sakura.utils.llm.model import Provider


@dataclass(frozen=True)
class ProjectPaths:
    tests_dir: Path
    resources_dir: Path
    project_root: Path
    output_root: Path
    project_output_dir: Path
    project_name: str


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


def get_project_paths(project_name: str) -> ProjectPaths:
    """Helper to construct paths for a specific project."""
    load_dotenv()
    tests_dir = Path(__file__).resolve().parent
    repo_root = tests_dir.parent
    resources_dir = tests_dir / "resources"

    candidate_path = resources_dir / project_name
    if not candidate_path.exists():
        candidate_path = repo_root / project_name

    project_root = candidate_path

    if not project_root.is_dir():
        raise RuntimeError(
            f"Project root directory {project_root} does not exist. "
            "Ensure test datasets are initialized in tests/resources/ or repo root."
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


def get_analysis(project_paths: ProjectPaths) -> JavaAnalysis:
    """Helper to create analysis for a project."""
    return CLDK(language="java").analysis(
        project_path=project_paths.project_root,
        analysis_backend_path=None,
        analysis_level=AnalysisLevel.symbol_table,
        analysis_json_path=project_paths.project_output_dir,
        eager=False,
    )


def init_test_config(project_paths: ProjectPaths) -> Config:
    """Helper to initialize config for a project."""
    llm_model = os.getenv("TEST_LLM_MODEL", "google/gemini-2.5-flash")
    emb_model = os.getenv("TEST_EMB_MODEL", "nomic-embed-text:v1.5")
    llm_provider = _resolve_provider("TEST_LLM_PROVIDER", Provider.OPENROUTER)
    emb_provider = _resolve_provider("TEST_EMB_PROVIDER", Provider.VLLM)
    llm_api_url = os.getenv("TEST_LLM_API_URL")
    emb_api_url = os.getenv("TEST_EMB_API_URL")
    llm_api_key = os.getenv("LLM_API_KEY")
    emb_api_key = os.getenv("EMB_API_KEY")

    Config.reset()
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


# --- Spring PetClinic Fixtures ---


@pytest.fixture(scope="session")
def petclinic_paths() -> ProjectPaths:
    return get_project_paths("spring-petclinic")


@pytest.fixture(scope="session")
def petclinic_analysis(petclinic_paths: ProjectPaths) -> JavaAnalysis:
    return get_analysis(petclinic_paths)


@pytest.fixture
def petclinic_config(petclinic_paths: ProjectPaths) -> Generator[Config, None, None]:
    """
    Function-scoped fixture to ensure config is clean for each test.
    """
    config = init_test_config(petclinic_paths)
    try:
        yield config
    finally:
        Config.reset()


@pytest.fixture(scope="session")
def petclinic_categorized_classes(
    petclinic_analysis: JavaAnalysis,
) -> tuple[list[str], list[str]]:
    _, application_classes, test_utility_classes = CommonAnalysis(
        petclinic_analysis
    ).categorize_classes()
    return application_classes, test_utility_classes


@pytest.fixture(scope="session")
def petclinic_test2nl_prompt(
    petclinic_analysis: JavaAnalysis,
    petclinic_session_config: Config,
    petclinic_categorized_classes: tuple[list[str], list[str]],
) -> Test2NLPrompt:
    application_classes, test_utility_classes = petclinic_categorized_classes
    return Test2NLPrompt(petclinic_analysis, application_classes, test_utility_classes)


@pytest.fixture(scope="session")
def petclinic_test2nl_pipeline(
    petclinic_analysis: JavaAnalysis,
    petclinic_paths: ProjectPaths,
    petclinic_session_config: Config,
) -> Test2NLPipeline:
    return Test2NLPipeline(
        petclinic_analysis,
        petclinic_paths.project_name,
        petclinic_paths.project_output_dir,
        petclinic_paths.project_root,
    )


@pytest.fixture(scope="session")
def petclinic_session_config(
    petclinic_paths: ProjectPaths,
) -> Generator[Config, None, None]:
    """Session-scoped config for fixtures that need Config at session level."""
    config = init_test_config(petclinic_paths)
    try:
        yield config
    finally:
        Config.reset()


@pytest.fixture(scope="session")
def petclinic_desc_generator(
    petclinic_analysis: JavaAnalysis,
    petclinic_session_config: Config,
    petclinic_categorized_classes: tuple[list[str], list[str]],
) -> DescriptionGenerator:
    application_classes, test_utility_classes = petclinic_categorized_classes
    return DescriptionGenerator(
        petclinic_analysis, application_classes, test_utility_classes
    )


@pytest.fixture(scope="session")
def petclinic_data_manager(petclinic_paths: ProjectPaths) -> StructuredDataManager:
    return StructuredDataManager(petclinic_paths.project_output_dir)
