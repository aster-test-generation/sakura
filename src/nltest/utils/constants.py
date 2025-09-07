"""
Common Constants
"""

import sys
from typing import Dict
from enum import Enum

MAVEN_CMD = "mvn.cmd" if sys.platform == "win32" else "mvn"
JACOCO_VERSION = "0.8.13"
MAVEN_COV_DIR = "target/coverage"
MAVEN_JACOCO_COV_FILE = f"{MAVEN_COV_DIR}/jacoco.exec"


class MutationOperators(str, Enum):
    defaults = "DEFAULTS"
    stronger = "STRONGER"
    all = "ALL"


RESOURCE_DIR = "resources/datasets"
ASTER_REPORTS_DIR = "reports"
DEFAULT_ANALYSIS_DIR = "resources/output"
HAMSTER_MODEL_DIR = "resources/hamster_models"
NL2TEST_DIR = "resources/nl2test"
MAVEN_CMD = "mvn.cmd" if sys.platform == "win32" else "mvn"

# directories for storing prompts and telemetry io
DEBUG_DIR = "nl2test_log"
AZURE_API_VERSION = "2024-12-01-preview"

# for recursive helper search depth limit
CONTEXT_SEARCH_DEPTH = 100

SETUP_ANNOTATIONS = {  # Also check `setUp()` method for JUnit 3
    "@Before",
    "@BeforeClass",
    "@BeforeEach",
    "@BeforeAll",
    "@BeforeMethod",
    "@BeforeTest",
    "@BeforeSuite",
    "@BeforeGroups",
}

TEARDOWN_ANNOTATIONS = {  # Also check `tearDown()` method for JUnit 3
    "@After",
    "@AfterClass",
    "@AfterEach",
    "@AfterAll",
    "@AfterMethod",
    "@AfterTest",
    "@AfterSuite",
    "@AfterGroups",
}

# Somewhat naive mapping - Proprietary is True and open-source is False
PARALLEL_TOOL_CALLABLE: Dict[bool, set[str]] = {
    False: {
        # Mistral models
        "mistralai/devstral-medium",
        "mistralai/devstral-small",
        # Deepseek models
        "deepseek/deepseek-chat-v3.1",
    },
    True: {
        # OpenAI models,
        "openai/gpt-5-mini",
        "openai/gpt-4o-mini",
        "openai/gpt-4.1-mini",
        # XAI models,
        "x-ai/grok-code-fast-1",
        # Google models
        "google/gemini-2.5-flash",
    },
}
