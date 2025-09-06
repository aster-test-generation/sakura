"""
Common Constants
"""

import sys
from typing import Dict

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

# Experimentally found through OpenRouter
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
