"""
Common Constants
"""
import sys

ASTER_REPORTS_DIR = "reports"
DEFAULT_ANALYSIS_DIR = "output"
MAVEN_CMD = "mvn.cmd" if sys.platform == "win32" else "mvn"

# directories for storing prompts and telemetry io
DEBUG_DIR = "nl2test_log"
AZURE_API_VERSION = "2024-12-01-preview"

# for recursive helper search depth limit
CONTEXT_SEARCH_DEPTH = 100

SETUP_ANNOTATIONS = {  # Also check `setUp()` method for JUnit 3
    "@Before", "@BeforeClass", "@BeforeEach", "@BeforeAll", "@BeforeMethod", "@BeforeTest", "@BeforeSuite",
    "@BeforeGroups"
}
