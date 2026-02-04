'\nCommon Constants\n'
_A=False
import sys
from enum import Enum
from pathlib import Path
from typing import Dict
MAVEN_CMD='mvn.cmd'if sys.platform=='win32'else'mvn'
JACOCO_VERSION='0.8.13'
MAVEN_COV_DIR='target/coverage'
MAVEN_JACOCO_COV_FILE=f"{MAVEN_COV_DIR}/jacoco.exec"
class MutationOperators(str,Enum):defaults='DEFAULTS';stronger='STRONGER';all='ALL'
RESOURCE_DIR='resources/datasets'
ASTER_REPORTS_DIR='reports'
DEFAULT_ANALYSIS_DIR='resources/output'
HAMSTER_MODEL_DIR='resources/hamster_models'
BUCKETED_TESTS_DIR='resources/bucketed_tests'
TEST_DIR='src/test/java'
def is_test_source_path(java_file:str|Path)->bool:
	A=java_file
	if not A:return _A
	D=str(A).replace('\\','/');B=[A for A in D.split('/')if A]
	for C in range(len(B)-2):
		if B[C:C+3]==['src','test','java']:return True
	return _A
DEBUG_DIR='nl2test_log'
AZURE_API_VERSION='2024-12-01-preview'
CONTEXT_SEARCH_DEPTH=100
ABSTRACTION_TEMPERATURES:dict[str,float]={'high':.7,'medium':.5,'low':.3}
SETUP_ANNOTATIONS={'@Before','@BeforeClass','@BeforeEach','@BeforeAll','@BeforeMethod','@BeforeTest','@BeforeSuite','@BeforeGroups'}
TEARDOWN_ANNOTATIONS={'@After','@AfterClass','@AfterEach','@AfterAll','@AfterMethod','@AfterTest','@AfterSuite','@AfterGroups'}
PARALLEL_TOOL_CALLABLE:Dict[bool,set[str]]={_A:{'mistralai/devstral-medium','mistralai/devstral-small','deepseek/deepseek-chat-v3.1'},True:{'openai/gpt-5-mini','openai/gpt-4o-mini','openai/gpt-4.1-mini','Azure/gpt-5-2025-08-07','Azure/gpt-4.1','x-ai/grok-code-fast-1','google/gemini-2.5-flash','GCP/gemini-2.5-flash','GCP/gemini-2.5-flash-lite','GCP/claude-4-sonnet'}}