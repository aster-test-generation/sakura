from unittest.mock import MagicMock

from langchain_core.tools import StructuredTool
from cldk import CLDK
from cldk.analysis import AnalysisLevel

from sakura.nl2test.generation.composition.orchestrators import (
    GherkinCompositionOrchestrator,
)
from sakura.nl2test.generation.composition.tools.base import BaseCompositionTools
from sakura.nl2test.generation.composition.tools.gherkin import GherkinCompositionTools
from sakura.nl2test.generation.composition.tools.grammatical import (
    GrammaticalCompositionTools,
)
from sakura.nl2test.models import (
    NL2TestInput,
    LocalizedScenario,
    AbstractionLevel,
    AgentState,
)
from sakura.nl2test.preprocessing.indexers import MethodIndexer, ClassIndexer
from sakura.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.compilation.maven import JavaMavenCompilation
from sakura.utils.evaluation.test_grader import TestGrader
from sakura.utils.file_io.test_file_manager import TestFileManager, TestFileInfo
from sakura.utils.models import NL2TestMetadata
from sakura.utils.pretty.prints import pretty_print


class TestCompositionAgent:
    def test_composition_system_prompt_formatting(self):
        """Verify system prompt formatting for composition Gherkin with placeholders."""
        prompt = LoadPrompt.load_prompt(
            "composition_agent_gherkin.jinja2", PromptFormat.JINJA2, "system"
        )

        # parallelizable True path
        iters_true = 4
        rendered_true = prompt.format(parallelizable=True, max_iters=iters_true)
        pretty_print("Parallelizable prompt", rendered_true)
        expected_true = (
            f"You must complete within at most {iters_true} model step(s) (iterations)."
        )
        assert expected_true in rendered_true
        assert (
            "You may parallelize tool calls that do not depend on each other"
            in rendered_true
        )
        assert "Do not parallelize tool calls" not in rendered_true

        # parallelizable False path
        iters_false = 6
        rendered_false = prompt.format(parallelizable=False, max_iters=iters_false)
        pretty_print("Not parallelizable prompt", rendered_false)
        expected_false = f"You must complete within at most {iters_false} model step(s) (iterations)."
        assert expected_false in rendered_false
        assert "Do not parallelize tool calls" in rendered_false
        assert (
            "You may parallelize tool calls that do not depend on each other"
            not in rendered_false
        )

        # duplicate_tools path — mirror orchestrator formatting
        # Create simple tools using LangChain's StructuredTool (a BaseTool subclass)
        dup_tools = [
            StructuredTool.from_function(
                func=lambda: None, name="view_test_code", description=""
            ),
            StructuredTool.from_function(
                func=lambda: None,
                name="compile_and_execute_tests",
                description="",
            ),
        ]
        duplicate_tools_str = ", ".join(f"`{t.name}`" for t in dup_tools)

        rendered_with_dups = prompt.format(
            parallelizable=True,
            max_iters=iters_true,
            duplicate_tools=duplicate_tools_str,
        )
        pretty_print("With duplicate_tools", rendered_with_dups)

        assert (
            f"Duplicate tool calls are allowed only for the following tool names: {duplicate_tools_str}"
            in rendered_with_dups
        )
        assert "Never repeat an identical {tool, args} pair" not in rendered_with_dups
        assert (
            "Duplicate tool calls are allowed only for the following tool names"
            not in rendered_true
        )

    def test_composition_chat_prompt_formatting(self):
        """Verify chat prompt formatting for composition Gherkin with placeholders."""
        prompt = LoadPrompt.load_prompt(
            "composition_agent_gherkin.jinja2", PromptFormat.JINJA2, "chat"
        )

        nl_description = "Compose a unit test for PetController update"
        instructions = "Use JUnit 5 and avoid Mockito unless necessary"
        localized_scenario = "{\n  'given': [], 'when': [], 'then': []\n}"

        rendered = prompt.format(
            nl_description=nl_description,
            instructions=instructions,
            localized_scenario=localized_scenario,
        )

        assert nl_description in rendered
        assert instructions in rendered
        assert "CURRENT LOCALIZED SCENARIO" in rendered
        assert "Compose a compilable and runnable Java test" in rendered

    def test_composition_agent_gherkin(
        self, petclinic_analysis, petclinic_config, petclinic_paths
    ):
        analysis = petclinic_analysis
        config = petclinic_config
        method_searcher = MethodIndexer(analysis).build_index()
        class_searcher = ClassIndexer(analysis).build_index()

        project_name = petclinic_paths.project_name
        project_root = petclinic_paths.project_root

        # Tighten iteration limits for this test
        config.set("composition", "max_iters", 6)

        nl2_input = NL2TestInput(
            id=-1,
            description=(
                "Create a test case that validates the successful processing of a pet update form. "
                "The test begins by setting up the test environment using the `@WebMvcTest` annotation to load the "
                "`PetController` and `PetTypeFormatter` within a Spring MVC test context, disabling the test in native image "
                "and AOT modes via `@DisabledInNativeImage` and `@DisabledInAotMode` respectively. The `OwnerRepository` "
                "dependency of the `PetController` is mocked using `@MockitoBean`, and the `MockMvc` is autowired for "
                "simulating HTTP requests. The setup method stubs the `findPetTypes` method of the mocked `OwnerRepository` "
                "to return a list containing a `PetType` object with a predefined ID and name, and also stubs the `findById` "
                "method to return an `Optional` containing an `Owner` object populated with two `Pet` objects, each having a "
                'unique ID and name. The test then performs a `POST` request to the "/owners/{ownerId}/pets/{petId}/edit" '
                "endpoint, substituting predefined `TEST_OWNER_ID` and `TEST_PET_ID` values into the URL, and including "
                "parameters for the pet's name, type, and birth date. The test uses `MockMvc` to simulate the request and "
                "verifies that the response has a 3xx redirection status code and that the view name is a redirection to the "
                "owner's details page using `andExpect` with `status().is3xxRedirection()` and `view().name()`. JUnit and "
                "Mockito are used for structuring the test and mocking dependencies, while Spring's `MockMvc` and its "
                "associated matchers are used for simulating and asserting on the web layer."
            ),
            project_name=project_name,
            qualified_class_name="org.springframework.samples.petclinic.owner.PetControllerTests",
            method_signature="testProcessUpdateFormSuccess()",
            abstraction_level=AbstractionLevel.LOW,
            is_bdd=False,
        )

        composition_agent = GherkinCompositionOrchestrator(
            analysis=analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            project_root=str(project_root),
        )

        localized_scenario_data = {
            "setup": [
                {
                    "id": 0,
                    "task": "Load Spring MVC test context for PetController and PetTypeFormatter using @WebMvcTest",
                    "uses": "",
                    "produces": "mock_mvc_context",
                    "candidate_methods": [],
                    "arg_bindings": [],
                    "comments": "@WebMvcTest is an annotation, not a method. It's used at the class level to configure the Spring application context for testing a Spring MVC controller.",
                    "external": False,
                },
                {
                    "id": 1,
                    "task": "Disable test in native image and AOT modes",
                    "uses": "mock_mvc_context",
                    "produces": "",
                    "candidate_methods": [],
                    "arg_bindings": [],
                    "comments": "@DisabledInNativeImage and @DisabledInAotMode are annotations used at the class level to disable tests in specific Spring Boot modes. They are not methods.",
                    "external": False,
                },
                {
                    "id": 2,
                    "task": "Mock OwnerRepository dependency using @MockitoBean",
                    "uses": "mock_mvc_context",
                    "produces": "mock_owner_repository",
                    "candidate_methods": [],
                    "arg_bindings": [],
                    "comments": "@MockitoBean is an annotation used to add Mockito mocks to the Spring application context. It's not a method call but a declaration.",
                    "external": False,
                },
                {
                    "id": 3,
                    "task": "Autowire MockMvc for simulating HTTP requests",
                    "uses": "mock_mvc_context",
                    "produces": "mock_mvc",
                    "candidate_methods": [],
                    "arg_bindings": [],
                    "comments": "@Autowired is an annotation used for dependency injection. MockMvc is typically injected into the test class, not called as a method.",
                    "external": False,
                },
                {
                    "id": 4,
                    "task": "Stub findPetTypes method of mocked OwnerRepository to return a list with a PetType",
                    "uses": "mock_owner_repository",
                    "produces": "pet_types",
                    "candidate_methods": [
                        {
                            "declaring_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                            "containing_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                            "method_signature": "findPetTypes()",
                            "return_type": "java.util.List<org.springframework.samples.petclinic.owner.PetType>",
                        }
                    ],
                    "arg_bindings": [],
                    "comments": "This step involves Mockito's `when().thenReturn()` syntax to stub the `findPetTypes` method.",
                    "external": False,
                },
                {
                    "id": 5,
                    "task": "Stub findById method of mocked OwnerRepository to return an Optional containing an Owner with two Pets",
                    "uses": "mock_owner_repository",
                    "produces": "owner_with_pets",
                    "candidate_methods": [
                        {
                            "declaring_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                            "containing_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                            "method_signature": "findById(java.lang.Integer)",
                            "return_type": "java.util.Optional<org.springframework.samples.petclinic.owner.Owner>",
                        }
                    ],
                    "arg_bindings": [{"arg_name": "id", "arg_value": "TEST_OWNER_ID"}],
                    "comments": "This step involves Mockito's `when().thenReturn()` syntax to stub the `findById` method.",
                    "external": False,
                },
            ],
            "gherkin_groups": [
                {
                    "given": [],
                    "when": [
                        {
                            "id": 6,
                            "task": "Perform POST request to /owners/{ownerId}/pets/{petId}/edit with pet details",
                            "uses": "mock_mvc, TEST_OWNER_ID, TEST_PET_ID, pet_name, pet_type, pet_birth_date",
                            "produces": "http_response",
                            "candidate_methods": [
                                {
                                    "declaring_class_name": "org.springframework.samples.petclinic.owner.PetController",
                                    "containing_class_name": "org.springframework.samples.petclinic.owner.PetController",
                                    "method_signature": "processUpdateForm(org.springframework.samples.petclinic.owner.Owner, org.springframework.samples.petclinic.owner.Pet, org.springframework.validation.BindingResult, org.springframework.web.servlet.mvc.support.RedirectAttributes)",
                                    "return_type": "java.lang.String",
                                }
                            ],
                            "arg_bindings": [
                                {"arg_name": "owner", "arg_value": "owner_with_pets"},
                                {"arg_name": "pet", "arg_value": "pet_details"},
                                {
                                    "arg_name": "result",
                                    "arg_value": "new BindingResult()",
                                },
                                {
                                    "arg_name": "redirectAttributes",
                                    "arg_value": "new RedirectAttributes()",
                                },
                            ],
                            "comments": "This step will use MockMvc.perform(post(...)) to simulate the HTTP POST request. The processUpdateForm method in PetController is the target method for this request.",
                            "external": False,
                        }
                    ],
                    "then": [
                        {
                            "id": 7,
                            "task": "Verify response has 3xx redirection status code",
                            "uses": "http_response",
                            "produces": "",
                            "candidate_methods": [],
                            "arg_bindings": [],
                            "comments": "This step will use MockMvcResultMatchers.status().is3xxRedirection(). This is a static method from Spring Test.",
                            "external": False,
                        },
                        {
                            "id": 8,
                            "task": "Verify view name is a redirection to the owner's details page",
                            "uses": "http_response",
                            "produces": "",
                            "candidate_methods": [],
                            "arg_bindings": [],
                            "comments": "This step will use MockMvcResultMatchers.view().name(). This is a static method from Spring Test.",
                            "external": False,
                        },
                    ],
                }
            ],
            "teardown": [],
        }

        localized_scenario = LocalizedScenario(**localized_scenario_data)

        instructions = "Compose the Java test for this scenario and finalize."
        agent_state = composition_agent.assign_task(
            localized_scenario, instructions=instructions
        )

        assert isinstance(agent_state, AgentState)

        # Extract results from AgentState
        updated_scenario = agent_state.localized_scenario
        final_comments = agent_state.final_comments
        package = agent_state.package
        class_name = agent_state.class_name

        assert updated_scenario is None or isinstance(
            updated_scenario, LocalizedScenario
        )
        assert final_comments is None or isinstance(final_comments, str)

        # Pretty print selected package and class name before assertions
        pretty_print("Selected package", package)
        pretty_print("Selected test class", class_name)

        # Ensure package and class name were selected
        assert package is not None
        assert class_name is not None

        pretty_print("Updated scenario", updated_scenario)
        pretty_print("Final comments", final_comments)

        # Build qualified test class name
        qualified_test_class_name = f"{package}.{class_name}" if package else class_name
        method_signature = agent_state.method_signature or ""

        # Regenerate analysis to pick up the new test class
        new_analysis = CLDK(language="java").analysis(
            project_path=project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=petclinic_paths.project_output_dir,
            eager=True,
        )

        # Get application classes for grading
        cmn = CommonAnalysis(new_analysis)
        _, application_classes, test_utility_classes = cmn.categorize_classes()

        # Gather compilation errors
        compilation_errors = JavaMavenCompilation(project_root).get_compilation_errors()
        erroneous_files = [err.file for err in compilation_errors]

        # Create grader and grade the test
        test_grader = TestGrader(
            analysis=new_analysis,
            project_root=project_root,
            project_erroneous_files=erroneous_files,
            application_classes=application_classes,
            test_utility_classes=test_utility_classes,
        )

        nl2_metadata = NL2TestMetadata(
            qualified_test_class_name=qualified_test_class_name,
            code="",
            method_signature=method_signature or None,
        )

        eval_result = test_grader.grade(nl2_input, nl2_metadata)

        # Load and attach test code
        fm = TestFileManager(project_root)
        info = TestFileInfo(qualified_class_name=qualified_test_class_name)
        try:
            code = fm.load(info, encode_class_name=False)
            eval_result.nl2test_metadata.code = code
        except FileNotFoundError:
            pass

        # Print evaluation results
        pretty_print("Evaluation result", eval_result)
        pretty_print("Compiles", eval_result.compiles)
        pretty_print("Structural eval", eval_result.structured_eval)
        pretty_print("Coverage eval", eval_result.coverage_eval)

        # Assert that the generated test compiles
        assert eval_result.compiles, "Generated test should compile"

        # Clean up generated test file
        try:
            fm.delete_single(info, encode_class_name=False)
        except Exception:
            pass


class TestCompositionToolInjection:
    """Tests verifying tool injection after refactoring to shared utilities."""

    @staticmethod
    def _create_mock_dependencies():
        """Create mock dependencies for tool builders."""
        mock_analysis = MagicMock()
        mock_analysis.get_class = MagicMock(return_value=None)
        mock_analysis.get_method = MagicMock(return_value=None)
        mock_method_searcher = MagicMock()
        mock_class_searcher = MagicMock()
        mock_llm = MagicMock()
        mock_llm.parse_tool_args = lambda x: x

        nl2_input = NL2TestInput(
            id=-1,
            description="Test description",
            project_name="test-project",
            qualified_class_name="org.example.TestClass",
            method_signature="testMethod()",
        )

        return {
            "analysis": mock_analysis,
            "method_searcher": mock_method_searcher,
            "class_searcher": mock_class_searcher,
            "structured_llm": mock_llm,
            "project_root": "/tmp/test",
            "nl2_input": nl2_input,
        }

    def test_base_composition_tools_contains_expected_tools(self):
        """Verify BaseCompositionTools creates expected base tools."""
        deps = self._create_mock_dependencies()
        tool_builder = BaseCompositionTools(**deps)
        tools, allow_duplicates = tool_builder.all()

        tool_names = {t.name for t in tools}
        expected_base_tools = {
            "query_class_db",
            "extract_method_code",
            "get_method_details",
            "get_class_fields",
            "get_class_imports",
            "get_class_constructors_and_factories",
            "get_getters_and_setters",
            "get_maven_dependencies",
            "view_test_code",
            "generate_test_code",
            "compile_and_execute_test",
            "finalize",
            "get_call_site_details",
        }

        assert expected_base_tools == tool_names, (
            f"Tool mismatch. Expected: {expected_base_tools}, Got: {tool_names}"
        )

    def test_gherkin_composition_tools_contains_all_tools(self):
        """Verify GherkinCompositionTools includes base + Gherkin-specific tools."""
        deps = self._create_mock_dependencies()
        tool_builder = GherkinCompositionTools(**deps)
        tools, allow_duplicates = tool_builder.all()

        tool_names = {t.name for t in tools}
        expected_tools = {
            "query_class_db",
            "extract_method_code",
            "get_method_details",
            "get_class_fields",
            "get_class_imports",
            "get_class_constructors_and_factories",
            "get_getters_and_setters",
            "get_maven_dependencies",
            "view_test_code",
            "generate_test_code",
            "compile_and_execute_test",
            "finalize",
            "get_call_site_details",
            "modify_scenario_comment",
        }

        assert expected_tools == tool_names, (
            f"Tool mismatch. Expected: {expected_tools}, Got: {tool_names}"
        )

    def test_grammatical_composition_tools_contains_all_tools(self):
        """Verify GrammaticalCompositionTools includes base + Grammatical-specific tools."""
        deps = self._create_mock_dependencies()
        tool_builder = GrammaticalCompositionTools(**deps)
        tools, allow_duplicates = tool_builder.all()

        tool_names = {t.name for t in tools}
        expected_tools = {
            "query_class_db",
            "extract_method_code",
            "get_method_details",
            "get_class_fields",
            "get_class_imports",
            "get_class_constructors_and_factories",
            "get_getters_and_setters",
            "get_maven_dependencies",
            "view_test_code",
            "generate_test_code",
            "compile_and_execute_test",
            "finalize",
            "get_call_site_details",
            "modify_scenario_comment",
        }

        assert expected_tools == tool_names, (
            f"Tool mismatch. Expected: {expected_tools}, Got: {tool_names}"
        )

    def test_composition_allow_duplicate_tools_correct(self):
        """Verify allow_duplicate_tools list contains expected tools."""
        deps = self._create_mock_dependencies()
        tool_builder = GherkinCompositionTools(**deps)
        _, allow_duplicates = tool_builder.all()

        allow_dup_names = {t.name for t in allow_duplicates}
        expected_duplicates = {"view_test_code", "compile_and_execute_test"}

        assert expected_duplicates == allow_dup_names, (
            f"Allow duplicates mismatch. Expected: {expected_duplicates}, Got: {allow_dup_names}"
        )

    def test_deferred_tool_generate_test_code_returns_inputs(self):
        """Verify generate_test_code returns expected input keys."""
        deps = self._create_mock_dependencies()
        tool_builder = BaseCompositionTools(**deps)
        tools, _ = tool_builder.all()

        gen_tool = next(t for t in tools if t.name == "generate_test_code")
        result = gen_tool.func(
            test_code="public void test() {}",
            qualified_class_name="org.example.TestClass",
            method_signature="testMethod()",
        )

        assert result["test_code"] == "public void test() {}"
        assert result["qualified_class_name"] == "org.example.TestClass"
        assert result["method_signature"] == "testMethod()"

    def test_deferred_tool_finalize_returns_comments(self):
        """Verify finalize returns comments as expected."""
        deps = self._create_mock_dependencies()
        tool_builder = BaseCompositionTools(**deps)
        tools, _ = tool_builder.all()

        finalize_tool = next(t for t in tools if t.name == "finalize")
        result = finalize_tool.func(comments="Test completed successfully")

        assert result == {"comments": "Test completed successfully"}

    def test_deferred_tool_modify_scenario_comment_gherkin(self):
        """Verify Gherkin modify_scenario_comment returns expected keys."""
        deps = self._create_mock_dependencies()
        tool_builder = GherkinCompositionTools(**deps)
        tools, _ = tool_builder.all()

        modify_tool = next(t for t in tools if t.name == "modify_scenario_comment")
        result = modify_tool.func(id=0, comment="Updated comment")

        assert result == {"id": 0, "comment": "Updated comment"}

    def test_deferred_tool_modify_scenario_comment_grammatical(self):
        """Verify Grammatical modify_scenario_comment returns expected keys."""
        deps = self._create_mock_dependencies()
        tool_builder = GrammaticalCompositionTools(**deps)
        tools, _ = tool_builder.all()

        modify_tool = next(t for t in tools if t.name == "modify_scenario_comment")
        result = modify_tool.func(order=1, note="Updated note")

        assert result == {"order": 1, "note": "Updated note"}

    def test_view_test_code_tool_shared_across_modes(self):
        """Verify view_test_code returns same structure across modes."""
        deps = self._create_mock_dependencies()

        gherkin_builder = GherkinCompositionTools(**deps)
        grammatical_builder = GrammaticalCompositionTools(**deps)

        gherkin_tools, _ = gherkin_builder.all()
        grammatical_tools, _ = grammatical_builder.all()

        gherkin_view = next(t for t in gherkin_tools if t.name == "view_test_code")
        grammatical_view = next(
            t for t in grammatical_tools if t.name == "view_test_code"
        )

        gherkin_result = gherkin_view.func(
            qualified_class_name="org.example.Test",
            method_signature="test()",
        )
        grammatical_result = grammatical_view.func(
            qualified_class_name="org.example.Test",
            method_signature="test()",
        )

        assert gherkin_result == grammatical_result

    def test_compile_and_execute_tool_shared_across_modes(self):
        """Verify compile_and_execute_test returns same structure across modes."""
        deps = self._create_mock_dependencies()

        gherkin_builder = GherkinCompositionTools(**deps)
        grammatical_builder = GrammaticalCompositionTools(**deps)

        gherkin_tools, _ = gherkin_builder.all()
        grammatical_tools, _ = grammatical_builder.all()

        gherkin_compile = next(
            t for t in gherkin_tools if t.name == "compile_and_execute_test"
        )
        grammatical_compile = next(
            t for t in grammatical_tools if t.name == "compile_and_execute_test"
        )

        assert gherkin_compile.func() == grammatical_compile.func() == {}


class TestCompositionForceFinalize:
    """Tests for force_finalize hooks and structured output in composition agent."""

    @staticmethod
    def _create_mock_finalize_tool():
        """Create a mock finalize tool to satisfy strict_finalize validation."""
        mock_tool = MagicMock()
        mock_tool.name = "finalize"
        return mock_tool

    @staticmethod
    def _create_mock_composition_agent(strict_finalize: bool = False):
        """Create a CompositionReActAgent with mocked dependencies."""
        from pathlib import Path

        from sakura.nl2test.generation.composition.agent import CompositionReActAgent

        mock_llm = MagicMock()
        mock_finalize = TestCompositionForceFinalize._create_mock_finalize_tool()
        mock_tools = [mock_finalize]

        agent = CompositionReActAgent(
            llm=mock_llm,
            tools=mock_tools,
            system_message="Test system message",
            project_root=Path("/tmp/test"),
            max_iters=5,
        )
        agent.strict_finalize = strict_finalize
        return agent

    def test_get_finalize_schema_returns_comments_args(self):
        """Verify _get_finalize_schema returns FinalizeCommentsArgs."""
        from sakura.nl2test.models.agents import FinalizeCommentsArgs

        agent = self._create_mock_composition_agent()
        schema = agent._get_finalize_schema()

        assert schema is FinalizeCommentsArgs

    def test_get_force_finalize_system_prompt_loads_template(self):
        """Verify _get_force_finalize_system_prompt loads jinja2 template."""
        agent = self._create_mock_composition_agent()
        prompt = agent._get_force_finalize_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "composition" in prompt.lower()
        assert "comments" in prompt.lower()

    def test_get_force_finalize_chat_prompt_loads_template(self):
        """Verify _get_force_finalize_chat_prompt loads jinja2 template."""
        agent = self._create_mock_composition_agent()
        prompt = agent._get_force_finalize_chat_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 50
        assert "comments" in prompt.lower()

    def test_process_force_finalize_result_sets_state(self):
        """Verify _process_force_finalize_result correctly updates state."""
        from sakura.nl2test.models.agents import FinalizeCommentsArgs

        agent = self._create_mock_composition_agent()
        state = AgentState()

        result = FinalizeCommentsArgs(
            comments="Test composition complete with some errors",
        )

        agent._process_force_finalize_result(result, state)

        assert state.finalize_called is True
        assert state.final_comments == "Test composition complete with some errors"

    def test_execute_force_end_with_mocked_llm(self):
        """Test _execute_force_end with mocked LLM returning valid structured output."""
        from sakura.nl2test.models.agents import FinalizeCommentsArgs

        agent = self._create_mock_composition_agent()

        mock_result = FinalizeCommentsArgs(
            comments="Force finalized: test compiles but has assertion failures",
        )

        agent.llm.invoke_structured_with_retries = MagicMock(return_value=mock_result)

        state = AgentState(messages=[])
        result_state = agent._execute_force_end(state)

        assert result_state.finalize_called is True
        assert "Force finalized" in result_state.final_comments
        agent.llm.invoke_structured_with_retries.assert_called_once()

    def test_execute_force_end_with_mocked_llm_failure_strict(self):
        """Test _execute_force_end raises exception when LLM fails in strict mode."""
        import pytest

        from sakura.utils.exceptions import ConfigurationException

        agent = self._create_mock_composition_agent()
        agent.strict_finalize = True

        agent.llm.invoke_structured_with_retries = MagicMock(return_value=None)

        state = AgentState(messages=[])

        with pytest.raises(ConfigurationException) as exc_info:
            agent._execute_force_end(state)

        assert "finalize_not_called" in str(exc_info.value.config_var)

    def test_execute_force_end_with_mocked_llm_failure_non_strict(self):
        """Test _execute_force_end auto-finalizes when LLM fails in non-strict mode."""
        agent = self._create_mock_composition_agent()
        agent.strict_finalize = False

        agent.llm.invoke_structured_with_retries = MagicMock(return_value=None)

        state = AgentState(messages=[])
        result_state = agent._execute_force_end(state)

        assert result_state.finalize_called is True
        assert "Auto-finalized" in result_state.final_comments
