from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nltest.nl2test.generation.localization import (
    GrammaticalLocalizationOrchestrator,
    GherkinLocalizationOrchestrator,
    LocalizationTools,
)
from nltest.nl2test.generation.localization.tools.base import BaseLocalizationTools
from nltest.nl2test.generation.localization.tools.gherkin import GherkinLocalizationTools
from nltest.nl2test.generation.localization.tools.grammatical import GrammaticalLocalizationTools
from nltest.nl2test.evaluation.localization_grader import LocalizationGrader
from nltest.nl2test.models import AgentState, AtomicBlock, AtomicBlockList, NL2TestInput, LocalizedScenario
from nltest.nl2test.models.decomposition import (
    DecompositionMode,
    Scenario,
    GrammaticalBlockList,
    AtomicBlockList,
)
from nltest.nl2test.preprocessing.indexers import MethodIndexer, ClassIndexer
from nltest.nl2test.preprocessing.nl_decomposer import NLDecomposer
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.llm import UsageTracker
from nltest.utils.pretty.prints import pretty_print


class TestLocalizationAgent:
    @pytest.fixture(autouse=True)
    def _inject(self, petclinic_analysis, petclinic_config):
        self.analysis = petclinic_analysis
        self.config = petclinic_config

    def test_gherkin_localization_system_prompt_formatting(self):
        """Ensure the system prompt renders with correct Jinja2 placeholders."""
        prompt = LoadPrompt.load_prompt(
            "localization_agent_gherkin.jinja2", PromptFormat.JINJA2, "system"
        )

        # Case 1: parallelizable True
        max_iters_parallel = 3
        rendered_parallel = prompt.format(
            parallelizable=True, max_iters=max_iters_parallel
        )
        pretty_print("Parallelizable prompt", rendered_parallel)
        expected_cap_parallel = (
            f"You must complete within at most {max_iters_parallel} model step(s) (iterations)."
        )
        assert expected_cap_parallel in rendered_parallel
        assert "You may parallelize tool calls" in rendered_parallel
        assert "Do not parallelize tool calls" not in rendered_parallel
        assert "Never repeat an identical {tool, args} pair" not in rendered_parallel

        # Case 2: parallelizable False
        max_iters_sequential = 7
        rendered_sequential = prompt.format(
            parallelizable=False, max_iters=max_iters_sequential
        )
        pretty_print("Not parallelizable", rendered_sequential)
        expected_cap_sequential = f"You must complete within at most {max_iters_sequential} model step(s) (iterations)."
        assert expected_cap_sequential in rendered_sequential
        assert "Do not parallelize tool calls" in rendered_sequential
        assert "You may parallelize tool calls" not in rendered_sequential
        assert "Never repeat an identical {tool, args} pair" in rendered_sequential

    def test_gherkin_localization_chat_prompt_formatting(self):
        """Ensure the chat prompt renders with required placeholders and includes a preview."""
        prompt = LoadPrompt.load_prompt(
            "localization_agent_gherkin.jinja2", PromptFormat.JINJA2, "chat"
        )

        nl_description = "Describe pet update behavior"
        instructions = "Be concise and prefer primary SUT methods"
        steps = "- Given owner exists\n- When pet is updated\n- Then response redirects"

        rendered = prompt.format(
            nl_description=nl_description,
            instructions=instructions,
            steps=steps,
        )

        # Print a small preview for debugging
        pretty_print("Chat prompt (gherkin localization)", rendered)

        # Assertions
        assert nl_description in rendered
        assert instructions in rendered
        assert "Given" in rendered
        assert "When" in rendered
        assert "Then" in rendered

    def test_localization_agent_simple_grammatical(self):
        nl_description = "Ensure pet is added to owner and ID is generated."
        tracker = UsageTracker()
        nl_decomposer = NLDecomposer(
            mode=DecompositionMode.GRAMMATICAL,
            usage_tracker=tracker,
        )
        grammatical_blocks: GrammaticalBlockList = nl_decomposer.decompose(
            nl_description
        )

        method_searcher = MethodIndexer(self.analysis).build_index()
        class_searcher = ClassIndexer(self.analysis).build_index()

        atomic_blocks = AtomicBlockList(
            atomic_blocks=[
                AtomicBlock.from_grammatical_block(gb)
                for gb in grammatical_blocks.grammatical_blocks
            ]
        )
        pretty_print("Initial atomic blocks", atomic_blocks)

        supervisor_instructions = (
            "Find the relevant methods and refine the atomic blocks."
        )

        nl2_input = NL2TestInput(
            description=nl_description,
            project_name="spring-petclinic",
            qualified_class_name="",
            method_signature="",
        )

        localization_agent = GrammaticalLocalizationOrchestrator(
            analysis=self.analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            usage_tracker=tracker,
        )

        agent_state = localization_agent.assign_task(
            atomic_blocks, instructions=supervisor_instructions
        )

        assert isinstance(agent_state, AgentState)

        # Extract results from AgentState
        refined_blocks = agent_state.atomic_blocks
        comments = agent_state.final_comments or ""
        prices = tracker.totals()

        assert refined_blocks is not None
        pretty_print("Refined atomic blocks", refined_blocks)
        pretty_print("Comments", comments)
        pretty_print("Token usage", prices)

    def test_localization_agent_simple_gherkin(self):
        nl_description = "Ensure pet is added to owner and ID is generated."
        tracker = UsageTracker()
        nl_decomposer = NLDecomposer(
            mode=DecompositionMode.GHERKIN,
            usage_tracker=tracker,
        )

        scenario: Scenario = nl_decomposer.decompose(nl_description)
        pretty_print("Initial scenario", scenario)

        # Build searchers
        method_searcher = MethodIndexer(self.analysis).build_index()
        class_searcher = ClassIndexer(self.analysis).build_index()

        supervisor_instructions = (
            "Find the relevant methods and refine the scenario blocks."
        )

        nl2_input = NL2TestInput(
            description=nl_description,
            project_name="spring-petclinic",
            qualified_class_name="",
            method_signature="",
        )

        # Run localization agent in GHERKIN mode
        localization_agent = GherkinLocalizationOrchestrator(
            analysis=self.analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            usage_tracker=tracker,
        )

        # Convert Scenario to LocalizedScenario with empty fields
        localized_input = LocalizedScenario.from_scenario(scenario)
        agent_state = localization_agent.assign_task(
            localized_input, instructions=supervisor_instructions
        )

        assert isinstance(agent_state, AgentState)

        # Extract results from AgentState
        localized_scenario = agent_state.localized_scenario
        comments = agent_state.final_comments or ""
        prices = tracker.totals()

        # Assertions and output
        assert localized_scenario is not None
        assert isinstance(localized_scenario, LocalizedScenario)
        assert isinstance(comments, str)
        pretty_print("Localized scenario", localized_scenario)
        pretty_print("Comments", comments)
        pretty_print("Token usage", prices)

    def test_localization_agent_complex_gherkin(self):
        qualified_class_name = (
            "org.springframework.samples.petclinic.owner.OwnerControllerTests"
        )
        method_signature = "testProcessCreationFormSuccess()"

        nl_description = 'Create a test case that validates the successful processing of a new owner creation form by the `OwnerController`. The test leverages Spring\'s `@WebMvcTest` with `MockMvc` to simulate HTTP requests and responses. The `OwnerRepository` dependency is mocked using `@MockitoBean`, and its behavior is pre-configured in the `setup` method: a predefined `Owner` (obtained via the `george()` helper method, which constructs and populates an `Owner` instance with associated `Pet` and `PetType` data) is returned when `owners.findByLastNameStartingWith()` is called with any `Pageable` and the last name "Franklin", and the same `Owner` is returned when `owners.findById()` is called with `TEST_OWNER_ID`. The test then performs a POST request to "/owners/new" using `mockMvc.perform()`, simulating form submission with parameters for firstName, lastName, address, city, and telephone. Finally, the test asserts that the HTTP response status is a 3xx redirection using `andExpect(status().is3xxRedirection())`, indicating successful form processing and redirection, using Spring\'s `MockMvcResultMatchers`. JUnit and Mockito are used for the test structure and mocking, respectively.'

        tracker = UsageTracker()
        nl_decomposer = NLDecomposer(
            mode=DecompositionMode.GHERKIN,
            usage_tracker=tracker,
        )
        scenario: Scenario = nl_decomposer.decompose(nl_description)
        pretty_print("Initial scenario", scenario)

        # Build searchers
        method_searcher = MethodIndexer(self.analysis).build_index()
        class_searcher = ClassIndexer(self.analysis).build_index()

        supervisor_instructions = (
            "Find the relevant methods and refine the scenario blocks."
        )

        nl2_input = NL2TestInput(
            description=nl_description,
            project_name="spring-petclinic",
            qualified_class_name="",
            method_signature="",
        )

        # Run localization agent in GHERKIN mode
        localization_agent = GherkinLocalizationOrchestrator(
            analysis=self.analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            usage_tracker=tracker,
        )

        # Convert Scenario to LocalizedScenario with empty fields
        localized_input = LocalizedScenario.from_scenario(scenario)
        agent_state = localization_agent.assign_task(
            localized_input, instructions=supervisor_instructions
        )

        assert isinstance(agent_state, AgentState)

        # Extract results from AgentState
        localized_scenario = agent_state.localized_scenario
        comments = agent_state.final_comments or ""
        prices = tracker.totals()

        # Assertions and output
        assert localized_scenario is not None
        assert isinstance(localized_scenario, LocalizedScenario)
        assert isinstance(comments, str)
        pretty_print("Localized scenario", localized_scenario)
        pretty_print("Comments", comments)
        pretty_print("Token usage", prices)


class TestLocalizationGrader:
    @pytest.fixture(autouse=True)
    def _inject(self, petclinic_analysis, petclinic_config, petclinic_paths):
        self.analysis = petclinic_analysis
        self.config = petclinic_config
        self.project_root = petclinic_paths.project_root

    def _petcontroller_localized_scenario_payload(self) -> dict:
        return {
            "testing_framework": "junit",
            "setup": [
                {
                    "id": 0,
                    "task": "Load Spring MVC test context for PetController and PetTypeFormatter using @WebMvcTest",
                    "uses": "",
                    "produces": "mock_mvc_context",
                    "candidate_methods": [],
                    "arg_bindings": [],
                    "comments": "@WebMvcTest is an annotation, not a method.",
                    "external": False,
                },
                {
                    "id": 1,
                    "task": "Disable test in native image and AOT modes",
                    "uses": "mock_mvc_context",
                    "produces": "",
                    "candidate_methods": [],
                    "arg_bindings": [],
                    "comments": "Disabled annotations are not methods.",
                    "external": False,
                },
                {
                    "id": 2,
                    "task": "Mock OwnerRepository.findPetTypes to return a PetType list",
                    "uses": "mocked_owner_repository",
                    "produces": "pet_types",
                    "candidate_methods": [],
                    "arg_bindings": [],
                    "comments": "Mockito stubbing for findPetTypes.",
                    "external": False,
                },
                {
                    "id": 3,
                    "task": "Mock OwnerRepository.findById to return an Owner",
                    "uses": "mocked_owner_repository, TEST_OWNER_ID",
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
                    "comments": "Mockito stubbing for findById.",
                    "external": False,
                },
                {
                    "id": 4,
                    "task": "Prepare Owner instance with Pet for updating",
                    "uses": "owner_with_pets, pet_types",
                    "produces": "owner_with_pet_to_update",
                    "candidate_methods": [],
                    "arg_bindings": [],
                    "comments": "Data preparation, no direct method.",
                    "external": False,
                },
                {
                    "id": 5,
                    "task": "Prepare pet details for update request",
                    "uses": "owner_with_pet_to_update",
                    "produces": "pet_details",
                    "candidate_methods": [],
                    "arg_bindings": [],
                    "comments": "Setup data.",
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
                            "comments": "processUpdateForm is the focal method.",
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
                            "comments": "Assertion helper; not a method under test.",
                            "external": False,
                        },
                        {
                            "id": 8,
                            "task": "Verify view name is a redirection to the owner's details page",
                            "uses": "http_response",
                            "produces": "",
                            "candidate_methods": [],
                            "arg_bindings": [],
                            "comments": "Assertion helper; not a method under test.",
                            "external": False,
                        },
                    ],
                }
            ],
            "teardown": [],
        }

    def test_localization_grader_localized_scenario(self):
        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.owner.PetControllerTests",
            method_signature="testProcessUpdateFormSuccess()",
            description="Validate successful processing of a pet update form via PetController.",
            project_name="spring-petclinic",
        )

        localized_scenario_data = self._petcontroller_localized_scenario_payload()

        localized_scenario = LocalizedScenario(**localized_scenario_data)

        project_root = Path(self.config.get("project", "base_project_dir"))
        common_analysis = CommonAnalysis(self.analysis)
        _, application_classes = (
            common_analysis.get_test_methods_classes_and_application_classes()
        )

        grader = LocalizationGrader(
            analysis=self.analysis,
            project_root=project_root,
            decomposition_mode=DecompositionMode.GHERKIN,
            application_classes=application_classes,
        )

        results = grader.grade(localized_scenario, nl2_input)

        pretty_print("Localization Results", results)

        assert results.qualified_class_name == nl2_input.qualified_class_name
        assert results.method_signature == nl2_input.method_signature
        assert results.tp >= 1
        assert 0.0 <= results.localization_recall <= 1.0
        assert (
            len(results.covered_focal_methods) + len(results.uncovered_focal_methods)
        ) == len(results.all_focal_methods)


class TestLocalizationTools:
    @pytest.fixture(autouse=True)
    def _inject(self, petclinic_analysis):
        self.analysis = petclinic_analysis

    def test_localization_call_site_tool(self):
        qualified_class_name = (
            "org.springframework.samples.petclinic.service.ClinicServiceTests"
        )
        method_signature = "shouldInsertPetIntoDatabaseAndGenerateId()"

        fake_method_searcher = MagicMock()
        fake_class_searcher = MagicMock()
        fake_llm = MagicMock()

        localization_tools = LocalizationTools(
            analysis=self.analysis,
            method_searcher=fake_method_searcher,
            class_searcher=fake_class_searcher,
            structured_llm=fake_llm,
        )

        call_site_tool = localization_tools._make_call_site_details_tool()

        cleaned_call_sites = call_site_tool.func(qualified_class_name, method_signature)
        pretty_print("Cleaned call site details", cleaned_call_sites)


class TestLocalizationToolInjection:
    """Tests verifying tool injection after refactoring to shared utilities."""

    @staticmethod
    def _create_mock_dependencies():
        """Create mock dependencies for tool builders."""
        mock_analysis = MagicMock()
        mock_analysis.get_class = MagicMock(return_value=None)
        mock_analysis.get_method = MagicMock(return_value=None)
        mock_method_searcher = MagicMock()
        mock_class_searcher = MagicMock()

        return {
            "analysis": mock_analysis,
            "method_searcher": mock_method_searcher,
            "class_searcher": mock_class_searcher,
        }

    def test_base_localization_tools_contains_expected_tools(self):
        """Verify BaseLocalizationTools creates expected base tools."""
        deps = self._create_mock_dependencies()
        tool_builder = BaseLocalizationTools(**deps)
        tools, allow_duplicates = tool_builder.all()

        tool_names = {t.name for t in tools}
        expected_base_tools = {
            "query_method_db",
            "query_class_db",
            "get_reachable_methods_in_class",
            "extract_method_code",
            "get_method_details",
            "get_class_details",
            "get_inherited_library_classes",
            "get_call_site_details",
        }

        assert expected_base_tools == tool_names, (
            f"Tool mismatch. Expected: {expected_base_tools}, Got: {tool_names}"
        )

    def test_gherkin_localization_tools_contains_all_tools(self):
        """Verify GherkinLocalizationTools includes base + Gherkin-specific tools."""
        deps = self._create_mock_dependencies()
        tool_builder = GherkinLocalizationTools(**deps)
        tools, allow_duplicates = tool_builder.all()

        tool_names = {t.name for t in tools}
        expected_tools = {
            "query_method_db",
            "query_class_db",
            "get_reachable_methods_in_class",
            "extract_method_code",
            "get_method_details",
            "get_class_details",
            "get_inherited_library_classes",
            "get_call_site_details",
            "finalize",
        }

        assert expected_tools == tool_names, (
            f"Tool mismatch. Expected: {expected_tools}, Got: {tool_names}"
        )

    def test_grammatical_localization_tools_contains_all_tools(self):
        """Verify GrammaticalLocalizationTools includes base + Grammatical-specific tools."""
        deps = self._create_mock_dependencies()
        tool_builder = GrammaticalLocalizationTools(**deps)
        tools, allow_duplicates = tool_builder.all()

        tool_names = {t.name for t in tools}
        expected_tools = {
            "query_method_db",
            "query_class_db",
            "get_reachable_methods_in_class",
            "extract_method_code",
            "get_method_details",
            "get_class_details",
            "get_inherited_library_classes",
            "get_call_site_details",
            "finalize",
        }

        assert expected_tools == tool_names, (
            f"Tool mismatch. Expected: {expected_tools}, Got: {tool_names}"
        )

    def test_localization_allow_duplicate_tools_empty(self):
        """Verify localization tools have no duplicate tools allowed."""
        deps = self._create_mock_dependencies()
        tool_builder = GherkinLocalizationTools(**deps)
        _, allow_duplicates = tool_builder.all()

        assert len(allow_duplicates) == 0, (
            f"Localization should have no duplicate tools, got: {[t.name for t in allow_duplicates]}"
        )

    def test_gherkin_finalize_tool_returns_scenario_and_comments(self):
        """Verify Gherkin finalize tool returns scenario and comments."""
        deps = self._create_mock_dependencies()
        tool_builder = GherkinLocalizationTools(**deps)
        tools, _ = tool_builder.all()

        finalize_tool = next(t for t in tools if t.name == "finalize")

        scenario = LocalizedScenario(
            testing_framework="junit",
            setup=[],
            gherkin_groups=[],
            teardown=[],
        )
        result = finalize_tool.func(scenario=scenario, comments="Final comments")

        assert result == (scenario, "Final comments")

    def test_grammatical_finalize_tool_returns_blocks_and_comments(self):
        """Verify Grammatical finalize tool returns blocks and comments."""
        deps = self._create_mock_dependencies()
        tool_builder = GrammaticalLocalizationTools(**deps)
        tools, _ = tool_builder.all()

        finalize_tool = next(t for t in tools if t.name == "finalize")

        blocks = AtomicBlockList(atomic_blocks=[])
        result = finalize_tool.func(current_blocks=blocks, comments="Final comments")

        assert result == (blocks, "Final comments")

    def test_shared_tools_have_consistent_names_across_modes(self):
        """Verify shared tools have same names in both modes."""
        deps = self._create_mock_dependencies()

        gherkin_builder = GherkinLocalizationTools(**deps)
        grammatical_builder = GrammaticalLocalizationTools(**deps)

        gherkin_tools, _ = gherkin_builder.all()
        grammatical_tools, _ = grammatical_builder.all()

        gherkin_names = {t.name for t in gherkin_tools}
        grammatical_names = {t.name for t in grammatical_tools}

        # All base tools should be in both
        base_tools = {
            "query_method_db",
            "query_class_db",
            "get_reachable_methods_in_class",
            "extract_method_code",
            "get_method_details",
            "get_class_details",
            "get_inherited_library_classes",
            "get_call_site_details",
        }

        assert base_tools.issubset(gherkin_names)
        assert base_tools.issubset(grammatical_names)

    def test_query_method_db_tool_validates_range(self):
        """Verify query_method_db tool exists and has correct name."""
        deps = self._create_mock_dependencies()
        tool_builder = BaseLocalizationTools(**deps)
        tools, _ = tool_builder.all()

        query_tool = next(t for t in tools if t.name == "query_method_db")
        assert query_tool is not None

    def test_get_reachable_methods_tool_exists(self):
        """Verify get_reachable_methods_in_class tool exists."""
        deps = self._create_mock_dependencies()
        tool_builder = BaseLocalizationTools(**deps)
        tools, _ = tool_builder.all()

        reachable_tool = next(t for t in tools if t.name == "get_reachable_methods_in_class")
        assert reachable_tool is not None

    def test_get_class_details_tool_exists(self):
        """Verify get_class_details tool exists."""
        deps = self._create_mock_dependencies()
        tool_builder = BaseLocalizationTools(**deps)
        tools, _ = tool_builder.all()

        class_details_tool = next(t for t in tools if t.name == "get_class_details")
        assert class_details_tool is not None

    def test_get_inherited_library_classes_tool_exists(self):
        """Verify get_inherited_library_classes tool exists."""
        deps = self._create_mock_dependencies()
        tool_builder = BaseLocalizationTools(**deps)
        tools, _ = tool_builder.all()

        inherited_tool = next(t for t in tools if t.name == "get_inherited_library_classes")
        assert inherited_tool is not None
