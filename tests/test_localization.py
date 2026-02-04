from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sakura.nl2test.evaluation.localization_grader import LocalizationGrader
from sakura.nl2test.generation.localization import (
    GherkinLocalizationOrchestrator,
    GrammaticalLocalizationOrchestrator,
)
from sakura.nl2test.generation.localization.tools.base import BaseLocalizationTools
from sakura.nl2test.generation.localization.tools.gherkin import (
    GherkinLocalizationTools,
)
from sakura.nl2test.generation.localization.tools.grammatical import (
    GrammaticalLocalizationTools,
)
from sakura.nl2test.models import (
    AgentState,
    AtomicBlock,
    AtomicBlockList,
    LocalizedScenario,
    NL2TestInput,
)
from sakura.nl2test.models.decomposition import (
    DecompositionMode,
    GrammaticalBlockList,
    Scenario,
)
from sakura.nl2test.preprocessing.indexers import ClassIndexer, MethodIndexer
from sakura.nl2test.preprocessing.nl_decomposer import NLDecomposer
from sakura.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.llm import UsageTracker
from sakura.utils.pretty.prints import pretty_print


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
        expected_cap_parallel = f"You must complete within at most {max_iters_parallel} model step(s) (iterations)."
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
        grammatical_blocks = nl_decomposer.decompose(nl_description)
        assert isinstance(grammatical_blocks, GrammaticalBlockList)

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

        scenario = nl_decomposer.decompose(nl_description)
        pretty_print("Initial scenario", scenario)
        assert isinstance(scenario, Scenario)

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
        nl_description = 'Create a test case that validates the successful processing of a new owner creation form by the `OwnerController`. The test leverages Spring\'s `@WebMvcTest` with `MockMvc` to simulate HTTP requests and responses. The `OwnerRepository` dependency is mocked using `@MockitoBean`, and its behavior is pre-configured in the `setup` method: a predefined `Owner` (obtained via the `george()` helper method, which constructs and populates an `Owner` instance with associated `Pet` and `PetType` data) is returned when `owners.findByLastNameStartingWith()` is called with any `Pageable` and the last name "Franklin", and the same `Owner` is returned when `owners.findById()` is called with `TEST_OWNER_ID`. The test then performs a POST request to "/owners/new" using `mockMvc.perform()`, simulating form submission with parameters for firstName, lastName, address, city, and telephone. Finally, the test asserts that the HTTP response status is a 3xx redirection using `andExpect(status().is3xxRedirection())`, indicating successful form processing and redirection, using Spring\'s `MockMvcResultMatchers`. JUnit and Mockito are used for the test structure and mocking, respectively.'
        tracker = UsageTracker()
        nl_decomposer = NLDecomposer(
            mode=DecompositionMode.GHERKIN,
            usage_tracker=tracker,
        )
        scenario = nl_decomposer.decompose(nl_description)
        pretty_print("Initial scenario", scenario)
        assert isinstance(scenario, Scenario)

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

    def _clinicservice_insert_pet_payload(self) -> dict:
        """
        Payload for ClinicServiceTests.shouldInsertPetIntoDatabaseAndGenerateId().

        This test validates that a new pet can be inserted into the database and
        that an ID is generated for it. The focal methods involve Owner and Pet
        entity operations plus repository calls.
        """
        return {
            "setup": [
                {
                    "id": 0,
                    "task": "Configure DataJpaTest context with OwnerRepository and PetTypeRepository",
                    "uses": "",
                    "produces": "owners_repository, types_repository",
                    "candidate_methods": [],
                    "arg_bindings": [],
                    "comments": "@DataJpaTest annotation configures test context.",
                    "external": False,
                },
                {
                    "id": 1,
                    "task": "Find owner with ID 6 from repository",
                    "uses": "owners_repository",
                    "produces": "owner6",
                    "candidate_methods": [
                        {
                            "declaring_class_name": "org.springframework.data.repository.CrudRepository",
                            "containing_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                            "method_signature": "findById(java.lang.Integer)",
                            "return_type": "java.util.Optional<org.springframework.samples.petclinic.owner.Owner>",
                        }
                    ],
                    "arg_bindings": [{"arg_name": "id", "arg_value": "6"}],
                    "comments": "Repository lookup for owner.",
                    "external": False,
                },
                {
                    "id": 2,
                    "task": "Get current pet count from owner",
                    "uses": "owner6",
                    "produces": "initial_pet_count",
                    "candidate_methods": [
                        {
                            "declaring_class_name": "org.springframework.samples.petclinic.owner.Owner",
                            "containing_class_name": "org.springframework.samples.petclinic.owner.Owner",
                            "method_signature": "getPets()",
                            "return_type": "java.util.List<org.springframework.samples.petclinic.owner.Pet>",
                        }
                    ],
                    "arg_bindings": [],
                    "comments": "Get pets to determine initial count.",
                    "external": False,
                },
            ],
            "gherkin_groups": [
                {
                    "given": [
                        {
                            "id": 3,
                            "task": "Create new Pet instance and set name to 'bowser'",
                            "uses": "",
                            "produces": "new_pet",
                            "candidate_methods": [
                                {
                                    "declaring_class_name": "org.springframework.samples.petclinic.owner.Pet",
                                    "containing_class_name": "org.springframework.samples.petclinic.owner.Pet",
                                    "method_signature": "setName(java.lang.String)",
                                    "return_type": "void",
                                }
                            ],
                            "arg_bindings": [
                                {"arg_name": "name", "arg_value": "bowser"}
                            ],
                            "comments": "Pet entity setter.",
                            "external": False,
                        },
                        {
                            "id": 4,
                            "task": "Retrieve all pet types from repository",
                            "uses": "types_repository",
                            "produces": "pet_types",
                            "candidate_methods": [
                                {
                                    "declaring_class_name": "org.springframework.samples.petclinic.owner.PetTypeRepository",
                                    "containing_class_name": "org.springframework.samples.petclinic.owner.PetTypeRepository",
                                    "method_signature": "findPetTypes()",
                                    "return_type": "java.util.Collection<org.springframework.samples.petclinic.owner.PetType>",
                                }
                            ],
                            "arg_bindings": [],
                            "comments": "Repository method to get pet types.",
                            "external": False,
                        },
                        {
                            "id": 5,
                            "task": "Set pet type and birth date",
                            "uses": "new_pet, pet_types",
                            "produces": "configured_pet",
                            "candidate_methods": [
                                {
                                    "declaring_class_name": "org.springframework.samples.petclinic.owner.Pet",
                                    "containing_class_name": "org.springframework.samples.petclinic.owner.Pet",
                                    "method_signature": "setType(org.springframework.samples.petclinic.owner.PetType)",
                                    "return_type": "void",
                                },
                                {
                                    "declaring_class_name": "org.springframework.samples.petclinic.owner.Pet",
                                    "containing_class_name": "org.springframework.samples.petclinic.owner.Pet",
                                    "method_signature": "setBirthDate(java.time.LocalDate)",
                                    "return_type": "void",
                                },
                            ],
                            "arg_bindings": [],
                            "comments": "Pet entity setters.",
                            "external": False,
                        },
                    ],
                    "when": [
                        {
                            "id": 6,
                            "task": "Add pet to owner and save owner to repository",
                            "uses": "owner6, configured_pet, owners_repository",
                            "produces": "saved_owner",
                            "candidate_methods": [
                                {
                                    "declaring_class_name": "org.springframework.samples.petclinic.owner.Owner",
                                    "containing_class_name": "org.springframework.samples.petclinic.owner.Owner",
                                    "method_signature": "addPet(org.springframework.samples.petclinic.owner.Pet)",
                                    "return_type": "void",
                                },
                            ],
                            "arg_bindings": [
                                {"arg_name": "pet", "arg_value": "configured_pet"}
                            ],
                            "comments": "Owner.addPet is the focal method for adding pet.",
                            "external": False,
                        },
                    ],
                    "then": [
                        {
                            "id": 7,
                            "task": "Verify pet count increased by one",
                            "uses": "saved_owner, initial_pet_count",
                            "produces": "",
                            "candidate_methods": [
                                {
                                    "declaring_class_name": "org.springframework.samples.petclinic.owner.Owner",
                                    "containing_class_name": "org.springframework.samples.petclinic.owner.Owner",
                                    "method_signature": "getPets()",
                                    "return_type": "java.util.List<org.springframework.samples.petclinic.owner.Pet>",
                                }
                            ],
                            "arg_bindings": [],
                            "comments": "Assertion on pet list size.",
                            "external": False,
                        },
                        {
                            "id": 8,
                            "task": "Retrieve pet by name and verify ID was generated",
                            "uses": "saved_owner",
                            "produces": "",
                            "candidate_methods": [
                                {
                                    "declaring_class_name": "org.springframework.samples.petclinic.owner.Owner",
                                    "containing_class_name": "org.springframework.samples.petclinic.owner.Owner",
                                    "method_signature": "getPet(java.lang.String)",
                                    "return_type": "org.springframework.samples.petclinic.owner.Pet",
                                },
                                {
                                    "declaring_class_name": "org.springframework.samples.petclinic.model.BaseEntity",
                                    "containing_class_name": "org.springframework.samples.petclinic.owner.Pet",
                                    "method_signature": "getId()",
                                    "return_type": "java.lang.Integer",
                                },
                            ],
                            "arg_bindings": [
                                {"arg_name": "name", "arg_value": "bowser"}
                            ],
                            "comments": "Verify pet ID is not null after save.",
                            "external": False,
                        },
                    ],
                }
            ],
            "teardown": [],
        }

    def _clinicservice_insert_pet_payload_relaxed(self) -> dict:
        """
        Payload with simplified class names and method signatures to test
        relaxed semantic matching.

        Uses simple class names (e.g., "Owner" instead of FQN) and method names
        without parameter types to verify the grader's heuristic matching.
        """
        return {
            "setup": [],
            "gherkin_groups": [
                {
                    "given": [
                        {
                            "id": 0,
                            "task": "Create and configure new pet",
                            "uses": "",
                            "produces": "new_pet",
                            "candidate_methods": [
                                {
                                    "declaring_class_name": "Pet",
                                    "containing_class_name": "Pet",
                                    "method_signature": "setName(String)",
                                    "return_type": "void",
                                }
                            ],
                            "arg_bindings": [],
                            "comments": "Relaxed matching test.",
                            "external": False,
                        },
                    ],
                    "when": [
                        {
                            "id": 1,
                            "task": "Add pet to owner",
                            "uses": "owner, new_pet",
                            "produces": "updated_owner",
                            "candidate_methods": [
                                {
                                    "declaring_class_name": "Owner",
                                    "containing_class_name": "Owner",
                                    "method_signature": "addPet(Pet)",
                                    "return_type": "void",
                                }
                            ],
                            "arg_bindings": [],
                            "comments": "Relaxed matching for Owner.addPet.",
                            "external": False,
                        },
                    ],
                    "then": [
                        {
                            "id": 2,
                            "task": "Verify pet ID generated",
                            "uses": "updated_owner",
                            "produces": "",
                            "candidate_methods": [
                                {
                                    "declaring_class_name": "Owner",
                                    "containing_class_name": "Owner",
                                    "method_signature": "getPet(String)",
                                    "return_type": "Pet",
                                }
                            ],
                            "arg_bindings": [],
                            "comments": "Relaxed matching for getPet.",
                            "external": False,
                        },
                    ],
                }
            ],
            "teardown": [],
        }

    def test_localization_grader_clinic_service_insert_pet(self):
        """Test grading for ClinicServiceTests.shouldInsertPetIntoDatabaseAndGenerateId."""
        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.service.ClinicServiceTests",
            method_signature="shouldInsertPetIntoDatabaseAndGenerateId()",
            description="Insert a new pet into the database and verify that an ID is generated.",
            project_name="spring-petclinic",
        )

        localized_scenario = LocalizedScenario(
            **self._clinicservice_insert_pet_payload()
        )

        project_root = Path(self.config.get("project", "base_project_dir"))
        common_analysis = CommonAnalysis(self.analysis)
        _, application_classes, test_utility_classes = (
            common_analysis.categorize_classes()
        )

        grader = LocalizationGrader(
            analysis=self.analysis,
            project_root=project_root,
            decomposition_mode=DecompositionMode.GHERKIN,
            application_classes=application_classes,
            test_utility_classes=test_utility_classes,
        )

        results = grader.grade(localized_scenario, nl2_input)

        pretty_print("ClinicService Insert Pet - Localization Results", results)

        assert results.qualified_class_name == nl2_input.qualified_class_name
        assert results.method_signature == nl2_input.method_signature
        assert results.tp >= 1, "Expected at least one true positive match"
        assert 0.0 <= results.localization_recall <= 1.0
        assert (
            len(results.covered_focal_methods) + len(results.uncovered_focal_methods)
        ) == len(results.all_focal_methods)

    def test_localization_grader_relaxed_matching(self):
        """
        Test that relaxed semantic matching works with simplified class/method names.

        This validates that predictions using simple class names (e.g., 'Owner')
        and method names without full parameter types (e.g., 'addPet(Pet)')
        correctly match against fully qualified ground truth focal methods.
        """
        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.service.ClinicServiceTests",
            method_signature="shouldInsertPetIntoDatabaseAndGenerateId()",
            description="Insert a new pet into the database and verify that an ID is generated.",
            project_name="spring-petclinic",
        )

        localized_scenario = LocalizedScenario(
            **self._clinicservice_insert_pet_payload_relaxed()
        )

        project_root = Path(self.config.get("project", "base_project_dir"))
        common_analysis = CommonAnalysis(self.analysis)
        _, application_classes, test_utility_classes = (
            common_analysis.categorize_classes()
        )

        grader = LocalizationGrader(
            analysis=self.analysis,
            project_root=project_root,
            decomposition_mode=DecompositionMode.GHERKIN,
            application_classes=application_classes,
            test_utility_classes=test_utility_classes,
        )

        results = grader.grade(localized_scenario, nl2_input)

        pretty_print("Relaxed Matching - Localization Results", results)

        # With relaxed matching, simple names like "Owner.addPet" should match
        # against "org.springframework.samples.petclinic.owner.Owner.addPet(...)"
        assert results.qualified_class_name == nl2_input.qualified_class_name
        assert results.method_signature == nl2_input.method_signature

        # The relaxed payload includes Owner.addPet, Owner.getPet, Pet.setName
        # These should match focal methods if they exist in ground truth
        if results.all_focal_methods:
            assert results.tp >= 1, (
                f"Relaxed matching should find at least one match. "
                f"Covered: {results.covered_focal_methods}, "
                f"All: {results.all_focal_methods}"
            )

        assert 0.0 <= results.localization_recall <= 1.0

    def test_localization_grader_state_missing_output_penalizes(self, monkeypatch):
        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.service.ClinicServiceTests",
            method_signature="shouldInsertPetIntoDatabaseAndGenerateId()",
            description="Insert a new pet into the database and verify that an ID is generated.",
            project_name="spring-petclinic",
        )

        project_root = Path(self.config.get("project", "base_project_dir"))
        common_analysis = CommonAnalysis(self.analysis)
        _, application_classes, test_utility_classes = (
            common_analysis.categorize_classes()
        )

        grader = LocalizationGrader(
            analysis=self.analysis,
            project_root=project_root,
            decomposition_mode=DecompositionMode.GHERKIN,
            application_classes=application_classes,
            test_utility_classes=test_utility_classes,
        )

        focal_methods = {
            ("org.example.Foo", "alpha()"),
            ("org.example.Bar", "beta(java.lang.String)"),
        }

        monkeypatch.setattr(grader, "_get_focal_methods", lambda _: focal_methods)

        results = grader.grade_from_state(None, nl2_input)
        expected_methods = grader._format_methods(focal_methods)

        assert results.localization_recall == 0.0
        assert results.tp == 0
        assert results.fn == len(focal_methods)
        assert results.all_focal_methods == expected_methods
        assert results.covered_focal_methods == []
        assert results.uncovered_focal_methods == expected_methods

    def test_localization_grader_state_empty_focal_methods_returns_one(
        self, monkeypatch
    ):
        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.service.ClinicServiceTests",
            method_signature="shouldInsertPetIntoDatabaseAndGenerateId()",
            description="Insert a new pet into the database and verify that an ID is generated.",
            project_name="spring-petclinic",
        )

        project_root = Path(self.config.get("project", "base_project_dir"))
        common_analysis = CommonAnalysis(self.analysis)
        _, application_classes, test_utility_classes = (
            common_analysis.categorize_classes()
        )

        grader = LocalizationGrader(
            analysis=self.analysis,
            project_root=project_root,
            decomposition_mode=DecompositionMode.GHERKIN,
            application_classes=application_classes,
            test_utility_classes=test_utility_classes,
        )

        monkeypatch.setattr(grader, "_get_focal_methods", lambda _: set())

        results = grader.grade_from_state(None, nl2_input)

        assert results.localization_recall == 1.0
        assert results.tp == 0
        assert results.fn == 0
        assert results.all_focal_methods == []
        assert results.covered_focal_methods == []
        assert results.uncovered_focal_methods == []


class TestLocalizationTools:
    @pytest.fixture(autouse=True)
    def _inject(self, petclinic_analysis, petclinic_config):
        self.analysis = petclinic_analysis
        self.config = petclinic_config

    def test_localization_call_site_tool(self):
        fake_method_searcher = MagicMock()
        fake_class_searcher = MagicMock()

        localization_tools = BaseLocalizationTools(
            analysis=self.analysis,
            method_searcher=fake_method_searcher,
            class_searcher=fake_class_searcher,
        )

        call_site_tool = localization_tools._make_call_site_details_tool()

        cleaned_call_sites = call_site_tool.invoke(
            {
                "qualified_class_name": "org.springframework.samples.petclinic.service.ClinicServiceTests",
                "method_signature": "shouldInsertPetIntoDatabaseAndGenerateId()",
            }
        )
        pretty_print("Cleaned call site details", cleaned_call_sites)

    def test_search_reachable_methods_tool(self):
        qualified_class_name = "org.springframework.samples.petclinic.owner.Owner"
        query = "add pet to owner"

        method_searcher = MethodIndexer(self.analysis).build_index()
        class_searcher = ClassIndexer(self.analysis).build_index()

        localization_tools = BaseLocalizationTools(
            analysis=self.analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
        )

        search_tool = localization_tools._make_search_reachable_methods_tool()
        results = search_tool.invoke(
            {
                "qualified_class_name": qualified_class_name,
                "query": query,
                "visibility_mode": "same_package_or_subclass",
            }
        )

        pretty_print("Reachable method search results", results)

        assert isinstance(results, list)
        assert results, "Expected reachable method search to return results"
        assert any(
            "addPet" in entry.get("method_signature", "") for entry in results
        ), "Expected addPet to appear in reachable search results"


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
            "search_reachable_methods_in_class",
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
            "search_reachable_methods_in_class",
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
            "search_reachable_methods_in_class",
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
            setup=[],
            gherkin_groups=[],
            teardown=[],
        )
        result = finalize_tool.invoke(
            {"scenario": scenario, "comments": "Final comments"}
        )

        assert result == (scenario, "Final comments")

    def test_grammatical_finalize_tool_returns_blocks_and_comments(self):
        """Verify Grammatical finalize tool returns blocks and comments."""
        deps = self._create_mock_dependencies()
        tool_builder = GrammaticalLocalizationTools(**deps)
        tools, _ = tool_builder.all()

        finalize_tool = next(t for t in tools if t.name == "finalize")

        blocks = AtomicBlockList(atomic_blocks=[])
        result = finalize_tool.invoke(
            {"current_blocks": blocks, "comments": "Final comments"}
        )

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
            "search_reachable_methods_in_class",
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

    def test_search_reachable_methods_tool_exists(self):
        """Verify search_reachable_methods_in_class tool exists."""
        deps = self._create_mock_dependencies()
        tool_builder = BaseLocalizationTools(**deps)
        tools, _ = tool_builder.all()

        reachable_tool = next(
            t for t in tools if t.name == "search_reachable_methods_in_class"
        )
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

        inherited_tool = next(
            t for t in tools if t.name == "get_inherited_library_classes"
        )
        assert inherited_tool is not None


class TestLocalizationForceFinalize:
    """Tests for force_finalize hooks and structured output in localization agent."""

    @staticmethod
    def _create_mock_finalize_tool():
        """Create a mock finalize tool to satisfy strict_finalize validation."""
        mock_tool = MagicMock()
        mock_tool.name = "finalize"
        return mock_tool

    @staticmethod
    def _create_mock_localization_agent(
        decomposition_mode: DecompositionMode, strict_finalize: bool = False
    ):
        """Create a LocalizationReActAgent with mocked dependencies."""
        from sakura.nl2test.generation.localization.agent import LocalizationReActAgent

        mock_llm = MagicMock()
        mock_finalize = TestLocalizationForceFinalize._create_mock_finalize_tool()
        mock_tools = [mock_finalize]

        agent = LocalizationReActAgent(
            llm=mock_llm,
            tools=mock_tools,
            system_message="Test system message",
            decomposition_mode=decomposition_mode,
            max_iters=5,
        )
        agent.strict_finalize = strict_finalize
        return agent

    def test_get_finalize_schema_gherkin_returns_scenario_args(self):
        """Verify _get_finalize_schema returns FinalizeScenarioArgs for Gherkin mode."""
        from sakura.nl2test.models.agents import FinalizeScenarioArgs

        agent = self._create_mock_localization_agent(DecompositionMode.GHERKIN)
        schema = agent._get_finalize_schema()

        assert schema is FinalizeScenarioArgs

    def test_get_finalize_schema_grammatical_returns_atomic_block_args(self):
        """Verify _get_finalize_schema returns FinalizeAtomicBlockArgs for Grammatical mode."""
        from sakura.nl2test.models.agents import FinalizeAtomicBlockArgs

        agent = self._create_mock_localization_agent(DecompositionMode.GRAMMATICAL)
        schema = agent._get_finalize_schema()

        assert schema is FinalizeAtomicBlockArgs

    def test_get_force_finalize_system_prompt_gherkin_loads_template(self):
        """Verify _get_force_finalize_system_prompt loads jinja2 template for Gherkin mode."""
        agent = self._create_mock_localization_agent(DecompositionMode.GHERKIN)
        prompt = agent._get_force_finalize_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert "localization" in prompt.lower()
        assert (
            "localized scenario" in prompt.lower()
            or "localized_scenario" in prompt.lower()
        )

    def test_get_force_finalize_chat_prompt_gherkin_loads_template(self):
        """Verify _get_force_finalize_chat_prompt loads jinja2 template for Gherkin mode."""
        agent = self._create_mock_localization_agent(DecompositionMode.GHERKIN)
        prompt = agent._get_force_finalize_chat_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 50
        assert (
            "conversation history" in prompt.lower()
            or "localized scenario" in prompt.lower()
        )

    def test_process_force_finalize_result_gherkin_sets_state(self):
        """Verify _process_force_finalize_result correctly updates state for Gherkin mode."""
        from sakura.nl2test.models.agents import FinalizeScenarioArgs

        agent = self._create_mock_localization_agent(DecompositionMode.GHERKIN)
        state = AgentState()

        localized_scenario = LocalizedScenario(
            setup=[],
            gherkin_groups=[],
            teardown=[],
        )
        result = FinalizeScenarioArgs(
            localized_scenario=localized_scenario,
            comments="Test localization complete",
        )

        agent._process_force_finalize_result(result, state)

        assert state.finalize_called is True
        assert state.final_comments == "Test localization complete"
        assert state.localized_scenario is not None
        assert isinstance(state.localized_scenario, LocalizedScenario)

    def test_process_force_finalize_result_grammatical_sets_state(self):
        """Verify _process_force_finalize_result correctly updates state for Grammatical mode."""
        from sakura.nl2test.models.agents import FinalizeAtomicBlockArgs

        agent = self._create_mock_localization_agent(DecompositionMode.GRAMMATICAL)
        state = AgentState()

        atomic_blocks = AtomicBlockList(atomic_blocks=[])
        result = FinalizeAtomicBlockArgs(
            current_blocks=atomic_blocks,
            comments="Grammatical localization complete",
        )

        agent._process_force_finalize_result(result, state)

        assert state.finalize_called is True
        assert state.final_comments == "Grammatical localization complete"
        assert state.atomic_blocks is not None
        assert isinstance(state.atomic_blocks, AtomicBlockList)

    def test_execute_force_end_with_mocked_llm_gherkin(self):
        """Test _execute_force_end with mocked LLM returning valid structured output."""
        from sakura.nl2test.models.agents import FinalizeScenarioArgs

        agent = self._create_mock_localization_agent(DecompositionMode.GHERKIN)

        localized_scenario = LocalizedScenario(
            setup=[],
            gherkin_groups=[],
            teardown=[],
        )
        mock_result = FinalizeScenarioArgs(
            localized_scenario=localized_scenario,
            comments="Force finalized successfully",
        )

        agent.llm.invoke_structured_with_retries = MagicMock(return_value=mock_result)

        state = AgentState(messages=[])
        result_state = agent._execute_force_end(state)

        assert result_state.finalize_called is True
        assert result_state.final_comments == "Force finalized successfully"
        assert result_state.localized_scenario is not None
        agent.llm.invoke_structured_with_retries.assert_called_once()

    def test_execute_force_end_with_mocked_llm_failure_strict(self):
        """Test _execute_force_end raises exception when LLM fails in strict mode."""
        from sakura.utils.exceptions import ConfigurationException

        agent = self._create_mock_localization_agent(DecompositionMode.GHERKIN)
        agent.strict_finalize = True

        agent.llm.invoke_structured_with_retries = MagicMock(return_value=None)

        state = AgentState(messages=[])

        with pytest.raises(ConfigurationException) as exc_info:
            agent._execute_force_end(state)

        assert "finalize_not_called" in str(exc_info.value.config_var)

    def test_execute_force_end_with_mocked_llm_failure_non_strict(self):
        """Test _execute_force_end auto-finalizes when LLM fails in non-strict mode."""
        agent = self._create_mock_localization_agent(DecompositionMode.GHERKIN)
        agent.strict_finalize = False

        agent.llm.invoke_structured_with_retries = MagicMock(return_value=None)

        state = AgentState(messages=[])
        result_state = agent._execute_force_end(state)

        assert result_state.finalize_called is True
        assert "Auto-finalized" in result_state.final_comments
