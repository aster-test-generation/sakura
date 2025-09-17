from unittest.mock import MagicMock
from pathlib import Path

from nltest.nl2test.generation.localization import (
    GrammaticalLocalizationOrchestrator,
    GherkinLocalizationOrchestrator,
    LocalizationTools,
)
from nltest.nl2test.evaluation.localization_grader_old import LocalizationGraderOld
from nltest.nl2test.evaluation.localization_grader import LocalizationGrader
from nltest.nl2test.models import (
    AtomicBlock,
    NL2TestInput,
    CandidateMethod,
    LocalizedScenario,
)
from nltest.nl2test.models.decomposition import (
    DecompositionMode,
    Scenario,
    GrammaticalBlockList,
    AtomicBlockList,
)
from nltest.nl2test.preprocessing.indexers import MethodIndexer, ClassIndexer
from nltest.nl2test.preprocessing.nl_decomposer import NLDecomposer
from nltest.utils.llm import usage_tracker
from nltest.utils.pretty.prints import pretty_print
from nltest.utils.analysis import CommonAnalysis

from tests._base_nl2test import BaseNL2Test
from nltest.cli import evaluate_localization
import os
from nltest.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat


class TestLocalizationAgent(BaseNL2Test):
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
            f"You must complete within at most {max_iters_parallel} tool invocation(s)."
        )
        self.assertIn(expected_cap_parallel, rendered_parallel)
        self.assertIn("You may parallelize tool calls", rendered_parallel)
        self.assertNotIn("You must call tools sequentially", rendered_parallel)
        # Literal braces should remain intact
        self.assertIn("Never repeat the same {tool, args} pair.", rendered_parallel)

        # Case 2: parallelizable False
        max_iters_sequential = 7
        rendered_sequential = prompt.format(
            parallelizable=False, max_iters=max_iters_sequential
        )
        pretty_print("Not parallelizable", rendered_sequential)
        expected_cap_sequential = f"You must complete within at most {max_iters_sequential} tool invocation(s)."
        self.assertIn(expected_cap_sequential, rendered_sequential)
        self.assertIn("You must call tools sequentially", rendered_sequential)
        self.assertNotIn("You may parallelize tool calls", rendered_sequential)

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
        self.assertIn(nl_description, rendered)
        self.assertIn(instructions, rendered)
        self.assertIn("Given", rendered)
        self.assertIn("When", rendered)
        self.assertIn("Then", rendered)

    def test_localization_agent_simple_grammatical(self):
        nl_description = "Ensure pet is added to owner and ID is generated."
        nl_decomposer = NLDecomposer(mode=DecompositionMode.GRAMMATICAL)
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
        )

        usage_tracker.start()
        refined_blocks, comments = localization_agent.assign_task(
            atomic_blocks, instructions=supervisor_instructions
        )
        prices = usage_tracker.stop()

        pretty_print("Refined atomic blocks", refined_blocks)
        pretty_print("Comments", comments)
        pretty_print("Token usage", prices)

    def test_localization_agent_simple_gherkin(self):
        nl_description = "Ensure pet is added to owner and ID is generated."
        nl_decomposer = NLDecomposer(mode=DecompositionMode.GHERKIN)

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
        )

        usage_tracker.start()
        # Convert Scenario to LocalizedScenario with empty fields
        localized_input = LocalizedScenario.from_scenario(scenario)
        localized_scenario, comments = localization_agent.assign_task(
            localized_input, instructions=supervisor_instructions
        )
        prices = usage_tracker.stop()

        # Assertions and output
        self.assertIsInstance(localized_scenario, LocalizedScenario)
        self.assertIsInstance(comments, str)
        pretty_print("Localized scenario", localized_scenario)
        pretty_print("Comments", comments)
        pretty_print("Token usage", prices)

    def test_localization_agent_complex_gherkin(self):
        qualified_class_name = (
            "org.springframework.samples.petclinic.owner.OwnerControllerTests"
        )
        method_signature = "testProcessCreationFormSuccess()"

        nl_description = 'Create a test case that validates the successful processing of a new owner creation form by the `OwnerController`. The test leverages Spring\'s `@WebMvcTest` with `MockMvc` to simulate HTTP requests and responses. The `OwnerRepository` dependency is mocked using `@MockitoBean`, and its behavior is pre-configured in the `setup` method: a predefined `Owner` (obtained via the `george()` helper method, which constructs and populates an `Owner` instance with associated `Pet` and `PetType` data) is returned when `owners.findByLastNameStartingWith()` is called with any `Pageable` and the last name "Franklin", and the same `Owner` is returned when `owners.findById()` is called with `TEST_OWNER_ID`. The test then performs a POST request to "/owners/new" using `mockMvc.perform()`, simulating form submission with parameters for firstName, lastName, address, city, and telephone. Finally, the test asserts that the HTTP response status is a 3xx redirection using `andExpect(status().is3xxRedirection())`, indicating successful form processing and redirection, using Spring\'s `MockMvcResultMatchers`. JUnit and Mockito are used for the test structure and mocking, respectively.'

        nl_decomposer = NLDecomposer(mode=DecompositionMode.GHERKIN)
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
        )

        usage_tracker.start()
        # Convert Scenario to LocalizedScenario with empty fields
        localized_input = LocalizedScenario.from_scenario(scenario)
        localized_scenario, comments = localization_agent.assign_task(
            localized_input, instructions=supervisor_instructions
        )
        prices = usage_tracker.stop()

        # Assertions and output
        self.assertIsInstance(localized_scenario, LocalizedScenario)
        self.assertIsInstance(comments, str)
        pretty_print("Localized scenario", localized_scenario)
        pretty_print("Comments", comments)
        pretty_print("Token usage", prices)


class TestLocalizationGrader(BaseNL2Test):
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
                    "best_candidate": {
                        "implementing_class_name": "",
                        "containing_class_name": "",
                        "method_signature": "",
                        "return_type": "",
                    },
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
                    "best_candidate": {
                        "implementing_class_name": "",
                        "containing_class_name": "",
                        "method_signature": "",
                        "return_type": "",
                    },
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
                    "best_candidate": {
                        "implementing_class_name": "",
                        "containing_class_name": "",
                        "method_signature": "",
                        "return_type": "java.util.List<org.springframework.samples.petclinic.owner.PetType>",
                    },
                    "arg_bindings": [],
                    "comments": "Mockito stubbing for findPetTypes.",
                    "external": False,
                },
                {
                    "id": 3,
                    "task": "Mock OwnerRepository.findById to return an Owner",
                    "uses": "mocked_owner_repository, TEST_OWNER_ID",
                    "produces": "owner_with_pets",
                    "candidate_methods": [],
                    "best_candidate": {
                        "implementing_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                        "containing_class_name": "org.springframework.samples.petclinic.owner.OwnerRepository",
                        "method_signature": "findById(java.lang.Integer)",
                        "return_type": "java.util.Optional<org.springframework.samples.petclinic.owner.Owner>",
                    },
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
                    "best_candidate": {
                        "implementing_class_name": "",
                        "containing_class_name": "",
                        "method_signature": "",
                        "return_type": "",
                    },
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
                    "best_candidate": {
                        "implementing_class_name": "",
                        "containing_class_name": "",
                        "method_signature": "",
                        "return_type": "",
                    },
                    "arg_bindings": [],
                    "comments": "Setup data.",
                    "external": False,
                },
            ],
            "steps": [
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
                                    "implementing_class_name": "org.springframework.samples.petclinic.owner.PetController",
                                    "containing_class_name": "org.springframework.samples.petclinic.owner.PetController",
                                    "method_signature": "processUpdateForm(org.springframework.samples.petclinic.owner.Owner, org.springframework.samples.petclinic.owner.Pet, org.springframework.validation.BindingResult, org.springframework.web.servlet.mvc.support.RedirectAttributes)",
                                    "return_type": "java.lang.String",
                                }
                            ],
                            "best_candidate": {
                                "implementing_class_name": "org.springframework.samples.petclinic.owner.PetController",
                                "containing_class_name": "org.springframework.samples.petclinic.owner.PetController",
                                "method_signature": "processUpdateForm(org.springframework.samples.petclinic.owner.Owner, org.springframework.samples.petclinic.owner.Pet, org.springframework.validation.BindingResult, org.springframework.web.servlet.mvc.support.RedirectAttributes)",
                                "return_type": "java.lang.String",
                            },
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
                            "best_candidate": {
                                "implementing_class_name": "",
                                "containing_class_name": "",
                                "method_signature": "",
                                "return_type": "",
                            },
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
                            "best_candidate": {
                                "implementing_class_name": "",
                                "containing_class_name": "",
                                "method_signature": "",
                                "return_type": "",
                            },
                            "arg_bindings": [],
                            "comments": "Assertion helper; not a method under test.",
                            "external": False,
                        },
                    ],
                }
            ],
            "teardown": [],
        }

    def test_localization_grader_old(self):
        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.owner.OwnerControllerTests",
            method_signature="testProcessCreationFormSuccess()",
            description="Test that the owner creation form processes successfully when valid data is submitted.",
            project_name="spring-petclinic",
        )

        candidate = CandidateMethod(
            implementing_class_name="org.springframework.samples.petclinic.owner.OwnerController",
            containing_class_name="org.springframework.samples.petclinic.owner.OwnerController",
            method_signature="processCreationForm(Owner, BindingResult, ModelMap)",
            return_type="void",
        )

        atomic_blocks = AtomicBlockList(
            atomic_blocks=[
                AtomicBlock(
                    order=0,
                    subjects=["form"],
                    verbs=["process"],
                    past_participles=[],
                    direct_objs=["creation"],
                    indirect_objs=[],
                    prep_phrases=[],
                    polarity="positive",
                    conditions=[],
                    simplified="form processes creation",
                    candidate_methods=[candidate],
                    best_candidate=candidate,
                    notes="",
                )
            ]
        )

        grader = LocalizationGraderOld(
            nl2_input, self.analysis, self.config.get("project", "base_project_dir")
        )

        # Test basic grading
        coverage_score, detailed_results = grader.grade(
            atomic_blocks, detailed_output=False
        )
        self.assertIsInstance(coverage_score, float)
        self.assertGreaterEqual(coverage_score, 0.0)
        self.assertLessEqual(coverage_score, 1.0)
        self.assertIsNone(detailed_results)

        # Test detailed grading
        coverage_score, detailed_results = grader.grade(
            atomic_blocks, detailed_output=True
        )
        self.assertIsInstance(coverage_score, float)
        self.assertGreaterEqual(coverage_score, 0.0)
        self.assertLessEqual(coverage_score, 1.0)
        self.assertIsNotNone(detailed_results)
        self.assertIsInstance(detailed_results, dict)

        expected_keys = [
            "test_class",
            "test_method",
            "total_focal_methods",
            "covered_focal_methods",
            "uncovered_focal_methods",
            "coverage_score",
            "focal_methods",
            "covered_methods",
            "uncovered_methods",
            "evaluation_algorithm",
            # New metrics
            "tp",
            "fp",
            "fn",
        ]
        for key in expected_keys:
            self.assertIn(key, detailed_results)

        self.assertEqual(
            detailed_results["evaluation_algorithm"], "optimal_coverage_one_to_one"
        )

        # Basic sanity checks on new metrics
        self.assertIsInstance(detailed_results["tp"], int)
        self.assertIsInstance(detailed_results["fp"], int)
        self.assertIsInstance(detailed_results["fn"], int)

    def test_localization_grader_scenario_old(self):
        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.owner.PetControllerTests",
            method_signature="testProcessUpdateFormSuccess()",
            description=(
                "Validate successful processing of a pet update form via PetController."
            ),
            project_name="spring-petclinic",
        )

        localized_scenario_data = self._petcontroller_localized_scenario_payload()

        localized_scenario = LocalizedScenario(**localized_scenario_data)

        grader = LocalizationGraderOld(
            nl2_input, self.analysis, self.config.get("project", "base_project_dir")
        )

        coverage_score, detailed_results = grader.grade(
            localized_scenario, detailed_output=True
        )

        pretty_print("detailed_results", detailed_results)

        self.assertIsInstance(coverage_score, float)
        self.assertGreater(coverage_score, 0.0)
        self.assertIsInstance(detailed_results, dict)

        self.assertGreaterEqual(detailed_results["tp"], 1)
        self.assertEqual(detailed_results["fp"], 1)

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

        self.assertEqual(results.qualified_class_name, nl2_input.qualified_class_name)
        self.assertEqual(results.method_signature, nl2_input.method_signature)
        self.assertGreaterEqual(results.tp, 1)
        self.assertGreaterEqual(results.localization_recall, 0.0)
        self.assertLessEqual(results.localization_recall, 1.0)
        self.assertEqual(
            len(results.covered_focal_methods) + len(results.uncovered_focal_methods),
            len(results.all_focal_methods),
        )


class TestLocalizationTools(BaseNL2Test):
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
