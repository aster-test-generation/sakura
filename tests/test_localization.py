from unittest.mock import MagicMock
from pathlib import Path

from nltest.nl2test.generation.localization import (
    GrammaticalLocalizationOrchestrator,
    GherkinLocalizationOrchestrator,
    LocalizationTools,
)
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

from tests._base_nl2test import BaseNL2Test


class TestLocalizationAgent(BaseNL2Test):
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
    def test_localization_grader(self):
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

        grader = LocalizationGrader(
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
