from pathlib import Path

from nltest.nl2test.models import (
    AtomicBlock,
    AtomicBlockList,
    NL2LocalizationOutput,
    NL2TestInput,
    LocalizedScenario,
    AbstractionLevel as NL2AbstractionLevel,
    LocalizationEvaluationResults,
)
from nltest.nl2test.models.decomposition import (
    DecompositionMode,
    GrammaticalBlockList,
)
from nltest.nl2test.pipeline import Pipeline
from nltest.utils.llm import usage_tracker
from nltest.utils.pretty.prints import pretty_print

from nltest.test2nl.model.models import AbstractionLevel

from tests._base_nl2test import BaseNL2Test
from tests._base_test2nl import BaseTest2NL


class TestPipelineDescriptions(BaseTest2NL):
    def test_all_low_abs(self):
        self.pipeline.reset_dataset()
        self.pipeline.run_descriptions_of_project(AbstractionLevel.LOW)


class TestPipelineLocalization(BaseNL2Test):
    def test_pipeline_run_localization_agent(self):
        """Test the pipeline's run_localization_agent method with a petclinic-based test case."""

        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.service.ClinicServiceTests",
            method_signature="shouldInsertPetIntoDatabaseAndGenerateId()",
            description="Ensure that the test validates the persistence and ID generation of a new pet associated with an existing owner by first retrieving an owner entity from the database using the OwnerRepository's findById method, confirming the owner exists, and capturing the initial count of pets. Next, instantiate a new Pet, configure its properties including name, birth date, and type (retrieved via the OwnerRepository's findPetTypes method and selected using EntityUtils), and associate it with the retrieved owner via the owner's addPet method, which only adds the pet if it is new. Verify the pet count increases by one on the owner instance, then persist the updated owner using the OwnerRepository's save method. Re-fetch the same owner from the database to confirm the pet count remains incremented, and finally assert that the newly added pet now has a non-null ID, confirming successful persistence and ID generation. The test uses JUnit 5 for execution and AssertJ for fluent assertions, interacting directly with the real OwnerRepository and Pet-related entities without mocking.",
            project_name="spring-petclinic",
            abstraction_level="low",
        )

        # Initialize pipeline
        test_dir = Path(__file__).resolve().parent
        project_root = (test_dir / "resources" / "spring-petclinic").resolve()
        pipeline = Pipeline(self.analysis, project_root)

        # Run preprocessing
        method_searcher, class_searcher = pipeline.run_preprocessing()
        self.assertIsNotNone(method_searcher)
        self.assertIsNotNone(class_searcher)

        # Decompose natural language (Pipeline returns GrammaticalBlockList in grammatical mode)
        blocks = pipeline.decompose_natural_language(nl2_input.description)
        self.assertIsInstance(blocks, GrammaticalBlockList)
        self.assertGreater(len(blocks.grammatical_blocks), 0)

        # Run localization agent directly with decomposed blocks
        usage_tracker.start()
        localized_blocks, comments = pipeline.run_localization_agent(nl2_input, blocks)
        prices = usage_tracker.stop()

        # Verify outputs
        self.assertIsInstance(localized_blocks, AtomicBlockList)
        self.assertIsInstance(comments, str)
        self.assertEqual(
            len(localized_blocks.atomic_blocks), len(blocks.grammatical_blocks)
        )

        # Verify that atomic blocks have been enhanced with candidate methods
        for i, block in enumerate(localized_blocks.atomic_blocks):
            self.assertIsInstance(block, AtomicBlock)
            self.assertEqual(block.order, i)
            self.assertIsInstance(block.candidate_methods, list)

        pretty_print("Localized blocks", localized_blocks)
        pretty_print("Comments", comments)
        pretty_print("Token usage", prices)

    def test_pipeline_run_localization_evaluation_pipeline_gherkin(self):
        """Test the complete run_localization_evaluation_pipeline method with a petclinic-based test case."""

        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.vet.VetControllerTests",
            method_signature="testShowResourcesVetList()",
            description="Ensure that the veterinarian controller correctly renders a JSON list of veterinarians by first configuring the VetRepository dependency as a Mockito mock during test setup to return two predefined veterinarian objects for both its unpaginated findAll method and its paginated findAll method (when invoked with any Pageable argument), then using the auto-wired MockMvc instance to perform an HTTP GET request to the /vets endpoint with an Accept header specifying JSON media type, and verifying that the response returns a 200 OK status, confirms the content type as JSON, and validates through JsonPath assertions that the first element in the vetList array of the response body contains an identifier matching the expected value for the initial veterinarian instance, all implemented using JUnit 5 for test lifecycle management, Spring Boot Test's @WebMvcTest for web layer testing configuration, Mockito for repository behavior stubbing via @MockitoBean, and Spring MVC Test's MockMvc framework for request execution and response validation with its built-in status, content, and jsonPath matchers.",
            project_name="spring-petclinic",
            abstraction_level=NL2AbstractionLevel("low"),
        )

        # Initialize pipeline
        test_dir = Path(__file__).resolve().parent
        project_root = (test_dir / "resources" / "spring-petclinic").resolve()
        pipeline = Pipeline(
            self.analysis, project_root, decomposition_mode=DecompositionMode.GHERKIN
        )

        # Run the complete evaluation pipeline
        usage_tracker.start()
        output = pipeline.run_localization_evaluation_pipeline(nl2_input)
        prices = usage_tracker.stop()

        # Verify output container and fields
        self.assertIsInstance(output, NL2LocalizationOutput)
        self.assertIsInstance(output.localized_blocks, LocalizedScenario)
        self.assertIsNotNone(output.evaluation_results)
        self.assertIsInstance(
            output.evaluation_results, LocalizationEvaluationResults
        )
        # coverage_score now lives inside evaluation_results
        self.assertIsInstance(output.evaluation_results.coverage_score, float)
        self.assertGreaterEqual(output.evaluation_results.coverage_score, 0.0)
        self.assertLessEqual(output.evaluation_results.coverage_score, 1.0)

        pretty_print(
            "Localized blocks from evaluation pipeline", output.localized_blocks
        )
        pretty_print("Coverage score", output.evaluation_results.coverage_score)
        pretty_print("Token usage", prices)
