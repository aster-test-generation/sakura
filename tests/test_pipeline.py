from pathlib import Path
import random

import pytest

from nltest.nl2test.models import (
    AtomicBlock,
    AtomicBlockList,
    NL2LocalizationOutput,
    NL2TestEval,
    NL2TestInput,
    LocalizedScenario,
    AbstractionLevel as NL2AbstractionLevel,
    LocalizationEval,
)
from nltest.nl2test.models.decomposition import (
    DecompositionMode,
    GrammaticalBlockList,
)
from nltest.nl2test.pipeline import Pipeline as NL2TestPipeline
from nltest.test2nl.pipeline import Pipeline as Test2NLPipeline
from nltest.utils.file_io.structured_data_manager import StructuredDataManager
from nltest.utils.pretty.prints import pretty_print

from nltest.test2nl.model.models import AbstractionLevel, Test2NLEntry

from nltest.utils.utilities import test2nl_entry_to_nl2test_input


class TestTest2NLPipeline:
    @pytest.fixture(autouse=True)
    def _inject(self, test2nl_context):
        self.pipeline = test2nl_context.pipeline
        self.analysis = test2nl_context.analysis
        self.data_manager = test2nl_context.data_manager
        self.project_name = test2nl_context.project_name
        self.project_root = test2nl_context.project_root
        self.output_dir = test2nl_context.output_dir
    def test_all_low_abs(self):
        self.pipeline.reset_dataset()
        self.pipeline.run_descriptions_of_project(AbstractionLevel.LOW)


class TestNL2TestPipeline:
    @pytest.fixture(autouse=True)
    def _inject(self, nl2test_context):
        self.analysis = nl2test_context.analysis
        self.config = nl2test_context.config
        self.project_root = nl2test_context.project_root
        self.project_name = nl2test_context.project_name
        self.output_dir = nl2test_context.output_dir

    def test_pipeline_run_localization_pipeline_grammatical(self):
        """Test the pipeline's run_localization_agent method with a petclinic-based test case."""

        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.service.ClinicServiceTests",
            method_signature="shouldInsertPetIntoDatabaseAndGenerateId()",
            description="Ensure that the test validates the persistence and ID generation of a new pet associated with an existing owner by first retrieving an owner entity from the database using the OwnerRepository's findById method, confirming the owner exists, and capturing the initial count of pets. Next, instantiate a new Pet, configure its properties including name, birth date, and type (retrieved via the OwnerRepository's findPetTypes method and selected using EntityUtils), and associate it with the retrieved owner via the owner's addPet method, which only adds the pet if it is new. Verify the pet count increases by one on the owner instance, then persist the updated owner using the OwnerRepository's save method. Re-fetch the same owner from the database to confirm the pet count remains incremented, and finally assert that the newly added pet now has a non-null ID, confirming successful persistence and ID generation. The test uses JUnit 5 for execution and AssertJ for fluent assertions, interacting directly with the real OwnerRepository and Pet-related entities without mocking.",
            project_name="spring-petclinic",
            abstraction_level="low",
        )

        pipeline = Test2NLPipeline(self.analysis, self.project_root)

        # Run preprocessing
        method_searcher, class_searcher = pipeline.run_preprocessing()
        assert method_searcher is not None
        assert class_searcher is not None

        # Decompose natural language (Pipeline returns GrammaticalBlockList in grammatical mode)
        blocks = pipeline.decompose_natural_language(nl2_input.description)
        assert isinstance(blocks, GrammaticalBlockList)
        assert len(blocks.grammatical_blocks) > 0

        # Run localization agent directly with decomposed blocks
        localized_blocks, comments = pipeline.run_localization_agent(nl2_input, blocks)
        prices = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

        # Verify outputs
        assert isinstance(localized_blocks, AtomicBlockList)
        assert isinstance(comments, str)
        assert len(localized_blocks.atomic_blocks) == len(
            blocks.grammatical_blocks
        )

        # Verify that atomic blocks have been enhanced with candidate methods
        for i, block in enumerate(localized_blocks.atomic_blocks):
            assert isinstance(block, AtomicBlock)
            assert block.order == i
            assert isinstance(block.candidate_methods, list)

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

        pipeline = NL2TestPipeline(
            self.analysis, self.project_root, decomposition_mode=DecompositionMode.GHERKIN
        )

        # Tighten iteration limits for this test
        self.config.set("localization", "max_iters", 10)
        self.config.set("composition", "max_iters", 6)
        self.config.set("supervisor", "max_iters", 4)

        # Run the complete evaluation pipeline
        output = pipeline.run_localization_evaluation_pipeline(nl2_input)
        prices = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

        # Verify output container and fields
        assert isinstance(output, NL2LocalizationOutput)
        assert isinstance(output.localized_blocks, LocalizedScenario)
        assert output.evaluation_results is not None
        assert isinstance(output.evaluation_results, LocalizationEval)
        assert isinstance(output.evaluation_results.localization_recall, float)
        assert 0.0 <= output.evaluation_results.localization_recall <= 1.0

        pretty_print(
            "Localized blocks from evaluation pipeline", output.localized_blocks
        )
        pretty_print(
            "Localization recall",
            output.evaluation_results.localization_recall,
        )
        pretty_print("Token usage", prices)

    def test_pipeline_run_nl2test_simple_gherkin(self):
        nl2_input = NL2TestInput(
            qualified_class_name="org.springframework.samples.petclinic.vet.VetControllerTests",
            method_signature="testShowResourcesVetList()",
            description="Ensure that the veterinarian controller correctly renders a JSON list of veterinarians by first configuring the VetRepository dependency as a Mockito mock during test setup to return two predefined veterinarian objects for both its unpaginated findAll method and its paginated findAll method (when invoked with any Pageable argument), then using the auto-wired MockMvc instance to perform an HTTP GET request to the /vets endpoint with an Accept header specifying JSON media type, and verifying that the response returns a 200 OK status, confirms the content type as JSON, and validates through JsonPath assertions that the first element in the vetList array of the response body contains an identifier matching the expected value for the initial veterinarian instance, all implemented using JUnit 5 for test lifecycle management, Spring Boot Test's @WebMvcTest for web layer testing configuration, Mockito for repository behavior stubbing via @MockitoBean, and Spring MVC Test's MockMvc framework for request execution and response validation with its built-in status, content, and jsonPath matchers.",
            project_name="spring-petclinic",
            abstraction_level=NL2AbstractionLevel("low"),
        )

        pipeline = NL2TestPipeline(
            self.analysis, self.project_root, decomposition_mode=DecompositionMode.GHERKIN
        )
        pipeline.run_preprocessing()

        # Tighten iteration limits for this test
        self.config.set("localization", "max_iters", 10)
        self.config.set("composition", "max_iters", 6)
        self.config.set("supervisor", "max_iters", 4)

        eval_result = pipeline.run_nl2test(nl2_input)
        prices = {
            "input_tokens": eval_result.input_tokens,
            "output_tokens": eval_result.output_tokens,
            "calls": eval_result.llm_calls,
        }

        assert isinstance(eval_result, NL2TestEval)
        assert eval_result.nl2test_input == nl2_input
        assert isinstance(eval_result.compiles, bool)
        assert isinstance(
            eval_result.nl2test_metadata.qualified_test_class_name, str
        )

        assert eval_result.localization_eval is not None
        assert isinstance(eval_result.localization_eval, LocalizationEval)
        assert eval_result.tool_log is not None
        assert isinstance(
            eval_result.tool_log.supervisor_tool_log.tool_counts, dict
        )

        pretty_print("NL2Test evaluation result", eval_result)
        pretty_print("Token usage", prices)

    def test_pipeline_run_nl2test_random_gherkin(self):
        # Load dataset entries from CSV using the same path pattern
        test_dir = Path(__file__).resolve().parent
        data_dir = test_dir / "output" / "resources" / "test2nl"
        sdm = StructuredDataManager(data_dir)
        entries = sdm.load("test2nl.csv", Test2NLEntry, format="csv")
        assert len(entries) > 0, "No Test2NL entries loaded from CSV"

        # Pick a random entry and convert to NL2TestInput
        entry = random.choice(entries)
        nl2_input = test2nl_entry_to_nl2test_input(entry)

        # Build pipeline
        project_name = nl2_input.project_name
        base_project_dir = Path(self.config.get("project", "base_project_dir"))
        project_root = base_project_dir / project_name
        output_dir = Path(self.config.get("project", "project_output_dir"))

        pipeline = NL2TestPipeline(
            self.analysis,
            project_root,
            decomposition_mode=DecompositionMode.GHERKIN,
            analysis_dir=output_dir,
        )

        # Tighten iteration limits for this test
        self.config.set("localization", "max_iters", 5)
        self.config.set("composition", "max_iters", 10)
        self.config.set("supervisor", "max_iters", 5)

        # Run end-to-end NL2Test
        pipeline.run_preprocessing()
        result = pipeline.run_nl2test(nl2_input)

        # Pretty print results
        pretty_print("NL2 test evaluation results", result)

        # Basic sanity assertions for NL2TestOutput
        assert result is not None
        assert result.nl2test_input == nl2_input
        assert isinstance(result.compiles, bool)
        assert isinstance(result.nl2test_metadata.qualified_test_class_name, str)
        assert isinstance(result.nl2test_metadata.method_signature, str)
        # Structured eval fields
        se = result.structured_eval
        assert 0.0 <= se.assertion_recall <= 1.0
        assert 0.0 <= se.assertion_precision <= 1.0
        assert 0.0 <= se.obj_creation_recall <= 1.0
        assert 0.0 <= se.obj_creation_precision <= 1.0
        assert 0.0 <= se.callable_recall <= 1.0
        assert 0.0 <= se.callable_precision <= 1.0
        assert 0.0 <= se.focal_recall <= 1.0
        assert 0.0 <= se.focal_precision <= 1.0
        # Coverage eval fields are in percent [0, 100]
        cv = result.coverage_eval
        assert 0.0 <= cv.class_coverage <= 100.0
        assert 0.0 <= cv.method_coverage <= 100.0
        assert 0.0 <= cv.line_coverage <= 100.0
        assert 0.0 <= cv.branch_coverage <= 100.0
