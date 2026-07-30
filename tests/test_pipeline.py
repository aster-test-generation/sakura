from pathlib import Path
import random

import pytest

from sakura.nl2test.models.decomposition import DecompositionMode
from sakura.nl2test.pipeline import Pipeline as NL2TestPipeline
from sakura.test2nl.pipeline import Pipeline as Test2NLPipeline
from sakura.utils.file_io.structured_data_manager import StructuredDataManager
from sakura.utils.pretty.prints import pretty_print

from sakura.test2nl.model.models import AbstractionLevel, Test2NLEntry

from sakura.utils.utilities import test2nl_entry_to_nl2test_input


class TestTest2NLPipeline:
    @pytest.fixture(autouse=True)
    def _inject(self, petclinic_analysis, petclinic_paths):
        self.analysis = petclinic_analysis
        self.project_name = petclinic_paths.project_name
        self.project_root = petclinic_paths.project_root
        self.output_dir = petclinic_paths.project_output_dir
        
        # Initialize pipeline directly
        self.pipeline = Test2NLPipeline(
            self.analysis,
            self.project_name,
            self.output_dir,
            self.project_root,
        )

    def test_all_low_abs(self):
        self.pipeline.reset_dataset()
        self.pipeline.run_descriptions_of_project(AbstractionLevel.LOW)


class TestNL2TestPipeline:
    @pytest.fixture(autouse=True)
    def _inject(self, petclinic_analysis, petclinic_config, petclinic_paths):
        self.analysis = petclinic_analysis
        self.config = petclinic_config
        self.project_root = petclinic_paths.project_root
        self.project_name = petclinic_paths.project_name
        self.output_dir = petclinic_paths.project_output_dir

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
            self.analysis, project_root=self.project_root, analysis_dir=self.output_dir,
            decomposition_mode=DecompositionMode.GHERKIN
        )

        # Tighten iteration limits for this test
        self.config.set("localization", "max_iters", 12) # Best so far is 12
        self.config.set("composition", "max_iters", 16) # 16
        self.config.set("supervisor", "max_iters", 8) # 8

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
