from __future__ import annotations

import random
from pathlib import Path

from nltest.utils.file_io.structured_data_manager import StructuredDataManager
from nltest.test2nl.model.models import Test2NLEntry
from nltest.utils.utilities import test2nl_entry_to_nl2test_input
from nltest.nl2test.preprocessing.nl_decomposer import NLDecomposer
from nltest.nl2test.models.decomposition import DecompositionMode, LocalizedScenario
from nltest.nl2test.generation.supervisor.orchestrators.gherkin import (
    GherkinSupervisorOrchestrator,
)
from nltest.nl2test.pipeline import Pipeline as NL2Pipeline
from nltest.utils.pretty.prints import pretty_print

from tests._base_nl2test import BaseNL2Test


class TestSupervisorAgent(BaseNL2Test):
    def test_supervisor_end_to_end(self):
        # Load dataset entries from CSV relative to this test file
        test_dir = Path(__file__).resolve().parent
        data_dir = test_dir / "output" / "resources" / "test2nl"
        sdm = StructuredDataManager(data_dir)
        entries = sdm.load("test2nl.csv", Test2NLEntry, format="csv")
        self.assertTrue(len(entries) > 0, "No Test2NL entries loaded from CSV")

        # Pick a random entry and convert to NL2TestInput
        entry = random.choice(entries)
        nl2_input = test2nl_entry_to_nl2test_input(entry)

        # Build project root from config (resources/{project_name})
        project_name = nl2_input.project_name
        base_project_dir = Path(self.config.get("project", "base_project_dir"))
        project_root = base_project_dir / project_name
        self.assertTrue(project_root.exists(), "Project root does not exist")

        # Decompose NL into Gherkin Scenario and wrap as LocalizedScenario
        decomposer = NLDecomposer(mode=DecompositionMode.GHERKIN)
        scenario = decomposer.decompose(nl2_input.description)
        localized = LocalizedScenario.from_scenario(scenario)

        # Instantiate Supervisor (GHERKIN mode) and run
        supervisor = GherkinSupervisorOrchestrator(
            analysis=self.analysis,
            nl2_input=nl2_input,
            base_project_dir=str(project_root),
        )

        instructions = (
            "Use delegate agents to localize and compose the test. Finalize when done."
        )
        updated_state = supervisor.assign_task(localized, instructions=instructions)

        # Validate updated AgentState includes selected package/class
        self.assertIsNotNone(updated_state.package)
        self.assertIsNotNone(updated_state.class_name)

    def test_pipeline_run_nl2test(self):
        # Load dataset entries from CSV using the same path pattern
        test_dir = Path(__file__).resolve().parent
        data_dir = test_dir / "output" / "resources" / "test2nl"
        sdm = StructuredDataManager(data_dir)
        entries = sdm.load("test2nl.csv", Test2NLEntry, format="csv")
        self.assertTrue(len(entries) > 0, "No Test2NL entries loaded from CSV")

        # Pick a random entry and convert to NL2TestInput
        entry = random.choice(entries)
        nl2_input = test2nl_entry_to_nl2test_input(entry)

        # Build pipeline
        project_name = nl2_input.project_name
        base_project_dir = Path(self.config.get("project", "base_project_dir"))
        project_root = base_project_dir / project_name
        output_dir = Path(self.config.get("project", "output_dir"))

        pipeline = NL2Pipeline(
            self.analysis,
            project_root,
            decomposition_mode=DecompositionMode.GHERKIN,
            analysis_dir=output_dir,
        )

        # Tighten iteration limits for this test
        self.config.set("localization", "max_iters", 10)
        self.config.set("composition", "max_iters", 6)
        self.config.set("supervisor", "max_iters", 3)

        # Run end-to-end NL2Test
        result = pipeline.run_nl2test(nl2_input)

        # Pretty print results
        pretty_print("NL2 test evaluation results", result)

        # Basic sanity assertions (non-None fields)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.gt_class_name)
        self.assertIsNotNone(result.gt_method_signature)
        self.assertIsNotNone(result.pred_class_name)
        self.assertIsNotNone(result.pred_method_signature)
        self.assertIsNotNone(result.structural_metrics)
