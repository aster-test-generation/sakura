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
from nltest.nl2test.preprocessing.indexers import MethodIndexer, ClassIndexer
from nltest.utils.pretty.prints import pretty_print
from nltest.utils.llm import UsageTracker

import pytest


class TestSupervisorAgent:
    @pytest.fixture(autouse=True)
    def _inject(self, nl2test_context):
        self.analysis = nl2test_context.analysis
        self.config = nl2test_context.config

    def test_supervisor_end_to_end(self):
        tracker = UsageTracker()
        # Load dataset entries from CSV relative to this test file
        test_dir = Path(__file__).resolve().parent
        data_dir = test_dir / "output" / "resources" / "test2nl"
        sdm = StructuredDataManager(data_dir)
        entries = sdm.load("test2nl.csv", Test2NLEntry, format="csv")
        assert len(entries) > 0, "No Test2NL entries loaded from CSV"

        # Tighten iteration limits for this test
        self.config.set("localization", "max_iters", 40)
        self.config.set("composition", "max_iters", 40)
        self.config.set("supervisor", "max_iters", 10)

        # Pick a random entry and convert to NL2TestInput
        entry = random.choice(entries)
        nl2_input = test2nl_entry_to_nl2test_input(entry)

        # Build project root from config (resources/{project_name})
        project_name = nl2_input.project_name
        base_project_dir = Path(self.config.get("project", "base_project_dir"))
        project_root = base_project_dir / project_name
        assert project_root.exists(), "Project root does not exist"

        # Decompose NL into Gherkin Scenario and wrap as LocalizedScenario
        decomposer = NLDecomposer(mode=DecompositionMode.GHERKIN, usage_tracker=tracker)
        scenario = decomposer.decompose(nl2_input.description)
        localized = LocalizedScenario.from_scenario(scenario)

        # Index database
        method_indexer = MethodIndexer(self.analysis)
        class_indexer = ClassIndexer(self.analysis)
        method_searcher = method_indexer.build_index()
        class_searcher = class_indexer.build_index()

        # Instantiate Supervisor (GHERKIN mode) and run
        supervisor = GherkinSupervisorOrchestrator(
            analysis=self.analysis,
            method_searcher=method_searcher,
            class_searcher=class_searcher,
            nl2_input=nl2_input,
            base_project_dir=str(project_root),
            usage_tracker=tracker,
        )

        supervisor_state, localization_state, composition_state = (
            supervisor.assign_task(localized)
        )

        pretty_print("Updated state", supervisor_state)

        # Validate updated AgentState includes selected package/class
        assert supervisor_state.package is not None
        assert supervisor_state.class_name is not None

        prices = tracker.totals()
        pretty_print("Token usage", prices)
