from pathlib import Path
from typing import List

from cldk.analysis.java import JavaAnalysis

from sakura.test2nl.evaluation import RoundTripEvaluator
from sakura.test2nl.generation import DescriptionGenerator, RoundTripGenerator
from sakura.test2nl.model.models import (
    AbstractionLevel,
    RoundTripTest,
    Test2NLEntry,
    TestDescriptionInfo,
)
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.file_io import StructuredDataManager, TestFileInfo, TestFileManager
from sakura.utils.models import Method


class Pipeline:
    def __init__(
        self,
        analysis: JavaAnalysis,
        project_name: str,
        output_dir: Path,
        project_root: Path | None = None,
    ):
        self.project_name = project_name
        self.data_manager = StructuredDataManager(output_dir)

        _, application_classes, test_utility_classes = CommonAnalysis(
            analysis
        ).categorize_classes()
        self.desc_generator = DescriptionGenerator(
            analysis, application_classes, test_utility_classes
        )
        self.rt_generator = RoundTripGenerator(analysis)
        if project_root:
            self.rt_evaluator = RoundTripEvaluator(project_root, output_dir)
        else:
            self.rt_evaluator = None

    def _next_description_id(self) -> int:
        """Ensure unique IDs across abstraction runs."""
        try:
            existing = self.data_manager.load("descriptions.json", TestDescriptionInfo)
        except FileNotFoundError:
            return 1

        max_id = 0
        for item in existing:
            try:
                val = int(getattr(item, "id", 0) or 0)
                if val > max_id:
                    max_id = val
            except (TypeError, ValueError):
                continue
        return max_id + 1

    def run_descriptions_of_project(
        self,
        abs_level: AbstractionLevel,
        num_trials: int = 1,
        max_entries: int = 0,
        only_interesting_tests: bool = False,
    ) -> List[TestDescriptionInfo]:
        test_descriptions: List[TestDescriptionInfo] = []

        for trial_num in range(1, num_trials + 1):
            # Calculate remaining entries we can generate
            remaining_entries = (
                max_entries - len(test_descriptions) if max_entries > 0 else 0
            )

            if max_entries > 0 and len(test_descriptions) >= max_entries:
                break

            # Generate descriptions for this trial with the remaining limit
            temp = self.desc_generator.generate(
                abs_level, remaining_entries, only_interesting_tests
            )
            for td in temp:
                td.trial_number = trial_num
            test_descriptions.extend(temp)

        next_id = self._next_description_id()

        test2nl_entries: List[Test2NLEntry] = []
        for td in test_descriptions:
            td.id = next_id
            entry = Test2NLEntry.from_test_description_info(td, self.project_name)
            test2nl_entries.append(entry)
            next_id += 1

        self.data_manager.save(
            "descriptions.json", test_descriptions, format="json", mode="append"
        )
        self.data_manager.save(
            "test2nl.csv", test2nl_entries, format="csv", mode="append"
        )

        return test_descriptions

    def run_roundtrip(self) -> List[RoundTripTest]:
        # DEPRECATED
        test_descriptions = self.data_manager.load(
            "descriptions.json", TestDescriptionInfo
        )
        roundtrip_trials = self.rt_generator.generate(test_descriptions)
        self.data_manager.save("roundtrip_tests.json", roundtrip_trials)
        return roundtrip_trials

    def run_rt_evaluation(self, gen_classes: bool = True):
        # DEPRECATED
        roundtrip_tests = self.data_manager.load("roundtrip_tests.json", RoundTripTest)
        if gen_classes:
            test_file_infos = [
                TestFileInfo.from_roundtrip_test(rt_test) for rt_test in roundtrip_tests
            ]
            TestFileManager(self.rt_evaluator.project_root).save_batch(test_file_infos)

        # Compile and regenerate analysis, then grade
        self.rt_evaluator.reanalyze()
        graded_rt_tests = self.rt_evaluator.grade(roundtrip_tests)

        self.data_manager.save("roundtrip_tests.json", graded_rt_tests)

    def run_descriptions_on_select(
        self, test_methods: List[Method], start_id: int = 0
    ) -> tuple[List[Test2NLEntry], List[TestDescriptionInfo], int]:
        test_descriptions: List[TestDescriptionInfo] = []
        test2nl_entries: List[Test2NLEntry] = []

        next_id = start_id

        for method in test_methods:
            for abstraction in AbstractionLevel:
                test_description = self.desc_generator.generate_for_method(
                    method.method_signature,
                    method.qualified_class_name,
                    abstraction,
                )
                if not test_description:
                    continue

                test_description.id = next_id
                test_descriptions.append(test_description)

                entry = Test2NLEntry.from_test_description_info(
                    test_description, self.project_name
                )
                test2nl_entries.append(entry)
                next_id += 1

        # Do not save; return the generated data and the next available ID
        return test2nl_entries, test_descriptions, next_id

    def run_description_of_method(
        self, select_method: Method, id: int, abstraction: AbstractionLevel
    ) -> tuple[Test2NLEntry | None, TestDescriptionInfo | None]:
        test_description = self.desc_generator.generate_for_method(
            select_method.method_signature,
            select_method.qualified_class_name,
            abstraction,
        )
        if not test_description:
            return None, None

        test_description.id = id

        entry = Test2NLEntry.from_test_description_info(
            test_description, self.project_name
        )
        return entry, test_description
