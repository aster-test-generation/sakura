from pathlib import Path
from typing import List

from cldk.analysis.java import JavaAnalysis

from nltest.utils.file_io import StructuredDataManager, TestFileManager, TestFileInfo
from nltest.test2nl.evaluation import RoundTripEvaluator
from nltest.test2nl.generation import DescriptionGenerator, RoundTripGenerator
from nltest.test2nl.model.models import AbstractionLevel, TestDescriptionInfo, RoundTripTest, Test2NLEntry


class Pipeline:
    def __init__(self, analysis: JavaAnalysis, project_root: Path, output_dir: Path):
        self.project_name = project_root.name
        self.data_manager = StructuredDataManager(output_dir)
        self.desc_generator = DescriptionGenerator(analysis)
        self.rt_generator = RoundTripGenerator(analysis)
        self.rt_evaluator = RoundTripEvaluator(project_root, output_dir)

    def run_descriptions(self, abs_level: AbstractionLevel, num_trials: int = 1) -> List[TestDescriptionInfo]:
        test_descriptions = []
        for trial_num in range(1, num_trials + 1, 1):
            temp_test_descriptions = self.desc_generator.generate(abs_level)
            for test_description_info in temp_test_descriptions:
                test_description_info.trial_number = trial_num
            test_descriptions.extend(temp_test_descriptions)

        test2nl_entries: List[Test2NLEntry] = []
        entry_id = 1
        for test_description_info in test_descriptions:
            test_description_info.entry_id = entry_id
            entry = Test2NLEntry.from_test_description_info(test_description_info, self.project_name)
            test2nl_entries.append(entry)
            entry_id += 1

        self.data_manager.save("description_trials.json", test_descriptions, format="json")
        self.data_manager.save("test2nl.csv", test2nl_entries, format="csv")

        return test_descriptions

    def run_roundtrip(self) -> List[RoundTripTest]:
        test_descriptions = self.data_manager.load("descriptions.json", TestDescriptionInfo)
        roundtrip_trials = self.rt_generator.generate(test_descriptions)
        self.data_manager.save("roundtrip_tests.json", roundtrip_trials)
        return roundtrip_trials

    def run_rt_evaluation(self, gen_classes: bool = True):
        roundtrip_tests = self.data_manager.load("roundtrip_tests.json", RoundTripTest)
        if gen_classes:
            test_file_infos = [TestFileInfo.from_roundtrip_test(rt_test) for rt_test in roundtrip_tests]
            TestFileManager(self.rt_evaluator.project_root).save_batch(test_file_infos)

        # Compile and regenerate analysis, then grade
        self.rt_evaluator.reanalyze()
        graded_rt_tests = self.rt_evaluator.grade(roundtrip_tests)

        self.data_manager.save("roundtrip_tests.json", graded_rt_tests)

    def run_all(self, abstraction_level: AbstractionLevel, regen_classes: bool = True):
        self.run_descriptions(abstraction_level)
        self.run_roundtrip()
        self.run_rt_evaluation(gen_classes=regen_classes)
