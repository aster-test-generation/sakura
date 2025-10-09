from pathlib import Path
from typing import List

from cldk import CLDK
from cldk.analysis import AnalysisLevel
from cldk.analysis.java import JavaAnalysis

from .test_grader import TestGrader

from nltest.test2nl.model.models import RoundTripTest
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.execution import JavaCompilation
from nltest.utils.pretty.prints import pretty_print
from nltest.utils.file_io import TestFileManager


class RoundTripEvaluator:
    def __init__(self, project_root: Path, output_dir: Path) -> None:
        self.project_root = project_root
        self.output_dir = output_dir
        self.analysis: JavaAnalysis | None = None

    def reanalyze(self):
        self.analysis = CLDK(language="java").analysis(
            project_path=self.project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=self.output_dir,
            eager=True,
        )
        return self.analysis

    def find_errors(self) -> List[str]:
        return JavaCompilation.get_erroneous_files(self.project_root)

    def grade(self, roundtrip_tests: List[RoundTripTest]):
        analysis = self.analysis
        erroneous_files = self.find_errors()
        pretty_print("Erroneous files", erroneous_files)
        grader = TestGrader(analysis, self.project_root, erroneous_files)

        for rt_test in roundtrip_tests:
            encoded_class = TestFileManager.encode_class_name(rt_test.generated_description.id)
            pkg = rt_test.generated_description.qualified_class_name.rsplit(".", 1)[0]
            qualified_name = f"{pkg}.{encoded_class}"

            if analysis.get_class(qualified_name) is None:
                pretty_print("ERROR LOADING", qualified_name)

            frameworks = CommonAnalysis(analysis).get_testing_frameworks_for_class(qualified_name)
            for method_sig in analysis.get_methods_in_class(qualified_name):
                if CommonAnalysis(analysis).is_test_method(method_sig, qualified_name, frameworks):
                    rt_test.score = grader.grade(
                        pred_method_sig=method_sig,
                        pred_class_name=qualified_name,
                        gt_method_sig=rt_test.method_signature,
                        gt_class_name=rt_test.qualified_class_name,
                    )

        return roundtrip_tests
