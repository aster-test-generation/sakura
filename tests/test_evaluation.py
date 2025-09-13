from __future__ import annotations

import os
from pathlib import Path
from unittest import TestCase

from dotenv import load_dotenv

from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.utils.analysis.common_analysis import CommonAnalysis
from nltest.utils.evaluation.test_grader import TestGrader
from nltest.utils.pretty.prints import pretty_print


class TestEvaluation(TestCase):
    def setUp(self) -> None:
        load_dotenv()

        project_name = "spring-petclinic"
        test_dir = Path(__file__).resolve().parent
        resources_dir = test_dir / "resources"
        output_base_dir = test_dir / "output"

        project_root = resources_dir / project_name
        if not (project_root.exists() and project_root.is_dir()):
            raise Exception(
                f"Project root directory {project_root.resolve()} does not exist."
            )
        output_dir = output_base_dir / project_name
        output_dir.mkdir(parents=True, exist_ok=True)

        self.project_root = project_root
        self.analysis = CLDK(language="java").analysis(
            project_path=project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=output_dir,
            eager=True,
        )

    def test_structural_grading_pettype_formatter(self):
        grader = TestGrader(self.analysis, self.project_root)

        # Ground truth
        gt_class = "org.springframework.samples.petclinic.owner.PetTypeFormatterTests"
        gt_method = "testPrint()"

        # Prediction
        pred_class = (
            "org.springframework.samples.petclinic.owner.PetTypeFormatterTests_New"
        )
        pred_method = "testPrintPetType()"

        score, metrics = grader.grade_structural(
            pred_method_sig=pred_method,
            pred_class_name=pred_class,
            gt_method_sig=gt_method,
            gt_class_name=gt_class,
        )

        pretty_print("Score", score)
        pretty_print("Metrics", metrics)

        self.assertIsInstance(score, float)
        self.assertIsInstance(metrics, dict)

        expected_keys = {
            "objects_created",
            "constructors",
            "application_calls",
            "library_calls",
            "assertions",
            "order_aware_assertion_coverage",
            "ground_truth_assertion_coverage",
            "longest_callable_subsequence",
            "ground_truth_callable_coverage",
            "compilation_score",
            "focal_method_coverage",
        }
        self.assertTrue(expected_keys.issubset(set(metrics.keys())))

        for k in expected_keys:
            v = metrics[k]
            self.assertIsInstance(v, float)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
