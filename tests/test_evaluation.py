from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from dotenv import load_dotenv

from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.utils.analysis.common_analysis import CommonAnalysis
from nltest.utils.evaluation import TestGrader
from nltest.utils.compilation.compilation_old import JavaCompilation
from nltest.utils.models import NL2TestInput, NL2TestMetadata
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

    def test_structural_grading_old(self):
        grader = TestGraderOld(self.analysis, self.project_root)

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

    def test_test_grader_petclinic(self):
        project_name = "spring-petclinic"
        test_dir = Path(__file__).resolve().parent
        resources_dir = test_dir / "resources"
        output_base_dir = test_dir / "output"
        project_root = resources_dir / project_name
        output_dir = output_base_dir / project_name

        self.assertTrue(project_root.exists() and project_root.is_dir())

        analysis = CLDK(language="java").analysis(
            project_path=project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=output_dir,
            eager=False,
        )

        qualified_class_name = "org.springframework.samples.petclinic.service.ClinicServiceTests"
        method_signature = "shouldUpdateOwner()"

        self.assertIsNotNone(self.analysis.get_class(qualified_class_name))
        self.assertIsNotNone(self.analysis.get_method(qualified_class_name, method_signature))

        common = CommonAnalysis(analysis)
        _, application_classes = (
            common.get_test_methods_classes_and_application_classes()
        )

        erroneous_files = JavaCompilation.get_erroneous_files(project_root)

        grader = TestGrader(
            analysis=analysis,
            project_root=project_root,
            project_erroneous_files=erroneous_files,
            application_classes=application_classes,
        )

        nl2_input = NL2TestInput(
            description="",
            project_name=project_name,
            qualified_class_name=qualified_class_name,
            method_signature=method_signature,
        )

        nl2_metadata = NL2TestMetadata(
            qualified_test_class_name=qualified_class_name,
            method_signature=method_signature,
            code="",
        )

        output = grader.grade(nl2_input, nl2_metadata)

        pretty_print("Grading Output", output)

        self.assertEqual(output.nl2test_input, nl2_input)
        self.assertEqual(output.nl2test_metadata, nl2_metadata)

        self.assertEqual(output.structured_eval.assertion_recall, 1.0)
        self.assertEqual(output.structured_eval.obj_creation_recall, 1.0)
        self.assertEqual(output.structured_eval.assertion_precision, 1.0)
        self.assertEqual(output.structured_eval.obj_creation_precision, 1.0)
        self.assertGreaterEqual(output.structured_eval.callable_recall, 0.0)
        self.assertLessEqual(output.structured_eval.callable_recall, 1.0)
        self.assertGreaterEqual(output.structured_eval.callable_precision, 0.0)
        self.assertLessEqual(output.structured_eval.callable_precision, 1.0)
        self.assertGreaterEqual(output.structured_eval.focal_recall, 0.0)
        self.assertLessEqual(output.structured_eval.focal_recall, 1.0)
        self.assertGreaterEqual(output.structured_eval.focal_precision, 0.0)
        self.assertLessEqual(output.structured_eval.focal_precision, 1.0)

        cov = output.coverage_eval
        self.assertGreaterEqual(cov.class_coverage, 0.0)
        self.assertLessEqual(cov.class_coverage, 1.0)
        self.assertGreaterEqual(cov.method_coverage, 0.0)
        self.assertLessEqual(cov.method_coverage, 1.0)
        self.assertGreaterEqual(cov.line_coverage, 0.0)
        self.assertLessEqual(cov.line_coverage, 1.0)
        self.assertGreaterEqual(cov.branch_coverage, 0.0)
        self.assertLessEqual(cov.branch_coverage, 1.0)
