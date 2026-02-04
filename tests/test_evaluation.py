from __future__ import annotations

from pathlib import Path

import pytest
from cldk import CLDK
from cldk.analysis import AnalysisLevel

from sakura.utils.analysis.java_analyzer import CommonAnalysis
from sakura.utils.compilation.compilation_old import JavaCompilation
from sakura.utils.evaluation import TestGrader
from sakura.utils.models import NL2TestInput, NL2TestMetadata
from sakura.utils.pretty.prints import pretty_print


@pytest.fixture(scope="session")
def eager_analysis(petclinic_paths):
    return CLDK(language="java").analysis(
        project_path=petclinic_paths.project_root,
        analysis_backend_path=None,
        analysis_level=AnalysisLevel.symbol_table,
        analysis_json_path=petclinic_paths.project_output_dir,
        eager=True,
    )


def _build_grader(analysis, project_root):
    common = CommonAnalysis(analysis)
    _, application_classes, test_utility_classes = common.categorize_classes()
    erroneous_files = JavaCompilation.get_erroneous_files(project_root)
    return TestGrader(
        analysis=analysis,
        project_root=project_root,
        project_erroneous_files=erroneous_files,
        application_classes=application_classes,
        test_utility_classes=test_utility_classes,
    )


def test_structural_grading(eager_analysis, petclinic_paths):
    grader = _build_grader(eager_analysis, petclinic_paths.project_root)

    gt_class = "org.springframework.samples.petclinic.owner.PetTypeFormatterTests"
    gt_method = "testPrint()"
    pred_class = "org.springframework.samples.petclinic.owner.PetTypeFormatterTests_New"
    pred_method = "testPrintPetType()"

    structural_eval = grader.grade_structural(
        pred_method_sig=pred_method,
        pred_class_name=pred_class,
        gt_method_sig=gt_method,
        gt_class_name=gt_class,
    )

    pretty_print("Structural eval", structural_eval)

    assert structural_eval is not None
    numeric_fields = [
        structural_eval.obj_creation_recall,
        structural_eval.obj_creation_precision,
        structural_eval.assertion_recall,
        structural_eval.assertion_precision,
        structural_eval.callable_recall,
        structural_eval.callable_precision,
        structural_eval.focal_recall,
        structural_eval.focal_precision,
    ]
    for value in numeric_fields:
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


def test_test_grader_petclinic(petclinic_paths):
    analysis = CLDK(language="java").analysis(
        project_path=petclinic_paths.project_root,
        analysis_backend_path=None,
        analysis_level=AnalysisLevel.symbol_table,
        analysis_json_path=petclinic_paths.project_output_dir,
        eager=False,
    )

    qualified_class_name = (
        "org.springframework.samples.petclinic.service.ClinicServiceTests"
    )
    method_signature = "shouldUpdateOwner()"

    second_qualified_class_name = (
        "org.springframework.samples.petclinic.owner.OwnerControllerTests"
    )
    second_method_signature = "testProcessUpdateOwnerFormHasErrors()"

    assert analysis.get_class(qualified_class_name) is not None
    assert analysis.get_method(qualified_class_name, method_signature) is not None
    assert analysis.get_class(second_qualified_class_name) is not None
    assert (
        analysis.get_method(second_qualified_class_name, second_method_signature)
        is not None
    )

    grader = _build_grader(analysis, petclinic_paths.project_root)

    nl2_input = NL2TestInput(
        description="",
        project_name=petclinic_paths.project_name,
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

    assert output.nl2test_input == nl2_input
    assert output.nl2test_metadata == nl2_metadata

    assert output.structured_eval is not None
    matching_metrics = [
        output.structured_eval.obj_creation_recall,
        output.structured_eval.obj_creation_precision,
        output.structured_eval.assertion_recall,
        output.structured_eval.assertion_precision,
        output.structured_eval.callable_recall,
        output.structured_eval.callable_precision,
        output.structured_eval.focal_recall,
        output.structured_eval.focal_precision,
    ]
    for metric in matching_metrics:
        assert metric == pytest.approx(1.0)

    cov = output.coverage_eval
    assert cov is not None
    assert 0.0 <= cov.class_coverage <= 1.0
    assert 0.0 <= cov.method_coverage <= 1.0
    assert 0.0 <= cov.line_coverage <= 1.0
    assert 0.0 <= cov.branch_coverage <= 1.0

    mismatched_metadata = NL2TestMetadata(
        qualified_test_class_name=second_qualified_class_name,
        method_signature=second_method_signature,
        code="",
    )
    mismatched_output = grader.grade(nl2_input, mismatched_metadata)
    pretty_print("Grading Output (Mismatched)", mismatched_output)

    assert mismatched_output.structured_eval is not None
    mismatched_metrics = [
        mismatched_output.structured_eval.obj_creation_recall,
        mismatched_output.structured_eval.obj_creation_precision,
        mismatched_output.structured_eval.assertion_recall,
        mismatched_output.structured_eval.assertion_precision,
        mismatched_output.structured_eval.callable_recall,
        mismatched_output.structured_eval.callable_precision,
        mismatched_output.structured_eval.focal_recall,
        mismatched_output.structured_eval.focal_precision,
    ]
    for metric in mismatched_metrics:
        assert metric < 1.0


class TestCoverageMultiModule:
    def test_multimodule_coverage_eval(self):
        tests_dir = Path(__file__).resolve().parent
        repo_root = tests_dir.parent
        datasets_dir = repo_root / "resources" / "datasets"
        project_name = "commons-numbers"
        project_root = datasets_dir / project_name
        assert project_root.is_dir()

        output_dir = tests_dir / "output" / project_name
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis = CLDK(language="java").analysis(
            project_path=project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=output_dir,
            eager=False,
        )

        test_class = "org.apache.commons.numbers.core.AdditionTest"
        test_method = "testIsZero()"

        assert analysis.get_class(test_class) is not None
        assert analysis.get_method(test_class, test_method) is not None

        common = CommonAnalysis(analysis)
        module_root = common.resolve_module_root(test_class)
        assert module_root is not None
        assert module_root.name == "commons-numbers-core"

        _, application_classes, test_utility_classes = common.categorize_classes()
        grader = TestGrader(
            analysis=analysis,
            project_root=project_root,
            project_erroneous_files=[],
            application_classes=application_classes,
            test_utility_classes=test_utility_classes,
        )

        coverage = grader.grade_coverage(
            pred_method_sig=test_method,
            pred_class_name=test_class,
            gt_method_sig=test_method,
            gt_class_name=test_class,
        )
        assert coverage is not None
        assert coverage.class_coverage == pytest.approx(1.0)
        assert coverage.method_coverage == pytest.approx(1.0)
        assert coverage.line_coverage == pytest.approx(1.0)
        assert coverage.branch_coverage == pytest.approx(1.0)
