from pathlib import Path
from unittest import TestCase

from nltest.utils.evaluation import TestGrader


class TestCoverageGrader(TestCase):
    def test_coverage_grader(self):
        test_dir = Path(__file__).resolve().parent
        project_root = (test_dir / "resources" / "spring-petclinic").resolve()
        test_grader = TestGrader(analysis=None, project_root=project_root)
        ground_truth_class_method_pairs = ("org.springframework.samples.petclinic.vet.VetControllerTests",
                                           "testShowVetListHtml")
        prediction_class_method_pairs = ("org.springframework.samples.petclinic.vet.VetControllerTests",
                                           "testShowVetListHtml")
        coverage = test_grader.grade_coverage_all([(prediction_class_method_pairs, ground_truth_class_method_pairs)])
        coverage_details = {'branch_coverage_percent': 100.0, 'class_coverage_percent': 100.0,
                            'line_coverage_percent': 100.0, 'method_coverage_percent': 100.0}
        self.assertEquals(coverage[0]["coverage_details"], coverage_details)
