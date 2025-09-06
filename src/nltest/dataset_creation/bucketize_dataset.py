import json
from pathlib import Path

import ray
from hamster.code_analysis.model.models import ProjectAnalysis
from tqdm import tqdm

from nltest.dataset_creation.create_hamster_model import CreateHamsterModel, BASE_PATH
from nltest.dataset_creation.model import Test, NL2TestDataset
from nltest.utils import constants


class BucketizeDataset:
    def __init__(self):
        pass

    @staticmethod
    @ray.remote
    def create_bucketize_dataset(project_hamster_path: str, report_store_path: str):
        parent = Path(project_hamster_path).parent
        parent_name = parent.name
        # Just in case check
        if parent_name == "__pycache__" or parent_name.startswith("."):
            return

        with open(project_hamster_path, "r") as f:
            file_content = json.load(f)
            project_analysis = ProjectAnalysis.model_validate(file_content)
        if project_analysis:
            dataset_name = project_analysis.dataset_name
            # Skip invalid dataset names
            if (
                not dataset_name
                or dataset_name == "__pycache__"
                or dataset_name.startswith(".")
            ):
                return
            tests_with_one_focal_methods = []
            tests_with_two_focal_methods = []
            tests_with_more_than_two_to_five_focal_methods = []
            tests_with_more_than_five_to_ten_focal_methods = []
            tests_with_more_than_ten_focal_methods = []
            for test_class in project_analysis.test_class_analyses:
                for test_method in test_class.test_method_analyses:
                    focal_method_count = sum(
                        [
                            len(focal_class.focal_method_names)
                            for focal_class in test_method.focal_classes
                        ]
                    )
                    test = Test(
                        qualified_class_name=test_class.qualified_class_name,
                        method_signature=test_method.method_signature,
                        focal_details=test_method.focal_classes,
                    )
                    if focal_method_count == 1:
                        tests_with_one_focal_methods.append(test)
                    elif focal_method_count == 2:
                        tests_with_two_focal_methods.append(test)
                    elif 5 >= focal_method_count > 2:
                        tests_with_more_than_two_to_five_focal_methods.append(test)
                    elif 10 >= focal_method_count > 5:
                        tests_with_more_than_five_to_ten_focal_methods.append(test)
                    elif focal_method_count > 10:
                        tests_with_more_than_ten_focal_methods.append(test)
            nl2test_dataset = NL2TestDataset(
                dataset_name=dataset_name,
                tests_with_one_focal_methods=tests_with_one_focal_methods,
                tests_with_two_focal_methods=tests_with_two_focal_methods,
                tests_with_more_than_two_to_five_focal_methods=tests_with_more_than_two_to_five_focal_methods,
                tests_with_more_than_five_to_ten_focal_methods=tests_with_more_than_five_to_ten_focal_methods,
                tests_with_more_than_ten_focal_methods=tests_with_more_than_ten_focal_methods,
            )

            nl2test_dataset_str = nl2test_dataset.model_dump_json()

            output_dir = Path(report_store_path) / Path(dataset_name).name
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / "nl2test.json", "w") as f:
                f.write(nl2test_dataset_str)


if __name__ == "__main__":
    # Define the directory paths
    hamster_model_dir = "tests/output/" + constants.HAMSTER_MODEL_DIR
    # hamster_model_dir = constants.HAMSTER_MODEL_DIR

    nl2test_dir = "tests/output/" + constants.NL2TEST_DIR
    # nl2test_dir = constants.NL2TEST_DIR

    # Ensure directories exist
    BASE_PATH.joinpath(hamster_model_dir).mkdir(parents=True, exist_ok=True)
    BASE_PATH.joinpath(nl2test_dir).mkdir(parents=True, exist_ok=True)

    # Change this to your specific folder path
    projects = CreateHamsterModel.get_subfolders(BASE_PATH.joinpath(hamster_model_dir))

    # Launch tasks
    futures = [
        BucketizeDataset.create_bucketize_dataset.remote(
            str(Path(sf).joinpath("hamster.json")),
            BASE_PATH.joinpath(nl2test_dir),
        )
        for sf in projects
    ]

    results = []
    with tqdm(total=len(futures), desc="Processing folders") as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            res = ray.get(done)
            results.extend(res)
            pbar.update(len(done))
