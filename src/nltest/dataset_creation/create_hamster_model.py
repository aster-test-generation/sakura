import os
import subprocess
from pathlib import Path

import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from hamster.code_analysis.test_statistics import ProjectAnalysisInfo
from tqdm import tqdm

from nltest.utils import constants

# Path constants
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # Project root
OUTPUT_FILE_NAME = "hamster.json"
TEST_RESOURCES_DIR = "resources/datasets"  # Relative to ROOT_DIR
TEST_OUTPUT_DIR = "output"  # Relative to ROOT_DIR


class CreateHamsterModel:
    def __init__(self):
        pass

    @staticmethod
    @ray.remote
    def create_hamster_model(project_path, analysis_dir, hamster_dir):
        try:
            # Just in case check
            proj_name = Path(project_path).name
            if proj_name == "__pycache__" or proj_name.startswith("."):
                return

            cldk = CLDK(language="java").analysis(
                project_path=project_path,
                analysis_backend_path=None,
                analysis_level=AnalysisLevel.symbol_table,
                analysis_json_path=ROOT_DIR.joinpath(
                    analysis_dir, Path(project_path).name
                ),
            )
            project_analysis = ProjectAnalysisInfo(
                analysis=cldk, dataset_name=Path(project_path).name
            ).gather_project_analysis_info()
            project_analysis_str = project_analysis.model_dump_json()

            output_dir = ROOT_DIR.joinpath(hamster_dir) / Path(project_path).name
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / OUTPUT_FILE_NAME, "w") as f:
                f.write(project_analysis_str)
        except Exception as e:
            print(f"Error processing dataset {project_path}")

    @staticmethod
    def get_subfolders(base_folder):
        """Get all immediate subfolders in the base folder."""
        return [
            os.path.join(base_folder, f)
            for f in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, f))
            and f != "__pycache__"
            and not f.startswith(".")
        ]


if __name__ == "__main__":
    # Define the directory paths
    resource_dir = TEST_RESOURCES_DIR
    # resource_dir = constants.RESOURCE_DIR

    # hamster_dir = constants.HAMSTER_MODEL_DIR
    hamster_dir = TEST_OUTPUT_DIR + "/" + constants.HAMSTER_MODEL_DIR

    # analysis_dir = constants.DEFAULT_ANALYSIS_DIR
    analysis_dir = TEST_OUTPUT_DIR + "/" + constants.DEFAULT_ANALYSIS_DIR

    # Ensure directories exist
    ROOT_DIR.joinpath(resource_dir).mkdir(parents=True, exist_ok=True)
    ROOT_DIR.joinpath(hamster_dir).mkdir(parents=True, exist_ok=True)
    ROOT_DIR.joinpath(analysis_dir).mkdir(parents=True, exist_ok=True)

    # Change this to your specific folder path
    subfolders = CreateHamsterModel.get_subfolders(ROOT_DIR.joinpath(resource_dir))

    # Launch tasks
    futures = [
        CreateHamsterModel.create_hamster_model.remote(sf, analysis_dir, hamster_dir)
        for sf in subfolders
    ]

    results = []
    with tqdm(total=len(futures), desc="Processing folders") as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            res = ray.get(done)
            results.extend(res)
            pbar.update(len(done))
