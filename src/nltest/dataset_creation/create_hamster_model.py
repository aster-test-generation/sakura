import os
import subprocess
from pathlib import Path

import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from hamster.code_analysis.test_statistics import ProjectAnalysisInfo
from tqdm import tqdm

from nltest.utils import constants

BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent
class CreateHamsterModel:
    def __init__(self):
        pass

    @staticmethod
    @ray.remote
    def create_hamster_model(project_path):
        try:
            cldk = CLDK(language="java").analysis(
                project_path=project_path,
                analysis_backend_path=None,
                analysis_level=AnalysisLevel.symbol_table,
                analysis_json_path=BASE_PATH.joinpath(constants.DEFAULT_ANALYSIS_DIR, Path(project_path).name),
            )
            project_analysis = (ProjectAnalysisInfo(analysis=cldk,
                                                    dataset_name=Path(project_path).name)
                                .gather_project_analysis_info())
            project_analysis_str = project_analysis.model_dump_json()

            output_dir = BASE_PATH.joinpath(constants.HAMSTER_MODEL_DIR) / Path(project_path).name
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / "hamster.json", 'w') as f:
                f.write(project_analysis_str)
        except Exception as e:
            print(f'Error processing dataset {project_path}')

    @staticmethod
    def get_subfolders(base_folder):
        """Get all immediate subfolders in the base folder."""
        return [os.path.join(base_folder, f) for f in os.listdir(base_folder)
                if os.path.isdir(os.path.join(base_folder, f))]

if __name__ == "__main__":
    # Change this to your specific folder path

    subfolders = CreateHamsterModel.get_subfolders(BASE_PATH.joinpath(constants.RESOURCE_DIR))

    # Launch tasks
    futures = [CreateHamsterModel.create_hamster_model.remote(sf) for sf in subfolders]

    results = []
    with tqdm(total=len(futures), desc="Processing folders") as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            res = ray.get(done)
            results.extend(res)
            pbar.update(len(done))
