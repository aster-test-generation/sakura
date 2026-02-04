import json
import os
from pathlib import Path
import random

from sakura.utils import constants

# Path constants
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # Project root
INPUT_FILE_NAME = "nl2test.json"
OUTPUT_FILE_NAME = "nl2test.json"
RESOURCES_DIR = "resources"  # Relative to ROOT_DIR
BUCKETED_TESTS_DIR = "bucketed_tests"  # Relative to RESOURCES_DIR
SAMPLED_TESTS_DIR = "sampled_tests"  # Relative to RESOURCES_DIR
RANDOM_SAMPLE_SIZE = 20


class RandomSampleDataset:
    def __init__(self):
        pass

    @staticmethod
    def create_random_sample_dataset(dataset_folder: str):
        with open(Path(dataset_folder).joinpath(INPUT_FILE_NAME), 'r') as f:
            file_content = json.load(f)

        # Dictionary to hold sampled data
        sampled_data = {}

        # Loop through each key in the JSON
        for key, values in file_content.items():
            if isinstance(values, list):
                # Pick min(n, len(values)) items randomly
                sampled_data[key] = random.sample(values, min(RANDOM_SAMPLE_SIZE, len(values)))
            else:
                # If the value is not a list, just copy it
                sampled_data[key] = values

        output_dir = ROOT_DIR / RESOURCES_DIR / SAMPLED_TESTS_DIR / Path(dataset_folder).name
        os.makedirs(output_dir, exist_ok=True)

        # Write the sampled data to a new JSON
        with open(output_dir / OUTPUT_FILE_NAME, 'w') as f:
            json.dump(sampled_data, f, indent=2)

    @staticmethod
    def get_subfolders(base_folder):
        """Get all immediate subfolders in the base folder."""
        return [os.path.join(base_folder, f) for f in os.listdir(base_folder)
                if os.path.isdir(os.path.join(base_folder, f))]


if __name__ == '__main__':
    projects = RandomSampleDataset.get_subfolders(ROOT_DIR / RESOURCES_DIR / BUCKETED_TESTS_DIR)
    for project in projects:
        RandomSampleDataset.create_random_sample_dataset(project)
