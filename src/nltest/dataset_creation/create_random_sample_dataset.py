import json
import os
from pathlib import Path
import random

from nltest.utils import constants

FILE_NAME = 'nl2test.json'
STORE_PATH = ""
BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent.joinpath('resources')
RANDOM_SAMPLE_SIZE = 20


class RandomSampleDataset:
    def __init__(self):
        pass

    @staticmethod
    def create_random_sample_dataset(dataset_folder: str):
        with open(Path(dataset_folder).joinpath(FILE_NAME), 'r') as f:
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

        os.makedirs(Path(BASE_PATH).joinpath('final_dataset')
                          .joinpath(Path(dataset_folder).name), exist_ok=True)

        # Write the sampled data to a new JSON
        with open(Path(BASE_PATH).joinpath('final_dataset')
                          .joinpath(Path(dataset_folder).name).joinpath(FILE_NAME), 'w') as f:
            json.dump(sampled_data, f, indent=2)

    @staticmethod
    def get_subfolders(base_folder):
        """Get all immediate subfolders in the base folder."""
        return [os.path.join(base_folder, f) for f in os.listdir(base_folder)
                if os.path.isdir(os.path.join(base_folder, f))]


if __name__ == '__main__':
    projects = RandomSampleDataset.get_subfolders(BASE_PATH.joinpath('nl2test'))
    for project in projects:
        RandomSampleDataset.create_random_sample_dataset(project)
