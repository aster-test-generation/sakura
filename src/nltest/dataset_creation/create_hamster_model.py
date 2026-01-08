from pathlib import Path

import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from hamster.code_analysis.test_statistics import ProjectAnalysisInfo
from tqdm import tqdm

# Path constants
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # Project root

# Relative to ROOT_DIR
RESOURCES_DIR = "resources/"

# Relative to RESOURCES_DIR
DATASETS_DIR = "datasets/"  # Source projects for CLDK analysis
ANALYSIS_DIR = "analysis/"  # CLDK analysis cache directory
HAMSTER_DIR = "hamster/"  # Output: hamster.json per project

# File names
OUTPUT_FILE_NAME = "hamster.json"


def get_subfolders(base_folder: Path) -> list[str]:
    """Get all immediate subfolders in the base folder."""
    return [
        str(base_folder / f.name)
        for f in base_folder.iterdir()
        if f.is_dir() and f.name != "__pycache__" and not f.name.startswith(".")
    ]


@ray.remote
def _create_hamster_model(
    project_path: str,
    analysis_path: str,
    output_path: str,
) -> None:
    """Create a hamster model from CLDK analysis for a project."""
    proj_name = Path(project_path).name
    if proj_name == "__pycache__" or proj_name.startswith("."):
        return

    try:
        cldk = CLDK(language="java").analysis(
            project_path=project_path,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=analysis_path,
            eager=True,  # ENSURE CLDK IS REGENERATED
        )
        project_analysis = ProjectAnalysisInfo(
            analysis=cldk, dataset_name=proj_name
        ).gather_project_analysis_info()
        project_analysis_str = project_analysis.model_dump_json()

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / OUTPUT_FILE_NAME, "w") as f:
            f.write(project_analysis_str)
    except Exception as e:
        print(f"Error processing dataset {project_path}: {e}")


def main() -> None:
    """Process all projects and create hamster models."""
    resources_path = ROOT_DIR / RESOURCES_DIR
    datasets_dir = resources_path / DATASETS_DIR
    analysis_dir = resources_path / ANALYSIS_DIR
    hamster_dir = resources_path / HAMSTER_DIR

    datasets_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    hamster_dir.mkdir(parents=True, exist_ok=True)

    projects = get_subfolders(datasets_dir)

    futures = []
    for project_folder in projects:
        project_name = Path(project_folder).name
        futures.append(
            _create_hamster_model.remote(  # pyright: ignore[reportCallIssue]
                project_folder,
                str(analysis_dir / project_name),
                str(hamster_dir / project_name),
            )
        )

    results = []
    with tqdm(total=len(futures), desc="Processing folders") as pbar:
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            res = ray.get(done)
            results.extend(res)
            pbar.update(len(done))


if __name__ == "__main__":
    main()
