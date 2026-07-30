import os
from pathlib import Path

from sakura.cli import (
    generate_descriptions,
    run_nl2test,
)


def test_cli_generate_descriptions_smoke():
    analysis_dir = "../resources/analysis/"
    organized_methods_dir = "../resources/filtered_bucketed_tests/"
    output_dir = "./output/test2nl/"

    orig_cwd = os.getcwd()
    try:
        os.chdir(Path(__file__).resolve().parent)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        generate_descriptions(
            analysis_dir=analysis_dir,
            output_dir=output_dir,
            organized_methods_dir=organized_methods_dir,
            organized_methods_file_name="nl2test.json",
            llm_model="google/gemini-2.5-flash",
            llm_provider="openrouter",
            llm_api_url=None,
            clear_dataset=True,
            max_methods=1,
            num_proj_parallel=1,
            per_proj_concurrency=2,
            max_inflight=0,
            exclude_groups=[],
        )
    finally:
        os.chdir(orig_cwd)


def test_cli_run_nl2test_smoke():
    base_project_dir = "../resources/datasets/"
    base_analysis_dir = "../resources/analysis/"
    output_dir = "./output/"
    test2nl_file = "../resources/test2nl/filtered_dataset/test2nl.csv"

    orig_cwd = os.getcwd()
    try:
        os.chdir(Path(__file__).resolve().parent)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        run_nl2test(
            base_project_dir=base_project_dir,
            base_analysis_dir=base_analysis_dir,
            output_dir=output_dir,
            reset_evaluation_results=True,
            test2nl_file=test2nl_file,
            llm_model="google/gemini-2.5-flash",
            emb_model="qwen/qwen3-embedding-8b",
            decomposition_mode="gherkin",
            supervisor_max_iters=7,
            localization_max_iters=16,
            composition_max_iters=16,
            num_proj_parallel=2,
            max_inflight=0,
            max_entries=2,
            llm_provider="openrouter",
            llm_api_url=None,
            emb_provider="openrouter",
            emb_api_url=None,
        )
    finally:
        os.chdir(orig_cwd)
