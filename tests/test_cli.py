import os
import tempfile
from pathlib import Path
from unittest import TestCase

from nltest.cli import (
    generate_descriptions,
    run_nl2test,
    _load_nl2_inputs_by_project_from_csv,
)


def _ensure_empty_test2nl_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        # Minimal header matching Test2NLEntry
        csv_path.write_text(
            "id,description,project_name,qualified_class_name,method_signature,abstraction_level,is_bdd\n",
            encoding="utf-8",
        )


class TestCLI(TestCase):
    def test_cli_generate_descriptions_smoke(self):
        analysis_dir = "./output/resources/output/"
        organized_methods_dir = "./output/resources/nl2test/"
        output_dir = "./output/resources/test2nl/"

        orig_cwd = os.getcwd()
        try:
            os.chdir(Path(__file__).resolve().parent)
            for p in [
                Path(analysis_dir),
                Path(organized_methods_dir),
                Path(output_dir),
            ]:
                p.mkdir(parents=True, exist_ok=True)
            generate_descriptions(
                analysis_dir=analysis_dir,
                output_dir=output_dir,
                organized_methods_dir=organized_methods_dir,
                organized_methods_file_name="nl2test.json",
                llm_model="google/gemini-2.5-flash",
                llm_provider="openrouter",
                llm_api_url=None,
                clear_dataset=True,
                max_methods=0,
                num_proj_parallel=1,
                per_proj_concurrency=2,
                max_inflight=0,
                exclude_groups=[],
            )
        finally:
            os.chdir(orig_cwd)

    def test_cli_run_nl2test_smoke(self):
        base_project_dir = "./resources/"
        base_analysis_dir = "./output/"
        output_dir = "./output"
        test2nl_file = "./output/resources/test2nl/test2nl.csv"

        orig_cwd = os.getcwd()
        try:
            os.chdir(Path(__file__).resolve().parent)
            # Ensure directories and an empty CSV exist
            Path(base_project_dir).mkdir(parents=True, exist_ok=True)
            Path(base_analysis_dir).mkdir(parents=True, exist_ok=True)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            _ensure_empty_test2nl_csv(Path(test2nl_file))

            run_nl2test(
                base_project_dir=base_project_dir,
                base_analysis_dir=base_analysis_dir,
                output_dir=output_dir,
                clear_output=True,
                test2nl_file=test2nl_file,
                llm_model="google/gemini-2.5-flash",
                emb_model="nomic-embed-text:v1.5",
                decomposition_mode="gherkin",
                supervisor_max_iters=5,
                localization_max_iters=5,
                composition_max_iters=5,
                num_proj_parallel=1,
                max_inflight=1,
                max_entries=1,
                llm_provider="openrouter",
                llm_api_url=None,
                emb_provider="ollama",
                emb_api_url=None,
            )
        finally:
            os.chdir(orig_cwd)

    def test_load_nl2_inputs_requires_csv_file_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test2nl_dir = Path(tmp_dir) / "test2nl"
            test2nl_dir.mkdir(parents=True, exist_ok=True)

            with self.assertRaisesRegex(
                    Exception, "Expected --test2nl-file to point to a CSV file"
            ):
                _load_nl2_inputs_by_project_from_csv(test2nl_dir, max_entries=0)
