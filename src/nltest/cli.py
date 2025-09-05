import os
from pathlib import Path
from dotenv import load_dotenv

import typer
import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel
from typing_extensions import Annotated

from nltest.test2nl.model.models import AbstractionLevel
from nltest.test2nl import Pipeline
from nltest.utils.config import init_config
from nltest.utils.llm.model import Provider
from nltest.utils.pretty.color_logger import RichLog
from nltest.utils.pretty.prints import pretty_print
from nltest.nl2test import Pipeline as NL2TestPipeline
from nltest.nl2test.models import NL2TestInput, NL2LocalizationOutput
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.test2nl.model.models import Test2NLEntry, TestDescriptionInfo
from nltest.utils.file_io.structured_data_manager import StructuredDataManager
from nltest.dataset_creation.model import NL2TestDataset, Test as DatasetTest
from nltest.utils.models import Method

app = typer.Typer(
    help="ASTER-Test2NL: [A]utomated Te[s][t] Cas[e] Generato[r] from Natural Language",
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    add_completion=False,
)

load_dotenv()


def _resolve_output_dir(output_dir: str | None) -> Path:
    return Path(output_dir or "./output")


def _log_section(title: str) -> None:
    sep = "=" * 60
    RichLog.info(f"\n{sep}")
    RichLog.info(title)
    RichLog.info(sep)


def _init_project_config(
    *,
    project_name: str,
    base_project_dir: Path,
    output_dir: Path,
    llm_model: str,
    emb_model: str | None = None,
    localization_max_iters: int | None = None,
) -> None:
    """
    Initialize Config singleton for a project.
    """
    init_config(
        project_name=project_name,
        base_project_dir=str(base_project_dir),
        output_dir=str(output_dir),
        llm_provider=Provider.OPENROUTER,
        llm_model=llm_model,
        emb_provider=Provider.OLLAMA if emb_model else None,
        emb_model=emb_model,
        llm_api_key=os.getenv("OPENROUTER_API_KEY"),
        emb_api_key=None,
        localization_max_iters=(localization_max_iters or 20),
    )


def _run_analysis(*, project_root: Path, analysis_dir: Path, eager: bool = True):
    """Run or load CLDK analysis for a project."""
    return CLDK(language="java").analysis(
        project_path=project_root,
        analysis_backend_path=None,
        analysis_level=AnalysisLevel.symbol_table,
        analysis_json_path=analysis_dir,
        eager=eager,
    )


@app.callback()
def main() -> None:
    return


@app.command()
def generate_descriptions(
    analysis_dir: str,
    organized_methods_dir: str,
    llm_model: str,
    clear_dataset: bool = True,
    max_entries: int = 0,
):
    output_dir = _resolve_output_dir(None)

    analysis_root = Path(analysis_dir)
    methods_root = Path(organized_methods_dir)

    if not (methods_root.exists() and methods_root.is_dir()):
        raise Exception(
            f"Organized methods directory {methods_root} does not exist or is not a directory."
        )

    if not (analysis_root.exists() and analysis_root.is_dir()):
        raise Exception(
            f"Analysis directory {analysis_root} does not exist or is not a directory."
        )

    organized_methods_file_name = "nl2test.json"
    project_dirs = [
        p
        for p in sorted(methods_root.iterdir())
        if p.is_dir() and (p / organized_methods_file_name).exists()
    ]

    if not project_dirs:
        RichLog.warn(
            f"No project directories with nl2test.json found under {methods_root}"
        )
        return

    data_manager = StructuredDataManager(output_dir)
    if clear_dataset:
        RichLog.info(f"Clearing the existing Test2NL dataset at the output directory.")
        targets = ["descriptions.json", "test2nl.csv"]
        data_manager.delete_many(targets)

    RichLog.info(
        f"Found {len(project_dirs)} project(s) with nl2test.json: {[p.name for p in project_dirs]}"
    )

    start_id = 0

    for project_dir in project_dirs:
        project_name = project_dir.name

        # Ensuring matching analysis.json exists
        analysis_project_dir = Path(analysis_root) / project_name
        analysis_json_path = analysis_project_dir / "analysis.json"

        if not (analysis_project_dir.exists() and analysis_project_dir.is_dir()):
            RichLog.warn(
                f"Skipping {project_name}: missing analysis directory {analysis_project_dir}"
            )
            continue

        if not analysis_json_path.exists():
            RichLog.warn(
                f"Skipping {project_name}: missing analysis.json at {analysis_json_path}"
            )
            continue

        _log_section(f"Preparing project: {project_name}")

        analysis = CLDK(language="java").analysis(
            project_path="",
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=analysis_project_dir,
            eager=False,
        )

        _init_project_config(
            project_name=project_name,
            base_project_dir=project_dir,
            output_dir=output_dir,
            llm_model=llm_model,
        )

        pipeline = Pipeline(analysis, project_name, output_dir)

        dataset_path = project_dir / organized_methods_file_name
        try:
            dataset = NL2TestDataset.model_validate_json(
                dataset_path.read_text("utf-8")
            )
        except Exception as exc:
            RichLog.warn(
                f"Skipping {project_name}: failed to load dataset from {dataset_path} ({exc})"
            )
            continue

        groups = [
            "tests_with_one_focal_methods",
            "tests_with_two_focal_methods",
            "tests_with_more_than_two_to_five_focal_methods",
            "tests_with_more_than_five_to_ten_focal_methods",
            "tests_with_more_than_ten_focal_methods",
        ]

        tests_to_process: list[Method] = []
        for group_name in groups:
            tests: list[DatasetTest] = getattr(dataset, group_name, []) or []
            for t in tests:
                tests_to_process.append(
                    Method(
                        qualified_class_name=t.qualified_class_name,
                        method_signature=t.method_signature,
                    )
                )

        RichLog.info(
            f"Prepared {len(tests_to_process)} test methods from {dataset_path} for {project_name}"
        )

        test2nl_entries, test_descriptions, start_id = (
            pipeline.run_descriptions_on_select(
                test_methods=tests_to_process, start_id=start_id
            )
        )

        pipeline.data_manager.save(
            "descriptions.json", test_descriptions, format="json", mode="append"
        )
        pipeline.data_manager.save(
            "test2nl.csv", test2nl_entries, format="csv", mode="append"
        )


@app.command()
def generate_descriptions_old(
    base_project_dir: Annotated[
        str,
        typer.Option(
            help="Path to the directory containing the project directories.",
            show_default=False,
        ),
    ] = "./resources",
    output_dir: Annotated[
        str,
        typer.Option(
            help="Path to the output directory for saving descriptions.",
            show_default=False,
        ),
    ] = None,
    llm_model: Annotated[
        str,
        typer.Option(
            help="LLM model to use for description generation.",
            show_default=False,
        ),
    ] = "mistralai/devstral-small",
    evaluate: Annotated[
        bool,
        typer.Option(
            help="Whether to evaluation the description for each test case.",
            show_default=False,
        ),
    ] = False,
    clear_dataset: Annotated[
        bool,
        typer.Option(
            help="Whether to delete the existing descriptions at the save location, or whether to append to them.",
            show_default=False,
        ),
    ] = True,
    max_entries: Annotated[
        int,
        typer.Option(
            help="Maximum number of Test2NL entries to generate per project (0 for unlimited).",
            show_default=False,
        ),
    ] = 0,
    only_interesting_tests: Annotated[
        bool,
        typer.Option(
            help="Whether to only generate descriptions for interesting/complicated tests (those with complex focal class/method relationships).",
            show_default=False,
        ),
    ] = False,
):
    base_project_dir = Path(base_project_dir)
    if not (base_project_dir.exists() and base_project_dir.is_dir()):
        raise Exception(f"Base project directory {base_project_dir} does not exist.")

    output_dir = _resolve_output_dir(output_dir)

    # Get all project directories in base_project_dir
    all_projects = sorted([p for p in base_project_dir.iterdir() if p.is_dir()])

    if not all_projects:
        RichLog.error(f"No project directories found in {base_project_dir}")
        return

    RichLog.info(
        f"Found {len(all_projects)} project(s) to process: {[p.name for p in all_projects]}"
    )

    # Process each project separately
    for project_root in all_projects:
        project_name = project_root.name
        _log_section(f"Processing project: {project_name}")

        # Create project-specific output directory
        project_output_dir = output_dir / project_name
        project_output_dir.mkdir(parents=True, exist_ok=True)

        _init_project_config(
            project_name=project_name,
            base_project_dir=project_root,
            output_dir=project_output_dir,
            llm_model=llm_model,
        )

        # Generate analysis of the current project
        RichLog.info(f"Gathering static analysis results for {project_name}")
        analysis = _run_analysis(
            project_root=project_root, analysis_dir=project_output_dir, eager=True
        )
        RichLog.info(
            f"Successfully finished gathering static analysis results for {project_name}"
        )

        # Create orchestration to handle workflow
        pipeline = Pipeline(analysis, project_root, project_output_dir)

        if clear_dataset:
            RichLog.info(
                f"Clearing the existing dataset for {project_name} at the output directory."
            )
            pipeline.reset_dataset()

        # Track total entries generated for this project
        total_entries_generated = 0

        for abs_level in AbstractionLevel:
            # Calculate remaining entries for this abstraction level
            remaining_entries = (
                max_entries - total_entries_generated if max_entries > 0 else 0
            )

            # Break early if we've reached the limit
            if max_entries > 0 and total_entries_generated >= max_entries:
                RichLog.info(
                    f"Reached maximum entries limit ({max_entries}) for {project_name}. Stopping early."
                )
                break

            if evaluate:
                pipeline.run_all(
                    abs_level,
                    regen_classes=True,
                    max_entries=remaining_entries,
                    only_interesting_tests=only_interesting_tests,
                )
                # Get the number of descriptions generated
                try:
                    descriptions = pipeline.data_manager.load(
                        "descriptions.json", TestDescriptionInfo
                    )
                    # Count descriptions for this abstraction level and project
                    entries_generated = len(
                        [d for d in descriptions if d.abstraction_level == abs_level]
                    )
                    total_entries_generated += entries_generated

                    RichLog.info(
                        f"Generated {entries_generated} entries for {abs_level.value} abstraction level"
                    )
                    RichLog.info(
                        f"Total entries generated for {project_name}: {total_entries_generated}"
                    )

                    # Break early if we've reached the limit
                    if max_entries > 0 and total_entries_generated >= max_entries:
                        RichLog.info(
                            f"Reached maximum entries limit ({max_entries}) for {project_name}."
                        )
                        break
                except FileNotFoundError:
                    pass
            else:
                # Pass the remaining entries limit to run_descriptions
                descriptions = pipeline.run_descriptions(
                    abs_level,
                    max_entries=remaining_entries,
                    only_interesting_tests=only_interesting_tests,
                )
                entries_generated = len(descriptions)
                total_entries_generated += entries_generated

                RichLog.info(
                    f"Generated {entries_generated} entries for {abs_level.value} abstraction level"
                )
                RichLog.info(
                    f"Total entries generated for {project_name}: {total_entries_generated}"
                )

                # Break early if we've reached the limit
                if max_entries > 0 and total_entries_generated >= max_entries:
                    RichLog.info(
                        f"Reached maximum entries limit ({max_entries}) for {project_name}."
                    )
                    break

        final_message = f"Completed processing project: {project_name}"
        if max_entries > 0:
            final_message += (
                f" (Generated {total_entries_generated}/{max_entries} entries)"
            )
        RichLog.info(final_message)

    RichLog.info(f"\n{'='*60}")
    RichLog.info(
        f"Completed description generation for all {len(all_projects)} projects"
    )
    RichLog.info(f"{'='*60}")


@app.command()
def evaluate_localization(
    base_project_dir: Annotated[
        str,
        typer.Option(
            help="Path to the base directory containing the projects.",
            show_default=False,
        ),
    ] = "./resources",
    output_dir: Annotated[
        str,
        typer.Option(
            help="Path to the output directory containing the test2nl.csv file.",
            show_default=False,
        ),
    ] = None,
    csv_file: Annotated[
        str,
        typer.Option(
            help="Name of the CSV file containing Test2NL entries (default: test2nl.csv).",
            show_default=False,
        ),
    ] = "test2nl.csv",
    llm_model: Annotated[
        str,
        typer.Option(
            help="LLM model to use for localization evaluation.",
            show_default=False,
        ),
    ] = "mistralai/devstral-small",
    emb_model: Annotated[
        str,
        typer.Option(
            help="Embedding model to use for vector search.",
            show_default=False,
        ),
    ] = "nomic-embed-text:v1.5",
    decomposition_mode: Annotated[
        DecompositionMode,
        typer.Option(
            help="Decomposition mode: grammatical or gherkin",
            show_default=True,
        ),
    ] = DecompositionMode.GRAMMATICAL,
    save_results: Annotated[
        bool,
        typer.Option(
            help="Whether to save detailed evaluation results to files.",
            show_default=False,
        ),
    ] = True,
    max_entries: Annotated[
        int,
        typer.Option(
            help="Maximum number of entries to process (0 for all). Entries are sorted by class-method pairs.",
            show_default=False,
        ),
    ] = 0,
    parallelize: Annotated[
        bool,
        typer.Option(
            help="Whether to parallelize per-entry localization with Ray.",
            show_default=False,
        ),
    ] = False,
    num_workers: Annotated[
        int,
        typer.Option(
            help="Number of Ray workers when parallelization is enabled.",
            show_default=True,
        ),
    ] = 4,
    localization_max_iters: Annotated[
        int,
        typer.Option(
            help="Maximum number of iterations for localization.",
            show_default=True,
        ),
    ] = 20,
):
    base_project_dir = Path(base_project_dir)
    if not (base_project_dir.exists() and base_project_dir.is_dir()):
        raise Exception(f"Base project directory {base_project_dir} does not exist.")

    output_dir = _resolve_output_dir(output_dir)

    if not output_dir.exists():
        raise Exception(f"Output directory {output_dir} does not exist.")

    # Load Test2NL entries from CSV
    data_manager = StructuredDataManager(output_dir)
    csv_path = output_dir / csv_file

    if not csv_path.exists():
        raise Exception(f"CSV file {csv_path} does not exist.")

    RichLog.info(f"Loading Test2NL entries from {csv_path}")
    test2nl_entries = data_manager.load(csv_file, Test2NLEntry, format="csv")

    # Sort entries by qualified_class_name and method_signature to group related entries together
    test2nl_entries.sort(
        key=lambda entry: (entry.qualified_class_name, entry.method_signature)
    )
    RichLog.info(
        f"Loaded and sorted {len(test2nl_entries)} Test2NL entries by class-method pairs"
    )

    # Limit entries if specified (applied to individual entries, not class-method pairs)
    if max_entries > 0:
        test2nl_entries = test2nl_entries[:max_entries]
        RichLog.info(
            f"Processing first {len(test2nl_entries)} entries (max_entries={max_entries})"
        )

    # Convert Test2NL entries to NL2TestInput objects and organize by project
    nl2test_inputs_by_project = {}
    for entry in test2nl_entries:
        nl2test_input = NL2TestInput(
            description=entry.description,
            project_name=entry.project_name,
            qualified_class_name=entry.qualified_class_name,
            method_signature=entry.method_signature,
            abstraction_level=entry.abstraction_level.value,
            is_bdd=entry.is_bdd,
            id=entry.id,
        )

        if entry.project_name not in nl2test_inputs_by_project:
            nl2test_inputs_by_project[entry.project_name] = []
        nl2test_inputs_by_project[entry.project_name].append(nl2test_input)

    RichLog.info(
        f"Organized inputs by {len(nl2test_inputs_by_project)} projects: {list(nl2test_inputs_by_project.keys())}"
    )

    # Only init ray if parallelize is enabled
    ray_available = False
    if parallelize:
        try:
            if not ray.is_initialized():
                # Limit Ray to the requested number of workers on this node
                ray.init(ignore_reinit_error=True, num_cpus=max(1, int(num_workers)))
            ray_available = True
        except Exception as exc:
            RichLog.warn(
                f"Ray is not available or failed to initialize ({exc}). Falling back to sequential."
            )
            parallelize = False

    # Process each project separately
    total_successful_evaluations = 0
    total_failed_evaluations = 0
    all_localization_outputs = []

    for project_name, nl2test_inputs in nl2test_inputs_by_project.items():
        _log_section(f"Processing project: {project_name}")

        # Create project-specific paths
        project_root = base_project_dir / project_name
        if not (project_root.exists() and project_root.is_dir()):
            RichLog.error(
                f"Project directory {project_root} does not exist. Skipping project {project_name}."
            )
            continue

        project_output_dir = output_dir / project_name

        # Initialize configuration for this project
        _init_project_config(
            project_name=project_name,
            base_project_dir=project_root,
            output_dir=project_output_dir,
            llm_model=llm_model,
            emb_model=emb_model,
            localization_max_iters=localization_max_iters,
        )

        # Generate analysis for this specific project
        RichLog.info(f"Gathering static analysis results for {project_name}")
        analysis = _run_analysis(
            project_root=project_root, analysis_dir=project_output_dir, eager=True
        )
        RichLog.info(
            f"Successfully finished gathering static analysis results for {project_name}"
        )

        # Process inputs for this project
        total_inputs = len(nl2test_inputs)
        successful_evaluations = 0
        failed_evaluations = 0

        RichLog.info(
            f"Starting localization evaluation for {total_inputs} inputs in {project_name}"
        )

        if parallelize and ray_available:
            decomposition_mode_value = decomposition_mode.value

            @ray.remote
            def _process_one(
                input_payload: dict,
                project_name: str,
                base_project_dir_str: str,
                output_dir_str: str,
                llm_model_str: str,
                emb_model_str: str,
                decomposition_mode_val: str,
                localization_max_iters_val: int,
            ) -> dict:
                try:
                    project_root_local = Path(base_project_dir_str) / project_name
                    project_output_dir_local = Path(output_dir_str) / project_name

                    # Each worker initializes its own config and loads analysis from JSON
                    _init_project_config(
                        project_name=project_name,
                        base_project_dir=Path(base_project_dir_str),
                        output_dir=project_output_dir_local,
                        llm_model=llm_model_str,
                        emb_model=emb_model_str,
                        localization_max_iters=localization_max_iters_val,
                    )

                    analysis_local = _run_analysis(
                        project_root=project_root_local,
                        analysis_dir=project_output_dir_local,
                        eager=False,  # Load from JSON produced earlier
                    )

                    pipeline_local = NL2TestPipeline(
                        analysis_local,
                        project_root_local,
                        decomposition_mode=DecompositionMode(decomposition_mode_val),
                    )

                    nl2_input_local = NL2TestInput(**input_payload)
                    output = pipeline_local.run_localization_evaluation_pipeline(
                        nl2_input_local
                    )
                    return {"success": True, "output": output.model_dump(mode="json")}
                except Exception as e:  # pragma: no cover - executed in Ray worker
                    return {
                        "success": False,
                        "error": str(e),
                        "input": input_payload,
                    }

            input_payloads = [x.model_dump(mode="json") for x in nl2test_inputs]
            futures = [
                _process_one.remote(
                    payload,
                    project_name,
                    str(base_project_dir),
                    str(output_dir),
                    llm_model,
                    emb_model,
                    decomposition_mode_value,
                    localization_max_iters,
                )
                for payload in input_payloads
            ]

            # Gather results
            results = ray.get(futures)
            for idx, result in enumerate(results, 1):
                if result.get("success"):
                    try:
                        loc_output_dict = result["output"]
                        localization_output = NL2LocalizationOutput(**loc_output_dict)
                        all_localization_outputs.append(loc_output_dict)
                        pretty_print(
                            "Localized Blocks:", localization_output.localized_blocks
                        )
                        RichLog.info(
                            f"Coverage score: {localization_output.coverage_score:.3f}"
                        )
                        successful_evaluations += 1
                    except Exception as parse_exc:
                        RichLog.error(
                            f"Failed to parse output {idx}/{total_inputs}: {parse_exc}"
                        )
                        failed_evaluations += 1
                else:
                    err = result.get("error", "Unknown error")
                    RichLog.error(f"Failed to process input {idx}: {err}")
                    failed_evaluations += 1
        else:
            # Create NL2Test pipeline for this project
            RichLog.info(f"Initializing NL2Test pipeline for {project_name}")
            pipeline = NL2TestPipeline(
                analysis, project_root, decomposition_mode=decomposition_mode
            )

            for i, nl2test_input in enumerate(nl2test_inputs, 1):
                try:
                    RichLog.info(
                        f"Processing input {i}/{total_inputs}: {nl2test_input.qualified_class_name}.{nl2test_input.method_signature} ({nl2test_input.abstraction_level})"
                    )

                    # Run localization evaluation pipeline
                    localization_output = pipeline.run_localization_evaluation_pipeline(
                        nl2test_input
                    )
                    all_localization_outputs.append(
                        localization_output.model_dump(mode="json")
                    )

                    pretty_print(
                        "Localized Blocks:", localization_output.localized_blocks
                    )
                    RichLog.info(
                        f"Coverage score: {localization_output.coverage_score:.3f}"
                    )
                    successful_evaluations += 1

                except Exception as e:
                    RichLog.error(f"Failed to process input {i}: {str(e)}")
                    failed_evaluations += 1
                    continue

        RichLog.info(f"Project {project_name} evaluation completed:")
        RichLog.info(f"  - Total inputs: {total_inputs}")
        RichLog.info(f"  - Successful evaluations: {successful_evaluations}")
        RichLog.info(f"  - Failed evaluations: {failed_evaluations}")

        total_successful_evaluations += successful_evaluations
        total_failed_evaluations += failed_evaluations

    RichLog.info(f"\n{'='*60}")
    RichLog.info(f"Overall localization evaluation completed:")
    RichLog.info(f"  - Total projects processed: {len(nl2test_inputs_by_project)}")
    RichLog.info(f"  - Total successful evaluations: {total_successful_evaluations}")
    RichLog.info(f"  - Total failed evaluations: {total_failed_evaluations}")
    RichLog.info(f"{'='*60}")

    if save_results and all_localization_outputs:
        # Save all localization outputs to a single file
        global_data_manager = StructuredDataManager(output_dir)
        filename = "nl2_localization_outputs.json"
        global_data_manager.save(filename, all_localization_outputs, format="json")
        RichLog.info(
            f"Saved {len(all_localization_outputs)} localization outputs to {output_dir / filename}"
        )
    elif save_results:
        RichLog.info(f"No successful evaluations to save")

    # Shutdown Ray if we started it
    if parallelize and ray_available:
        try:
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    app()
