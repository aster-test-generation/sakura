import os
from pathlib import Path
from collections import deque
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
from nltest.nl2test.models import NL2TestInput, NL2LocalizationOutput, NL2TestEval
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.test2nl.model.models import Test2NLEntry, TestDescriptionInfo
from nltest.utils.file_io.structured_data_manager import StructuredDataManager
from nltest.dataset_creation.model import NL2TestDataset, Test as DatasetTest
from nltest.utils.models import Method
from nltest.ray.localization_actor import LocalizationActor
from nltest.ray.nl2test_actor import NL2TestActor
from nltest.ray.test2nl_actor import Test2NLActor

app = typer.Typer(
    help="ASTER-NLTest: [A]utomated Te[s][t] Cas[e] Generato[r] from Natural Language",
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    add_completion=False,
)

load_dotenv()


# Common directories to ignore when scanning for projects
IGNORED_DIRS = {
    "__pycache__",
    ".git",
    ".idea",
    ".vscode",
}


@app.callback()
def main() -> None:
    return


def _load_nl2_inputs_by_project_from_csv(
    test2nl_file: str, max_entries: int
) -> dict[str, list[NL2TestInput]]:
    """Load Test2NL CSV and convert to NL2TestInput grouped by project."""
    if test2nl_file is None:
        raise Exception("Parameter --test2nl-file is required and was not provided.")

    csv_path = Path(test2nl_file)
    if not csv_path.exists():
        raise Exception(f"CSV file {csv_path} does not exist.")

    RichLog.info(f"Loading Test2NL entries from {csv_path}")
    data_manager = StructuredDataManager(csv_path.parent)
    test2nl_entries = data_manager.load(csv_path.name, Test2NLEntry, format="csv")

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
    nl2test_inputs_by_project: dict[str, list[NL2TestInput]] = {}
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

    return nl2test_inputs_by_project


@app.command()
def generate_descriptions(
    analysis_dir: Annotated[
        str,
        typer.Option(
            help="Path to the directory containing all project analysis directories (each with an analysis.json).",
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            help="Path to the output directory where test2nl.csv and descriptions.json are saved.",
            show_default=False,
        ),
    ],
    organized_methods_dir: Annotated[
        str,
        typer.Option(
            help="Path to the directory containing per-project folders of filtered methods (each with nl2test.json).",
            show_default=False,
        ),
    ],
    llm_model: Annotated[
        str,
        typer.Option(
            help="LLM model ID to use for generating Test2NL descriptions.",
            show_default=False,
        ),
    ],
    clear_dataset: Annotated[
        bool,
        typer.Option(
            help="Whether to clear existing Test2NL data at the output directory before appending.",
            show_default=False,
        ),
    ] = True,
    max_methods: Annotated[
        int,
        typer.Option(
            help="Maximum number of test methods to process across all projects (0 for unlimited). Note: generates 3x entries due to three abstraction levels per method.",
            show_default=False,
        ),
    ] = 0,
    num_proj_parallel: Annotated[
        int,
        typer.Option(
            help="Maximum number of projects to process concurrently.",
            show_default=True,
        ),
    ] = 2,
    per_proj_concurrency: Annotated[
        int,
        typer.Option(
            help="Maximum concurrent generate_descriptions_one calls per project.",
            show_default=True,
        ),
    ] = 2,
    max_inflight: Annotated[
        int,
        typer.Option(
            help="Global cap on in-flight tasks across all projects (0 uses 2 * num_proj_parallel * per_proj_concurrency).",
            show_default=True,
        ),
    ] = 0,
    exclude_groups: Annotated[
        list[str],
        typer.Option(
            help="Dataset group names to exclude (repeat the option to exclude multiple).",
            show_default=True,
        ),
    ] = [],
    llm_provider: Annotated[
        str | None,
        typer.Option(
            help="LLM provider (guides default API URL). One of: openrouter, vllm, ollama, openai, gcp. Either this or --llm-api-url must be provided.",
            show_default=False,
        ),
    ] = None,
    llm_api_url: Annotated[
        str | None,
        typer.Option(
            help="OpenAI-compatible base URL for the LLM API (must support the OpenAI API format).",
            show_default=False,
        ),
    ] = None,
):
    if clear_dataset is False:
        raise NotImplementedError(
            "Behavior for retrieving the max ID for continuing dataset appends is not implemented."
        )

    if llm_provider is None and llm_api_url is None:
        raise Exception(
            "Either --llm-provider or --llm-api-url must be provided. Provider only guides the API URL default."
        )
    if llm_provider is not None:
        try:
            llm_provider = Provider(llm_provider.strip().lower())
        except Exception:
            raise Exception(
                f"Invalid --llm-provider: {llm_provider}. Must be one of {[p.value for p in Provider]}"
            )

    output_dir = Path(output_dir)
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
        if p.is_dir()
        and p.name not in IGNORED_DIRS
        and not p.name.startswith(".")
        and (p / organized_methods_file_name).exists()
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

    # Build per-project payloads from organized dataset and incorporate max_methods cap
    payloads_by_project: dict[str, list[dict]] = {}
    total_methods_planned = 0
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

        # Filter out any groups requested for exclusion
        if exclude_groups:
            valid_group_set = set(groups)
            unknown = [g for g in exclude_groups if g not in valid_group_set]
            if unknown:
                RichLog.warn(
                    f"Unknown group(s) in exclude_groups: {unknown}. Valid groups: {sorted(valid_group_set)}"
                )
            groups = [g for g in groups if g not in set(exclude_groups)]

        if not groups:
            RichLog.warn(
                f"All dataset groups were excluded for project {project_name}; skipping."
            )
            payloads_by_project[project_name] = []
            continue

        methods: list[dict] = []
        for group_name in groups:
            tests: list[DatasetTest] = getattr(dataset, group_name, []) or []
            for t in tests:
                methods.append(
                    {
                        "qualified_class_name": t.qualified_class_name,
                        "method_signature": t.method_signature,
                        "id": 0,  # PLACEHOLDER - real ID assigned when saving
                    }
                )

        if max_methods > 0:
            remaining = max_methods - total_methods_planned
            if remaining <= 0:
                RichLog.info(
                    f"Reached global max_methods limit ({max_methods}). Skipping remaining projects."
                )
                break
            if len(methods) > remaining:
                RichLog.info(
                    f"Limiting tests for {project_name} to first {remaining} out of {len(methods)} due to max_methods={max_methods}"
                )
                methods = methods[:remaining]

        payloads_by_project[project_name] = methods
        total_methods_planned += len(methods)

    if not payloads_by_project:
        RichLog.warn("No valid project payloads to process. Exiting.")
        return

    # Initialize Ray
    try:
        if not ray.is_initialized():
            ray.init()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Ray: {exc}") from exc

    # Project scheduling using bounded in-flight tasks
    start_id = 0
    pending_projects = deque(payloads_by_project.keys())

    active_projects: dict[str, dict] = {}
    future_to_project: dict[ray.ObjectRef, str] = {}
    inflight_futures: set[ray.ObjectRef] = set()

    # Effective in-flight budget across all projects
    effective_max_inflight = (
        max_inflight
        if max_inflight and max_inflight > 0
        else 2 * max(1, int(num_proj_parallel)) * max(1, int(per_proj_concurrency))
    )

    def launch_projects_up_to_limit() -> None:
        while (
            len(active_projects) < max(1, int(num_proj_parallel)) and pending_projects
        ):
            project_name = pending_projects.popleft()
            sep = "=" * 60
            RichLog.info(f"\n{sep}")
            RichLog.info(f"Starting Test2NL actor for project: {project_name}")
            RichLog.info(sep)

            actor = Test2NLActor.options(
                max_concurrency=max(1, int(per_proj_concurrency))
            ).remote(
                project_name=project_name,
                analysis_root_dir=str(analysis_root),
                output_dir=str(output_dir),
                llm_model=llm_model,
                llm_provider=llm_provider,
                llm_api_url=llm_api_url,
                base_project_dir=str(methods_root / project_name),
            )

            base_payloads = list(payloads_by_project.get(project_name, []))
            payloads: list[dict] = []
            for p in base_payloads:
                for abs_level in AbstractionLevel:
                    expanded = dict(p)
                    expanded["abstraction_level"] = abs_level.value
                    payloads.append(expanded)

            active_projects[project_name] = {
                "actor": actor,
                "pending": deque(payloads),
                "inflight": set(),
                "total": len(payloads),
                "produced": 0,
            }
            if len(payloads) == 0:
                RichLog.warn(
                    f"No inputs for project {project_name}; will finalize immediately."
                )

    def schedule_tasks() -> None:
        nonlocal inflight_futures
        if len(inflight_futures) >= effective_max_inflight:
            return
        allowed = effective_max_inflight - len(inflight_futures)
        if allowed <= 0:
            return
        per_proj_cap = max(1, int(per_proj_concurrency))
        for pname, state in list(active_projects.items()):
            if allowed <= 0:
                break
            while (
                allowed > 0
                and len(state["inflight"]) < per_proj_cap
                and state["pending"]
            ):
                payload = state["pending"].popleft()
                fut = state["actor"].generate_descriptions_one.remote(payload)
                state["inflight"].add(fut)
                inflight_futures.add(fut)
                future_to_project[fut] = pname
                allowed -= 1

    def finalize_project_if_done(pname: str) -> None:
        state = active_projects.get(pname)
        if state is None:
            return
        if state["pending"] or state["inflight"]:
            return
        RichLog.info(
            f"Project {pname} completed: total_methods={state['total']}, produced={state['produced']}"
        )
        active_projects.pop(pname, None)

    # Launch initial projects and schedule tasks
    launch_projects_up_to_limit()
    schedule_tasks()

    # Main scheduling loop
    while active_projects or pending_projects or inflight_futures:
        launch_projects_up_to_limit()
        schedule_tasks()

        if not inflight_futures:
            # No in-flight work. Finalize completed projects, then continue
            for pname in list(active_projects.keys()):
                finalize_project_if_done(pname)
            launch_projects_up_to_limit()
            if not inflight_futures and not active_projects and not pending_projects:
                break
            continue

        ready_refs, _ = ray.wait(list(inflight_futures), num_returns=1)
        for ref in ready_refs:
            project_name = future_to_project.pop(ref, None)
            inflight_futures.discard(ref)
            if project_name is None:
                continue

            state = active_projects.get(project_name)
            if state is None:
                continue

            state["inflight"].discard(ref)

            try:
                result = ray.get(ref)
            except Exception as exc:
                RichLog.error(f"[{project_name}] Ray task failed: {exc}")
                finalize_project_if_done(project_name)
                continue

            if result.get("success"):
                entry_dict = result.get("entry")
                desc_dict = result.get("description")
                if entry_dict and desc_dict:
                    entry_dict["id"] = start_id
                    desc_dict["id"] = start_id

                    data_manager.save(
                        "descriptions.json", [desc_dict], format="json", mode="append"
                    )
                    data_manager.save(
                        "test2nl.csv", [entry_dict], format="csv", mode="append"
                    )

                    start_id += 1
                    state["produced"] += 1
                else:
                    RichLog.warn(
                        f"[{project_name}] Received success but missing payloads; skipping."
                    )
            else:
                err = result.get("error", "Unknown error")
                RichLog.error(f"[{project_name}] Failed to generate description: {err}")

            finalize_project_if_done(project_name)

    # Shutdown Ray
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


@app.command()
def generate_descriptions_for_entire_projects(
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

    output_dir = Path(output_dir)

    # Get all project directories in base_project_dir (ignore cache/hidden folders)
    all_projects = sorted(
        [
            p
            for p in base_project_dir.iterdir()
            if p.is_dir() and p.name not in IGNORED_DIRS and not p.name.startswith(".")
        ]
    )

    if not all_projects:
        RichLog.error(f"No project directories found in {base_project_dir}")
        return

    RichLog.info(
        f"Found {len(all_projects)} project(s) to process: {[p.name for p in all_projects]}"
    )

    # Process each project separately
    for project_root in all_projects:
        project_name = project_root.name
        sep = "=" * 60
        RichLog.info(f"\n{sep}")
        RichLog.info(f"Processing project: {project_name}")
        RichLog.info(sep)

        # Create project-specific output directory
        project_output_dir = output_dir / project_name
        project_output_dir.mkdir(parents=True, exist_ok=True)

        init_config(
            project_name=project_name,
            base_project_dir=str(project_root),
            output_dir=str(project_output_dir),
            llm_provider=Provider.OPENROUTER,
            llm_model=llm_model,
            emb_provider=None,
            emb_model=None,
            llm_api_key=os.getenv("OPENROUTER_API_KEY"),
            emb_api_key=None,
            localization_max_iters=20,
        )

        # Generate analysis of the current project
        analysis = CLDK(language="java").analysis(
            project_path=project_root,
            analysis_backend_path=None,
            analysis_level=AnalysisLevel.symbol_table,
            analysis_json_path=project_output_dir,
            eager=True,
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

            # Single check: stop early if no remaining entries are allowed
            if max_entries > 0 and remaining_entries <= 0:
                RichLog.info(
                    f"Reached maximum entries limit ({max_entries}) for {project_name}. Stopping early."
                )
                break

            # Generate descriptions (non-evaluation path)
            descriptions = pipeline.run_descriptions_of_project(
                abs_level,
                max_entries=remaining_entries,
                only_interesting_tests=only_interesting_tests,
            )
            entries_generated = len(descriptions)
            total_entries_generated += entries_generated

            RichLog.info(
                f"Generated {entries_generated} at {abs_level.value}; total for {project_name}: {total_entries_generated}"
            )

        final_message = f"Completed processing project: {project_name}"
        if max_entries > 0:
            final_message += (
                f" (Generated {total_entries_generated}/{max_entries} entries)"
            )
        RichLog.info(final_message)

    RichLog.info(f"Completed description generation for {len(all_projects)} projects")


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
            help="Path to the output directory for saving localization results.",
            show_default=False,
        ),
    ] = None,
    test2nl_file: Annotated[
        str,
        typer.Option(
            help="Path to the Test2NL CSV file to evaluate (e.g., /path/to/test2nl.csv).",
            show_default=False,
        ),
    ] = None,
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
        str,
        typer.Option(
            help="Decomposition mode: grammatical or gherkin",
            show_default=True,
        ),
    ] = "gherkin",
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
    num_proj_parallel: Annotated[
        int,
        typer.Option(
            help="Maximum number of projects to process concurrently.",
            show_default=True,
        ),
    ] = 2,
    per_proj_concurrency: Annotated[
        int,
        typer.Option(
            help="Maximum concurrent localize_one calls per project.",
            show_default=True,
        ),
    ] = 2,
    localization_max_iters: Annotated[
        int,
        typer.Option(
            help="Maximum number of iterations for localization.",
            show_default=True,
        ),
    ] = 20,
    max_inflight: Annotated[
        int,
        typer.Option(
            help="Global cap on in-flight tasks across all projects (0 uses 2 * num_proj_parallel * per_proj_concurrency).",
            show_default=True,
        ),
    ] = 0,
    llm_provider: Annotated[
        str | None,
        typer.Option(
            help="LLM provider (guides default API URL). One of: openrouter, vllm, ollama, openai, gcp. Either this or --llm-api-url must be provided.",
            show_default=False,
        ),
    ] = None,
    llm_api_url: Annotated[
        str | None,
        typer.Option(
            help="OpenAI-compatible base URL for the LLM API (must support the OpenAI API format).",
            show_default=False,
        ),
    ] = None,
    emb_provider: Annotated[
        str | None,
        typer.Option(
            help="Embedding provider (guides default API URL). One of: vllm, ollama, openai, openrouter, gcp. Either this or --emb-api-url must be provided.",
            show_default=False,
        ),
    ] = None,
    emb_api_url: Annotated[
        str | None,
        typer.Option(
            help="Base URL for the Embedding API if using an HTTP endpoint.",
            show_default=False,
        ),
    ] = None,
):
    try:
        decomposition_mode = DecompositionMode(decomposition_mode.strip().lower())
    except Exception:
        raise Exception(
            f"Invalid --decomposition-mode: {decomposition_mode}. Must be one of {[d.value for d in DecompositionMode]}"
        )

    if llm_provider is None and llm_api_url is None:
        raise Exception(
            "Either --llm-provider or --llm-api-url must be provided. Provider only guides the API URL default."
        )
    if emb_provider is None and emb_api_url is None:
        raise Exception(
            "Either --emb-provider or --emb-api-url must be provided for embeddings. Provider only guides the API URL default."
        )

    if llm_provider is not None:
        try:
            llm_provider = Provider(llm_provider.strip().lower())
        except Exception:
            raise Exception(
                f"Invalid --llm-provider: {llm_provider}. Must be one of {[p.value for p in Provider]}"
            )

    if emb_provider is not None:
        try:
            emb_provider = Provider(emb_provider.strip().lower())
        except Exception:
            raise Exception(
                f"Invalid --emb-provider: {emb_provider}. Must be one of {[p.value for p in Provider]}"
            )

    base_project_dir = Path(base_project_dir)
    if not (base_project_dir.exists() and base_project_dir.is_dir()):
        raise Exception(f"Base project directory {base_project_dir} does not exist.")

    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise Exception(f"Output directory {output_dir} does not exist.")

    # Load and prepare NL2Test inputs grouped by project
    nl2test_inputs_by_project = _load_nl2_inputs_by_project_from_csv(
        test2nl_file, max_entries
    )

    # Initialize Ray
    try:
        if not ray.is_initialized():
            ray.init()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Ray: {exc}") from exc

    # Project scheduling using bounded in-flight tasks
    total_successful_evaluations = 0
    total_failed_evaluations = 0
    all_localization_outputs: list[dict] = []

    # Build pending projects deque
    pending_projects = deque()
    for project_name in nl2test_inputs_by_project.keys():
        project_root = base_project_dir / project_name
        if not (project_root.exists() and project_root.is_dir()):
            RichLog.error(
                f"Project directory {project_root} does not exist. Skipping project {project_name}."
            )
            continue
        pending_projects.append(project_name)

    if not pending_projects:
        RichLog.error("No valid projects to process. Exiting.")
        return

    # Global scheduler state
    active_projects: dict[str, dict] = {}
    future_to_project: dict[ray.ObjectRef, str] = {}
    inflight_futures: set[ray.ObjectRef] = set()

    # Effective in-flight budget across all projects
    effective_max_inflight = (
        max_inflight
        if max_inflight and max_inflight > 0
        else 2 * max(1, int(num_proj_parallel)) * max(1, int(per_proj_concurrency))
    )

    def launch_projects_up_to_limit() -> None:
        while (
            len(active_projects) < max(1, int(num_proj_parallel)) and pending_projects
        ):
            project_name = pending_projects.popleft()
            sep = "=" * 60
            RichLog.info(f"\n{sep}")
            RichLog.info(f"Starting actor for project: {project_name}")
            RichLog.info(sep)

            actor = LocalizationActor.options(
                max_concurrency=max(1, int(per_proj_concurrency))
            ).remote(
                project_name=project_name,
                base_project_dir=str(base_project_dir),
                output_dir=str(output_dir),
                llm_model=llm_model,
                emb_model=emb_model,
                llm_provider=llm_provider,
                llm_api_url=llm_api_url,
                emb_provider=emb_provider,
                emb_api_url=emb_api_url,
                decomposition_mode=decomposition_mode.value,
                localization_max_iters=localization_max_iters,
            )

            payloads = [
                x.model_dump(mode="json")
                for x in nl2test_inputs_by_project.get(project_name, [])
            ]
            active_projects[project_name] = {
                "actor": actor,
                "pending": deque(payloads),
                "inflight": set(),
                "total": len(payloads),
                "success": 0,
                "failed": 0,
            }
            if len(payloads) == 0:
                RichLog.warn(
                    f"No inputs for project {project_name}; will finalize immediately."
                )

    def schedule_tasks() -> None:
        # Fill global in-flight capacity while respecting per-project caps
        nonlocal inflight_futures
        if len(inflight_futures) >= effective_max_inflight:
            return
        allowed = effective_max_inflight - len(inflight_futures)
        if allowed <= 0:
            return
        per_proj_cap = max(1, int(per_proj_concurrency))
        for pname, state in list(active_projects.items()):
            if allowed <= 0:
                break
            while (
                allowed > 0
                and len(state["inflight"]) < per_proj_cap
                and state["pending"]
            ):
                payload = state["pending"].popleft()
                fut = state["actor"].localize_one.remote(payload)
                state["inflight"].add(fut)
                inflight_futures.add(fut)
                future_to_project[fut] = pname
                allowed -= 1

    def finalize_project_if_done(pname: str) -> None:
        nonlocal total_successful_evaluations, total_failed_evaluations
        state = active_projects.get(pname)
        if state is None:
            return
        if state["pending"] or state["inflight"]:
            return
        total_inputs = state["total"]
        successful_evaluations = state["success"]
        failed_evaluations = state["failed"]
        RichLog.info(
            f"Project {pname} completed: total={total_inputs}, success={successful_evaluations}, failed={failed_evaluations}"
        )
        total_successful_evaluations += successful_evaluations
        total_failed_evaluations += failed_evaluations
        active_projects.pop(pname, None)

    # Launch initial projects and schedule tasks
    launch_projects_up_to_limit()
    schedule_tasks()

    # Main scheduling loop
    while active_projects or pending_projects or inflight_futures:
        launch_projects_up_to_limit()
        schedule_tasks()

        if not inflight_futures:
            # No in-flight work; finalize completed projects, then continue
            for pname in list(active_projects.keys()):
                finalize_project_if_done(pname)
            launch_projects_up_to_limit()
            if not inflight_futures and not active_projects and not pending_projects:
                break
            continue

        ready_refs, _ = ray.wait(list(inflight_futures), num_returns=1)
        for ref in ready_refs:
            project_name = future_to_project.pop(ref, None)
            inflight_futures.discard(ref)
            if project_name is None:
                continue
            state = active_projects.get(project_name)
            if state is None:
                continue
            state["inflight"].discard(ref)
            try:
                result = ray.get(ref)
            except Exception as exc:
                RichLog.error(f"[{project_name}] Ray task failed: {exc}")
                state["failed"] += 1
                finalize_project_if_done(project_name)
                continue

            if result.get("success"):
                try:
                    loc_output_dict = result["output"]
                    localization_output = NL2LocalizationOutput(**loc_output_dict)
                    all_localization_outputs.append(loc_output_dict)
                    pretty_print(
                        "Localized Blocks:", localization_output.localized_blocks
                    )
                    state["success"] += 1
                except Exception as parse_exc:
                    RichLog.error(
                        f"[{project_name}] Failed to parse output: {parse_exc}"
                    )
                    state["failed"] += 1
            else:
                err = result.get("error", "Unknown error")
                RichLog.error(f"[{project_name}] Failed to process input: {err}")
                state["failed"] += 1

            finalize_project_if_done(project_name)

    RichLog.info(
        f"Overall localization: projects={len(nl2test_inputs_by_project)}, success={total_successful_evaluations}, failed={total_failed_evaluations}"
    )

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

    # Shutdown Ray
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


@app.command()
def run_nl2test(
    base_project_dir: Annotated[
        str,
        typer.Option(
            help="Path to the base directory containing all project directories.",
            show_default=False,
        ),
    ] = "./resources",
    base_analysis_dir: Annotated[
        str,
        typer.Option(
            help="Path to the base directory containing per-project analysis.json directories.",
            show_default=False,
        ),
    ] = None,
    output_dir: Annotated[
        str,
        typer.Option(
            help="Path to the output directory for saving NL2Test generation results.",
            show_default=False,
        ),
    ] = None,
    clear_output: Annotated[
        bool,
        typer.Option(
            help="Whether to remove existing NL2Test evaluation results before running.",
            show_default=True,
        ),
    ] = True,
    test2nl_file: Annotated[
        str,
        typer.Option(
            help="Path to the Test2NL CSV file to use as inputs (e.g., /path/to/test2nl.csv).",
            show_default=False,
        ),
    ] = None,
    llm_model: Annotated[
        str,
        typer.Option(
            help="LLM model to use for NL2Test generation.",
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
        str,
        typer.Option(
            help="Decomposition mode (must be 'gherkin' for now).",
            show_default=True,
        ),
    ] = "gherkin",
    supervisor_max_iters: Annotated[
        int,
        typer.Option(
            help="Maximum iterations for the supervisor agent.",
            show_default=True,
        ),
    ] = 10,
    localization_max_iters: Annotated[
        int,
        typer.Option(
            help="Maximum iterations for the localization agent.",
            show_default=True,
        ),
    ] = 40,
    composition_max_iters: Annotated[
        int,
        typer.Option(
            help="Maximum iterations for the composition agent.",
            show_default=True,
        ),
    ] = 30,
    num_proj_parallel: Annotated[
        int,
        typer.Option(
            help="Maximum number of projects to process concurrently.",
            show_default=True,
        ),
    ] = 2,
    max_inflight: Annotated[
        int,
        typer.Option(
            help="Global cap on in-flight project tasks (0 uses num_proj_parallel).",
            show_default=True,
        ),
    ] = 0,
    max_entries: Annotated[
        int,
        typer.Option(
            help="Maximum number of Test2NL entries to process (0 for all).",
            show_default=False,
        ),
    ] = 0,
    llm_provider: Annotated[
        str | None,
        typer.Option(
            help="LLM provider (guides default API URL). One of: openrouter, vllm, ollama, openai, gcp. Either this or --llm-api-url must be provided.",
            show_default=False,
        ),
    ] = None,
    llm_api_url: Annotated[
        str | None,
        typer.Option(
            help="OpenAI-compatible base URL for the LLM API (must support the OpenAI API format).",
            show_default=False,
        ),
    ] = None,
    emb_provider: Annotated[
        str | None,
        typer.Option(
            help="Embedding provider (guides default API URL). One of: vllm, ollama, openai, openrouter, gcp. Either this or --emb-api-url must be provided.",
            show_default=False,
        ),
    ] = None,
    emb_api_url: Annotated[
        str | None,
        typer.Option(
            help="Base URL for the Embedding API if using an HTTP endpoint.",
            show_default=False,
        ),
    ] = None,
):
    try:
        decomposition_mode = DecompositionMode(decomposition_mode.strip().lower())
    except Exception:
        raise Exception(
            f"Invalid --decomposition-mode: {decomposition_mode}. Must be one of {[d.value for d in DecompositionMode]}"
        )
    if decomposition_mode != DecompositionMode.GHERKIN:
        raise Exception(
            "Only 'gherkin' decomposition is supported for run_nl2test at the moment."
        )

    if llm_provider is None and llm_api_url is None:
        raise Exception(
            "Either --llm-provider or --llm-api-url must be provided. Provider only guides the API URL default."
        )
    if emb_provider is None and emb_api_url is None:
        raise Exception(
            "Either --emb-provider or --emb-api-url must be provided for embeddings. Provider only guides the API URL default."
        )
    if llm_provider is not None:
        try:
            llm_provider = Provider(llm_provider.strip().lower())
        except Exception:
            raise Exception(
                f"Invalid --llm-provider: {llm_provider}. Must be one of {[p.value for p in Provider]}"
            )
    if emb_provider is not None:
        try:
            emb_provider = Provider(emb_provider.strip().lower())
        except Exception:
            raise Exception(
                f"Invalid --emb-provider: {emb_provider}. Must be one of {[p.value for p in Provider]}"
            )

    base_project_dir = Path(base_project_dir)
    if not (base_project_dir.exists() and base_project_dir.is_dir()):
        raise Exception(f"Base project directory {base_project_dir} does not exist.")

    base_analysis_dir = Path(base_analysis_dir) if base_analysis_dir else None
    if base_analysis_dir is None or not (
        base_analysis_dir.exists() and base_analysis_dir.is_dir()
    ):
        raise Exception(f"Base analysis directory {base_analysis_dir} does not exist.")

    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise Exception(f"Output directory {output_dir} does not exist.")

    # Load and prepare NL2Test inputs grouped by project
    nl2test_inputs_by_project = _load_nl2_inputs_by_project_from_csv(
        test2nl_file, max_entries
    )

    # Build pending projects, ensuring both project dir and analysis.json exist
    pending_projects = deque()
    for project_name in nl2test_inputs_by_project.keys():
        project_root = base_project_dir / project_name
        analysis_project_dir = base_analysis_dir / project_name
        analysis_json_path = analysis_project_dir / "analysis.json"
        if not (project_root.exists() and project_root.is_dir()):
            RichLog.error(f"Project directory {project_root} does not exist.")
            raise Exception(f"Project directory {project_root} does not exist.")
        if not analysis_json_path.exists():
            RichLog.error(
                f"Missing analysis.json for {project_name} at {analysis_json_path}."
            )
            raise Exception(
                f"Missing analysis.json for {project_name} at {analysis_json_path}."
            )
        pending_projects.append(project_name)

    if not pending_projects:
        RichLog.error("No valid projects to process. Exiting.")
        return

    # Initialize Ray
    try:
        if not ray.is_initialized():
            ray.init()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Ray: {exc}") from exc

    # Project-level scheduling, only between-project parallelism
    results_filename = "nl2test_evaluation_results.json"
    project_data_managers: dict[str, StructuredDataManager] = {}
    cleared_projects: set[str] = set()

    def get_project_data_manager(project_name: str) -> StructuredDataManager:
        manager = project_data_managers.get(project_name)
        if manager is None:
            manager = StructuredDataManager(output_dir / project_name)
            project_data_managers[project_name] = manager
        return manager

    total_success = 0
    total_failed = 0

    inflight_futures: set[ray.ObjectRef] = set()
    future_to_project: dict[ray.ObjectRef, str] = {}
    project_payload_counts: dict[str, int] = {}

    effective_max_inflight = (
        max_inflight
        if max_inflight and max_inflight > 0
        else max(1, int(num_proj_parallel))
    )

    def launch_projects_up_to_limit() -> None:
        while pending_projects and len(inflight_futures) < effective_max_inflight:
            project_name = pending_projects.popleft()
            sep = "=" * 60
            RichLog.info(f"\n{sep}")
            RichLog.info(f"Starting NL2Test actor for project: {project_name}")
            RichLog.info(sep)

            project_manager = get_project_data_manager(project_name)
            if clear_output and project_name not in cleared_projects:
                cleared = project_manager.delete(results_filename)
                if cleared:
                    RichLog.info(
                        f"[{project_name}] Removed existing evaluation results at "
                        f"{project_manager.base_dir / results_filename}"
                    )
                cleared_projects.add(project_name)

            actor = NL2TestActor.options(max_concurrency=1).remote(
                project_name=project_name,
                base_project_dir=str(base_project_dir),
                base_analysis_dir=str(base_analysis_dir),
                output_dir=str(output_dir),
                llm_model=llm_model,
                emb_model=emb_model,
                llm_provider=llm_provider,
                llm_api_url=llm_api_url,
                emb_provider=emb_provider,
                emb_api_url=emb_api_url,
                decomposition_mode=decomposition_mode.value,
                supervisor_max_iters=supervisor_max_iters,
                localization_max_iters=localization_max_iters,
                composition_max_iters=composition_max_iters,
            )

            payloads = [
                x.model_dump(mode="json")
                for x in nl2test_inputs_by_project.get(project_name, [])
            ]
            project_payload_counts[project_name] = len(payloads)

            fut = actor.run_nl2test_batch.remote(payloads)
            inflight_futures.add(fut)
            future_to_project[fut] = project_name

    # Start initial batch
    launch_projects_up_to_limit()

    # Main scheduling loop
    while inflight_futures or pending_projects:
        if not inflight_futures:
            launch_projects_up_to_limit()
            if not inflight_futures and not pending_projects:
                break

        ready_refs, _ = ray.wait(list(inflight_futures), num_returns=1)
        for ref in ready_refs:
            project_name = future_to_project.pop(ref, None)
            inflight_futures.discard(ref)
            if project_name is None:
                continue
            try:
                results = ray.get(ref)
            except Exception as exc:
                RichLog.error(f"[{project_name}] NL2Test batch failed: {exc}")
                total_failed += project_payload_counts.get(project_name, 0)
                launch_projects_up_to_limit()
                continue

            # Persist and aggregate results
            project_success = 0
            project_failed = 0
            batch_to_save: list[NL2TestEval] = []
            for item in results or []:
                if item.get("success"):
                    res = item.get("result")
                    if res is not None:
                        # Ensure NL2TestEval objects persist correctly even if serialized across Ray execution
                        if isinstance(res, dict):
                            res = NL2TestEval(**res)
                        batch_to_save.append(res)
                    project_success += 1
                else:
                    project_failed += 1

            if batch_to_save:
                project_manager = get_project_data_manager(project_name)
                project_manager.save(
                    results_filename, batch_to_save, format="json", mode="append"
                )

            total_success += project_success
            total_failed += project_failed
            RichLog.info(
                f"[{project_name}] Completed NL2Test: success={project_success}, failed={project_failed} (inputs={project_payload_counts.get(project_name, 0)})"
            )

            # Launch more if capacity allows
            launch_projects_up_to_limit()

    RichLog.info(
        f"Overall NL2Test: projects={len(nl2test_inputs_by_project)}, success={total_success}, failed={total_failed}"
    )

    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    app()
