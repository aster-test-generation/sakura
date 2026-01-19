import logging
import shutil
from collections import deque
from pathlib import Path

import ray
import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

from nltest.dataset_creation.model import NL2TestDataset
from nltest.dataset_creation.model import Test as DatasetTest
from nltest.nl2test.models import NL2TestEval, NL2TestInput
from nltest.nl2test.models.decomposition import DecompositionMode
from nltest.ray_utils.nl2test_actor import NL2TestActor
from nltest.ray_utils.test2nl_actor import Test2NLActor
from nltest.test2nl.model.models import AbstractionLevel, Test2NLEntry
from nltest.utils.file_io.structured_data_manager import StructuredDataManager
from nltest.utils.llm.model import Provider
from nltest.utils.pretty.color_logger import RichLog
from nltest.utils.vcs.git_utils import GitUtilities

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

NL2TEST_DEBUG = True


@app.callback()
def main() -> None:
    return


def _load_nl2_inputs_by_project_from_csv(
    test2nl_file: str | Path, max_entries: int, num_proj_parallel: int
) -> dict[str, list[NL2TestInput]]:
    """Load Test2NL CSV (explicit file path) and convert to NL2TestInput grouped by project."""
    if test2nl_file is None:
        raise Exception("Parameter --test2nl-file is required and was not provided.")

    csv_path = Path(test2nl_file)
    if not csv_path.exists():
        raise Exception(f"CSV file {csv_path} does not exist.")
    if not csv_path.is_file():
        raise Exception(
            f"Expected --test2nl-file to point to a CSV file, but {csv_path} is not a file."
        )
    if csv_path.suffix.lower() != ".csv":
        raise Exception(
            f"Expected --test2nl-file to have a .csv extension, but got {csv_path.name}."
        )

    RichLog.info(f"Loading Test2NL entries from {csv_path}")
    data_manager = StructuredDataManager(csv_path.parent)
    test2nl_entries = data_manager.load(csv_path.name, Test2NLEntry, format="csv")
    total_entries = len(test2nl_entries)

    # Debug mode: load from spanning_subset.csv
    if NL2TEST_DEBUG:
        debug_csv_path = (
            Path(__file__).parent.parent.parent
            / "resources/test2nl/filtered_dataset/spanning_subset_20.csv"
        )
        if not debug_csv_path.exists():
            raise Exception(f"Debug CSV file not found: {debug_csv_path}")
        debug_data_manager = StructuredDataManager(debug_csv_path.parent)
        test2nl_entries = debug_data_manager.load(
            debug_csv_path.name, Test2NLEntry, format="csv"
        )
        RichLog.info(
            f"NL2TEST_DEBUG enabled: loaded {len(test2nl_entries)} entries from {debug_csv_path}"
        )
    elif max_entries > 0:
        test2nl_entries = test2nl_entries[:max_entries]
        RichLog.info(
            f"Processing subset of {len(test2nl_entries)} entries "
            f"(max_entries={max_entries}, total_entries={total_entries})"
        )

    # Sort entries by qualified_class_name and method_signature to group related entries together
    test2nl_entries.sort(
        key=lambda entry: (entry.qualified_class_name, entry.method_signature)
    )
    RichLog.info(
        f"Loaded and sorted {len(test2nl_entries)} Test2NL entries by class-method pairs"
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


def _clear_nl2test_output_artifacts(
    output_dir: Path, reset_evaluation_results: bool
) -> None:
    if not output_dir.exists():
        return

    root_log = output_dir / "nl2test.log"
    if root_log.exists():
        root_log.unlink()

    for project_dir in output_dir.iterdir():
        if not project_dir.is_dir():
            continue

        if reset_evaluation_results:
            results_path = project_dir / "nl2test_evaluation_results.json"
            if results_path.exists():
                results_path.unlink()

        temp_analysis = project_dir / "temp_analysis"
        if temp_analysis.exists():
            shutil.rmtree(temp_analysis)

        for log_file in project_dir.glob("*.log"):
            log_file.unlink()


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
    llm_model: Annotated[
        str,
        typer.Option(
            help="LLM model ID to use for generating Test2NL descriptions.",
            show_default=False,
        ),
    ],
    organized_methods_dir: Annotated[
        str,
        typer.Option(
            help="Path to the directory containing per-project folders of filtered methods (each holding the filtered JSON file).",
            show_default=False,
        ),
    ],
    organized_methods_file_name: Annotated[
        str,
        typer.Option(
            help="Name of the JSON file containing filtered methods for each project within organized_methods_dir.",
            show_default=False,
        ),
    ] = "nl2test.json",
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
    """
    Generate Test2NL descriptions for methods that passed the filtering pipeline.

    The organized_methods_dir must contain per-project folders with a JSON filter
    that enumerates the methods to process. Only methods listed in that file are
    used when producing descriptions.
    """
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
    organized_methods_file_name = organized_methods_file_name.strip()
    if not organized_methods_file_name:
        raise Exception("Parameter --organized-methods-file-name cannot be empty.")
    if Path(organized_methods_file_name).suffix.lower() != ".json":
        raise Exception(
            "Parameter --organized-methods-file-name must point to a JSON file name."
        )

    if not (methods_root.exists() and methods_root.is_dir()):
        raise Exception(
            f"Organized methods directory {methods_root} does not exist or is not a directory."
        )

    if not (analysis_root.exists() and analysis_root.is_dir()):
        raise Exception(
            f"Analysis directory {analysis_root} does not exist or is not a directory."
        )

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
            f"No project directories with {organized_methods_file_name} found under {methods_root}"
        )
        return

    data_manager = StructuredDataManager(output_dir)
    if clear_dataset:
        RichLog.info("Clearing the existing Test2NL dataset at the output directory.")
        targets = ["descriptions.json", "test2nl.csv"]
        data_manager.delete_many(targets)

    RichLog.info(
        f"Found {len(project_dirs)} project(s) with {organized_methods_file_name}: {[p.name for p in project_dirs]}"
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
            RichLog.info(str(sep))

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
def run_nl2test(
    base_project_dir: Annotated[
        str,
        typer.Option(
            help="Path to the base directory containing all project directories.",
            show_default=False,
        ),
    ],
    base_analysis_dir: Annotated[
        str,
        typer.Option(
            help="Path to the base directory containing per-project analysis.json directories.",
            show_default=False,
        ),
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            help="Path to the output directory for saving NL2Test generation results.",
            show_default=False,
        ),
    ],
    reset_evaluation_results: Annotated[
        bool,
        typer.Option(
            help="Whether to remove existing NL2Test evaluation results before running.",
            show_default=True,
        ),
    ] = True,
    use_stored_index: Annotated[
        bool,
        typer.Option(
            help="Reuse cached FAISS indexes when available instead of rebuilding them.",
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
    can_parallel_tool: Annotated[
        bool,
        typer.Option(
            help="Whether the LLM client may issue parallel tool calls.",
            show_default=True,
        ),
    ] = True,
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
    debug: Annotated[
        bool,
        typer.Option(
            help="Enable debug logging for more verbose output.",
            show_default=True,
        ),
    ] = False,
    log_file: Annotated[
        str | None,
        typer.Option(
            help="Optional log file name to write under --output-dir.",
            show_default=False,
        ),
    ] = None,
    exclude_test_dirs: Annotated[
        bool,
        typer.Option(
            help="Skip files under Maven test directories when preparing indexes.",
            show_default=True,
        ),
    ] = False,
    enable_reasoning: Annotated[
        bool,
        typer.Option(
            help="Enable extended reasoning for LLM calls (for reasoning models like MiniMax M2.1).",
            show_default=True,
        ),
    ] = True,
    reasoning_effort: Annotated[
        str,
        typer.Option(
            help="Reasoning effort level. One of: none, minimal, low, medium, high, xhigh.",
            show_default=True,
        ),
    ] = "medium",
    exclude_reasoning: Annotated[
        bool,
        typer.Option(
            help="Exclude reasoning content from LLM responses (set to False for MiniMax models to see thinking between tool calls).",
            show_default=True,
        ),
    ] = True,
    max_tokens: Annotated[
        int,
        typer.Option(
            help="Maximum tokens for LLM completion output. Increase for reasoning models (e.g., 32768 or 65536 for MiniMax M2.1).",
            show_default=True,
        ),
    ] = 16384,
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

    valid_reasoning_efforts = {"none", "minimal", "low", "medium", "high", "xhigh"}
    reasoning_effort = reasoning_effort.strip().lower()
    if reasoning_effort not in valid_reasoning_efforts:
        raise Exception(
            f"Invalid --reasoning-effort: {reasoning_effort}. Must be one of {sorted(valid_reasoning_efforts)}"
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

    base_project_dir = Path(base_project_dir).expanduser().resolve()
    if not (base_project_dir.exists() and base_project_dir.is_dir()):
        raise Exception(f"Base project directory {base_project_dir} does not exist.")

    base_analysis_dir = Path(base_analysis_dir).expanduser().resolve()
    if not (base_analysis_dir.exists() and base_analysis_dir.is_dir()):
        raise Exception(f"Base analysis directory {base_analysis_dir} does not exist.")

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise Exception(f"Output path {output_dir} is not a directory.")
    if not output_dir.exists():
        # Ensure output directory exists so results can be written.
        output_dir.mkdir(parents=True, exist_ok=True)

    RichLog.info(f"Resetting submodules under {base_project_dir}")
    GitUtilities.reset_submodules_in_dir(base_project_dir)

    # Load and prepare NL2Test inputs grouped by project
    nl2test_inputs_by_project = _load_nl2_inputs_by_project_from_csv(
        test2nl_file, max_entries, num_proj_parallel
    )

    _clear_nl2test_output_artifacts(output_dir, reset_evaluation_results)

    # Configure logging
    actor_log_file_name: str | None = None
    if debug:
        RichLog.set_level(logging.DEBUG)
        RichLog.debug("Debug logging enabled.")
    if log_file:
        file_name = Path(log_file).name
        actor_log_file_name = file_name
        try:
            # Always save the log file under the provided output_dir.
            file_path = output_dir / file_name
            RichLog.add_file_handler(str(file_path), overwrite=True)
            RichLog.info(f"Writing logs to file: {file_path}")
        except Exception as exc:
            RichLog.warn(f"Failed to add file handler at {log_file}: {exc}")

    RichLog.info(
        "NL2Test run configuration: "
        f"llm_model={llm_model}, emb_model={emb_model}, provider={llm_provider}, emb_provider={emb_provider}, "
        f"projects_dir={base_project_dir}, analysis_dir={base_analysis_dir}, out={output_dir}, max_inflight={max_inflight or num_proj_parallel}"
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
            RichLog.info(str(sep))

            actor = NL2TestActor.options(max_concurrency=1).remote(
                project_name=project_name,
                base_project_dir=str(base_project_dir),
                base_analysis_dir=str(base_analysis_dir),
                output_dir=str(output_dir),
                llm_model=llm_model,
                can_parallel_tool=can_parallel_tool,
                emb_model=emb_model,
                llm_provider=llm_provider,
                llm_api_url=llm_api_url,
                emb_provider=emb_provider,
                emb_api_url=emb_api_url,
                decomposition_mode=decomposition_mode.value,
                supervisor_max_iters=supervisor_max_iters,
                localization_max_iters=localization_max_iters,
                composition_max_iters=composition_max_iters,
                use_stored_index=use_stored_index,
                debug=bool(debug),
                log_file_name=actor_log_file_name,
                exclude_test_dirs=exclude_test_dirs,
                reasoning_enabled=enable_reasoning,
                reasoning_effort=reasoning_effort,
                exclude_reasoning=exclude_reasoning,
                max_tokens=max_tokens,
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

            if results and isinstance(results, list):
                compilation_failure = next(
                    (
                        item
                        for item in results
                        if isinstance(item, dict)
                        and item.get("project_compilation_failed")
                    ),
                    None,
                )
                if compilation_failure:
                    error_message = compilation_failure.get("error")
                    RichLog.error(
                        f"[{project_name}] {error_message or 'Project failed baseline compilation; skipping project.'}"
                    )
                    files_with_errors = (
                        compilation_failure.get("files_with_errors") or []
                    )
                    if files_with_errors:
                        RichLog.error(
                            f"[{project_name}] Files with compilation errors: {files_with_errors}"
                        )
                    error_details = compilation_failure.get("error_details") or []
                    if error_details:
                        RichLog.error(
                            f"[{project_name}] Compilation error details:\n"
                            + "\n".join(error_details)
                        )
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
                    # Log details to help pinpoint failing inputs
                    err = item.get("error")
                    err_type = item.get("error_type") or "Exception"
                    payload = item.get("input") or {}
                    RichLog.error(
                        f"[{project_name}] Failed input id={payload.get('id')} "
                        f"{payload.get('qualified_class_name')}::{payload.get('method_signature')} "
                        f"-> {err_type}: {err}"
                    )
                    tb = item.get("traceback")
                    if tb:
                        # Tracebacks can be long; emit only at debug level unless debug is off.
                        RichLog.debug(tb)
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
