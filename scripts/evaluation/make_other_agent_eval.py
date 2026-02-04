"""Evaluate agent-generated test outputs against Test2NL entries."""

from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import ray
from cldk import CLDK
from cldk.analysis import AnalysisLevel

from sakura.test2nl.model.models import AbstractionLevel, Test2NLEntry
from sakura.utils.analysis import CommonAnalysis
from sakura.utils.compilation.maven import JavaMavenCompilation
from sakura.utils.evaluation import TestGrader
from sakura.utils.file_io.test_file_manager import TestFileInfo, TestFileManager
from sakura.utils.models import (
    NL2TestCoverageEval,
    NL2TestInput,
    NL2TestMetadata,
    NL2TestStructuralEval,
    OutOfBoxAgentEval,
)
from sakura.utils.pretty.color_logger import RichLog
from sakura.utils.utilities import test2nl_entry_to_nl2test_input
from sakura.utils.vcs.git_utils import GitUtilities

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
AGENT_OUTPUT_DIR = ROOT_DIR / "resources" / "outputs" / "gemini_cli_flash_output"
TEST2NL_DIR = ROOT_DIR / "resources" / "test2nl" / "filtered_dataset"
TEST2NL_FILE_NAME = "test2nl.csv"
PROJECTS_DIR = ROOT_DIR / "resources" / "datasets"
TEMP_ANALYSIS_DIR = ROOT_DIR / "resources" / "temp_analysis"
OUTPUT_DIR = (
    ROOT_DIR / "resources" / "outputs" / "gemini_cli_flash_output" / "evaluation"
)
OUTPUT_FILE_NAME = "nl2test_evaluation_results.json"
RUN_DIR_NAME = "run_001"
GENERATED_TESTS_DIR_NAME = "generated_tests"
METADATA_FILE_NAME = "metadata.json"

MAX_ENTRIES = 0  # 0 for all entries
NUM_PROJ_PARALLEL = 20  # Maximum number of projects to process concurrently
TARGET_PROJECTS: list[str] = ["commons-fileupload"]  # Empty for all projects, or specify e.g. ["commons-io", "commons-lang"]
SKIP_DIRS = {"evaluation"}  # Directories to skip when iterating agent outputs


@dataclass(frozen=True)
class ProjectContext:
    project_name: str
    project_root: Path
    analysis_dir: Path
    common: CommonAnalysis
    test_grader: TestGrader


def load_test2nl_entries(csv_path: Path) -> list[Test2NLEntry]:
    entries: list[Test2NLEntry] = []
    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            abstraction_value = row.get("abstraction_level", "").strip()
            abstraction_level = (
                AbstractionLevel(abstraction_value) if abstraction_value else None
            )
            is_bdd_value = row.get("is_bdd", "False").strip().lower()
            entries.append(
                Test2NLEntry(
                    id=int(row["id"]),
                    description=row["description"],
                    project_name=row["project_name"],
                    qualified_class_name=row["qualified_class_name"],
                    method_signature=row["method_signature"],
                    abstraction_level=abstraction_level,
                    is_bdd=is_bdd_value == "true",
                )
            )
    return entries


def index_test2nl_entries(
    entries: Iterable[Test2NLEntry],
) -> dict[int, Test2NLEntry]:
    return {entry.id: entry for entry in entries}


def build_empty_eval(
    nl2_input: NL2TestInput,
    *,
    failed_test_file_generation: bool = False,
    failed_code_generation: bool = False,
) -> OutOfBoxAgentEval:
    return OutOfBoxAgentEval(
        compiles=False,
        nl2test_input=nl2_input,
        nl2test_metadata=NL2TestMetadata(qualified_test_class_name="", code=""),
        structured_eval=None,
        coverage_eval=None,
        localization_eval=None,
        tool_log=None,
        input_tokens=0,
        output_tokens=0,
        llm_calls=0,
        failed_test_file_generation=failed_test_file_generation,
        failed_code_generation=failed_code_generation,
    )


def zero_structural_eval() -> NL2TestStructuralEval:
    return NL2TestStructuralEval(
        obj_creation_recall=0.0,
        obj_creation_precision=0.0,
        assertion_recall=0.0,
        assertion_precision=0.0,
        callable_recall=0.0,
        callable_precision=0.0,
        focal_recall=0.0,
        focal_precision=0.0,
    )


def zero_coverage_eval() -> NL2TestCoverageEval:
    return NL2TestCoverageEval(
        class_coverage=0.0,
        method_coverage=0.0,
        line_coverage=0.0,
        branch_coverage=0.0,
    )


def ensure_clean_submodule(project_root: Path) -> None:
    if GitUtilities.has_working_tree_changes(project_root):
        RichLog.warn(f"Local changes detected in {project_root}; resetting submodule.")
        GitUtilities.reset_submodule(project_root)
    if GitUtilities.has_working_tree_changes(project_root):
        raise RuntimeError(
            f"Submodule {project_root} still has local changes after reset."
        )


def build_project_context(project_name: str) -> ProjectContext | None:
    project_root = PROJECTS_DIR / project_name
    if not project_root.is_dir():
        RichLog.warn(f"Project directory not found: {project_root}")
        return None

    ensure_clean_submodule(project_root)

    analysis_dir = TEMP_ANALYSIS_DIR / project_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    analysis = CLDK(language="java").analysis(
        project_path=project_root,
        analysis_backend_path=None,
        analysis_level=AnalysisLevel.symbol_table,
        analysis_json_path=analysis_dir,
        eager=False,
    )

    common = CommonAnalysis(analysis)
    _, application_classes, test_utility_classes = common.categorize_classes()
    test_grader = TestGrader(
        analysis=analysis,
        project_root=project_root,
        application_classes=application_classes,
        test_utility_classes=test_utility_classes,
    )

    return ProjectContext(
        project_name=project_name,
        project_root=project_root,
        analysis_dir=analysis_dir,
        common=common,
        test_grader=test_grader,
    )


def regenerate_analysis(context: ProjectContext, *, eager: bool = True) -> Any:
    context.analysis_dir.mkdir(parents=True, exist_ok=True)
    return CLDK(language="java").analysis(
        project_path=context.project_root,
        analysis_backend_path=None,
        analysis_level=AnalysisLevel.symbol_table,
        analysis_json_path=context.analysis_dir,
        eager=eager,
    )


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_token_counts(metadata: dict[str, Any]) -> tuple[int, int]:
    tokens = metadata.get("tokens", {}) if isinstance(metadata, dict) else {}
    if not isinstance(tokens, dict):
        return 0, 0
    input_tokens = int(tokens.get("input_tokens", 0))
    output_tokens = int(tokens.get("output_tokens", 0))
    thoughts_tokens = int(
        tokens.get("thoughts_tokens", tokens.get("thought_tokens", 0))
    )
    return input_tokens, output_tokens + thoughts_tokens


def resolve_test_base_rel(test_base_dir: Path, project_root: Path) -> Path:
    if test_base_dir.is_absolute():
        try:
            return test_base_dir.relative_to(project_root)
        except ValueError:
            return test_base_dir
    return test_base_dir


def strip_test_base_marker(path: Path) -> Path:
    marker = ("src", "test", "java")
    parts = path.parts
    for idx in range(len(parts) - len(marker) + 1):
        if parts[idx : idx + len(marker)] == marker:
            return Path(*parts[idx + len(marker) :])
    return path


def derive_qualified_class_name(
    relative_path: Path, test_base_dir: Path, project_root: Path
) -> str:
    test_base_dir = Path(test_base_dir)
    project_root = Path(project_root)
    candidate_path = relative_path

    if candidate_path.is_absolute():
        if test_base_dir.is_absolute():
            try:
                class_path = candidate_path.relative_to(test_base_dir)
                return ".".join(class_path.with_suffix("").parts)
            except ValueError:
                pass
        try:
            candidate_path = candidate_path.relative_to(project_root)
        except ValueError:
            class_path = strip_test_base_marker(candidate_path)
            return ".".join(class_path.with_suffix("").parts)

    test_base_rel = resolve_test_base_rel(test_base_dir, project_root)
    try:
        class_path = candidate_path.relative_to(test_base_rel)
    except ValueError:
        class_path = strip_test_base_marker(candidate_path)
    return ".".join(class_path.with_suffix("").parts)


def resolve_generated_test_path(
    generated_tests_dir: Path, metadata: dict[str, Any], project_root: Path
) -> Path | None:
    candidates: list[Path] = []
    generated_files = (
        metadata.get("generated_files") if isinstance(metadata, dict) else []
    )
    if isinstance(generated_files, list) and generated_files:
        first = generated_files[0] if isinstance(generated_files[0], dict) else None
        if first:
            path_value = first.get("path") or first.get("absolute_path")
            if path_value:
                candidate = Path(path_value)
                candidates.append(candidate)
                if candidate.is_absolute():
                    try:
                        rel_candidate = candidate.relative_to(project_root)
                    except ValueError:
                        rel_candidate = None
                    if rel_candidate is not None:
                        candidates.append(generated_tests_dir / rel_candidate)
                else:
                    candidates.append(generated_tests_dir / candidate)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    java_files = list(generated_tests_dir.rglob("*.java"))
    if len(java_files) == 1:
        return java_files[0]
    if len(java_files) > 1:
        RichLog.warn(
            f"Multiple generated test files found in {generated_tests_dir}; skipping."
        )
    return None


def first_generated_file(metadata: dict[str, Any]) -> dict[str, Any] | None:
    generated_files = (
        metadata.get("generated_files") if isinstance(metadata, dict) else []
    )
    if not isinstance(generated_files, list) or not generated_files:
        return None
    entry = generated_files[0]
    return entry if isinstance(entry, dict) else None


def load_generated_code(
    generated_file_path: Path | None, metadata: dict[str, Any]
) -> str:
    if generated_file_path and generated_file_path.is_file():
        return generated_file_path.read_text(encoding="utf-8")
    first = first_generated_file(metadata)
    if first:
        content = first.get("content")
        if isinstance(content, str):
            return content
    raise FileNotFoundError("Generated test code could not be loaded.")


def matches_error_path(error_path: str, qualified_class_name: str) -> bool:
    pred_rel_path = qualified_class_name.replace(".", "/") + ".java"
    pred_simple_file = qualified_class_name.rsplit(".", 1)[-1] + ".java"
    normalized = error_path.replace("\\", "/")
    if "/" in normalized:
        return normalized.endswith(pred_rel_path)
    return normalized.endswith(pred_simple_file)


def extract_code_from_stdout(metadata: dict[str, Any]) -> tuple[str, str] | None:
    """Extract Java code from stdout.

    Looks for:
    1. A JSON object in stdout containing a "response" field with a markdown code block
    2. A write_file tool call with Java code in the content field

    Returns:
        Tuple of (code, file_path) or None if extraction fails.
        file_path may be empty string if extracted from response field.
    """
    import re

    stdout = metadata.get("stdout")
    if not isinstance(stdout, str) or not stdout.strip():
        return None

    # First, try extracting from response field with markdown code block
    response_match = re.search(r'"response"\s*:\s*"((?:[^"\\]|\\.)*)"', stdout)
    if response_match:
        response_raw = response_match.group(1)
        response = response_raw.encode().decode("unicode_escape")
        code_match = re.search(r"```(?:java)?\s*\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip(), ""

    # Second, try extracting from write_file tool call
    # Match the toolCall JSON structure to get both content and file_path
    tool_call_pattern = r'\[MESSAGE_BUS\] publish: (\{[^\n]+\})'
    for match in re.finditer(tool_call_pattern, stdout):
        try:
            msg = json.loads(match.group(1))
            tool_call = msg.get("toolCall", {})
            if tool_call.get("name") == "write_file":
                args = tool_call.get("args", {})
                content = args.get("content", "")
                file_path = args.get("file_path", "")
                if content and ("class " in content or "interface " in content or "package " in content):
                    return content, file_path
        except (json.JSONDecodeError, KeyError):
            continue

    return None


def parse_java_package_and_class(code: str) -> tuple[str, str] | None:
    """Extract package name and class name from Java source code.

    Returns (package_name, class_name) or None if parsing fails.
    Package name may be empty string for default package.
    """
    import re

    package_match = re.search(
        r"^\s*package\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*;",
        code,
        re.MULTILINE,
    )
    package_name = package_match.group(1) if package_match else ""

    class_match = re.search(
        r"(?:public\s+)?(?:abstract\s+|final\s+)?(?:class|interface|enum|record)\s+"
        r"([A-Za-z_]\w*)",
        code,
    )
    if not class_match:
        return None

    class_name = class_match.group(1)
    return package_name, class_name


def evaluate_entry(
    context: ProjectContext, entry: Test2NLEntry, entry_dir: Path
) -> OutOfBoxAgentEval:
    nl2_input = test2nl_entry_to_nl2test_input(entry)
    result = build_empty_eval(nl2_input)

    run_dir = entry_dir / RUN_DIR_NAME
    metadata_path = run_dir / METADATA_FILE_NAME
    generated_tests_dir = run_dir / GENERATED_TESTS_DIR_NAME

    if not metadata_path.is_file():
        RichLog.warn(f"Metadata file not found: {metadata_path}")
        return result

    metadata = load_metadata(metadata_path)
    input_tokens, output_tokens = extract_token_counts(metadata)
    result.input_tokens = input_tokens
    result.output_tokens = output_tokens

    generated_file_entry = first_generated_file(metadata)
    generated_code: str | None = None
    derived_qualified_name: str | None = None
    code_from_stdout = False
    stdout_file_path: str | None = None

    if not generated_file_entry:
        stdout_result = extract_code_from_stdout(metadata)
        if stdout_result:
            stdout_code, stdout_file_path = stdout_result
            RichLog.info(
                f"No generated_files entry, but found code in stdout: {metadata_path}"
            )
            result.failed_test_file_generation = True
            generated_code = stdout_code
            code_from_stdout = True

            # If we have a file_path from write_file, defer deriving qualified name
            # until test_base_dir is available. Otherwise, parse from code.
            if not stdout_file_path:
                parsed = parse_java_package_and_class(stdout_code)
                if parsed:
                    package_name, class_name = parsed
                    derived_qualified_name = (
                        f"{package_name}.{class_name}" if package_name else class_name
                    )
                else:
                    RichLog.warn("Could not parse package/class from stdout code")
                    result.structured_eval = zero_structural_eval()
                    result.coverage_eval = zero_coverage_eval()
                    return result
        else:
            RichLog.warn(f"No generated_files and no code in stdout: {metadata_path}")
            result.failed_code_generation = True
            result.structured_eval = zero_structural_eval()
            result.coverage_eval = zero_coverage_eval()
            return result
    else:
        generated_content = generated_file_entry.get("content")
        generated_path = generated_file_entry.get("path") or generated_file_entry.get(
            "absolute_path"
        )
        if generated_path is None and generated_content is None:
            RichLog.warn(f"Generated test content is empty in {metadata_path}")
            result.structured_eval = zero_structural_eval()
            result.coverage_eval = zero_coverage_eval()
            return result
        if generated_path is None:
            if not isinstance(generated_content, str) or not generated_content.strip():
                RichLog.warn(f"Generated test content is empty in {metadata_path}")
                result.structured_eval = zero_structural_eval()
                result.coverage_eval = zero_coverage_eval()
                return result

        if not generated_tests_dir.is_dir():
            RichLog.warn(f"Generated tests directory not found: {generated_tests_dir}")
            result.structured_eval = zero_structural_eval()
            result.coverage_eval = zero_coverage_eval()
            return result

    ensure_clean_submodule(context.project_root)

    module_root = context.common.resolve_module_root(nl2_input.qualified_class_name)
    test_base_dir = context.common.resolve_test_base_dir(
        module_root, project_root=context.project_root
    )

    # If code came from stdout with a file_path, derive qualified name from path
    if code_from_stdout and stdout_file_path:
        derived_qualified_name = derive_qualified_class_name(
            Path(stdout_file_path), test_base_dir, context.project_root
        )

    if not code_from_stdout:
        generated_file_path = resolve_generated_test_path(
            generated_tests_dir, metadata, context.project_root
        )
        if not generated_file_path:
            RichLog.warn(f"No generated Java file found in {generated_tests_dir}")
            result.structured_eval = zero_structural_eval()
            result.coverage_eval = zero_coverage_eval()
            return result

        try:
            generated_relative = generated_file_path.relative_to(generated_tests_dir)
        except ValueError:
            generated_relative = generated_file_path

        derived_qualified_name = derive_qualified_class_name(
            generated_relative, test_base_dir, context.project_root
        )

        try:
            generated_code = load_generated_code(generated_file_path, metadata)
        except FileNotFoundError as exc:
            RichLog.warn(f"{exc} ({generated_file_path})")
            result.structured_eval = zero_structural_eval()
            result.coverage_eval = zero_coverage_eval()
            return result

    if derived_qualified_name is None or generated_code is None:
        RichLog.warn("Internal error: qualified name or code not set")
        result.structured_eval = zero_structural_eval()
        result.coverage_eval = zero_coverage_eval()
        return result

    fm = TestFileManager(context.project_root, test_base_dir=test_base_dir)
    resolved_module_root = module_root or context.project_root
    test_info = TestFileInfo(
        qualified_class_name=derived_qualified_name,
        test_code=generated_code,
        id=nl2_input.id,
    )

    saved_qualified_name = ""
    try:
        saved_qualified_name, _ = fm.save_single(
            test_info,
            sync_names=True,
            encode_class_name=False,
        )
        result.nl2test_metadata = NL2TestMetadata(
            qualified_test_class_name=saved_qualified_name,
            code="",
            method_signature=nl2_input.method_signature,
        )

        compilation_errors = JavaMavenCompilation(
            context.project_root, module_root=resolved_module_root
        ).get_compilation_errors()
        result.compiles = not any(
            matches_error_path(error.file, saved_qualified_name)
            for error in compilation_errors
        )

        try:
            new_analysis = regenerate_analysis(context, eager=True)
        except Exception as exc:
            RichLog.warn(
                f"Regenerating analysis failed for {saved_qualified_name} (id={nl2_input.id}): {exc}"
            )
        else:
            context.test_grader.set_analysis(new_analysis)
            structured_eval, coverage_eval = context.test_grader.grade(
                nl2_input, result.nl2test_metadata, compiles=result.compiles
            )
            result.structured_eval = structured_eval
            result.coverage_eval = coverage_eval

        try:
            saved_code = fm.load(
                TestFileInfo(qualified_class_name=saved_qualified_name),
                encode_class_name=False,
            )
            result.nl2test_metadata.code = saved_code
        except FileNotFoundError:
            pass
    finally:
        if saved_qualified_name:
            try:
                fm.delete_single(
                    TestFileInfo(qualified_class_name=saved_qualified_name),
                    encode_class_name=False,
                )
            except Exception:
                GitUtilities.reset_submodule(context.project_root)

    return result


def collect_project_results(
    context: ProjectContext,
    entries: dict[int, Test2NLEntry],
    project_dir: Path,
    entry_ids_to_process: list[int] | None = None,
) -> list[OutOfBoxAgentEval]:
    results: list[OutOfBoxAgentEval] = []
    entry_dirs = [path for path in project_dir.iterdir() if path.is_dir()]

    for entry_dir in sorted(entry_dirs, key=lambda path: path.name):
        try:
            entry_id = int(entry_dir.name)
        except ValueError:
            RichLog.warn(
                f"Skipping non-numeric entry directory {entry_dir} in {project_dir}."
            )
            continue

        if entry_ids_to_process is not None and entry_id not in entry_ids_to_process:
            continue

        entry = entries.get(entry_id)
        if not entry:
            RichLog.warn(f"No Test2NL entry found for id={entry_id}")
            continue
        if entry.project_name != context.project_name:
            RichLog.warn(
                f"Entry id={entry_id} belongs to {entry.project_name}, not {context.project_name}."
            )

        results.append(evaluate_entry(context, entry, entry_dir))
    return results


@ray.remote
def evaluate_project_remote(
    project_name: str,
    entry_lookup: dict[int, dict[str, Any]],
    entry_ids_to_process: list[int],
) -> dict[str, Any]:
    """Ray remote function to evaluate a single project's entries sequentially."""
    entries = {
        entry_id: Test2NLEntry(
            id=data["id"],
            description=data["description"],
            project_name=data["project_name"],
            qualified_class_name=data["qualified_class_name"],
            method_signature=data["method_signature"],
            abstraction_level=(
                AbstractionLevel(data["abstraction_level"])
                if data.get("abstraction_level")
                else None
            ),
            is_bdd=data.get("is_bdd", False),
        )
        for entry_id, data in entry_lookup.items()
    }

    project_dir = AGENT_OUTPUT_DIR / project_name
    context = build_project_context(project_name)
    if not context:
        return {
            "project_name": project_name,
            "success": False,
            "error": f"Could not build project context for {project_name}",
            "results": [],
        }

    results = collect_project_results(
        context, entries, project_dir, entry_ids_to_process
    )
    return {
        "project_name": project_name,
        "success": True,
        "results": [r.model_dump() for r in results],
    }


def write_results(project_name: str, results: list[OutOfBoxAgentEval]) -> None:
    output_dir = OUTPUT_DIR / project_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / OUTPUT_FILE_NAME
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump([entry.model_dump() for entry in results], handle, indent=2)
    RichLog.info(f"Saved evaluation results to {output_file}")


def main() -> None:
    test2nl_file = TEST2NL_DIR / TEST2NL_FILE_NAME
    if not test2nl_file.is_file():
        raise FileNotFoundError(f"Test2NL CSV not found: {test2nl_file}")

    entries = load_test2nl_entries(test2nl_file)
    entry_lookup = index_test2nl_entries(entries)
    entry_lookup_serializable = {
        entry_id: {
            "id": entry.id,
            "description": entry.description,
            "project_name": entry.project_name,
            "qualified_class_name": entry.qualified_class_name,
            "method_signature": entry.method_signature,
            "abstraction_level": (
                entry.abstraction_level.value if entry.abstraction_level else None
            ),
            "is_bdd": entry.is_bdd,
        }
        for entry_id, entry in entry_lookup.items()
    }

    if not AGENT_OUTPUT_DIR.is_dir():
        raise FileNotFoundError(f"Agent output directory not found: {AGENT_OUTPUT_DIR}")

    project_dirs = [
        path
        for path in AGENT_OUTPUT_DIR.iterdir()
        if path.is_dir() and path.name not in SKIP_DIRS
    ]
    if TARGET_PROJECTS:
        target_set = set(TARGET_PROJECTS)
        project_dirs = [p for p in project_dirs if p.name in target_set]
        RichLog.info(
            f"TARGET_PROJECTS: processing {len(project_dirs)} projects: {TARGET_PROJECTS}"
        )
    project_dirs = sorted(project_dirs, key=lambda path: path.name)

    # Build per-project entry ID lists from agent output directories
    entries_by_project: dict[str, list[int]] = {}
    for project_dir in project_dirs:
        project_name = project_dir.name
        entry_ids: list[int] = []
        for entry_dir in project_dir.iterdir():
            if not entry_dir.is_dir():
                continue
            try:
                entry_id = int(entry_dir.name)
                if entry_id in entry_lookup:
                    entry_ids.append(entry_id)
            except ValueError:
                continue
        if entry_ids:
            entries_by_project[project_name] = sorted(entry_ids)

    # Apply MAX_ENTRIES limit across all projects
    if MAX_ENTRIES > 0:
        total_assigned = 0
        limited_entries_by_project: dict[str, list[int]] = {}
        for project_name, entry_ids in entries_by_project.items():
            if total_assigned >= MAX_ENTRIES:
                break
            remaining = MAX_ENTRIES - total_assigned
            limited_ids = entry_ids[:remaining]
            if limited_ids:
                limited_entries_by_project[project_name] = limited_ids
                total_assigned += len(limited_ids)
        entries_by_project = limited_entries_by_project
        RichLog.info(
            f"MAX_ENTRIES={MAX_ENTRIES}: processing {total_assigned} entries across {len(entries_by_project)} projects"
        )

    if not entries_by_project:
        RichLog.warn("No entries to process. Exiting.")
        return

    # Initialize Ray
    try:
        if not ray.is_initialized():
            ray.init()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Ray: {exc}") from exc

    # Project-level parallel scheduling
    pending_projects = deque(entries_by_project.keys())
    inflight_futures: set[ray.ObjectRef] = set()
    future_to_project: dict[ray.ObjectRef, str] = {}

    def launch_projects_up_to_limit() -> None:
        while len(inflight_futures) < NUM_PROJ_PARALLEL and pending_projects:
            project_name = pending_projects.popleft()
            entry_ids = entries_by_project[project_name]
            RichLog.info(
                f"Starting evaluation for project: {project_name} ({len(entry_ids)} entries)"
            )
            fut = evaluate_project_remote.remote(
                project_name, entry_lookup_serializable, entry_ids
            )
            inflight_futures.add(fut)
            future_to_project[fut] = project_name

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
                result: dict[str, Any] = ray.get(ref)
            except Exception as exc:
                RichLog.error(f"[{project_name}] Evaluation failed: {exc}")
                launch_projects_up_to_limit()
                continue

            if not result.get("success"):
                RichLog.warn(f"[{project_name}] {result.get('error', 'Unknown error')}")
                launch_projects_up_to_limit()
                continue

            results_data = result.get("results", [])
            if results_data:
                results = [OutOfBoxAgentEval(**r) for r in results_data]
                write_results(project_name, results)
                RichLog.info(
                    f"[{project_name}] Completed evaluation: {len(results)} results"
                )
            else:
                RichLog.warn(f"No evaluations produced for {project_name}.")

            launch_projects_up_to_limit()

    # Shutdown Ray
    try:
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
