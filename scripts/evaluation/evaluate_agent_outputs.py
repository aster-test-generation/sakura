"""Evaluate agent-generated test outputs against Test2NL entries."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from cldk import CLDK
from cldk.analysis import AnalysisLevel

from nltest.test2nl.model.models import AbstractionLevel, Test2NLEntry
from nltest.utils.analysis import CommonAnalysis
from nltest.utils.compilation.maven import JavaMavenCompilation
from nltest.utils.evaluation import TestGrader
from nltest.utils.file_io.test_file_manager import TestFileInfo, TestFileManager
from nltest.utils.models import (
    NL2TestCoverageEval,
    NL2TestEval,
    NL2TestInput,
    NL2TestMetadata,
    NL2TestStructuralEval,
)
from nltest.utils.pretty.color_logger import RichLog
from nltest.utils.utilities import test2nl_entry_to_nl2test_input
from nltest.utils.vcs.git_utils import GitUtilities

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
AGENT_OUTPUT_DIR = ROOT_DIR / "resources" / "agent_outputs"
TEST2NL_DIR = ROOT_DIR / "resources" / "test2nl" / "filtered_dataset"
TEST2NL_FILE_NAME = "test2nl.csv"
PROJECTS_DIR = ROOT_DIR / "resources" / "datasets"
TEMP_ANALYSIS_DIR = ROOT_DIR / "resources" / "temp_analysis"
OUTPUT_DIR = ROOT_DIR / "resources" / "agent_outputs" / "evaluation"
OUTPUT_FILE_NAME = "nl2test_evaluation_results.json"
RUN_DIR_NAME = "run_001"
GENERATED_TESTS_DIR_NAME = "generated_tests"
METADATA_FILE_NAME = "metadata.json"


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


def build_empty_eval(nl2_input: NL2TestInput) -> NL2TestEval:
    return NL2TestEval(
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


def evaluate_entry(
    context: ProjectContext, entry: Test2NLEntry, entry_dir: Path
) -> NL2TestEval:
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
    if not generated_file_entry:
        RichLog.warn(f"No generated_files entry in metadata: {metadata_path}")
        result.structured_eval = zero_structural_eval()
        result.coverage_eval = zero_coverage_eval()
        return result

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

        if not result.nl2test_metadata.code.strip():
            result.structured_eval = zero_structural_eval()
            result.coverage_eval = zero_coverage_eval()
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
    context: ProjectContext, entries: dict[int, Test2NLEntry], project_dir: Path
) -> list[NL2TestEval]:
    results: list[NL2TestEval] = []
    entry_dirs = [path for path in project_dir.iterdir() if path.is_dir()]

    for entry_dir in sorted(entry_dirs, key=lambda path: path.name):
        try:
            entry_id = int(entry_dir.name)
        except ValueError:
            RichLog.warn(
                f"Skipping non-numeric entry directory {entry_dir} in {project_dir}."
            )
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


def write_results(project_name: str, results: list[NL2TestEval]) -> None:
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

    if not AGENT_OUTPUT_DIR.is_dir():
        raise FileNotFoundError(f"Agent output directory not found: {AGENT_OUTPUT_DIR}")

    project_dirs = [path for path in AGENT_OUTPUT_DIR.iterdir() if path.is_dir()]
    for project_dir in sorted(project_dirs, key=lambda path: path.name):
        context = build_project_context(project_dir.name)
        if not context:
            continue
        results = collect_project_results(context, entry_lookup, project_dir)
        if results:
            write_results(context.project_name, results)
        else:
            RichLog.warn(f"No evaluations produced for {context.project_name}.")


if __name__ == "__main__":
    main()
