"""Analyze motivation examples with complexity metrics and detailed data extraction.

This script provides comprehensive analysis of motivation examples identified by
identify_motivation_examples.py. It integrates Hamster complexity analysis,
extracts detailed data (code, evaluations, ground truth), and supports
flexible filtering and output options.

Usage:
    # Rank all motivation examples by complexity
    python analyze_motivation_examples.py --rank-by complexity

    # Print details for specific entry IDs
    python analyze_motivation_examples.py --ids 282 293 --print

    # Get top 10 compile gap examples by cyclomatic complexity
    python analyze_motivation_examples.py --category compile_gap --top 10 --rank-by cc

    # Export detailed data for top examples to JSON
    python analyze_motivation_examples.py --top 20 --output detailed_examples.json

    # Find best examples where NL2Test code is more similar to ground truth
    python analyze_motivation_examples.py --find-best-examples --min-complexity 10
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from cldk import CLDK
from cldk.analysis.java import JavaAnalysis

from sakura.utils.analysis.java_analyzer import CommonAnalysis

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

MOTIVATION_FILE = ROOT_DIR / "outputs" / "motivation" / "motivation_examples.json"
HAMSTER_DIR = ROOT_DIR / "resources" / "hamster"
NL2TEST_EVAL_DIR = ROOT_DIR / "outputs" / "raw_outputs" / "nl2test_gemini_pro_output"
GEMINI_CLI_EVAL_DIR = ROOT_DIR / "outputs" / "raw_outputs" / "gemini_cli_pro_output"
DATASETS_DIR = ROOT_DIR / "resources" / "datasets"
ANALYSIS_DIR = ROOT_DIR / "resources" / "analysis"
TEMP_ANALYSIS_DIR = ROOT_DIR / "resources" / "temp_analysis"

EVAL_FILE_NAME = "nl2test_evaluation_results.json"


def load_motivation_ids() -> dict[str, list[int]]:
    """Load all motivation example IDs organized by category."""
    with MOTIVATION_FILE.open("r") as f:
        motivation_data = json.load(f)

    all_ids: dict[str, list[int]] = {}

    for category in ["high_quality", "perfect", "perfect_high_localization"]:
        key = f"compile_gap_{category}"
        cat_data = motivation_data.get("compile_gap", {}).get(category, {})
        ids = []
        for bucket_data in cat_data.get("by_bucket", {}).values():
            ids.extend(bucket_data.get("entry_ids", []))
        all_ids[key] = ids

    for category in [
        "beats_other",
        "significant",
        "perfect",
        "perfect_significant",
        "perfect_high_localization",
        "perfect_significant_high_localization",
    ]:
        key = f"quality_gap_{category}"
        cat_data = motivation_data.get("quality_gap", {}).get(category, {})
        ids = []
        for bucket_data in cat_data.get("by_bucket", {}).values():
            ids.extend(bucket_data.get("entry_ids", []))
        all_ids[key] = ids

    return all_ids


def load_evaluations(eval_dir: Path) -> dict[int, dict[str, Any]]:
    """Load all evaluation entries from directory, indexed by ID."""
    entries_by_id: dict[int, dict[str, Any]] = {}

    if not eval_dir.exists():
        return entries_by_id

    for project_dir in sorted(eval_dir.iterdir()):
        if not project_dir.is_dir():
            continue

        eval_file = project_dir / EVAL_FILE_NAME
        if not eval_file.exists():
            continue

        try:
            with eval_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        if not isinstance(data, list):
            continue

        for entry_data in data:
            if not isinstance(entry_data, dict):
                continue
            entry_id = entry_data.get("nl2test_input", {}).get("id")
            if entry_id is not None and entry_id >= 0:
                entries_by_id[entry_id] = entry_data

    return entries_by_id


def load_hamster_analysis(project_name: str) -> dict[str, Any] | None:
    """Load Hamster analysis for a project."""
    hamster_file = HAMSTER_DIR / project_name / "hamster.json"
    if not hamster_file.exists():
        return None
    try:
        with hamster_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_test_method_analysis(
    hamster_data: dict[str, Any] | None,
    qualified_class_name: str,
    method_signature: str,
) -> dict[str, Any] | None:
    """Get TestMethodAnalysis from Hamster data."""
    if not hamster_data:
        return None

    for test_class in hamster_data.get("test_class_analyses", []):
        if test_class.get("qualified_class_name") == qualified_class_name:
            for test_method in test_class.get("test_method_analyses", []):
                if test_method.get("method_signature") == method_signature:
                    return test_method
    return None


def strip_java_comments_and_literals(code: str) -> str:
    """Strip Java comments and string/char literal contents.

    This helps keep identifier and type extraction focused on code, not
    documentation text or expected output strings.
    """
    out: list[str] = []
    i = 0
    state: str = "normal"

    while i < len(code):
        ch = code[i]
        nxt = code[i + 1] if i + 1 < len(code) else ""

        if state == "normal":
            if ch == "/" and nxt == "/":
                out.append(" ")
                out.append(" ")
                i += 2
                state = "line_comment"
                continue
            if ch == "/" and nxt == "*":
                out.append(" ")
                out.append(" ")
                i += 2
                state = "block_comment"
                continue
            if ch == '"':
                out.append('"')
                i += 1
                state = "string"
                continue
            if ch == "'":
                out.append("'")
                i += 1
                state = "char"
                continue
            out.append(ch)
            i += 1
            continue

        if state == "line_comment":
            if ch == "\n":
                out.append("\n")
                i += 1
                state = "normal"
                continue
            out.append(" ")
            i += 1
            continue

        if state == "block_comment":
            if ch == "*" and nxt == "/":
                out.append(" ")
                out.append(" ")
                i += 2
                state = "normal"
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue

        if state == "string":
            if ch == "\n":
                out.append("\n")
                i += 1
                state = "normal"
                continue
            if ch == "\\":
                out.append(" ")
                if i + 1 < len(code):
                    out.append(" ")
                i += 2
                continue
            if ch == '"':
                out.append('"')
                i += 1
                state = "normal"
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue

        # state == "char"
        if ch == "\n":
            out.append("\n")
            i += 1
            state = "normal"
            continue
        if ch == "\\":
            out.append(" ")
            if i + 1 < len(code):
                out.append(" ")
            i += 2
            continue
        if ch == "'":
            out.append("'")
            i += 1
            state = "normal"
            continue
        out.append("\n" if ch == "\n" else " ")
        i += 1

    return "".join(out)


def extract_java_identifiers(code: str | None) -> set[str]:
    """Extract Java identifiers (class names, method calls, variables) from code."""
    if not code:
        return set()
    code = strip_java_comments_and_literals(code)
    pattern = r"\b([A-Z][a-zA-Z0-9]*|[a-z][a-zA-Z0-9]*)\b"
    identifiers = set(re.findall(pattern, code))
    java_keywords = {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "default",
        "break",
        "continue",
        "return",
        "try",
        "catch",
        "finally",
        "throw",
        "throws",
        "new",
        "class",
        "interface",
        "extends",
        "implements",
        "public",
        "private",
        "protected",
        "static",
        "final",
        "abstract",
        "synchronized",
        "volatile",
        "transient",
        "native",
        "strictfp",
        "void",
        "boolean",
        "byte",
        "char",
        "short",
        "int",
        "long",
        "float",
        "double",
        "true",
        "false",
        "null",
        "this",
        "super",
        "instanceof",
        "import",
        "package",
        "assert",
        "enum",
    }
    return identifiers - java_keywords


def extract_method_calls(code: str | None) -> set[str]:
    """Extract method call names from Java code."""
    if not code:
        return set()
    code = strip_java_comments_and_literals(code)
    pattern = r"\.([a-z][a-zA-Z0-9]*)\s*\("
    return set(re.findall(pattern, code))


def extract_class_references(code: str | None) -> set[str]:
    """Extract class name references (PascalCase identifiers) from Java code."""
    if not code:
        return set()
    code = strip_java_comments_and_literals(code)
    pattern = r"\b([A-Z][a-zA-Z0-9]+)\b"
    classes = set(re.findall(pattern, code))
    common_types = {
        "String",
        "Integer",
        "Long",
        "Double",
        "Float",
        "Boolean",
        "Object",
        "Class",
        "Exception",
        "Test",
        "Override",
        "List",
        "Map",
        "Set",
        "Array",
        "Arrays",
        "Collections",
    }
    return classes - common_types


def extract_assertion_types(code: str | None) -> set[str]:
    """Extract assertion method names from code."""
    if not code:
        return set()
    code = strip_java_comments_and_literals(code)
    pattern = r"\b(assert[A-Za-z]+|assertEquals|assertTrue|assertFalse|assertNull|assertNotNull|assertThrows|assertArrayEquals|assertSame|assertNotSame)\b"
    return set(re.findall(pattern, code))


def compute_jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_sequence_similarity(code1: str | None, code2: str | None) -> float:
    """Compute sequence similarity between two code strings using SequenceMatcher."""
    if not code1 or not code2:
        return 0.0
    code1 = strip_java_comments_and_literals(code1)
    code2 = strip_java_comments_and_literals(code2)
    code1_normalized = re.sub(r"\s+", " ", code1.strip())
    code2_normalized = re.sub(r"\s+", " ", code2.strip())
    return SequenceMatcher(None, code1_normalized, code2_normalized).ratio()


def compute_code_similarity(
    generated_code: str | None, ground_truth_code: str | None
) -> dict[str, float]:
    """Compute multiple similarity metrics between generated and ground truth code."""
    gen_identifiers = extract_java_identifiers(generated_code)
    gt_identifiers = extract_java_identifiers(ground_truth_code)

    gen_methods = extract_method_calls(generated_code)
    gt_methods = extract_method_calls(ground_truth_code)

    gen_classes = extract_class_references(generated_code)
    gt_classes = extract_class_references(ground_truth_code)

    gen_assertions = extract_assertion_types(generated_code)
    gt_assertions = extract_assertion_types(ground_truth_code)

    return {
        "identifier_similarity": compute_jaccard_similarity(
            gen_identifiers, gt_identifiers
        ),
        "method_call_similarity": compute_jaccard_similarity(gen_methods, gt_methods),
        "class_reference_similarity": compute_jaccard_similarity(
            gen_classes, gt_classes
        ),
        "assertion_similarity": compute_jaccard_similarity(
            gen_assertions, gt_assertions
        ),
        "sequence_similarity": compute_sequence_similarity(
            generated_code, ground_truth_code
        ),
        "generated_classes": sorted(gen_classes),
        "ground_truth_classes": sorted(gt_classes),
        "generated_methods": sorted(gen_methods),
        "ground_truth_methods": sorted(gt_methods),
        "missing_classes": sorted(gt_classes - gen_classes),
        "extra_classes": sorted(gen_classes - gt_classes),
        "missing_methods": sorted(gt_methods - gen_methods),
        "extra_methods": sorted(gen_methods - gt_methods),
    }


def compute_combined_similarity(similarity_metrics: dict[str, Any]) -> float:
    """Compute a weighted combined similarity score."""
    weights = {
        "identifier_similarity": 0.2,
        "method_call_similarity": 0.3,
        "class_reference_similarity": 0.25,
        "assertion_similarity": 0.1,
        "sequence_similarity": 0.15,
    }
    total = sum(
        similarity_metrics.get(key, 0) * weight for key, weight in weights.items()
    )
    return total


# Global cache for CLDK analysis instances per project
_analysis_cache: dict[str, tuple[JavaAnalysis, CommonAnalysis] | None] = {}


def get_cached_analysis(
    project_name: str,
) -> tuple[JavaAnalysis, CommonAnalysis] | None:
    """Get or create cached CLDK analysis for a project."""
    if project_name in _analysis_cache:
        return _analysis_cache[project_name]

    project_dir = DATASETS_DIR / project_name
    if not project_dir.exists():
        _analysis_cache[project_name] = None
        return None

    analysis_project_dir = ANALYSIS_DIR / project_name
    if not (analysis_project_dir / "analysis.json").exists():
        analysis_project_dir = TEMP_ANALYSIS_DIR / project_name
    if not (analysis_project_dir / "analysis.json").exists():
        _analysis_cache[project_name] = None
        return None

    try:
        cldk_instance = CLDK(language="java")
        analysis: JavaAnalysis = cldk_instance.analysis(
            project_path=str(project_dir),
            analysis_json_path=str(analysis_project_dir),
            eager=False,
        )
        common = CommonAnalysis(analysis)
        _analysis_cache[project_name] = (analysis, common)
        return _analysis_cache[project_name]
    except Exception:
        _analysis_cache[project_name] = None
        return None


def get_ground_truth_code(
    project_name: str, qualified_class_name: str, method_signature: str
) -> tuple[
    str | None, int | None, dict[str, list[str]] | None, dict[str, list[str]] | None
]:
    """Get ground truth test code and fixture methods.

    Returns:
        Tuple of (test_code, ncloc, setup_methods, teardown_methods)
    """
    cached = get_cached_analysis(project_name)
    if cached is None:
        return None, None, None, None

    analysis, common = cached

    try:
        cldk_class_name = CommonAnalysis.get_cldk_class_name(qualified_class_name)
        cldk_method_signature = CommonAnalysis.get_cldk_method_sig(
            qualified_class_name, method_signature
        )
        method_details = analysis.get_method(cldk_class_name, cldk_method_signature)
        if not method_details:
            method_details = analysis.get_method(
                cldk_class_name,
                CommonAnalysis.simplify_method_signature(cldk_method_signature),
            )
        if not method_details:
            return None, None, None, None

        test_code = common.get_complete_method_code(
            method_details.declaration, method_details.code
        )
        ncloc = common.get_ncloc(method_details.declaration, method_details.code)
        setup_methods = common.get_setup_methods(cldk_class_name)
        teardown_methods = common.get_teardown_methods(cldk_class_name)

        return test_code, ncloc, setup_methods, teardown_methods

    except Exception:
        return None, None, None, None


def analyze_entry(
    entry_id: int,
    nl2test_entry: dict[str, Any],
    gemini_cli_entry: dict[str, Any] | None,
    hamster_cache: dict[str, dict[str, Any] | None],
    categories: list[str],
    include_code: bool = False,
) -> dict[str, Any]:
    """Analyze a single entry with complexity metrics and optional code."""
    nl2test_input = nl2test_entry.get("nl2test_input", {})
    project_name = nl2test_input.get("project_name", "")
    qualified_class_name = nl2test_input.get("qualified_class_name", "")
    method_signature = nl2test_input.get("method_signature", "")
    description = nl2test_input.get("description", "")
    abstraction_level = nl2test_input.get("abstraction_level", "")

    # Load hamster analysis
    if project_name not in hamster_cache:
        hamster_cache[project_name] = load_hamster_analysis(project_name)

    hamster_data = hamster_cache[project_name]
    test_analysis = get_test_method_analysis(
        hamster_data, qualified_class_name, method_signature
    )

    # Extract complexity metrics from Hamster
    cyclomatic_complexity = 0
    ncloc = 0
    if test_analysis:
        cyclomatic_complexity = test_analysis.get("cyclomatic_complexity", 0)
        ncloc = test_analysis.get("ncloc", 0)

    # Get evaluation metrics
    nl2test_compiles = nl2test_entry.get("compiles", False)
    gemini_cli_compiles = (
        gemini_cli_entry.get("compiles", False) if gemini_cli_entry else False
    )
    gemini_cli_failed = (
        gemini_cli_entry.get("failed_code_generation", False)
        if gemini_cli_entry
        else False
    )

    loc_eval = nl2test_entry.get("localization_eval", {})
    localization_recall = loc_eval.get("localization_recall", 0) if loc_eval else 0
    focal_method_count = len(loc_eval.get("all_focal_methods", [])) if loc_eval else 0

    structural_eval = nl2test_entry.get("structured_eval", {})
    coverage_eval = nl2test_entry.get("coverage_eval", {})

    result = {
        "entry_id": entry_id,
        "project_name": project_name,
        "qualified_class_name": qualified_class_name,
        "method_signature": method_signature,
        "abstraction_level": abstraction_level,
        "description_length": len(description),
        "categories": categories,
        "cyclomatic_complexity": cyclomatic_complexity,
        "ncloc": ncloc,
        "focal_method_count": focal_method_count,
        "localization_recall": localization_recall,
        "nl2test_compiles": nl2test_compiles,
        "gemini_cli_compiles": gemini_cli_compiles,
        "gemini_cli_failed_code_gen": gemini_cli_failed,
        "structural_eval": structural_eval,
        "coverage_eval": coverage_eval,
    }

    # Combined complexity score
    result["combined_score"] = (
        cyclomatic_complexity * max(1, ncloc) * max(1, focal_method_count)
    )

    if include_code:
        result["description"] = description
        nl2test_code = nl2test_entry.get("nl2test_metadata", {}).get("code", "")
        gemini_cli_code = (
            gemini_cli_entry.get("nl2test_metadata", {}).get("code", "")
            if gemini_cli_entry
            else ""
        )
        result["nl2test_code"] = nl2test_code
        result["gemini_cli_code"] = gemini_cli_code

        gt_code, gt_ncloc, setup, teardown = get_ground_truth_code(
            project_name, qualified_class_name, method_signature
        )
        result["ground_truth_code"] = gt_code
        result["ground_truth_ncloc"] = gt_ncloc
        result["setup_methods"] = setup
        result["teardown_methods"] = teardown

        # Compute similarity metrics
        nl2test_similarity = compute_code_similarity(nl2test_code, gt_code)
        gemini_cli_similarity = compute_code_similarity(gemini_cli_code, gt_code)

        result["nl2test_similarity"] = nl2test_similarity
        result["gemini_cli_similarity"] = gemini_cli_similarity

        nl2test_combined = compute_combined_similarity(nl2test_similarity)
        gemini_cli_combined = compute_combined_similarity(gemini_cli_similarity)

        result["nl2test_combined_similarity"] = nl2test_combined
        result["gemini_cli_combined_similarity"] = gemini_cli_combined
        result["similarity_advantage"] = nl2test_combined - gemini_cli_combined

        # Detect if Gemini CLI uses wrong target class
        gt_classes = set(nl2test_similarity.get("ground_truth_classes", []))
        gemini_classes = set(gemini_cli_similarity.get("generated_classes", []))
        nl2test_classes = set(nl2test_similarity.get("generated_classes", []))

        # Check if key classes from ground truth are missing in Gemini but present in NL2Test
        result["gemini_missing_key_classes"] = sorted(gt_classes - gemini_classes)
        result["nl2test_missing_key_classes"] = sorted(gt_classes - nl2test_classes)
        result["gemini_extra_classes"] = sorted(gemini_classes - gt_classes)
        result["nl2test_extra_classes"] = sorted(nl2test_classes - gt_classes)

    return result


def print_entry_details(entry: dict[str, Any], show_similarity: bool = True) -> None:
    """Print formatted details for an entry."""
    print("\n" + "=" * 100)
    print(f"ENTRY ID: {entry['entry_id']}")
    print("=" * 100)

    print(f"\nProject: {entry['project_name']}")
    print(f"Class: {entry['qualified_class_name']}")
    print(f"Method: {entry['method_signature']}")
    print(f"Abstraction Level: {entry['abstraction_level']}")

    print(
        f"\nComplexity: CC={entry['cyclomatic_complexity']}, NCLOC={entry['ncloc']}, "
        f"Focal Methods={entry['focal_method_count']}, Combined Score={entry['combined_score']:,}"
    )
    print(f"Localization Recall: {entry['localization_recall']}")
    print(f"Categories: {', '.join(entry['categories'][:3])}...")

    compile_status = "NL2Test: " + (
        "COMPILES" if entry["nl2test_compiles"] else "FAILS"
    )
    compile_status += " | Gemini CLI: "
    if entry["gemini_cli_failed_code_gen"]:
        compile_status += "NO CODE"
    elif entry["gemini_cli_compiles"]:
        compile_status += "COMPILES"
    else:
        compile_status += "FAILS"
    print(f"Compilation: {compile_status}")

    # Print similarity metrics if available
    if show_similarity and "nl2test_combined_similarity" in entry:
        print("\n" + "-" * 50)
        print("SIMILARITY TO GROUND TRUTH:")
        print("-" * 50)
        nl2test_sim = entry.get("nl2test_combined_similarity", 0)
        gemini_sim = entry.get("gemini_cli_combined_similarity", 0)
        advantage = entry.get("similarity_advantage", 0)
        print(f"  NL2Test Combined Similarity:    {nl2test_sim:.3f}")
        print(f"  Gemini CLI Combined Similarity: {gemini_sim:.3f}")
        print(f"  NL2Test Advantage:              {advantage:+.3f}")

        nl2test_detail = entry.get("nl2test_similarity", {})
        gemini_detail = entry.get("gemini_cli_similarity", {})
        print("\n  Detailed Similarity Breakdown:")
        print(f"    {'Metric':<25} {'NL2Test':>10} {'Gemini CLI':>12} {'Diff':>10}")
        print(f"    {'-' * 25} {'-' * 10} {'-' * 12} {'-' * 10}")
        for metric in [
            "identifier_similarity",
            "method_call_similarity",
            "class_reference_similarity",
            "assertion_similarity",
            "sequence_similarity",
        ]:
            nl2_val = nl2test_detail.get(metric, 0)
            gem_val = gemini_detail.get(metric, 0)
            diff = nl2_val - gem_val
            metric_name = metric.replace("_similarity", "").replace("_", " ").title()
            print(
                f"    {metric_name:<25} {nl2_val:>10.3f} {gem_val:>12.3f} {diff:>+10.3f}"
            )

        # Show class targeting issues
        gemini_missing = entry.get("gemini_missing_key_classes", [])
        gemini_extra = entry.get("gemini_extra_classes", [])
        nl2test_missing = entry.get("nl2test_missing_key_classes", [])

        if gemini_missing or gemini_extra:
            print("\n  Class Targeting Issues (Gemini CLI):")
            if gemini_missing:
                print(f"    Missing from GT: {', '.join(gemini_missing[:5])}")
            if gemini_extra:
                print(f"    Extra (not in GT): {', '.join(gemini_extra[:5])}")
        if nl2test_missing:
            print(f"\n  NL2Test Missing Classes: {', '.join(nl2test_missing[:5])}")

    if entry.get("description"):
        print("\n" + "-" * 50)
        print("DESCRIPTION:")
        print("-" * 50)
        desc = entry.get("description", "")
        print(desc[:500] + "..." if len(desc) > 500 else desc)

    if entry.get("ground_truth_code"):
        print("\n" + "-" * 50)
        print(f"GROUND TRUTH TEST (ncloc={entry.get('ground_truth_ncloc', 'N/A')}):")
        print("-" * 50)
        print(entry["ground_truth_code"])

    if entry.get("nl2test_code"):
        print("\n" + "-" * 50)
        print("NL2TEST GENERATED CODE:")
        print("-" * 50)
        print(entry["nl2test_code"])

    if entry.get("gemini_cli_code"):
        print("\n" + "-" * 50)
        print("GEMINI CLI GENERATED CODE:")
        print("-" * 50)
        print(entry["gemini_cli_code"] if entry["gemini_cli_code"] else "(no code)")


def print_summary_table(entries: list[dict[str, Any]], title: str) -> None:
    """Print a summary table of entries."""
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    for i, e in enumerate(entries):
        compile_status = "NL2Test:" + ("Y" if e["nl2test_compiles"] else "N")
        compile_status += " Gemini:"
        if e["gemini_cli_failed_code_gen"]:
            compile_status += "-"
        elif e["gemini_cli_compiles"]:
            compile_status += "Y"
        else:
            compile_status += "N"

        print(
            f"\n#{i + 1}: Entry {e['entry_id']} | CC={e['cyclomatic_complexity']} | "
            f"NCLOC={e['ncloc']} | Focal={e['focal_method_count']} | {compile_status}"
        )
        print(f"    Project: {e['project_name']}")
        print(f"    Method: {e['method_signature']}")
        print(
            f"    Abstraction: {e['abstraction_level']} | Loc Recall: {e['localization_recall']}"
        )


def print_similarity_table(entries: list[dict[str, Any]], title: str) -> None:
    """Print a summary table with similarity metrics."""
    print("\n" + "=" * 160)
    print(title)
    print("=" * 160)

    print(
        f"\n{'#':<3} {'ID':<6} {'CC':<3} {'NCLOC':<5} {'Focal':<5} {'LocRec':<7} "
        f"{'StructAvg':<10} {'CovAvg':<8} {'SimAdv':<8} {'Compile':<12} Project/Method"
    )
    print("-" * 160)

    for i, e in enumerate(entries):
        compile_status = ""
        if e["nl2test_compiles"]:
            compile_status += "NL2T:Y "
        else:
            compile_status += "NL2T:N "
        if e["gemini_cli_failed_code_gen"]:
            compile_status += "Gem:-"
        elif e["gemini_cli_compiles"]:
            compile_status += "Gem:Y"
        else:
            compile_status += "Gem:N"

        advantage = e.get("similarity_advantage", 0)

        # Compute structural and coverage averages
        struct = e.get("structural_eval", {})
        struct_avg = 0.0
        if struct:
            recalls = [
                struct.get("obj_creation_recall", 0),
                struct.get("assertion_recall", 0),
                struct.get("callable_recall", 0),
                struct.get("focal_recall", 0),
            ]
            struct_avg = sum(recalls) / len(recalls) if recalls else 0.0

        cov = e.get("coverage_eval", {})
        cov_avg = 0.0
        if cov:
            metrics = [
                cov.get("class_coverage", 0),
                cov.get("method_coverage", 0),
                cov.get("line_coverage", 0),
                cov.get("branch_coverage", 0),
            ]
            cov_avg = sum(metrics) / len(metrics) if metrics else 0.0

        print(
            f"{i + 1:<3} {e['entry_id']:<6} {e['cyclomatic_complexity']:<3} {e['ncloc']:<5} "
            f"{e['focal_method_count']:<5} {e['localization_recall']:<7.2f} "
            f"{struct_avg:<10.3f} {cov_avg:<8.3f} {advantage:<+8.3f} {compile_status:<12} "
            f"{e['project_name']}/{e['method_signature'][:30]}"
        )

        # Show class targeting issues if significant
        gemini_missing = e.get("gemini_missing_key_classes", [])
        if gemini_missing:
            print(
                f"    -> Gemini CLI missing key classes: {', '.join(gemini_missing[:3])}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze motivation examples with complexity metrics."
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        help="Specific entry IDs to analyze",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=[
            "compile_gap",
            "quality_gap",
            "compile_gap_perfect_high_localization",
            "quality_gap_perfect_significant_high_localization",
        ],
        help="Filter by category prefix",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top entries to show (default: 20)",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        choices=[
            "cc",
            "ncloc",
            "combined",
            "focal",
            "similarity_advantage",
            "best_example",
        ],
        default="combined",
        help="Metric to rank by (default: combined)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_details",
        help="Print detailed output including code",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--find-best-examples",
        action="store_true",
        help="Find best examples where NL2Test is significantly more similar to ground truth",
    )
    parser.add_argument(
        "--min-complexity",
        type=int,
        default=0,
        help="Minimum combined complexity score (CC * NCLOC * focal) to include",
    )
    parser.add_argument(
        "--min-localization",
        type=float,
        default=0.0,
        help="Minimum localization recall to include (default: 0.0)",
    )
    parser.add_argument(
        "--min-similarity-advantage",
        type=float,
        default=0.0,
        help="Minimum similarity advantage (NL2Test - Gemini) to include (default: 0.0)",
    )
    args = parser.parse_args()

    print("Loading motivation example IDs...")
    all_category_ids = load_motivation_ids()

    # Determine which IDs to analyze
    if args.ids:
        target_ids = set(args.ids)
    else:
        target_ids = set()
        for cat_name, cat_ids in all_category_ids.items():
            if args.category is None or cat_name.startswith(args.category):
                target_ids.update(cat_ids)

    print(f"Analyzing {len(target_ids)} entries...")

    print("Loading NL2Test evaluations...")
    nl2test_entries = load_evaluations(NL2TEST_EVAL_DIR)
    print(f"Loaded {len(nl2test_entries)} NL2Test entries")

    print("Loading Gemini CLI evaluations...")
    gemini_cli_entries = load_evaluations(GEMINI_CLI_EVAL_DIR)
    print(f"Loaded {len(gemini_cli_entries)} Gemini CLI entries")

    # Always include code for similarity analysis when using best_example ranking or find-best-examples
    include_code = (
        args.print_details
        or args.output
        or args.find_best_examples
        or args.rank_by in ["similarity_advantage", "best_example"]
    )

    hamster_cache: dict[str, dict[str, Any] | None] = {}
    results = []

    for entry_id in sorted(target_ids):
        if entry_id not in nl2test_entries:
            continue

        # Find categories for this entry
        entry_categories = [
            cat for cat, ids in all_category_ids.items() if entry_id in ids
        ]

        result = analyze_entry(
            entry_id,
            nl2test_entries[entry_id],
            gemini_cli_entries.get(entry_id),
            hamster_cache,
            entry_categories,
            include_code=include_code,
        )
        results.append(result)

    # Apply filters if specified
    if args.min_complexity > 0:
        results = [r for r in results if r["combined_score"] >= args.min_complexity]
        print(
            f"After complexity filter (>= {args.min_complexity}): {len(results)} entries"
        )

    if args.min_localization > 0:
        results = [
            r for r in results if r["localization_recall"] >= args.min_localization
        ]
        print(
            f"After localization filter (>= {args.min_localization}): {len(results)} entries"
        )

    if args.min_similarity_advantage > 0:
        results = [
            r
            for r in results
            if r.get("similarity_advantage", 0) >= args.min_similarity_advantage
        ]
        print(
            f"After similarity advantage filter (>= {args.min_similarity_advantage}): {len(results)} entries"
        )

    # Helper to compute average of structural recall metrics
    def get_structural_score(entry: dict[str, Any]) -> float:
        struct = entry.get("structural_eval", {})
        if not struct:
            return 0.0
        recalls = [
            struct.get("obj_creation_recall", 0),
            struct.get("assertion_recall", 0),
            struct.get("callable_recall", 0),
            struct.get("focal_recall", 0),
        ]
        return sum(recalls) / len(recalls) if recalls else 0.0

    # Helper to compute average of coverage metrics
    def get_coverage_score(entry: dict[str, Any]) -> float:
        cov = entry.get("coverage_eval", {})
        if not cov:
            return 0.0
        metrics = [
            cov.get("class_coverage", 0),
            cov.get("method_coverage", 0),
            cov.get("line_coverage", 0),
            cov.get("branch_coverage", 0),
        ]
        return sum(metrics) / len(metrics) if metrics else 0.0

    # Sort by specified metric
    sort_keys = {
        "cc": lambda x: x["cyclomatic_complexity"],
        "ncloc": lambda x: x["ncloc"],
        "combined": lambda x: x["combined_score"],
        "focal": lambda x: x["focal_method_count"],
        "similarity_advantage": lambda x: x.get("similarity_advantage", 0),
        "best_example": lambda x: (
            # Heavily weight NCLOC, CC, localization recall, and eval scores
            # Formula: (loc_recall^2) * (struct + cov) * sim_adv * (CC^1.5) * (NCLOC^1.2) * focal
            (x["localization_recall"] ** 2)
            * (get_structural_score(x) + get_coverage_score(x))
            * max(0.01, x.get("similarity_advantage", 0))
            * (max(1, x["cyclomatic_complexity"]) ** 1.5)
            * (max(1, x["ncloc"]) ** 1.2)
            * max(1, x["focal_method_count"])
        ),
    }
    sort_key = sort_keys[args.rank_by]

    results.sort(key=sort_key, reverse=True)

    # Limit to top N
    top_results = results[: args.top]

    # Output
    if args.print_details:
        for entry in top_results:
            print_entry_details(entry, show_similarity=include_code)
    elif args.find_best_examples or args.rank_by in [
        "similarity_advantage",
        "best_example",
    ]:
        rank_label = {
            "similarity_advantage": "SIMILARITY ADVANTAGE (NL2Test - Gemini)",
            "best_example": "BEST EXAMPLE SCORE (SimAdv * Complexity * LocRecall)",
        }.get(args.rank_by, f"TOP {len(top_results)} BEST EXAMPLES")
        print_similarity_table(
            top_results, f"TOP {len(top_results)} ENTRIES BY {rank_label}"
        )
    else:
        rank_label = {
            "cc": "CYCLOMATIC COMPLEXITY",
            "ncloc": "NCLOC",
            "combined": "COMBINED SCORE (CC * NCLOC * Focal)",
            "focal": "FOCAL METHOD COUNT",
        }[args.rank_by]
        print_summary_table(
            top_results, f"TOP {len(top_results)} ENTRIES BY {rank_label}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(top_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Print category-specific summaries
    if not args.ids and not args.print_details:
        compile_gap = [
            r
            for r in results
            if r["nl2test_compiles"]
            and not r["gemini_cli_compiles"]
            and not r["gemini_cli_failed_code_gen"]
        ]
        quality_gap = [
            r for r in results if r["nl2test_compiles"] and r["gemini_cli_compiles"]
        ]

        if compile_gap:
            compile_gap.sort(key=sort_key, reverse=True)
            print_summary_table(
                compile_gap[:10],
                "TOP 10 COMPILE GAP EXAMPLES (NL2Test compiles, Gemini CLI doesn't)",
            )

        if quality_gap:
            quality_gap.sort(key=sort_key, reverse=True)
            print_summary_table(
                quality_gap[:10],
                "TOP 10 QUALITY GAP EXAMPLES (Both compile, NL2Test better)",
            )


if __name__ == "__main__":
    main()
