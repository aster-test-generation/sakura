from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import javalang
from cldk import CLDK
from cldk.analysis.java import JavaAnalysis

from sakura.utils.analysis.java_analyzer import CommonAnalysis

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

NL2TEST_EVAL_DIR = ROOT_DIR / "outputs" / "raw_outputs" / "nl2test_gemini_pro_output"
GEMINI_CLI_EVAL_DIR = ROOT_DIR / "outputs" / "raw_outputs" / "gemini_cli_pro_output"

HAMSTER_DIR = ROOT_DIR / "resources" / "hamster"
DATASETS_DIR = ROOT_DIR / "resources" / "datasets"
ANALYSIS_DIR = ROOT_DIR / "resources" / "analysis"
TEMP_ANALYSIS_DIR = ROOT_DIR / "resources" / "temp_analysis"

EVAL_FILE_NAME = "nl2test_evaluation_results.json"


@dataclass(frozen=True)
class Similarity:
    combined: float
    type_sim: float
    method_sim: float
    assertion_sim: float


@dataclass(frozen=True)
class Candidate:
    entry_id: int
    project_name: str
    qualified_class_name: str
    method_signature: str
    abstraction_level: str
    ncloc: int
    cyclomatic_complexity: int
    localization_recall: float
    nl2test_compiles: bool
    gemini_cli_compiles: bool
    nl2test_similarity: Similarity
    gemini_cli_similarity: Similarity
    similarity_advantage: float
    focal_class_coverage_nl2test: float
    focal_class_coverage_gemini: float
    score: float


def load_evaluations(eval_dir: Path) -> dict[int, dict[str, Any]]:
    """Load evaluation entries from project subdirectories, indexed by entry id."""
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


def build_hamster_index() -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    """Build (project -> (qcn, sig) -> test_method_analysis) index from Hamster JSON."""
    hamster_index: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}

    if not HAMSTER_DIR.exists():
        return hamster_index

    for project_dir in sorted(HAMSTER_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        hamster_file = project_dir / "hamster.json"
        if not hamster_file.exists():
            continue

        try:
            with hamster_file.open("r", encoding="utf-8") as f:
                hamster_data = json.load(f)
        except Exception:
            continue

        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for test_class in hamster_data.get("test_class_analyses", []):
            qcn = test_class.get("qualified_class_name")
            for test_method in test_class.get("test_method_analyses", []):
                sig = test_method.get("method_signature")
                if qcn and sig:
                    by_key[(qcn, sig)] = test_method
        hamster_index[project_dir.name] = by_key

    return hamster_index


def strip_java_comments_and_literals(code: str) -> str:
    """Strip Java comments and string/char literal contents."""
    out: list[str] = []
    i = 0
    state = "normal"

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


def _wrap_method_in_dummy_class(method_code: str) -> str:
    return f"public class Dummy {{\n{method_code}\n}}"


def _fallback_semantic_sets(code: str) -> tuple[set[str], set[str], set[str]]:
    stripped = strip_java_comments_and_literals(code)
    types = set(re.findall(r"\b([A-Z][A-Za-z0-9_]*)\b", stripped))
    methods = set(re.findall(r"(?:\.|\b)([a-z][A-Za-z0-9_]*)\s*\(", stripped))
    assertions = {m for m in methods if m.startswith("assert")}
    return types, methods, assertions


def extract_semantic_sets(
    code: str, parse_as_compilation_unit: bool
) -> tuple[set[str], set[str], set[str]]:
    """Extract (types, methods, assertions) from Java code."""
    try:
        tree = javalang.parse.parse(
            code if parse_as_compilation_unit else _wrap_method_in_dummy_class(code)
        )
    except Exception:
        return _fallback_semantic_sets(code)

    types: set[str] = set()
    methods: set[str] = set()
    assertions: set[str] = set()

    for _, node in tree:
        if isinstance(node, javalang.tree.ReferenceType) and node.name:
            types.add(node.name)
        elif isinstance(node, javalang.tree.ClassCreator):
            typ = node.type
            if hasattr(typ, "name") and typ.name:
                types.add(typ.name)
        elif isinstance(
            node, (javalang.tree.MethodInvocation, javalang.tree.SuperMethodInvocation)
        ):
            member = node.member
            if member:
                methods.add(member)
                if member.startswith("assert"):
                    assertions.add(member)

    return types, methods, assertions


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_similarity(
    generated_code: str, ground_truth_method_code: str
) -> Similarity:
    gt_types, gt_methods, gt_assertions = extract_semantic_sets(
        ground_truth_method_code, parse_as_compilation_unit=False
    )
    gen_types, gen_methods, gen_assertions = extract_semantic_sets(
        generated_code, parse_as_compilation_unit=True
    )

    type_sim = jaccard_similarity(gen_types, gt_types)
    method_sim = jaccard_similarity(gen_methods, gt_methods)
    assertion_sim = jaccard_similarity(gen_assertions, gt_assertions)
    combined = 0.4 * method_sim + 0.4 * type_sim + 0.2 * assertion_sim
    return Similarity(
        combined=combined,
        type_sim=type_sim,
        method_sim=method_sim,
        assertion_sim=assertion_sim,
    )


_analysis_cache: dict[str, tuple[JavaAnalysis, CommonAnalysis] | None] = {}
_ground_truth_cache: dict[tuple[str, str, str], str | None] = {}


def get_cached_analysis(
    project_name: str,
) -> tuple[JavaAnalysis, CommonAnalysis] | None:
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


def get_ground_truth_method_code(
    project_name: str, qualified_class_name: str, method_signature: str
) -> str | None:
    key = (project_name, qualified_class_name, method_signature)
    if key in _ground_truth_cache:
        return _ground_truth_cache[key]

    cached = get_cached_analysis(project_name)
    if cached is None:
        _ground_truth_cache[key] = None
        return None

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
            _ground_truth_cache[key] = None
            return None

        test_code = common.get_complete_method_code(
            method_details.declaration, method_details.code
        )
        _ground_truth_cache[key] = test_code
        return test_code
    except Exception:
        _ground_truth_cache[key] = None
        return None


def extract_focal_class_simple_names(
    localization_eval: dict[str, Any] | None,
) -> set[str]:
    if not localization_eval:
        return set()
    all_focal_methods = localization_eval.get("all_focal_methods", []) or []
    classes: set[str] = set()
    for item in all_focal_methods:
        if not isinstance(item, str):
            continue
        head = item.split("(")[0]
        class_part = head.rsplit(".", 1)[0] if "." in head else ""
        cls = class_part.rsplit(".", 1)[-1] if class_part else ""
        if cls:
            classes.add(cls)
    return classes


def token_coverage(code: str, tokens: set[str]) -> float:
    if not tokens:
        return 0.0
    stripped = strip_java_comments_and_literals(code)
    hits = sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", stripped))
    return hits / len(tokens)


def compute_candidate_score(
    ncloc: int,
    cyclomatic_complexity: int,
    nl2test_sim: Similarity,
    gemini_sim: Similarity,
    focal_cov_gap: float,
    nl2test_compiles: bool,
    gemini_compiles: bool,
) -> float:
    advantage = nl2test_sim.combined - gemini_sim.combined
    if advantage <= 0:
        return 0.0

    complexity = (1 + math.log1p(max(0, ncloc))) * (
        1 + (max(0, cyclomatic_complexity) / 3.0)
    )
    compile_bonus = 1.15 if nl2test_compiles and not gemini_compiles else 1.0
    focal_bonus = 1.0 + max(0.0, focal_cov_gap)

    return (
        advantage
        * (nl2test_sim.combined**2)
        * (1 - gemini_sim.combined)
        * complexity
        * compile_bonus
        * focal_bonus
    )


def export_examples(
    candidates: list[Candidate],
    export_dir: Path,
    nl2test_entries: dict[int, dict[str, Any]],
    gemini_entries: dict[int, dict[str, Any]],
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    for c in candidates:
        nl_entry = nl2test_entries.get(c.entry_id)
        gem_entry = gemini_entries.get(c.entry_id)
        if nl_entry is None or gem_entry is None:
            continue

        nl_code = nl_entry.get("nl2test_metadata", {}).get("code", "")
        gem_code = gem_entry.get("nl2test_metadata", {}).get("code", "")
        gt_code = get_ground_truth_method_code(
            c.project_name, c.qualified_class_name, c.method_signature
        )
        if not nl_code or not gem_code or not gt_code:
            continue

        safe_method = re.sub(r"[^A-Za-z0-9_]+", "_", c.method_signature)[:80]
        folder = export_dir / f"{c.entry_id}_{c.project_name}_{safe_method}"
        folder.mkdir(parents=True, exist_ok=True)

        (folder / "ground_truth.java").write_text(gt_code, encoding="utf-8")
        (folder / "nl2test.java").write_text(nl_code, encoding="utf-8")
        (folder / "gemini_cli.java").write_text(gem_code, encoding="utf-8")
        (folder / "summary.json").write_text(
            json.dumps(asdict(c), indent=2), encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank semantic motivation candidates (NL2Test vs Gemini CLI)."
    )
    parser.add_argument(
        "--nl2test-dir",
        type=Path,
        default=NL2TEST_EVAL_DIR,
        help="Directory containing NL2Test evaluation results",
    )
    parser.add_argument(
        "--gemini-cli-dir",
        type=Path,
        default=GEMINI_CLI_EVAL_DIR,
        help="Directory containing Gemini CLI evaluation results",
    )
    parser.add_argument(
        "--min-ncloc",
        type=int,
        default=20,
        help="Minimum Hamster NCLOC for ground truth test (default: 20)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of top candidates to print (default: 25)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum candidate score to include (default: 0.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write ranked candidates to JSON (optional)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Export Java snippets for the top candidates (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading evaluation entries...")
    nl2test_entries = load_evaluations(args.nl2test_dir)
    gemini_entries = load_evaluations(args.gemini_cli_dir)
    matching_ids = sorted(set(nl2test_entries) & set(gemini_entries))
    print(f"NL2Test entries: {len(nl2test_entries)}")
    print(f"Gemini CLI entries: {len(gemini_entries)}")
    print(f"Matched entries: {len(matching_ids)}")

    print("Indexing Hamster complexity data...")
    hamster_index = build_hamster_index()

    candidates: list[Candidate] = []
    for entry_id in matching_ids:
        nl_entry = nl2test_entries[entry_id]
        gem_entry = gemini_entries[entry_id]

        nl2test_input = nl_entry.get("nl2test_input", {})
        project_name = nl2test_input.get("project_name", "")
        qualified_class_name = nl2test_input.get("qualified_class_name", "")
        method_signature = nl2test_input.get("method_signature", "")
        abstraction_level = nl2test_input.get("abstraction_level", "")

        if not project_name or not qualified_class_name or not method_signature:
            continue

        test_analysis = hamster_index.get(project_name, {}).get(
            (qualified_class_name, method_signature)
        )
        if not test_analysis:
            continue

        ncloc = int(test_analysis.get("ncloc", 0) or 0)
        cyclomatic_complexity = int(test_analysis.get("cyclomatic_complexity", 0) or 0)
        if ncloc < args.min_ncloc:
            continue

        nl_code = nl_entry.get("nl2test_metadata", {}).get("code", "")
        gem_code = gem_entry.get("nl2test_metadata", {}).get("code", "")
        if not nl_code or not gem_code:
            continue

        gt_code = get_ground_truth_method_code(
            project_name, qualified_class_name, method_signature
        )
        if not gt_code:
            continue

        loc_eval = nl_entry.get("localization_eval") or {}
        localization_recall = float(loc_eval.get("localization_recall", 0.0) or 0.0)

        nl_sim = compute_similarity(nl_code, gt_code)
        gem_sim = compute_similarity(gem_code, gt_code)
        advantage = nl_sim.combined - gem_sim.combined

        focal_classes = extract_focal_class_simple_names(loc_eval)
        nl_focal_cov = token_coverage(nl_code, focal_classes)
        gem_focal_cov = token_coverage(gem_code, focal_classes)
        focal_gap = nl_focal_cov - gem_focal_cov

        nl2test_compiles = bool(nl_entry.get("compiles", False))
        gemini_compiles = bool(gem_entry.get("compiles", False))
        score = compute_candidate_score(
            ncloc,
            cyclomatic_complexity,
            nl_sim,
            gem_sim,
            focal_gap,
            nl2test_compiles,
            gemini_compiles,
        )
        if score < args.min_score:
            continue

        candidates.append(
            Candidate(
                entry_id=entry_id,
                project_name=project_name,
                qualified_class_name=qualified_class_name,
                method_signature=method_signature,
                abstraction_level=abstraction_level,
                ncloc=ncloc,
                cyclomatic_complexity=cyclomatic_complexity,
                localization_recall=localization_recall,
                nl2test_compiles=nl2test_compiles,
                gemini_cli_compiles=gemini_compiles,
                nl2test_similarity=nl_sim,
                gemini_cli_similarity=gem_sim,
                similarity_advantage=advantage,
                focal_class_coverage_nl2test=nl_focal_cov,
                focal_class_coverage_gemini=gem_focal_cov,
                score=score,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    top_candidates = candidates[: args.top]

    print("\n" + "=" * 160)
    print(
        f"TOP {len(top_candidates)} SEMANTIC MOTIVATION CANDIDATES (min_ncloc={args.min_ncloc})"
    )
    print("=" * 160)
    print(
        f"\n{'#':<3} {'ID':<6} {'NCLOC':<5} {'CC':<3} {'LocRec':<7} "
        f"{'NL_Sim':<7} {'Gem_Sim':<7} {'Adv':<7} {'FocalGap':<8} {'Compile':<12} Project/Method"
    )
    print("-" * 160)
    for i, c in enumerate(top_candidates):
        compile_status = (
            f"NL2T:{'Y' if c.nl2test_compiles else 'N'} "
            f"Gem:{'Y' if c.gemini_cli_compiles else 'N'}"
        )
        focal_gap = c.focal_class_coverage_nl2test - c.focal_class_coverage_gemini
        print(
            f"{i + 1:<3} {c.entry_id:<6} {c.ncloc:<5} {c.cyclomatic_complexity:<3} {c.localization_recall:<7.2f} "
            f"{c.nl2test_similarity.combined:<7.3f} {c.gemini_cli_similarity.combined:<7.3f} {c.similarity_advantage:<+7.3f} "
            f"{focal_gap:<+8.2f} {compile_status:<12} {c.project_name}/{c.method_signature[:40]}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in top_candidates], f, indent=2)
        print(f"\nWrote: {args.output}")

    if args.export_dir:
        export_examples(
            top_candidates, args.export_dir, nl2test_entries, gemini_entries
        )
        print(f"\nExported examples to: {args.export_dir}")


if __name__ == "__main__":
    main()
