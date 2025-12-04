import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Dict, Any, Set, Tuple, List

CSV_REQUIRED_COLS = [
    "abstraction_level",
    "description",
    "id",
    "is_bdd",
    "method_signature",
    "project_name",
    "qualified_class_name",
]

def iter_subset_entries(json_data: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    """Yield (qualified_class_name, method_signature) from all keys starting with 'tests_' that are lists."""
    for key, value in json_data.items():
        if not (key.startswith("tests_") and isinstance(value, list)):
            continue
        for item in value:
            qcn = item.get("qualified_class_name")
            ms = item.get("method_signature")
            if qcn and ms:
                yield (qcn, ms)

def collect_pairs(root: Path, subset_glob: str = "subset_*.json", subset_name: str | None = None) -> Set[Tuple[str, str, str]]:
    """
    Return a set of (project_name, qualified_class_name, method_signature) from subset JSONs under root.
    If subset_name is provided, only files with that exact name are used. Otherwise, subset_glob is used.
    """
    triples: Set[Tuple[str, str, str]] = set()
    pattern = subset_name if subset_name else subset_glob

    for path in root.rglob(pattern):
        # Determine the project_name as the immediate child under root
        try:
            rel = path.parent.relative_to(root)
            project_name = rel.parts[0] if rel.parts else path.parent.name
        except Exception:
            project_name = path.parent.name

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue

        for qcn, ms in iter_subset_entries(data):
            triples.add((project_name, qcn, ms))

    return triples

def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in CSV_REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV missing required columns: {missing}")
        return list(reader)

def write_csv_rows(csv_path: Path, rows: Iterable[Dict[str, str]], fieldnames: List[str], minimal: bool):
    if minimal:
        fieldnames = ["project_name", "qualified_class_name", "method_signature"]
        rows = ({k: r[k] for k in fieldnames} for r in rows)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    parser = argparse.ArgumentParser(description="Filter CSV by ALL class/method pairs found in subset JSONs.")
    parser.add_argument("root", type=str, help="Root folder containing project subfolders")
    parser.add_argument("input_csv", type=str, help="Path to the master CSV")
    parser.add_argument("--subset-glob", type=str, default="subset_*.json", help="Glob used to discover subset files (default: subset_*.json)")
    parser.add_argument("--subset-name", type=str, default=None, help="Exact subset file name (e.g., subset_5.json). If set, ignores --subset-glob")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: input dir / selected_from_subsets.csv)")
    parser.add_argument("--minimal", action="store_true", help="Only write project_name, qualified_class_name, method_signature")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    csv_in = Path(args.input_csv).expanduser().resolve()
    csv_out = Path(args.output).expanduser().resolve() if args.output else (csv_in.parent / "selected_from_subsets.csv")

    if not root.exists():
        raise SystemExit(f"[ERROR] Root not found: {root}")
    if not csv_in.exists():
        raise SystemExit(f"[ERROR] Input CSV not found: {csv_in}")

    pairs = collect_pairs(root, subset_glob=args.subset_glob, subset_name=args.subset_name)
    if not pairs:
        raise SystemExit(f"[ERROR] No subset entries found using pattern '{args.subset_name or args.subset_glob}' under {root}")

    print(f"[INFO] Collected {len(pairs)} unique (project, class, method) triples from subsets.")

    rows = read_csv_rows(csv_in)
    print(f"[INFO] Loaded {len(rows)} rows from CSV. Matching...")

    selected = [
        r for r in rows
        if (r.get("project_name"), r.get("qualified_class_name"), r.get("method_signature")) in pairs
    ]
    print(f"[OK] Matched {len(selected)} rows. Writing {csv_out}")

    # Preserve original field order unless minimal
    fieldnames = list(rows[0].keys()) if rows else CSV_REQUIRED_COLS
    write_csv_rows(csv_out, selected, fieldnames, args.minimal)
    print(f"[DONE] Wrote {csv_out}")

if __name__ == "__main__":
    main()
# Example python scripts/subset_test2nl_csv.py ./resources/sampled_tests ./resources/test2nl/partitioned_dataset/test2nl.csv --subset-name nl2test_subset_5.json