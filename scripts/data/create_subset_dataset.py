import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

def make_subset(data: Dict[str, Any], n: int) -> Dict[str, Any]:
    """Return a copy of `data` with the first n elements kept for keys starting with 'tests_' when the value is a list."""
    subset = {}
    for k, v in data.items():
        if k.startswith("tests_") and isinstance(v, list):
            subset[k] = v[:n]
        else:
            subset[k] = v
    return subset

def process_file(path: Path, n: int) -> None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[SKIP] {path}: could not read/parse JSON ({e})")
        return

    subset = make_subset(data, n)
    out_path = path.parent / f"nl2test_subset_{n}.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(subset, f, indent=2, ensure_ascii=False)
        print(f"[OK] Wrote {out_path}")
    except Exception as e:
        print(f"[FAIL] {out_path}: could not write file ({e})")

def main():
    parser = argparse.ArgumentParser(description="Create subset_N.json files by trimming test groups to first N items.")
    parser.add_argument("root", type=str, help="Root folder containing subfolders with j.json files")
    parser.add_argument("N", type=int, help="Number of items to keep from each test group list")
    parser.add_argument("--filename", type=str, default="nl2test.json", help="Name of the JSON file in each subfolder (default: j.json)")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"[ERROR] Root path does not exist: {root}")
        return
    if args.N < 0:
        print(f"[ERROR] N must be non-negative; got {args.N}")
        return

    matches = list(root.rglob(args.filename))
    if not matches:
        print(f"[WARN] No '{args.filename}' files found under {root}")
        return

    print(f"Found {len(matches)} '{args.filename}' files under {root}. Processing with N={args.N}...\n")
    for p in matches:
        process_file(p, args.N)

if __name__ == "__main__":
    main()
# Example  python scripts/create_subset_dataset.py ./resources/sampled_tests 5