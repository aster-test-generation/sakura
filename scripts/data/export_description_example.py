"""
Export the low, medium, and high abstraction descriptions of a single Test2NL test
as standalone Markdown files, alongside a metadata file identifying the test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from sakura.test2nl.model.models import AbstractionLevel, Test2NLEntry
from sakura.utils.file_io.structured_data_manager import StructuredDataManager

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

TEST2NL_FILE = ROOT_DIR / "resources/test2nl/filtered_dataset/test2nl.csv"
OUTPUT_DIR = ROOT_DIR / "outputs/description_example"
METADATA_FILE_NAME = "metadata.json"

# Default exemplar: chosen for the cleanest separation between the three
# abstraction levels across the criteria in the abstraction table.
DEFAULT_PROJECT_NAME = "commons-dbcp"
DEFAULT_QUALIFIED_CLASS_NAME = "org.apache.commons.dbcp2.datasources.UserPassKeyTest"
DEFAULT_METHOD_SIGNATURE = "testClear()"

LEVEL_ORDER = (AbstractionLevel.LOW, AbstractionLevel.MEDIUM, AbstractionLevel.HIGH)


def load_test2nl_entries(test2nl_file: Path) -> list[Test2NLEntry]:
    """Load every Test2NL entry from the dataset CSV."""
    manager = StructuredDataManager(test2nl_file.parent)
    return manager.load(test2nl_file.name, Test2NLEntry, format="csv")


def normalize_abstraction_level(value: object) -> AbstractionLevel | None:
    """Coerce a raw abstraction level value into an AbstractionLevel."""
    if isinstance(value, AbstractionLevel):
        return value
    try:
        return AbstractionLevel(str(value).strip().lower())
    except ValueError:
        return None


def select_entries(
    entries: Iterable[Test2NLEntry],
    project_name: str,
    qualified_class_name: str,
    method_signature: str,
) -> dict[AbstractionLevel, Test2NLEntry]:
    """Return the entry for each abstraction level of a single test."""
    selected: dict[AbstractionLevel, Test2NLEntry] = {}
    for entry in entries:
        if (
            entry.project_name != project_name
            or entry.qualified_class_name != qualified_class_name
            or entry.method_signature != method_signature
        ):
            continue
        level = normalize_abstraction_level(entry.abstraction_level)
        if level is None:
            continue
        if level in selected:
            raise ValueError(
                f"Duplicate {level.value} abstraction entry for "
                f"{qualified_class_name}.{method_signature}"
            )
        selected[level] = entry

    if not selected:
        raise ValueError(
            "No Test2NL entries found for "
            f"{project_name} :: {qualified_class_name} :: {method_signature}"
        )

    missing = [level.value for level in LEVEL_ORDER if level not in selected]
    if missing:
        raise ValueError(
            f"Missing {', '.join(missing)} abstraction entries for "
            f"{qualified_class_name}.{method_signature}"
        )

    return selected


def description_file_name(level: AbstractionLevel) -> str:
    """Return the Markdown file name for an abstraction level."""
    return f"{level.value}_abstraction.md"


def render_description(entry: Test2NLEntry, level: AbstractionLevel) -> str:
    """Render a single description as a readable Markdown document."""
    return "\n".join(
        [
            f"# {level.value.capitalize()} Abstraction Description",
            "",
            entry.description.strip(),
            "",
        ]
    )


def build_metadata(
    selected: dict[AbstractionLevel, Test2NLEntry], test2nl_file: Path
) -> dict:
    """Build the metadata record describing the exported example."""
    reference = selected[LEVEL_ORDER[0]]
    try:
        source_csv = str(test2nl_file.relative_to(ROOT_DIR))
    except ValueError:
        source_csv = str(test2nl_file)

    return {
        "project_name": reference.project_name,
        "qualified_class_name": reference.qualified_class_name,
        "method_signature": reference.method_signature,
        "source_csv": source_csv,
        "descriptions": {
            level.value: {
                "id": selected[level].id,
                "file": description_file_name(level),
            }
            for level in LEVEL_ORDER
        },
    }


def write_example(
    output_dir: Path,
    selected: dict[AbstractionLevel, Test2NLEntry],
    test2nl_file: Path,
) -> list[Path]:
    """Write the description files and the metadata file, returning their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for level in LEVEL_ORDER:
        path = output_dir / description_file_name(level)
        path.write_text(render_description(selected[level], level), encoding="utf-8")
        written.append(path)

    metadata_path = output_dir / METADATA_FILE_NAME
    metadata = build_metadata(selected, test2nl_file)
    metadata_path.write_text(
        json.dumps(metadata, indent=4, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    written.append(metadata_path)

    return written


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Export the low, medium, and high abstraction descriptions of one "
            "Test2NL test as Markdown files plus a metadata file."
        )
    )
    parser.add_argument(
        "--test2nl-file",
        type=Path,
        default=TEST2NL_FILE,
        help="Path to the Test2NL CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write the description example into.",
    )
    parser.add_argument(
        "--project-name",
        default=DEFAULT_PROJECT_NAME,
        help="Project name of the test to export.",
    )
    parser.add_argument(
        "--qualified-class-name",
        default=DEFAULT_QUALIFIED_CLASS_NAME,
        help="Qualified class name of the test to export.",
    )
    parser.add_argument(
        "--method-signature",
        default=DEFAULT_METHOD_SIGNATURE,
        help="Method signature of the test to export.",
    )
    return parser


def main() -> None:
    """Run the description example exporter."""
    parser = build_parser()
    args = parser.parse_args()

    test2nl_file: Path = args.test2nl_file
    if not test2nl_file.is_file():
        raise FileNotFoundError(f"Test2NL CSV not found: {test2nl_file}")

    entries = load_test2nl_entries(test2nl_file)
    selected = select_entries(
        entries,
        project_name=args.project_name,
        qualified_class_name=args.qualified_class_name,
        method_signature=args.method_signature,
    )

    written = write_example(args.output_dir, selected, test2nl_file)

    print(f"Loaded {len(entries)} Test2NL entries from {test2nl_file}")
    print(
        f"Exported {args.qualified_class_name}.{args.method_signature} "
        f"({args.project_name})"
    )
    for path in written:
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
