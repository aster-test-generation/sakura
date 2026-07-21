from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sakura.test2nl.model.models import Test2NLEntry
from sakura.utils.file_io.structured_data_manager import StructuredDataManager


LEVELS = ("low", "medium", "high")
REQUIRED_COLUMNS = {
    "abstraction_level",
    "description",
    "id",
    "is_bdd",
    "method_signature",
    "project_name",
    "qualified_class_name",
}
SAFE_USER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class Grades(BaseModel):
    model_config = ConfigDict(extra="forbid")

    naturalness: int = Field(ge=1, le=5)
    fidelity: int = Field(ge=1, le=5)
    abstraction_fit: int = Field(ge=1, le=5)


class GradingSession:
    """Own the selected entries, reviewer progress, and durable grade file."""

    def __init__(self, user: str, descriptions_dir: Path, repo_root: Path) -> None:
        self.user = self.validate_user(user)
        self.descriptions_dir = descriptions_dir.resolve()
        self.repo_root = repo_root.resolve()
        self.output_path = self.descriptions_dir / "graded" / f"{self.user}.json"
        self.project_descriptions = self._load_project_descriptions()

        entries_by_level = self._load_entries()
        self.selection = self._load_selection(entries_by_level)
        self.entries = self._balanced_entries(entries_by_level, self.selection)
        self.grades: dict[int, Grades] = {}
        self._load_existing()

    @staticmethod
    def validate_user(user: str) -> str:
        if (
            not SAFE_USER.fullmatch(user)
            or user in {".", ".."}
            or Path(user).name != user
        ):
            raise ValueError(
                "--user must contain only letters, numbers, periods, underscores, "
                "or hyphens and may not be a path"
            )
        return user

    def _load_project_descriptions(self) -> dict[str, str]:
        path = self.descriptions_dir / "project_txt.json"
        if not path.is_file():
            raise FileNotFoundError(f"Project description file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid project description JSON: {path}") from exc
        if not isinstance(data, dict) or not all(
            isinstance(key, str) and isinstance(value, str) and value.strip()
            for key, value in data.items()
        ):
            raise ValueError(f"Project descriptions must map project names to text: {path}")
        return data

    def _load_entries(self) -> dict[str, list[Test2NLEntry]]:
        result: dict[str, list[Test2NLEntry]] = {}
        seen_ids: set[int] = set()
        projects: set[str] = set()
        for level in LEVELS:
            path = self.descriptions_dir / f"{level}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Description CSV not found: {path}")
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                columns = set(reader.fieldnames or [])
                missing = REQUIRED_COLUMNS - columns
                if missing:
                    raise ValueError(
                        f"{path} is missing columns: {', '.join(sorted(missing))}"
                    )
                rows = [Test2NLEntry.model_validate(row) for row in reader]
            for entry in rows:
                entry_level = (
                    entry.abstraction_level.value
                    if entry.abstraction_level is not None
                    else None
                )
                if entry_level != level:
                    raise ValueError(
                        f"Entry {entry.id} in {path.name} has level {entry_level!r}"
                    )
                if entry.id in seen_ids:
                    raise ValueError(f"Duplicate description ID across CSVs: {entry.id}")
                seen_ids.add(entry.id)
                projects.add(entry.project_name)
            result[level] = rows

        missing_projects = projects - set(self.project_descriptions)
        if missing_projects:
            raise ValueError(
                "project_txt.json is missing projects: "
                + ", ".join(sorted(missing_projects))
            )
        return result

    def _load_selection(
        self, entries_by_level: dict[str, list[Test2NLEntry]]
    ) -> dict[str, Any]:
        subset_path = self.descriptions_dir / "subset" / f"{self.user}.json"
        if not subset_path.exists():
            return {
                "kind": "full",
                "path": None,
                "ids": {
                    level: [entry.id for entry in entries_by_level[level]]
                    for level in LEVELS
                },
            }
        try:
            raw = json.loads(subset_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid subset JSON: {subset_path}") from exc
        if not isinstance(raw, dict) or set(raw) != set(LEVELS):
            raise ValueError(
                f"Subset must contain exactly these keys: {', '.join(LEVELS)}"
            )

        known = {
            level: {entry.id for entry in entries_by_level[level]} for level in LEVELS
        }
        selected: dict[str, list[int]] = {}
        seen: set[int] = set()
        for level in LEVELS:
            ids = raw[level]
            if not isinstance(ids, list):
                raise ValueError(f"Subset key {level!r} must contain an ID list")
            selected[level] = []
            for entry_id in ids:
                if isinstance(entry_id, bool) or not isinstance(entry_id, int):
                    raise ValueError(f"Subset {level!r} IDs must be integers")
                if entry_id not in known[level]:
                    raise ValueError(
                        f"Subset ID {entry_id} does not exist in {level}.csv"
                    )
                if entry_id in seen:
                    raise ValueError(f"Duplicate subset ID: {entry_id}")
                seen.add(entry_id)
                selected[level].append(entry_id)
        if not seen:
            raise ValueError("Subset must select at least one entry")
        return {
            "kind": "subset",
            "path": self._display_path(subset_path),
            "ids": selected,
        }

    @staticmethod
    def _balanced_entries(
        entries_by_level: dict[str, list[Test2NLEntry]], selection: dict[str, Any]
    ) -> list[Test2NLEntry]:
        lookup = {
            level: {entry.id: entry for entry in entries_by_level[level]}
            for level in LEVELS
        }
        lists = [
            [lookup[level][entry_id] for entry_id in selection["ids"][level]]
            for level in LEVELS
        ]
        result: list[Test2NLEntry] = []
        for index in range(max(map(len, lists), default=0)):
            for entries in lists:
                if index < len(entries):
                    result.append(entries[index])
        return result

    def _load_existing(self) -> None:
        if not self.output_path.exists():
            return
        try:
            data = json.loads(self.output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid existing grade file: {self.output_path}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Existing grade file must be an object: {self.output_path}")
        if data.get("schema_version") != 1 or data.get("user") != self.user:
            raise ValueError(f"Incompatible existing grade file: {self.output_path}")
        saved_entries = data.get("entries")
        if not isinstance(saved_entries, list) or not all(
            isinstance(entry, dict) for entry in saved_entries
        ):
            raise ValueError(f"Existing grade entries must be a list: {self.output_path}")
        current_ids = [entry.id for entry in self.entries]
        saved_ids = [entry.get("id") for entry in saved_entries]
        if saved_ids != current_ids:
            raise ValueError(
                "Existing grade file does not match the current selected ID order: "
                f"{self.output_path}"
            )
        for saved in saved_entries:
            if saved.get("grades") is not None:
                self.grades[saved["id"]] = Grades.model_validate(saved["grades"])

    @property
    def completed_count(self) -> int:
        return len(self.grades)

    @property
    def first_incomplete_index(self) -> int:
        for index, entry in enumerate(self.entries):
            if entry.id not in self.grades:
                return index
        return len(self.entries)

    def entry_payload(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.entries):
            raise IndexError(f"Entry position out of range: {index}")
        entry = self.entries[index]
        payload = entry.model_dump(mode="json")
        payload["grades"] = (
            self.grades[entry.id].model_dump() if entry.id in self.grades else None
        )
        payload["project_description"] = self.project_descriptions[
            entry.project_name
        ]
        payload["position"] = index
        payload["total"] = len(self.entries)
        return payload

    def set_grades(self, entry_id: int, raw_grades: Any) -> Grades:
        if entry_id not in {entry.id for entry in self.entries}:
            raise ValueError(f"Entry ID is not in this session: {entry_id}")
        grades = Grades.model_validate(raw_grades)
        self.grades[entry_id] = grades
        self.save()
        return grades

    def save(self) -> None:
        document = {
            "schema_version": 1,
            "user": self.user,
            "selection": {
                "kind": self.selection["kind"],
                "path": self.selection["path"],
            },
            "entries": [],
        }
        for entry in self.entries:
            row = entry.model_dump(mode="json")
            row["grades"] = (
                self.grades[entry.id].model_dump()
                if entry.id in self.grades
                else None
            )
            document["entries"].append(row)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        StructuredDataManager._atomic_write_text(
            self.output_path,
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        )

    def session_payload(self) -> dict[str, Any]:
        return {
            "user": self.user,
            "total": len(self.entries),
            "completed": self.completed_count,
            "resume_position": self.first_incomplete_index,
            "output_path": self._display_path(self.output_path),
        }

    def _display_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.repo_root).as_posix()
        except ValueError:
            return str(path.resolve())
