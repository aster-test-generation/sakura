import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, List, Literal, Sequence, Type, TypeVar, Union

from pydantic import BaseModel

from sakura.utils.pretty.color_logger import RichLog

SubModel = TypeVar("SubModel", bound=BaseModel)


class StructuredDataManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _as_list_of_dicts(
        data: Union[Sequence[BaseModel], Sequence[dict], BaseModel, dict],
    ) -> List[dict]:
        """Normalize input data to a list of dictionaries"""
        if data is None:
            return []

        def model_to_dict(x: Any) -> dict:
            if isinstance(x, BaseModel):
                return x.model_dump(mode="json")
            if isinstance(x, dict):
                return x
            if hasattr(x, "__dict__"):
                return dict(x.__dict__)
            raise TypeError(f"Unsupported item type: {type(x)}")

        if isinstance(data, (list, tuple)):
            return [model_to_dict(item) for item in data]

        return [model_to_dict(data)]

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        """Write through a unique same-directory temporary file."""
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                fd = -1
                f.write(content)
            tmp_path.replace(path)
        finally:
            if fd != -1:
                os.close(fd)
            tmp_path.unlink(missing_ok=True)

    def save(
        self,
        file_name: str,
        data: Union[Sequence[BaseModel], Sequence[dict], BaseModel, dict],
        *,
        format: Literal["json", "csv"] = "json",
        mode: Literal["write", "append"] = "write",
    ) -> None:
        path = self.base_dir / file_name
        rows = self._as_list_of_dicts(data)

        if format == "json":
            if mode == "append" and path.exists():
                with path.open("r", encoding="utf-8") as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
                existing.extend(rows)
                self._atomic_write_text(
                    path, json.dumps(existing, indent=4, ensure_ascii=True)
                )
            else:
                self._atomic_write_text(
                    path, json.dumps(rows, indent=4, ensure_ascii=True)
                )

        elif format == "csv":
            file_exists = path.exists()
            append_mode = mode == "append" and file_exists

            if not rows:
                return

            # If appending, attempt to reuse existing header
            existing_fieldnames = None
            if append_mode:
                try:
                    with path.open("r", encoding="utf-8", newline="") as rf:
                        reader = csv.DictReader(rf)
                        existing_fieldnames = (
                            list(reader.fieldnames) if reader.fieldnames else None
                        )
                except Exception:
                    existing_fieldnames = None

                if existing_fieldnames is None:
                    append_mode = False

            fieldnames = existing_fieldnames or sorted(
                {k for row in rows for k in row.keys()}
            )

            with path.open(
                "a" if append_mode else "w", encoding="utf-8", newline=""
            ) as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not append_mode and fieldnames:
                    writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

        else:
            raise ValueError(f"Unsupported format: {format}")

    def load(
        self, file_name: str, model_cls: Type[SubModel], *, format: str = "json"
    ) -> List[SubModel]:
        path = self.base_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if format == "json":
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            return [model_cls(**item) for item in data]
        elif format == "csv":
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)
            return [model_cls(**item) for item in data]
        else:
            raise ValueError(f"Unsupported format: {format}")

    def delete(self, file_name: str) -> bool:
        path = self.base_dir / file_name
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            RichLog.error(f"Failed to delete {path}: {exc}")
            return False

    def delete_many(self, file_names: Sequence[str]) -> None:
        for fn in file_names:
            self.delete(fn)
