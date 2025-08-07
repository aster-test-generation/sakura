import json
import csv
from pathlib import Path
from typing import List, TypeVar, Type

from pydantic import BaseModel

from nltest.utils.pretty.prints import pretty_print

SubModel = TypeVar("SubModel", bound=BaseModel)


class StructuredDataManager:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, file_name: str, objects: List[BaseModel], *, format: str = "json") -> bool:
        path = self.base_dir / file_name
        try:
            if format == "json":
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([obj.model_dump(mode="json") for obj in objects], f, indent=4)
            elif format == "csv":
                data = [obj.model_dump(mode="json") for obj in objects]
                if data:
                    headers = data[0].keys()
                    with open(path, "w", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=headers)
                        writer.writeheader()
                        writer.writerows(data)
            else:
                raise ValueError(f"Unsupported format: {format}")
            return True
        except Exception as e:
            pretty_print("Error during save", f"Error during save: {e}")
            return False

    def load(self, file_name: str, model_cls: Type[SubModel], *, format: str = "json") -> List[SubModel]:
        path = self.base_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if format == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [model_cls(**item) for item in data]
        elif format == "csv":
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                data = list(reader)
            return [model_cls(**item) for item in data]
        else:
            raise ValueError(f"Unsupported format: {format}")