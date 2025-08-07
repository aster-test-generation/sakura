import json
from enum import Enum
from typing import List, Any


def pretty_print(header: str, body: Any):
    header = header.upper()
    divider = '=' * 20 + f' {header} ' + '=' * 20
    end_divider = '=' * len(divider)

    def serialize(obj: Any) -> Any:
        # If Pydantic model
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump(mode="json")

        # Catch enums
        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            return [serialize(item) for item in obj]

        # Return normal as is
        return obj

    serialized = serialize(body)

    print()
    print(divider)
    if isinstance(serialized, (dict, list)):
        print(json.dumps(serialized, indent=4))
    else:
        print(serialized)
    print(end_divider)