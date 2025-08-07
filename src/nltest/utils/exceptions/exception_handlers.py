from typing import Dict, Any

from langchain_core.tools import ToolException


class ToolExceptionHandler:
    @staticmethod
    def handle_error(error: ToolException) -> Dict[str, Any]:
        return {
            "status": "error",
            "error_type": type(error).__name__,
            "message": str(error),
            "details": getattr(error, "extra_info", {})
        }