from __future__ import annotations

from typing import Any, Dict, Set

from nltest.utils.analysis.common_analysis import CommonAnalysis


class CLDKArgNormalizer:
    """
    Shared utility for normalizing tool arguments for CLDK compatibility.
    Handles inner class name normalization and constructor method signature normalization.
    """

    NORMALIZE_CLASS_TOOLS: Set[str] = {
        "get_method_details",
        "get_class_fields",
        "get_class_imports",
        "get_class_constructors_and_factories",
        "get_getters_and_setters",
        "extract_method_code",
        "get_call_site_details",
        "get_reachable_methods_in_class",
        "get_class_details",
        "get_inherited_library_classes",
    }

    NORMALIZE_METHOD_SIG_TOOLS: Set[str] = {
        "get_call_site_details",
        "get_method_details",
        "extract_method_code",
    }

    @staticmethod
    def normalize_args(tool_name: str, raw_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize tool arguments for CLDK compatibility.

        Handles:
        - Inner class name normalization (e.g., Outer$Inner -> Outer.Inner for CLDK)
        - Constructor method signature normalization

        Args:
            tool_name: The name of the tool being called
            raw_args: The original tool arguments

        Returns:
            Normalized arguments dict (may be same object if no changes needed)
        """
        updated_args: Dict[str, Any] = raw_args

        if tool_name in CLDKArgNormalizer.NORMALIZE_CLASS_TOOLS:
            qualified_class_name = raw_args.get("qualified_class_name")
            if isinstance(qualified_class_name, str):
                normalized_class = CommonAnalysis.get_cldk_class_name(qualified_class_name)
                if normalized_class != qualified_class_name:
                    if updated_args is raw_args:
                        updated_args = dict(updated_args)
                    updated_args["qualified_class_name"] = normalized_class

        if tool_name in CLDKArgNormalizer.NORMALIZE_METHOD_SIG_TOOLS:
            qualified_class_name = updated_args.get("qualified_class_name")
            method_signature = updated_args.get("method_signature")
            if qualified_class_name and method_signature:
                normalized_sig = CommonAnalysis.get_cldk_method_sig(
                    qualified_class_name, method_signature
                )
                if normalized_sig != method_signature:
                    if updated_args is raw_args:
                        updated_args = dict(updated_args)
                    updated_args["method_signature"] = normalized_sig

        return updated_args
