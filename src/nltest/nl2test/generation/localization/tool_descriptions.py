QUERY_METHOD_DESC_OLD = """
Semantic search over application methods (vector index).
Args:
  query: Natural language or code-like phrase describing the desired behavior or the likely method name.
  i, j: 1-based inclusive window into the ranked results (i > 0, j >= i).
Use when:
  You need candidate methods for a sentence or intent; first-pass recall before filtering.
Limitations:
  Application code only; excludes inherited methods from external libraries.
Returns:
  List of dicts with method_signature, declaring_class_name (class that directly implements the method), and containing_class_name (class that inherits/contains the method).
  On failure, a structured error dict is returned.
"""

QUERY_METHOD_DESC = """
Semantic search over application methods (vector index). 
Returns the method signature, declaring class name (class that directly implement the method), and containing class name (the relevant class that inherits/contains the method) for each method matching the query.
Prioritize a maximum window of three results for any query unless a justified exception is needed.
"""

REACHABLE_METHODS_DESC_OLD = """
List methods visible from a class with metadata, honoring Java access rules and going through the inheritance chain.
Args:
  qualified_class_name: Fully qualified class to inspect.
  visibility_mode: One of public, same_package, same_package_or_subclass. It must be one of these three string values.
Use when:
  Filtering candidate methods to what the class under test can actually call; understanding inheritance and visibility.
Limitations:
  Includes inherited methods from application classes only; external library parents are excluded.
Returns:
  Dict mapping declaring_class_name (class that directly implements the method) -> list of method metadata dicts (e.g., signature, modifiers, and other details) that are reachable from the qualified_class_name used in the query.
  On failure, a structured error dict is returned.
"""

REACHABLE_METHODS_DESC = """
List all application methods visible from a class, honoring Java access rules and going through the inheritance chain. 
Returns the method signature, declaring class name, modifiers, and visibility details for each method.
"""

CLASS_DETAILS_DESC_OLD = """
Fetch structural info for a class (no inheritance traversal).
Args:
  qualified_class_name: Fully qualified class.
Use when:
  You need direct extends/implements, modifiers, or annotations to understand context or testability.
Limitations:
  Does not walk the inheritance graph and does not include fields by default.
Returns:
  Dict with class_name, modifiers, extends_list, implements_list, annotations.
  On failure, a structured error dict is returned.
"""

CLASS_DETAILS_DESC = """
Fetch class-level metadata, including its modifiers, superclass, implemented interfaces, and annotations.
"""

INHERITED_LIBRARY_CLASSES_DESC_OLD = """
Identify external superclasses/interfaces inherited by a class that are not part of the application codebase.
Args:
  qualified_class_name: Fully qualified class to inspect.
Use when:
  You suspect reachable behavior comes from libraries (methods not found in the application index).
Limitations:
  Returns names only; no method listing. Use docs or other tools to inspect those libraries.
Returns:
  List of fully qualified class/interface names for inherited library types.
  On failure, a structured error dict is returned.
"""

INHERITED_LIBRARY_CLASSES_DESC = """
Identify all inherited types for a class by traversing its inheritance chain in order.
"""

FINALIZE_ATOMIC_BLOCKS_DESC_OLD = """
End localization by returning the final AtomicBlockList and brief comments.
Args:
  current_blocks: The finalized AtomicBlockList.
  comments: Short notes on selection rationale and any non-localized tasks retained.
Use when:
  The blocks are localized to candidate methods (with bindings) and a best candidate where feasible, or when at the iteration limit. This call ends the run.
Returns:
  Tuple (AtomicBlockList, comments).
"""

FINALIZE_ATOMIC_BLOCKS_DESC = """
End localization by returning the final atomic block list with a concise 1–4 sentence status note summarizing key finds and unresolved issues, stated directly and bluntly. 
"""

FINALIZE_LOCALIZED_SCENARIO_DESC_OLD = """
End localization by returning the final LocalizedScenario and brief comments.
Args:
  scenario: The finalized LocalizedScenario (setup_steps + gherkin_steps + teardown_steps).
  comments: Short notes on choices, alternates, and any non-localized tasks retained.
Use when:
  Steps are localized to candidate methods (with bindings) and a best candidate where feasible, or when at the iteration limit. This call ends the run.
Returns:
  Tuple (LocalizedScenario, comments).
"""

FINALIZE_LOCALIZED_SCENARIO_DESC = """
End localization by returning the final localized scenario with a concise 1–4 sentence status note summarizing key finds and unresolved issues, stated directly and bluntly. Include the required comments field; if nothing notable, write "No issues.".
"""
