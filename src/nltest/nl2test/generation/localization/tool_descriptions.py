QUERY_METHOD_DESC = """
Semantic search over application methods (vector index).
Args:
  query: Natural language or code-like phrase describing the desired behavior or the likely method name.
  i, j: 1-based inclusive window into the ranked results (i > 0, j >= i).
Use when:
  You need candidate methods for a sentence or intent; first-pass recall before filtering.
Limitations:
  Application code only; excludes inherited methods from external libraries.
Returns:
  List of dicts with method_signature, implementing_class_name (class that directly implements the method), and containing_class_name (class that inherits/contains the method).
  On failure, a structured error dict is returned.
"""

QUERY_CLASS_DESC = """
Semantic search over application classes (vector index).
Args:
  query: Natural language or code-like phrase describing the class you want or the likely class name.
  i, j: 1-based inclusive window into the ranked results (i > 0, j >= i).
Use when:
  You want classes likely to contain or relate to target methods.
Limitations:
  Application classes only; library classes are not indexed.
Returns:
  List of dicts with implementing_class_name.
  On failure, a structured error dict is returned.
"""

REACHABLE_METHODS_DESC = """
List methods visible from a class with metadata, honoring Java access rules and going through the inheritance chain.
Args:
  qualified_class_name: Fully qualified class to inspect.
  visibility_mode: One of public, same_package, same_package_or_subclass. It must be one of these three string values.
Use when:
  Filtering candidate methods to what the class under test can actually call; understanding inheritance and visibility.
Limitations:
  Includes inherited methods from application classes only; external library parents are excluded.
Returns:
  Dict mapping implementing_class_name (class that directly implements the method) -> list of method metadata dicts (e.g., signature, modifiers, and other details) that are reachable from the qualified_class_name used in the query.
  On failure, a structured error dict is returned.
"""

EXTRACT_CODE_DESC = """
Get a method's complete source (declaration + body) from the analyzed codebase.
Args:
  qualified_class_name: Class that declares the method.
  method_signature: Exact signature to extract.
Use when:
  Verifying behavior, generating tests, or creating examples for an identified method.
Limitations:
  Fails if the method is not found; returns source as-is (no formatting changes).
Returns:
  String containing the full method code.
  On failure, a structured error dict is returned.
"""

METHOD_DETAILS_DESC = """
Fetch fast metadata for a method.
Args:
  qualified_class_name: Declaring class.
  method_signature: Exact signature.
Use when:
  You need signature, parameter types, return type, modifiers, or visibility to filter/rank candidates, or to form test scaffolds.
Limitations:
  Only the declaration site; does not include inherited versions or overrides from other classes.
Returns:
  Dict with method_signature, modifiers, return_type, parameter_types, comments, visibility.
  Visibility options: "public" (accessible from anywhere), "same_package_or_subclass" (accessible from same package or subclasses), "same_package" (accessible only from same package).
  On failure, a structured error dict is returned.
"""

CLASS_DETAILS_DESC = """
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

CALL_SITE_DETAILS_DESC = """
List callees invoked inside a specific method (static analysis).
Args:
  qualified_class_name: Declaring class of the caller.
  method_signature: Signature of the caller method.
Use when:
  Expanding candidates by usage, understanding side effects, or mapping sentences to downstream calls.
Limitations:
  Static only; may miss reflection/dynamic dispatch. Quality depends on the symbol table and parsed code.
Returns:
  List of dicts with callee qualified_class_name, method_signature, return_type, parameter_types, modifiers, num_times_called.
  On failure, a structured error dict is returned.
"""

FINALIZE_ATOMIC_BLOCKS_DESC = """
End localization by returning the final AtomicBlockList and brief comments.
Args:
  current_blocks: The finalized AtomicBlockList.
  comments: Short notes on selection rationale and any non-localized tasks retained.
Use when:
  The blocks are localized to candidate methods (with bindings) and a best candidate where feasible, or when at the iteration limit. This call ends the run.
Returns:
  Tuple (AtomicBlockList, comments).
"""

FINALIZE_LOCALIZED_SCENARIO_DESC = """
End localization by returning the final LocalizedScenario and brief comments.
Args:
  scenario: The finalized LocalizedScenario (setup + steps + teardown).
  comments: Short notes on choices, alternates, and any non-localized tasks retained.
Use when:
  Steps are localized to candidate methods (with bindings) and a best candidate where feasible, or when at the iteration limit. This call ends the run.
Returns:
  Tuple (LocalizedScenario, comments).
"""

INHERITED_LIBRARY_CLASSES_DESC = """
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
