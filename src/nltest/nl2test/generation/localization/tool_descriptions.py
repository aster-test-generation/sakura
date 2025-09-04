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

MODIFY_BLOCKS_DESC = """
Edit the current AtomicBlocks to better reflect the intended test steps or method assignments. Do thorough static analysis to inform edits.
Do not call this tool in parallel.
Args:
  instructions: Detailed, clear edits (merge/split/reorder/add/remove) and any method assignments; reference blocks by order_id. Indicate if certain blocks are just descriptive, and not relevant for method mapping. Indicate modifications in a number list if there are multiple. Refer to `notes` for adding any comments for a specific block, and `candidate_methods` for suggesting methods to assign to blocks. Use fully qualified class and method signature (without modifiers) names, like the outputs from the static analysis.
  current_blocks: Current list of AtomicBlock objects. Take this from the output of the last `modify_atomic_blocks` call, or the initial state from the first chat prompt if this is the first call.
Use when:
  You want to align block structure to the discovered methods and narrative from the sentence(s). Use static analysis results to inform edits.
Limitations:
  LLM-driven; results may be non-deterministic. Be explicit about which blocks change and why. Use structured language like {"implementing_class_name": "...", "containing_class_name": "...", "method_signature": "..."} to reference methods.
Returns:
  Updated list of AtomicBlock objects.
  On failure, a structured error dict is returned.
"""

FINALIZE_ATOMIC_BLOCKS_DESC = """
Finish and return the final grammatical AtomicBlocks with localized methods and clear comments for downstream execution after all modifications.
Focus on concise, actionable comments capturing rationale, caveats, or localization notes relevant to implementation.
Args:
  comments: Short rationale or execution notes for the finalized plan.
  current_blocks: The finalized list of AtomicBlock objects.
Use when:
  You are done editing the grammatical blocks and ready to hand off to the next stage (e.g., test generation/execution).
  Only call this once you are satisfied with potential methods and comments for the grammatical blocks.
Limitations:
  Requires precise formatted block outputs that must be carefully checked and cross-referenced with the static analysis results.
Returns:
  Tuple of (final AtomicBlock list, comments string).
  On failure, a structured error dict is returned.
"""

FINALIZE_LOCALIZED_SCENARIO_DESC = """
Finish and return the final LocalizedScenario plus clear comments for downstream execution after all modifications.
The `scenario` argument is the localized scenario consisting of setup, steps, and teardown that have already been localized.
Args:
  comments: Short rationale or execution notes for the finalized plan.
  scenario: LocalizedScenario aggregating setup, steps, and teardown.
Use when:
  You are done editing the scenario structure and method localization and ready to hand off to the next stage (e.g., test generation/execution).
Limitations:
  Requires precise formatted scenario outputs that must be carefully checked and cross-referenced with the static analysis results.
Returns:
  Tuple of (final LocalizedScenario, comments string).
  On failure, a structured error dict is returned.
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
