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

GET_CLASS_FIELDS_DESC = """
List declared fields for a class.
Args:
  qualified_class_name: Fully qualified class to inspect.
Use when:
  Planning object setup, mocks, or state assertions where fields matter.
Limitations:
  No inheritance traversal; returns only fields declared in the class.
Returns:
  List of dicts with variable_names (list of names in the declaration), type, and modifiers.
  On failure, a structured error dict is returned.
"""

GET_CLASS_IMPORTS_DESC = """
Get the import statements for a class's compilation unit.
Args:
  qualified_class_name: Fully qualified class to inspect.
Use when:
  Ensuring generated tests include all necessary imports for referenced types and frameworks.
Limitations:
  Reads only the class's compilation unit; does not infer transitive imports.
Returns:
  List of fully qualified import names (without the leading 'import ' prefix).
  On failure, a structured error dict is returned.
"""

GET_CLASS_CONSTRUCTORS_AND_FACTORIES_DESC = """
List constructors and simple factory methods for a class.
Args:
  qualified_class_name: Fully qualified class to inspect.
Use when:
  Understanding how to instantiate the class under test in generated tests.
Limitations:
  Factory detection is heuristic: static methods returning the class type; may miss builders or complex factories.
Returns:
  List of dicts with method_signature and type ('constructor' or 'factory').
  On failure, a structured error dict is returned.
"""

GET_GETTERS_AND_SETTERS_DESC = """
Find methods that look like simple getters/setters.
Args:
  qualified_class_name: Fully qualified class to inspect.
Use when:
  Identifying trivial accessors that might not need explicit tests or can be used to set up/verify state.
Limitations:
  Pattern-based; considers short methods whose signature starts with 'get' or 'set'. May include false positives.
Returns:
  List of method signatures classified as getters or setters.
  On failure, a structured error dict is returned.
"""

VIEW_TEST_CODE_DESC = """
View the current saved test file content for the working test class.
Args:
  None.
Use when:
  Inspecting the current code before edits or after generation to verify content.
Limitations:
  Depends on current agent state for which test class is active; returns a message if none saved.
Returns:
  String with the test source code, or a message if the file is missing.
"""

GENERATE_TEST_CODE_DESC = """
Create or overwrite the test file content and target class name.
Args:
  test_code: Complete Java test source including package and imports.
  qualified_class_name: Fully qualified name for the test class (e.g., 'com.example.MyTest').
Use when:
  You are ready to write or update the test file for the current scenario/blocks.
Limitations:
  The tool echoes your inputs; the agent saves the file and updates active package/class state.
Returns:
  Dict echoing {test_code, qualified_class_name}; the agent persists and reports the save location.
"""

COMPILE_AND_EXECUTE_TESTS_DESC = """
Compile the project and execute the currently active test class, returning feedback.
Args:
  None.
Use when:
  Validating that the generated test compiles and runs; use iteratively to fix issues.
Limitations:
  Runs Maven compile and tests; execution feedback is available only if compilation succeeds for the target class.
Returns:
  JSON string with keys: erroneous_classes (list of '.java' filenames with compile errors), target_class_file ('.java' filename), has_errors_for_target (bool), and execution_feedback (dict with returncode/stdout/stderr/command) when available.
"""

FINALIZE_DESC = """
Finalize the composition with optional final comments.
Args:
  comments: Short rationale, caveats, or next steps.
Use when:
  All end conditions are met and you want to finish the composition.
Limitations:
  Does not modify files or state beyond recording the comments and ending the run.
Returns:
  String containing the final comments.
"""

MODIFY_SCENARIO_COMMENT_DESC = """
Update the comment for a specific step in a Gherkin LocalizedScenario.
Args:
  id: Step identifier to update.
  comment: New comment for the step.
Use when:
  Annotating or clarifying a step during composition based on findings or changes.
Limitations:
  Only updates comments; does not change step ordering or bindings.
Returns:
  Tuple (id, comment) echoed back; the agent applies the update to the scenario.
"""

MODIFY_ATOMIC_BLOCKS_DESC = """
Replace the working AtomicBlockList to reflect refined decomposition for composition.
Args:
  atomic_blocks: The full, updated list of atomic blocks.
Use when:
  You must merge, split, or reorder blocks to better match executable test steps.
Limitations:
  Overwrites the working list; provide the complete desired list, not deltas.
Returns:
  The updated AtomicBlockList.
"""

MODIFY_ATOMIC_BLOCK_NOTE_DESC = """
Update the note for a specific atomic block by order.
Args:
  order: The block order identifier to update.
  note: New note for the block.
Use when:
  Capturing decisions, requirements, or clarifications per block during composition.
Limitations:
  Only updates a note; does not change block content or ordering.
Returns:
  Tuple (order, note) echoed back; the agent applies the update to the working blocks.
"""
