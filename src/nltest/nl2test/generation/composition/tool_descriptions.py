EXTRACT_CODE_DESC = """
Get full source code (declaration + body) for a specific method.
Args:
  qualified_class_name: Fully qualified implementing class.
  method_signature: Exact method signature of the method to analyze, including any qualified parameter types.
Use when:
  Understanding implementation details, side effects, or parameter semantics to write assertions or setup.
Tips:
  - Prefer `get_method_details` first for quick checks; use `extract_method_code` only when behavior or parameter meaning is unclear.
Limitations:
  - Returns raw source as-is.
  - Cannot be used to retrieve any inherited library methods. Fails if the method cannot be found in the application.
Returns:
  String with full method source, or a structured error dict on failure.
"""

METHOD_DETAILS_DESC = """
Fetch declaration-site metadata for a method.
Args:
  qualified_class_name: Fully qualified implementing class.
  method_signature: Exact method signature of the method to analyze, including any qualified parameter types.
Use when:
  Selecting overloads, confirming parameter/return types, visibility, and static/instance modifiers.
Tips:
  - This is the fastest path to fix signature mismatches that cause compilation failures.
  - Use to decide whether to instantiate the class or call statically.
Limitations:
  - Only the declaration site.
  - Cannot be used to retrieve any inherited library methods.
Returns:
  Dict with:
    method_signature
    modifiers (e.g., ["public", "static"])
    return_type (fully qualified when available)
    parameter_types (list; fully qualified when available)
    comments (docstring or extracted comments if available)
    visibility ("public" | "same_package_or_subclass" | "same_package")
  Or a structured error dict on failure.
"""

CALL_SITE_DETAILS_DESC = """
List callees invoked inside a specific method through static analysis.
Args:
  qualified_class_name: Fully qualified implementing class.
  method_signature: Exact method signature of the method to analyze, including any qualified parameter types.
Use when:
  Tracing downstream effects to design assertions or mocks, or to confirm that a facade method delegates as expected.
Limitations:
  - Static analysis only; reflective/dynamic calls may be missed. 
  - Only looks at call sites within the select method at depth one, and does not expand further.
Returns:
  List of dicts with:
    callee qualified_class_name
    method_signature
    return_type
    parameter_types
    modifiers
    num_times_called
  Or a structured error dict on failure.
"""

GET_CLASS_FIELDS_DESC = """
List declared fields for a class.
Args:
  qualified_class_name: Fully qualified class to inspect.
Use when:
  Verifying or setting state for assertions; retrieving values; designing minimal fakes for dependencies.
Tips:
  - Combine with `get_getters_and_setters` to read/write state without reflection.
Limitations:
  No inheritance traversal; only fields declared directly in the class.
Returns:
  List of dicts with:
    variable_names (list of declared names per field line)
    type
    modifiers
  Or a structured error dict on failure.
"""

GET_CLASS_IMPORTS_DESC = """
Get import statements for a class's compilation unit.
Args:
  qualified_class_name: Fully qualified class to inspect.
Use when:
  Mirroring imports in your test for external types or frameworks used by the SUT.
Tips:
  - Use this to detect presence of libraries and adapt test dependencies accordingly.
Limitations:
  Reads only the select class's compilation unit; does not expand transitive imports.
Returns:
  List of fully qualified import names (without the leading 'import ').
  Or a structured error dict on failure.
"""

GET_CLASS_CONSTRUCTORS_AND_FACTORIES_DESC = """
List constructors and obvious factory methods for a class.
Args:
  qualified_class_name: Fully qualified class to inspect.
Use when:
  Determining how to instantiate SUT or collaborators.
Tips:
  - Factories are heuristically detected: static methods returning the class type.
  - Prefer the simplest constructor/factory that satisfies parameter availability.
Limitations:
  May miss builder patterns or complex factories.
Returns:
  List of dicts with:
    method_signature
    type ("constructor" or "factory")
  Or a structured error dict on failure.
"""

GET_GETTERS_AND_SETTERS_DESC = """
Find simple getters and setters within a class.
Args:
  qualified_class_name: Fully qualified class to analyze.
Use when:
  Setting up objects or asserting on resulting state.
Limitations:
  - Pattern-based; may include false positives.
  - Only includes basic one-line getters and setters.
Returns:
  List of method signatures, each labeled as getter or setter when available.
  Or a structured error dict on failure.
"""

VIEW_TEST_CODE_DESC = """
View the currently saved test file content for the active test class.
Args:
  None
Use when:
  Inspecting what has been generated or verifying that fixes in a newly generated code input were applied as intended.
Limitations:
  Can usually refer to the tool history for the state of the generated test file.
Returns:
  String containing the test source code, or a message when missing.
"""

GENERATE_TEST_CODE_DESC = """
Create or overwrite the test file with newly generated code and set the active test class.
Args:
  test_code: Complete Java test code including package, imports, class, and methods. This will replace all previous test code in the file.
  qualified_class_name: Fully qualified name for the test class. Be careful with ensuring the package before the simple class name is compliant with the localized methods.
Use when:
  Writing or updating the test to reflect the localized scenario.
Rules:
  - Always send the full file; this overwrites existing content.
  - Ensure package mirrors the primary SUT package to access package-private members.
  - Prefer explicit imports; avoid wildcard imports.
  - Include minimal helper fakes as nested static classes if needed.
Returns:
  Dict echoing { test_code, qualified_class_name } and the persisted save location.
"""

COMPILE_AND_EXECUTE_TESTS_DESC = """
Compile the Maven project and execute the active test class.
Args:
  None
Use when:
  Getting feedback on compilation and runtime assertions; use iteratively in a fix loop.
Notes:
  - This runs Maven build steps for test compilation and execution.
  - Execution feedback is only available if the target class compiles.
Returns:
  JSON with:
    erroneous_classes: list of '.java' filenames with compile errors
    target_class_file: the '.java' filename of the active test
    has_errors_for_target: boolean
    execution_feedback: when available, { returncode, stdout, stderr, command }
"""

FINALIZE_DESC = """
Finalize composition with a brief comment log.
Args:
  comments: A concise rationale describing what compiled, what ran, any skipped steps, and localization feedback if steps should be revised. State the confidence in the test suite and whether further localization and composition should be done or would be productive.
Use when:
  End conditions are met or iteration limit reached. The test case compiles successfully and contains the desired logic for the test scenario.
Returns:
  String echoing the final comments.
"""

MODIFY_SCENARIO_COMMENT_DESC = """
Update the comment for a localized step.
Args:
  id: Step identifier to update.
  comment: A concise note explaining adjustments or difficulties (e.g., switched to alternate method, added stub, assertion rationale, or localization problems).
Use when:
  Capturing decisions or clarifications tied to individual steps.
Limitations:
  Only updates the step comment; does not change ordering or bindings.
Returns:
  Tuple (id, comment).
"""

MODIFY_ATOMIC_BLOCKS_DESC = """
Replace the working AtomicBlock list to reflect composition-oriented refinements.
Args:
  atomic_blocks: Full replacement list of atomic blocks.
Use when:
  You need to re-chunk steps for code generation while preserving overall scenario semantics.
Caution:
  This overwrites the current list; provide the entire desired list, not deltas.
Returns:
  The updated AtomicBlock list.
"""

MODIFY_ATOMIC_BLOCK_NOTE_DESC = """
Update the note for a specific atomic block by order.
Args:
  order: Atomic block order identifier.
  note: Short note capturing decisions or requirements for that block.
Use when:
  Recording per-block guidance during the composition loop.
Returns:
  Tuple (order, note).
"""
