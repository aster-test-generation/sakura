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

QUERY_CLASS_DESC = """
Semantic search over application classes (vector index).
Args:
  query: Natural language or code-like phrase describing the class you want or the likely class name.
  i, j: 1-based inclusive window into the ranked results (i > 0, j >= i).
Use when:
  - You want to find similar classes in the application to the query.
  - Identify the package and location of application classes relevant to the test method for resolving imports.
Limitations:
  Application classes only; library classes are not indexed.
Returns:
  List of dicts with implementing_class_name.
  On failure, a structured error dict is returned.
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

GET_MAVEN_DEPENDENCIES_DESC = """
List direct Maven dependencies declared in the project's root pom.xml.
Args:
  None
Use when:
  Detecting external libraries to align imports, mocks, or test utilities.
Notes:
  - Reads only the top-level <dependencies> section of the root POM.
  - Raises an error if the root pom.xml is missing.
  - Returns an empty list if the POM exists but cannot be parsed.
Limitations:
  Does not include transitive dependencies, managed versions, profiles, or parent inheritance.
Returns:
  List of dicts with: { group_id, artifact_id }.
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
Strict formatting for arguments:
  - Provide raw Java source for `test_code`. Do NOT wrap it in Markdown fences (``` ... ```), triple quotes (''' ... '''), or any other wrapper/annotations.
  - Do NOT include JSON, Markdown, XML, or commentary in `test_code`. Only valid Java source.
  - Ensure newlines are literal (no escaped newline sequences) and the string is valid JSON.
Rules:
  - Always send the full file; this overwrites existing content.
  - Ensure package mirrors the primary SUT package to access package-private members.
  - Prefer explicit imports; avoid wildcard imports.
  - Include minimal helper fakes as nested static classes if needed.
Returns:
  Dict echoing { test_code, qualified_class_name } and the persisted save location.
"""

COMPILE_AND_EXECUTE_TEST_DESC = """
Compile the Maven project and execute the active test class.
Args:
  None
Use when:
  Getting feedback on compilation and runtime assertions; use iteratively in a fix loop.
Notes:
  - Runs Maven steps for test compilation and execution.
  - Execution details are only available if the target class compiles.
Returns:
  JSON with:
    compilation: {
      target_class_file: '.java' filename of the active test
      has_errors_for_target: boolean
      any_compilation_errors: boolean (any project errors)
      errors_for_target_class: list of { file, line, column|null, message, details[] }
      error_summary: { total_errors, files_with_errors[], error_counts_by_file{ file: count } }
    }
    execution: when compiled and tests pass, {
      executed: true
      status: 'tests_passed'
      message: 'All tests in class passed.'
      num_tests_run: int
      num_failures: int
      num_errors: int
    }
    execution: when compiled and tests fail or error, {
      executed: true
      status: string (e.g., 'TEST_FAILURES' | 'TEST_ERRORS' | 'execution_failed')
      message: concise one-liner reason
      num_tests_run: int
      num_failures: int
      num_errors: int
      issues: list of short strings summarizing top failures, e.g.:
        "failure: com.example.MyTests.myTest @ MyTests.java:42 -> expected X but was Y"
    }
    execution: when not compiled, {
      executed: false
      status: 'compilation_errors'
      message: reason to fix compilation
    }
"""

FINALIZE_DESC = """
End composition with a concise status comment.
Args:
  comments: 1–4 sentences on selected package/class, key fixes, any excluded steps, and unresolved items.
Use when:
  The test compiles and reflects the localized scenario/description, or the iteration limit is reached. This call ends the run.
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
