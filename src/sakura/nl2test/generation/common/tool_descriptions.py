EXTRACT_CODE_DESC_OLD = """
Get full source code (declaration + body) for a specific method.
Args:
  qualified_class_name: Fully qualified declaring class.
  method_signature: Exact method signature of the method to analyze, including any qualified parameter types.
  start_line: 1-based inclusive start line.
  end_line: 1-based inclusive end line.
Use when:
  Understanding implementation details, side effects, or parameter semantics to write assertions or setup.
Tips:
  - Prefer `get_method_details` first for quick checks; use `extract_method_code` only when behavior or parameter meaning is unclear.
Limitations:
  - Returns raw source code slice as-is.
  - Cannot be used to retrieve any inherited library methods. Fails if the method cannot be found in the application.
Returns:
  Object with { source, start_line, end_line, total_lines, note } summarizing the requested slice, or a structured error dict on failure.
"""

EXTRACT_CODE_DESC = """
More costly: return the spliced source for a specific application method.
Use only when get_method_details is insufficient to determine behavior, side effects, or parameter meaning.
"""

METHOD_DETAILS_DESC_OLD = """
Fetch declaration-site metadata for a method.
Args:
  qualified_class_name: Fully qualified declaring class.
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

METHOD_DETAILS_DESC = """
Fetch method metadata for a specific method.
Returns: signature, modifiers/visibility, return type, parameter names/types, and doc/comments (if present).
Use to confirm method semantics and arguments for binding.
"""

CALL_SITE_DETAILS_DESC_OLD = """
List callees invoked inside a specific method through static analysis.
Args:
  qualified_class_name: Fully qualified declaring class.
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

CALL_SITE_DETAILS_DESC = """
List the qualified class name, method signature, return type, parameter types, modifiers, and number of times called for each callee invoked inside a specific method.
Use to locate application helper methods, confirm wrapper delegation, and validate behavior.
"""

QUERY_CLASS_DESC_OLD = """
Semantic search over application classes (vector index).
Args:
  query: Natural language or code-like phrase describing the class you want or the likely class name.
  i, j: 1-based inclusive window into the ranked results (i > 0, j >= i).
Use when:
  Finding relevant classes for the target behavior or resolving package/import context for generated tests.
Limitations:
  Application classes only; library classes are not indexed.
Returns:
  List of dicts with declaring_class_name.
  On failure, a structured error dict is returned.
"""

QUERY_CLASS_DESC = """
Semantic search over application classes (application source only).
Prefer <= 3 results unless you justify more.
"""

VIEW_TEST_CODE_DESC_OLD = """
View the entire current test file for the active test class.
Args:
  None
Use when:
  Inspecting what has been generated or verifying that requested changes were applied as intended.
Limitations:
  Returns full file content; use your editor or search to focus.
Returns:
  Object with { source, total_lines } and optionally { qualified_class_name }.
  On failure, a structured error dict is returned.
"""

VIEW_TEST_CODE_DESC = """
View the spliced source code for the active test file.
Use sparingly (expensive) to verify generated code and to reconcile compilation or execution feedback.
"""

COMPILE_AND_EXECUTE_TEST_DESC_OLD = """
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

COMPILE_AND_EXECUTE_TEST_DESC = """
Compile the Maven project and execute the currently active test class. 
Returns compiler results and, if compilation succeeds, the test execution results.
"""
