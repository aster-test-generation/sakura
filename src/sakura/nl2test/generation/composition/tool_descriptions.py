GET_CLASS_FIELDS_DESC_OLD = """
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

GET_CLASS_FIELDS_DESC = """
Return declared fields for a class: name, type, modifiers. 
Use to understand state you can read/write.
"""

GET_CLASS_IMPORTS_DESC_OLD = """
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

GET_CLASS_IMPORTS_DESC = """
Get the fully qualified import names for a class's compilation unit.
"""

GET_MAVEN_DEPENDENCIES_DESC_OLD = """
List Maven dependencies declared in the module pom.xml and inherited parent pom.xml files.
Use when:
  Detecting external libraries to align imports, mocks, or test utilities.
Notes:
  - Reads only the top-level <dependencies> section of each POM.
  - Follows parent pom.xml references when available.
  - Raises an error if the module pom.xml is missing.
  - Returns an empty list if the module POM exists but cannot be parsed.
Limitations:
  Does not include transitive dependencies, dependencyManagement-only entries, profiles, or remote parents.
Returns:
  List of dicts with: { group_id, artifact_id }.
  Or a structured error dict on failure.
"""

GET_MAVEN_DEPENDENCIES_DESC = """
Return direct Maven dependencies (groupId and artifactId) from module + parents. 
Use to confirm testing and mocking framework availability.
"""

GET_CLASS_CONSTRUCTORS_AND_FACTORIES_DESC_OLD = """
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

GET_CLASS_CONSTRUCTORS_AND_FACTORIES_DESC = """
Return constructor signatures and obvious static factory methods for a class. 
Use to instantiate required inputs.
"""

GET_GETTERS_AND_SETTERS_DESC_OLD = """
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

GET_GETTERS_AND_SETTERS_DESC = """
Return simple getter/setter signatures found in a class. 
Use for state setup and for reading or updating simple properties.
"""

GENERATE_TEST_CODE_DESC_OLD = """
Create or overwrite the test file with newly generated code and set the active test class.
Args:
  test_code: Complete Java test code including package, imports, class, and methods. This will replace all previous test code in the file.
  qualified_class_name: Fully qualified name for the test class. Be careful with ensuring the package before the simple class name is compliant with the localized methods.
  method_signature: Method header for the single test method annotated with a test annotation generated in your test code (for example, "testFindById()" or "findById(java.lang.Integer)"). Do not include modifiers, annotations, helper, setup, or teardown methods here.
Use when:
  Writing or updating the test to reflect the localized scenario.
Strict formatting for arguments:
  - Provide raw Java source for `test_code`. Do NOT wrap it in Markdown fences (``` ... ```), triple quotes (''' ... '''), or any other wrapper/annotations.
  - Do NOT include JSON, Markdown, XML, or commentary in `test_code`. Only valid Java source.
  - Ensure newlines are literal (no escaped newline sequences) and the string is valid JSON.
Rules:
  - Produce exactly one test method annotated with a test annotation and report its header via `method_signature`.
  - Always send the full file; this overwrites existing content.
  - Ensure package mirrors the primary SUT package to access package-private members.
  - Prefer explicit imports; avoid wildcard imports.
  - Include minimal helper fakes as nested static classes if needed.
  - Helper, setup, or teardown methods are allowed, but they must not be annotated as additional tests.
Returns:
  Dict echoing { test_code, qualified_class_name, method_signature } and the persisted save location.
"""

GENERATE_TEST_CODE_DESC = """
Write the complete Java test file (package, imports, one public test class, exactly one @Test method).
"""

FINALIZE_DESC_OLD = """
End composition with a concise status comment.
Args:
  comments: 1–4 sentences on selected package/class, key fixes, any excluded steps, and unresolved items.
Use when:
  The test compiles and reflects the localized scenario/description, or the iteration limit is reached. This call ends the run.
Returns:
  String echoing the final comments.
"""

FINALIZE_DESC = """
End composition with a concise 1-4 sentence status note of fixes, exclusions, and unresolved issues.
"""

MODIFY_SCENARIO_COMMENT_DESC_OLD = """
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

MODIFY_SCENARIO_COMMENT_DESC = """
Update the comment for a localized step to capture decisions or clarifications tied to individual steps.
"""

# DEPRECATED
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

MODIFY_ATOMIC_BLOCK_NOTE_DESC_OLD = """
Update the note for a specific atomic block by order.
Args:
  order: Atomic block order identifier.
  note: Short note capturing decisions or requirements for that block.
Use when:
  Recording per-block guidance during the composition loop.
Returns:
  Tuple (order, note).
"""

MODIFY_ATOMIC_BLOCK_NOTE_DESC = """
Update the note for a specific atomic block to capture decisions or clarifications tied to individual blocks.
"""
