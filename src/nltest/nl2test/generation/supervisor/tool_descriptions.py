CALL_LOCALIZATION_AGENT_GHERKIN_DESC = """
Delegate to the localization agent to begin localization or refine an already localized scenario.
Args:
  blocks: LocalizedScenario aggregating setup, Given/When/Then steps, and teardown to localize/update.
  instructions: Concrete, strict, and explicit guidance describing which steps to adjust and what to inspect. If modifications are to certain steps only, list the steps by `id` and explicitly mention to not make other changes. You can be more high-level when asking for localization for all steps, like on the first tool call.
Use when:
  You need 1 to 3 candidate methods per step (best-first), updated argument bindings, re-scoring or replacement of candidates, step split/merge, or improved notes and provenance for Gherkin-style steps.
  This should be used to first localize the steps to any relevant methods, and also to make any localization changes based on execution feedback as iterations progress.
Returns:
  Tuple (updated LocalizedScenario, feedback from the localization agent).
"""

CALL_LOCALIZATION_AGENT_GRAMMATICAL_DESC = """
Delegate to the Localization agent to refine AtomicBlocks.
Args:
  blocks: AtomicBlockList representing the working grammatical blocks.
  instructions: Actionable guidance describing what to modify or inspect.
Use when:
  You need the localization agent to re-score candidate methods, adjust bindings, or improve notes for grammatical blocks.
Returns:
  Tuple (updated AtomicBlockList, comments).
"""

CALL_COMPOSITION_AGENT_GHERKIN_DESC = """
Delegate to the composition agent to progress test generation from a localized scenario.
Args:
  blocks: The LocalizedScenario object to compose into executable Java test code. Use the most recent one in your message history.
  instructions: Concrete, informative guidance on test generation and specific implementation details.
Use when:
  Generating the code for the test scenario, using the localized test description decomposition.
  Selecting or revising the test package/class, integrating localized step bindings, adding imports and test annotations, aligning the method body with scenario order, or rewriting code to address diagnostics tied to specific steps.
Returns:
  Tuple (updated LocalizedScenario, composition_comments, current_package of generated test|null, current_class of generated test|null).
"""

CALL_COMPOSITION_AGENT_GRAMMATICAL_DESC = """
Delegate to the Composition agent to progress test generation from AtomicBlocks.
Args:
  blocks: AtomicBlockList to compose into executable test code.
  instructions: Actionable guidance describing the next composition step.
Use when:
  Iterating on package/class selection or rewriting test code to match the blocks.
Returns:
  Tuple (updated AtomicBlockList, composition_comments, current_package|null, current_class|null).
"""

VIEW_TEST_CODE_DESC = """
View the currently saved test file content for the active test class.
Args:
  None
Use when:
  Inspecting the latest generated test source after a composition step, verifying that requested fixes or integrations were applied, or capturing context for subsequent instructions.
  Determining if the test code remains faithful to the natural language description of the test case.
Returns:
  String containing the current test source code, or a descriptive message when the file is missing or not yet generated.
"""

COMPILE_AND_EXECUTE_CODE_DESC = """
Compile the project and execute the active test class.
Args:
  None
Use when:
  Obtaining concrete feedback on compilation status and runtime behavior to drive the next targeted localization or composition iteration; use iteratively as part of a fix loop after code changes.
Returns:
  Dict with compilation details (target_class_file, has_errors_for_target, any_compilation_errors, errors_for_target_class[], error_summary{}) and, when compilation succeeds, `execution` details including test run status, counts, failures, and error messages.
"""

FINALIZE_DESC = """
Mark the supervision process as complete when goals are satisfied or the iteration budget is exhausted.
Args:
  None
Use when:
  You have validated that the composed test aligns with the description and scenario and compiles, or you have reached the iteration limit and must report the best-achieved state.
Returns:
  A simple acknowledgement string summarizing the end state (e.g., package/class, compile status, brief notes).
"""
