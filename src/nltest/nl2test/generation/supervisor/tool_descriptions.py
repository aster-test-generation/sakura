CALL_LOCALIZATION_AGENT_GHERKIN_DESC_OLD = """
Delegate to the localization agent to begin localization or refine an already localized scenario.
Args:
  instructions: Concrete, strict, and explicit guidance describing which steps to adjust and what to inspect. If modifications are to certain steps only, list the steps by `id` and explicitly mention to not make other changes. You can be more high-level when asking for localization for all steps, like on the first tool call.
Notes:
  The Supervisor automatically injects the current LocalizedScenario from its own state; you must not supply `blocks`.
Use when:
  You need 1 to 3 candidate methods per step (best-first), updated argument bindings, re-scoring or replacement of candidates, step split/merge, or improved notes and provenance for Gherkin-style steps.
  This should be used to first localize the steps to any relevant methods, and also to make any localization changes based on execution feedback as iterations progress.
Returns:
  Tuple (updated LocalizedScenario, feedback from the localization agent).
"""

CALL_LOCALIZATION_AGENT_GHERKIN_DESC = """
Delegate to the localization agent to begin localization or refine an already localized scenario.
Receives the localization agent's feedback regarding localization difficulties.
"""

CALL_LOCALIZATION_AGENT_GRAMMATICAL_DESC_OLD = """
Delegate to the Localization agent to refine AtomicBlocks.
Args:
  instructions: Actionable guidance describing what to modify or inspect.
Notes:
  The Supervisor automatically injects the current AtomicBlockList from its own state; you must not supply `blocks`.
Use when:
  You need the localization agent to re-score candidate methods, adjust bindings, or improve notes for grammatical blocks.
Returns:
  Tuple (updated AtomicBlockList, comments).
"""

CALL_LOCALIZATION_AGENT_GRAMMATICAL_DESC = """
Delegate to the localization agent to begin localization or refine an already localized atomic block list.
Receives the localization agent's feedback regarding localization difficulties.
"""

CALL_COMPOSITION_AGENT_GHERKIN_DESC_OLD = """
Delegate to the composition agent to progress test generation from a localized scenario.
Args:
  instructions: Concrete, informative guidance on test generation and specific implementation details.
Notes:
  The Supervisor automatically injects the current LocalizedScenario from its own state; do not supply `blocks`.
Use when:
  Generating the code for the test scenario, using the localized test description decomposition.
  Selecting or revising the test package/class, integrating localized step bindings, adding imports and test annotations, aligning the method body with scenario order, or rewriting code to address diagnostics tied to specific steps.
Returns:
  Tuple (updated LocalizedScenario, composition_comments, current_package of generated test|null, current_class of generated test|null).
"""

CALL_COMPOSITION_AGENT_GHERKIN_DESC = """
Delegate to the composition agent to conduct test generation from a localized scenario. 
Receives the composition agent's feedback regarding test composition difficulties from the scenario. 
"""

CALL_COMPOSITION_AGENT_GRAMMATICAL_DESC_OLD = """
Delegate to the Composition agent to progress test generation from AtomicBlocks.
Args:
  instructions: Actionable guidance describing the next composition step.
Notes:
  The Supervisor automatically injects the latest AtomicBlockList from its state; do not supply `blocks`.
Use when:
  Iterating on package/class selection or rewriting test code to match the blocks.
Returns:
  Tuple (updated AtomicBlockList, composition_comments, current_package|null, current_class|null).
"""

CALL_COMPOSITION_AGENT_GRAMMATICAL_DESC = """
Delegate to the composition agent to conduct test generation from an atomic block list. 
Receives the composition agent's feedback regarding test composition difficulties from the atomic blocks. 
"""

FINALIZE_DESC_OLD = """
End supervision when goals are met or iterations are exhausted.
Args:
  None
Use when:
  The composed test aligns with the description and scenario and compiles, or when at the iteration limit. This call ends the run.
Returns:
  A short acknowledgement string (e.g., selected package/class, compile status, brief notes). Keep this concise.
"""

FINALIZE_DESC = """
End supervision and complete test generation once fault localization is complete and a compilable test case that satisfies all provided test descriptions has been produced.
"""
