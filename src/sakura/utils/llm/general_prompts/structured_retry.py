STRUCTURED_OUTPUT_RETRY_PROMPT = """
---
**Schema validation failed (attempt {attempt}):**
{error_excerpt}

**Your output (truncated):**
{output_excerpt}

**Common fixes:**
- If error mentions "should be a valid dictionary": use an object `{{...}}` not a string
- If error mentions "should be a valid list": use an array `[...]` not a single value
- Ensure all required fields are present and correctly typed

Regenerate the output with valid structured data matching the schema exactly.
"""

LENGTH_EXCEEDED_RETRY_PROMPT = """
---
**Response truncated due to length limit (attempt {attempt}):**
Your output exceeded the maximum token limit and was cut off before completion.

**Partial output received:**
{partial_output}

**Required action:**
- Produce a MORE CONCISE response that fits within token limits
- Remove unnecessary details, verbose explanations, or redundant content
- Focus only on the essential information required by the schema
- If generating code, minimize comments and use shorter variable names

Regenerate the output with valid structured data, keeping it brief.
"""
