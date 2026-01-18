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
