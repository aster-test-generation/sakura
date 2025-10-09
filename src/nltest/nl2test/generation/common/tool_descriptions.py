EXTRACT_CODE_DESC = """
Extract the full source code for a specific method.
Args:
  qualified_class_name: Fully qualified class name containing the method.
  method_signature: Exact method signature to extract.
Use when:
  You need to verify behavior, parameters, or control flow for a candidate method.
Returns:
  String with the full method source.
"""

METHOD_DETAILS_DESC = """
Get details for a specific method including signature, modifiers, return type, parameter types, comments, and visibility.
Args:
  qualified_class_name: Fully qualified class name containing the method.
  method_signature: Exact method signature to inspect.
Use when:
  You must confirm method visibility, types, or comments before using it as a candidate.
Returns:
  Dict { method_signature, modifiers[], return_type, parameter_types[], comments[], visibility }.
"""

CALL_SITE_DETAILS_DESC = """
List direct callees of a method with basic metadata.
Args:
  qualified_class_name: Class owning the method.
  method_signature: Exact signature for the caller method.
Use when:
  You need to see which downstream APIs a method calls to justify suitability.
Returns:
  List of { qualified_class_name, method_signature, return_type, parameter_types[], modifiers[], num_times_called }.
"""

