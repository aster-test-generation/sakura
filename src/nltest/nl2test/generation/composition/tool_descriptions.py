EXTRACT_CODE_DESC = """
Extract a method's full source code.

Args:
    qualified_class_name: Class that contains the method.
    method_signature: Signature of the method.

Returns:
    Method code, or an error string.
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
  Visibility options: "public", "same_package_or_subclass", or "same_package".
"""
