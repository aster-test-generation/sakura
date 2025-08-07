QUERY_METHOD_DESC = """
Semantic search in the method index. Does not contain inherited library methods.
Args:
    query: Query string.
    i: 1-based start rank, must be > 0.
    j: 1-based end rank (inclusive), must be >= i.
Returns:
    List of {'method_signature', 'qualified_class_name'} dicts, or {'status': 'error', 'error_type': str, 'message': str, 'details': dict} on error.
"""

QUERY_CLASS_DESC = """
Semantic search in the class index.
Args:
    query: Query string.
    i: 1-based start rank, must be > 0.
    j: 1-based end rank (inclusive), must be >= i.
Returns:
    List of {'qualified_class_name'} dicts, or {'status': 'error', 'error_type': str, 'message': str, 'details': dict} on error.
"""

REACHABLE_DESC = """
List all visible methods reachable from a class, including inherited ones from other application classes. 
Does not include inherited library methods from library classes, which must be discovered from analyzing class details.
Args:
    qualified_class_name: Fully qualified class name.
Returns:
    Mapping {qualified_class_name: [method_signature]}, or {'status': 'error', 'error_type': str, 'message': str, 'details': dict} on error.
"""

EXTRACT_CODE_DESC = """
Extract a method's full source code.
Args:
    qualified_class_name: Class that contains the method.
    method_signature: Signature of the method.
Returns:
    Method code, or {'status': 'error', 'error_type': str, 'message': str, 'details': dict} on error.
"""

METHOD_DETAILS_DESC = """
Return metadata for a method.
Args:
    qualified_class_name: Class that contains the method.
    method_signature: Signature of the method.
Returns:
    Dict with the method_signature, modifiers, return_type, parameter_types, comments,
    or {'status': 'error', 'error_type': str, 'message': str, 'details': dict} on error.
"""

CLASS_DETAILS_DESC = """
Return structural data for a class.
Args:
    qualified_class_name: Fully qualified class name.
Returns:
    Dict with the class_name, modifiers, extends_list, implements_list,
    annotations, or {'status': 'error', 'error_type': str, 'message': str, 'details': dict} on error.
"""

CALL_SITE_DETAILS_DESC = """
List call sites invoked inside a method.
Args:
    qualified_class_name: Class that contains the method.
    method_signature: Signature of the method.
Returns:
    List of call site dicts or {'status': 'error', 'error_type': str, 'message': str, 'details': dict} on error.
"""

MODIFY_BLOCKS_DESC = """
Transform AtomicBlocks by merging, splitting, reordering, removing, or adding blocks, or assigning relevant methods to a block.
Identify blocks for change by their order ID.
Args:
    instructions: High level text describing the desired modifications. Include relevant methods for the changed atomic block(s) if available.
Returns:
    A success message, or {'status': 'error', 'error_type': str, 'message': str, 'details': dict} on error.
"""