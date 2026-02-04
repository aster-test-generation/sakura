def pretty_indent(raw_code_str: str, indent_size: int = 4):
    # Removes new lines and allows for better character-by-character processing
    compact = " ".join(raw_code_str.strip().split())

    result_lines = []
    indent_level = 0
    token = ""
    first_brace = True

    def flush_token():
        # Adds existing token sequence into new line with correct indent level
        nonlocal token
        t = token.strip()
        if t:
            result_lines.append(" " * (indent_level * indent_size) + t)
        token = ""

    i = 0
    while i < len(compact):
        ch = compact[i]

        if ch == '{':
            if first_brace:
                # Append to the signature line
                token += " {"
                flush_token()
                indent_level += 1
                first_brace = False
                i += 1

            else:
                # Flush old line before bracket
                flush_token()

                # Bracket is on own line
                result_lines.append(" " * (indent_level * indent_size) + "{")
                indent_level += 1
                i += 1

        elif ch == '}':
            first_brace = False # Edge case where code started with '}'

            # Flush line before bracket
            flush_token()

            # Indent level is decreased before bracket print
            indent_level = max(indent_level - 1, 0)
            result_lines.append(" " * (indent_level * indent_size) + "}")
            i += 1

        elif ch == ';':
            # Each semicolon defines new line
            token += ';'
            flush_token()
            i += 1

        else:
            token += ch
            i += 1

    # Flushes trailing sequence
    flush_token()

    return "\n".join(result_lines)