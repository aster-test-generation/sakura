# Marks a newline that terminates a // comment before whitespace collapsing
# destroys it; the newline is the only terminator a line comment has.
_LINE_BREAK = "\u0000"


def _line_comment_start(line: str) -> int:
    # Returns the index where a // comment starts, ignoring string literals
    # and same-line block comments. -1 when the line has no line comment.
    in_string = ""
    i = 0
    while i < len(line) - 1:
        ch = line[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == in_string:
                in_string = ""
        elif ch in "\"'":
            in_string = ch
        elif ch == "/" and line[i + 1] == "/":
            return i
        elif ch == "/" and line[i + 1] == "*":
            end = line.find("*/", i + 2)
            if end == -1:
                return -1
            i = end + 1
        i += 1
    return -1


def _mark_line_comment_breaks(raw_code_str: str) -> str:
    marked = []
    for line in raw_code_str.splitlines():
        if _line_comment_start(line) != -1:
            line += _LINE_BREAK
        marked.append(line)
    return " ".join(marked)


def pretty_indent(raw_code_str: str, indent_size: int = 4):
    # Removes new lines and allows for better character-by-character
    # processing. Lines ending in a // comment are marked first so the
    # comment cannot swallow the code that followed it.
    compact = " ".join(_mark_line_comment_breaks(raw_code_str).strip().split())

    result_lines = []
    indent_level = 0
    token = ""
    first_brace = True

    def flush_token():
        # Adds existing token sequence into new line with correct indent level
        nonlocal token
        t = token.replace(_LINE_BREAK, " ").strip()
        if t:
            result_lines.append(" " * (indent_level * indent_size) + t)
        token = ""

    i = 0
    while i < len(compact):
        ch = compact[i]

        if ch == _LINE_BREAK:
            flush_token()
            i += 1

        elif compact.startswith("//", i):
            # A line comment runs to its original line break; braces and
            # semicolons inside it must not affect line splitting or indent
            end = compact.find(_LINE_BREAK, i)
            end = len(compact) if end == -1 else end
            token += compact[i:end]
            flush_token()
            i = min(end + 1, len(compact))

        elif compact.startswith("/*", i):
            end = compact.find("*/", i + 2)
            end = len(compact) if end == -1 else end + 2
            token += compact[i:end]
            i = end

        elif ch in "\"'":
            # String and char literals are opaque to line splitting
            j = i + 1
            while j < len(compact):
                if compact[j] == "\\":
                    j += 2
                elif compact[j] == ch:
                    j += 1
                    break
                else:
                    j += 1
            token += compact[i:j]
            i = j

        elif ch == '{':
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
