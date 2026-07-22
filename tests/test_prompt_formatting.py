from sakura.utils.pretty.prompt_formatting import pretty_indent


def test_indents_nested_blocks_and_splits_on_semicolons() -> None:
    code = "void run() { if (ready) { start(); } stop(); }"
    assert pretty_indent(code) == (
        "void run()  {\n"
        "    if (ready)\n"
        "    {\n"
        "        start();\n"
        "    }\n"
        "    stop();\n"
        "}"
    )


def test_line_comment_does_not_swallow_following_code() -> None:
    code = "void run() {\n    // @formatter:off\n    first();\n    // note\n    second();\n}"
    lines = pretty_indent(code).splitlines()
    assert "    // @formatter:off" in lines
    assert "    first();" in lines
    assert "    // note" in lines
    assert "    second();" in lines


def test_trailing_line_comment_keeps_next_statement_uncommented() -> None:
    code = "void run() {\n    first(); // explains first\n    second();\n}"
    lines = pretty_indent(code).splitlines()
    assert "    first();" in lines
    assert "    // explains first" in lines
    assert "    second();" in lines


def test_symbols_inside_comments_do_not_affect_structure() -> None:
    code = (
        "void run()  {\n"
        "    // pending; if (x) { y() }\n"
        "    /* block; with { braces } */ done();\n"
        "}"
    )
    assert pretty_indent(code) == (
        "void run()  {\n"
        "    // pending; if (x) { y() }\n"
        "    /* block; with { braces } */ done();\n"
        "}"
    )


def test_symbols_inside_string_literals_do_not_split_lines() -> None:
    code = 'void run() {\n    String s = "a;b{c}d // no comment";\n    done();\n}'
    assert pretty_indent(code) == (
        "void run()  {\n"
        '    String s = "a;b{c}d // no comment";\n'
        "    done();\n"
        "}"
    )
