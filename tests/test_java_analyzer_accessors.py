from typing import cast

import pytest
from cldk.analysis.java import JavaAnalysis
from cldk.models.java.models import JCallable, JCallableParameter

from sakura.test2nl.extractors import MethodExtractor
from sakura.utils.analysis import CommonAnalysis


def _parameter(name: str = "value") -> JCallableParameter:
    return JCallableParameter(
        name=name,
        type="java.lang.String",
        annotations=[],
        modifiers=[],
        start_line=1,
        end_line=1,
        start_column=1,
        end_column=1,
    )


def _callable(
    signature: str,
    code: str,
    return_type: str | None,
    *,
    parameters: list[JCallableParameter] | None = None,
    accessed_fields: list[str] | None = None,
    is_implicit: bool = False,
) -> JCallable:
    return JCallable(
        signature=signature,
        is_implicit=is_implicit,
        is_constructor=False,
        comments=[],
        annotations=[],
        modifiers=["public"],
        declaration=f"public {return_type or ''} {signature}".strip(),
        parameters=parameters or [],
        return_type=return_type,
        code=code,
        start_line=1,
        end_line=1 + code.count("\n"),
        code_start_line=1,
        referenced_types=[],
        accessed_fields=accessed_fields or [],
        call_sites=[],
        variable_declarations=[],
        crud_operations=[],
        crud_queries=[],
        cyclomatic_complexity=1,
    )


@pytest.mark.parametrize(
    "method",
    [
        _callable(
            "getName()",
            "{\n    return this.name;\n}",
            "java.lang.String",
            accessed_fields=["example.Person.name"],
        ),
        _callable("getCount()", "{ return count; }", "int"),
        _callable("isActive()", "{ return active; }", "boolean"),
        _callable("isReady()", "{ return this.ready; }", "java.lang.Boolean"),
        _callable(
            "setName(java.lang.String)",
            "{ this.name = value; }",
            "void",
            parameters=[_parameter()],
            accessed_fields=["example.Person.name"],
        ),
        _callable(
            "setCount(java.lang.String)",
            "{ count = value; }",
            "void",
            parameters=[_parameter()],
        ),
    ],
)
def test_recognizes_direct_javabeans_accessors(method: JCallable) -> None:
    assert CommonAnalysis.is_getter_or_setter(method) is True


@pytest.mark.parametrize(
    "method",
    [
        _callable("setup()", "{ repository.reset(); }", "void"),
        _callable(
            "setUp(java.lang.String)",
            "{ this.value = value; }",
            "void",
            parameters=[_parameter()],
        ),
        _callable("getaway()", "{ return destination; }", "java.lang.String"),
        _callable("island()", "{ return island; }", "boolean"),
        _callable(
            "getValue(java.lang.String)",
            "{ return value; }",
            "int",
            parameters=[_parameter()],
        ),
        _callable("getValue()", "{ return value; }", "void"),
        _callable("isValue()", "{ return value; }", "java.lang.String"),
        _callable("setValue()", "{ this.value = value; }", "void"),
        _callable(
            "setValue(java.lang.String)",
            "{ this.value = value; return this; }",
            "example.Person",
            parameters=[_parameter()],
        ),
        _callable("getValue()", "{ return value + 1; }", "int"),
        _callable("getValue()", "{ return loadValue(); }", "int"),
        _callable(
            "setValue(java.lang.String)",
            "{ this.value = normalize(value); }",
            "void",
            parameters=[_parameter()],
        ),
        _callable(
            "setValue(java.lang.String)",
            "{ value = value; }",
            "void",
            parameters=[_parameter()],
        ),
        _callable(
            "getValue()",
            "{ return value; }",
            "int",
            accessed_fields=["example.Person.other"],
        ),
        _callable(
            "getValue()",
            "{ return value; }",
            "int",
            accessed_fields=["example.Person.value", "example.Person.other"],
        ),
        _callable("getValue()", "{}", "int"),
        _callable("getValue()", "", "int"),
        _callable("getValue()", "{ return value; }", "int", is_implicit=True),
    ],
)
def test_rejects_non_accessors(method: JCallable) -> None:
    assert CommonAnalysis.is_getter_or_setter(method) is False


class _MethodAnalysis:
    def __init__(self, method: JCallable) -> None:
        self.method = method

    def get_method(
        self, qualified_class_name: str, method_signature: str
    ) -> JCallable | None:
        if qualified_class_name == "example.Fixture" and method_signature == "setup()":
            return self.method
        return None


def test_method_extractor_retains_short_setup_method_code() -> None:
    setup_method = _callable("setup()", "{\n    repository.reset();\n}", "void")
    analysis = cast(JavaAnalysis, _MethodAnalysis(setup_method))

    context = MethodExtractor(analysis, []).extract(
        "example.Fixture", "setup()", complete_methods=True
    )

    assert context.is_getter_or_setter is False
    assert context.code is not None
    assert "repository.reset()" in context.code
