from dataclasses import dataclass
from typing import cast

from cldk.analysis.java import JavaAnalysis

from sakura.utils.analysis import CommonAnalysis


@dataclass
class FakeClass:
    is_interface: bool = False
    extends_list: list[str] | None = None
    implements_list: list[str] | None = None


class FakeJavaAnalysis:
    def __init__(self, classes: dict[str, FakeClass]) -> None:
        self.classes = classes

    def get_class(self, qualified_class_name: str) -> FakeClass | None:
        return self.classes.get(qualified_class_name)


def build_common_analysis(classes: dict[str, FakeClass]) -> CommonAnalysis:
    common_analysis = object.__new__(CommonAnalysis)
    common_analysis.analysis = cast(JavaAnalysis, FakeJavaAnalysis(classes))
    return common_analysis


def test_implements_interface_follows_parent_interfaces() -> None:
    common_analysis = build_common_analysis(
        {
            "Concrete": FakeClass(implements_list=["Child"]),
            "Child": FakeClass(is_interface=True, extends_list=["Parent"]),
            "Parent": FakeClass(is_interface=True),
        }
    )

    assert common_analysis.implements_interface("Concrete", "Parent")


def test_implements_interface_follows_superclass_and_parent_interfaces() -> None:
    common_analysis = build_common_analysis(
        {
            "Concrete": FakeClass(extends_list=["Base"]),
            "Base": FakeClass(implements_list=["Child"]),
            "Child": FakeClass(is_interface=True, extends_list=["Parent"]),
            "Parent": FakeClass(is_interface=True),
        }
    )

    assert common_analysis.implements_interface("Concrete", "Parent")


def test_implements_interface_handles_cyclic_interface_ancestry() -> None:
    common_analysis = build_common_analysis(
        {
            "Concrete": FakeClass(implements_list=["First"]),
            "First": FakeClass(is_interface=True, extends_list=["Second"]),
            "Second": FakeClass(is_interface=True, extends_list=["First"]),
        }
    )

    assert not common_analysis.implements_interface("Concrete", "Missing")
