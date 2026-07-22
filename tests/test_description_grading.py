from __future__ import annotations

import csv
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import sakura.dataset_creation.description_grading.context as context_module
from sakura.cli import grade_descriptions
from sakura.dataset_creation.description_grading.context import JavaContextRenderer
from sakura.dataset_creation.description_grading.server import GradingHTTPServer
from sakura.dataset_creation.description_grading.session import GradingSession
from sakura.test2nl.model.models import (
    AbstractionLevel,
    ClassContext,
    FieldDeclaration,
    MethodContext,
    Test2NLContext as SourceContext,
    Test2NLEntry as DescriptionEntry,
)
from sakura.test2nl.prompts.test2nl_prompt import Test2NLPrompt as PromptFormatter


FIELDNAMES = [
    "abstraction_level",
    "description",
    "id",
    "is_bdd",
    "method_signature",
    "project_name",
    "qualified_class_name",
]


def _write_sample(root: Path, counts: tuple[int, int, int] = (2, 2, 2)) -> Path:
    descriptions = root / "outputs" / "descriptions_sample"
    descriptions.mkdir(parents=True)
    (descriptions / "project_txt.json").write_text(
        json.dumps({"example": "A brief example project description."}),
        encoding="utf-8",
    )
    next_id = 1
    for level, count in zip(("low", "medium", "high"), counts):
        with (descriptions / f"{level}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            for index in range(count):
                writer.writerow(
                    {
                        "abstraction_level": level,
                        "description": f"{level} description {index}",
                        "id": next_id,
                        "is_bdd": False,
                        "method_signature": f"test{next_id}()",
                        "project_name": "example",
                        "qualified_class_name": "example.ExampleTest",
                    }
                )
                next_id += 1
    return descriptions


def test_full_sample_uses_shared_shuffled_order(tmp_path: Path) -> None:
    descriptions = _write_sample(tmp_path, (3, 1, 2))
    session = GradingSession("alice", descriptions, tmp_path)
    ids = [entry.id for entry in session.entries]

    assert sorted(ids) == [1, 2, 3, 4, 5, 6]
    assert ids != [1, 2, 3, 4, 5, 6]
    assert session.selection["kind"] == "full"

    other = GradingSession("bob", descriptions, tmp_path)
    assert [entry.id for entry in other.entries] == ids


def test_subset_selects_ids_in_stable_shuffled_order_and_validates_ids(
    tmp_path: Path,
) -> None:
    descriptions = _write_sample(tmp_path)
    subset_dir = descriptions / "subset"
    subset_dir.mkdir()
    (subset_dir / "alice.json").write_text(
        json.dumps({"low": [2, 1], "medium": [4], "high": [6, 5]}),
        encoding="utf-8",
    )

    session = GradingSession("alice", descriptions, tmp_path)
    ids = [entry.id for entry in session.entries]
    assert sorted(ids) == [1, 2, 4, 5, 6]
    resumed = GradingSession("alice", descriptions, tmp_path)
    assert [entry.id for entry in resumed.entries] == ids

    (subset_dir / "bob.json").write_text(
        json.dumps({"low": [3], "medium": [], "high": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not exist in low.csv"):
        GradingSession("bob", descriptions, tmp_path)


@pytest.mark.parametrize("user", ["../alice", "..", "a/b", " space"])
def test_rejects_unsafe_user_names(tmp_path: Path, user: str) -> None:
    descriptions = _write_sample(tmp_path)
    with pytest.raises(ValueError, match="--user"):
        GradingSession(user, descriptions, tmp_path)


def test_grades_save_only_ids_and_grades_and_resume(tmp_path: Path) -> None:
    descriptions = _write_sample(tmp_path, (1, 1, 1))
    session = GradingSession("alice", descriptions, tmp_path)
    first_id = session.entries[0].id

    session.set_grades(
        first_id, {"fidelity": 4, "perceived_level": "low_medium"}
    )
    saved = json.loads(session.output_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 3
    assert len(saved["entries"]) == 3
    assert all(set(entry) == {"id", "grades"} for entry in saved["entries"])
    assert saved["entries"][0]["id"] == first_id
    assert saved["entries"][0]["grades"]["fidelity"] == 4
    assert saved["entries"][0]["grades"]["perceived_level"] == "low_medium"
    assert saved["entries"][1]["grades"] is None
    assert not session.output_path.with_suffix(".json.tmp").exists()

    resumed = GradingSession("alice", descriptions, tmp_path)
    assert resumed.completed_count == 1
    assert resumed.first_incomplete_index == 1


def test_resume_rejects_a_changed_selection(tmp_path: Path) -> None:
    descriptions = _write_sample(tmp_path, (1, 1, 1))
    session = GradingSession("alice", descriptions, tmp_path)
    session.set_grades(
        session.entries[0].id,
        {"fidelity": 4, "perceived_level": "medium"},
    )
    subset_dir = descriptions / "subset"
    subset_dir.mkdir()
    (subset_dir / "alice.json").write_text(
        json.dumps({"low": [1], "medium": [], "high": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        GradingSession("alice", descriptions, tmp_path)


def test_rejects_out_of_range_or_incomplete_grades(tmp_path: Path) -> None:
    session = GradingSession("alice", _write_sample(tmp_path), tmp_path)
    with pytest.raises(Exception):
        session.set_grades(
            session.entries[0].id,
            {"fidelity": 5, "perceived_level": "medium"},
        )
    with pytest.raises(Exception):
        session.set_grades(
            session.entries[0].id,
            {"fidelity": 4, "perceived_level": "medium-high"},
        )
    with pytest.raises(Exception):
        session.set_grades(session.entries[0].id, {"fidelity": 4})


def test_java_renderer_groups_only_collected_members() -> None:
    context = SourceContext(
        qualified_class_name="example.ExampleTest",
        class_annotations=["@ExtendWith(MockitoExtension.class)"],
        field_declarations=[
            FieldDeclaration(
                variables=["service"], type="example.Service", modifiers=["private"]
            )
        ],
        setup_methods=[MethodContext(method_signature="setUp()", code="void setUp() {}")],
        method_annotations=["@Test"],
        test_method=MethodContext(
            method_signature="testValue()", code="void testValue() {\n    verify();\n}"
        ),
        helper_methods=[
            MethodContext(
                method_signature="verify()",
                qualified_class_name="example.BaseTest",
                code="void verify() {}",
            )
        ],
        application_classes=[
            ClassContext(
                simple_class_name="Service",
                qualified_class_name="example.Service",
                relevant_class_methods=[MethodContext(method_signature="run(int)")],
            )
        ],
    )

    rendered = JavaContextRenderer.render(context)
    assert "class ExampleTest" in rendered
    assert "private example.Service service;" in rendered
    assert "@Test\n    void testValue()" in rendered
    assert "class BaseTest" in rendered
    assert "run(int);" in rendered
    assert "unrelated" not in rendered


def test_prompt_format_serializes_the_shared_context(monkeypatch) -> None:
    context = SourceContext(
        qualified_class_name="example.ExampleTest",
        class_annotations=["@Example"],
        field_declarations=[
            FieldDeclaration(variables=["value"], type="int", modifiers=["private"])
        ],
        method_annotations=["@Test"],
        test_method=MethodContext(
            method_signature="testValue()", code="void testValue() {}"
        ),
    )

    class FakeBuilder:
        @staticmethod
        def build(signature: str, owner: str):
            assert signature == "testValue()"
            assert owner == "example.ExampleTest"
            return context

    captured = {}

    class FakeTemplate:
        @staticmethod
        def render(**kwargs):
            captured.update(kwargs)
            return "rendered"

    monkeypatch.setattr(
        "sakura.test2nl.prompts.test2nl_prompt.LoadPrompt.load_jinja2_template",
        lambda *_args: FakeTemplate(),
    )
    prompt = PromptFormatter.__new__(PromptFormatter)
    prompt.context_builder = FakeBuilder()

    assert (
        prompt.format(
            "testValue()", "example.ExampleTest", AbstractionLevel.MEDIUM
        )
        == "rendered"
    )
    assert captured["class_annotations"] == "@Example"
    assert captured["method_annotations"] == "@Test"
    assert '"method_signature":"testValue()"' in captured["test_method"]
    assert '"variables":["value"]' in captured["field_declarations"][0]


def test_inherited_method_resolves_to_declaring_class(monkeypatch) -> None:
    class FakeAnalysis:
        @staticmethod
        def get_method(owner: str, signature: str):
            if owner == "example.BaseTest" and signature == "testValue(int)":
                return object()
            return None

    simplify = context_module.CommonAnalysis.simplify_method_signature

    class FakeCommonAnalysis:
        simplify_method_signature = staticmethod(simplify)

        def __init__(self, _analysis) -> None:
            pass

        @staticmethod
        def get_testing_frameworks_for_class(_owner):
            return [object()]

    class FakeReachability:
        def __init__(self, _analysis) -> None:
            pass

        @staticmethod
        def get_reachable_test_methods(_owner, _frameworks):
            return {"example.BaseTest": ["testValue(int)"]}

    monkeypatch.setattr(context_module, "CommonAnalysis", FakeCommonAnalysis)
    monkeypatch.setattr(context_module, "Reachability", FakeReachability)
    entry = DescriptionEntry(
        id=1,
        description="description",
        project_name="example",
        qualified_class_name="example.ConcreteTest",
        method_signature="testValue(int)",
        abstraction_level="low",
    )

    assert context_module.DescriptionContextService._resolve_method(
        FakeAnalysis(), entry  # type: ignore[arg-type]
    ) == ("example.BaseTest", "testValue(int)")


class _StaticContext:
    def render_entry(self, _entry) -> str:
        return "class ExampleTest {}\n"


def _request(
    base_url: str,
    token: str,
    path: str,
    *,
    method: str = "GET",
    body: dict | None = None,
) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base_url + path,
        data=data,
        method=method,
        headers={
            "Content-Type": "application/json",
            "X-Session-Token": token,
            "Origin": base_url,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_http_session_entry_grade_and_shutdown(tmp_path: Path) -> None:
    session = GradingSession(
        "alice", _write_sample(tmp_path, (1, 1, 1)), tmp_path
    )
    token = "test-token"
    server = GradingHTTPServer(
        ("127.0.0.1", 0), session, _StaticContext(), token  # type: ignore[arg-type]
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        status, metadata = _request(base, token, "/api/session")
        assert status == 200
        assert metadata["resume_position"] == 0

        status, entry = _request(base, token, "/api/entries/0")
        assert status == 200
        assert entry["code_context"] == "class ExampleTest {}\n"
        assert "abstraction_level" not in entry

        status, metadata = _request(
            base,
            token,
            f"/api/entries/{entry['id']}/grades",
            method="PUT",
            body={"fidelity": 4, "perceived_level": "high"},
        )
        assert status == 200
        assert metadata["completed"] == 1

        status, _ = _request(base, "wrong-token", "/api/session")
        assert status == 403

        status, _ = _request(base, token, "/api/shutdown", method="POST", body={})
        assert status == 200
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        if thread.is_alive():
            server.shutdown()
        server.server_close()


def test_real_project_descriptions_follow_style_rules() -> None:
    root = Path(__file__).resolve().parents[1]
    descriptions_dir = root / "outputs" / "descriptions_sample"
    project_text = json.loads(
        (descriptions_dir / "project_txt.json").read_text(encoding="utf-8")
    )
    projects: set[str] = set()
    for level in ("low", "medium", "high"):
        with (descriptions_dir / f"{level}.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            projects.update(row["project_name"] for row in csv.DictReader(handle))

    assert set(project_text) == projects
    assert all("—" not in text for text in project_text.values())
    assert all(text.count(":") <= 1 for text in project_text.values())


def test_cli_grade_command_uses_repository_root(monkeypatch) -> None:
    called = {}

    def fake_run(*, user: str, repo_root: Path, reset: bool, eager: bool) -> None:
        called.update(user=user, repo_root=repo_root, reset=reset, eager=eager)

    monkeypatch.setattr("sakura.cli.run_description_grader", fake_run)
    grade_descriptions(user="alice")

    assert called["user"] == "alice"
    assert called["repo_root"] == Path(__file__).resolve().parents[1]
    assert called["reset"] is False
    assert called["eager"] is False


def test_reset_archives_incompatible_grades_and_starts_over(tmp_path: Path) -> None:
    descriptions = _write_sample(tmp_path, (1, 1, 1))
    session = GradingSession("alice", descriptions, tmp_path)
    session.set_grades(session.entries[0].id, {"fidelity": 4, "perceived_level": "high"})

    outdated = json.loads(session.output_path.read_text(encoding="utf-8"))
    outdated["schema_version"] = 1
    session.output_path.write_text(json.dumps(outdated), encoding="utf-8")
    with pytest.raises(ValueError, match="--reset"):
        GradingSession("alice", descriptions, tmp_path)

    fresh = GradingSession("alice", descriptions, tmp_path, reset=True)
    assert fresh.completed_count == 0
    assert fresh.archived_backup is not None
    assert fresh.archived_backup.name == "alice.json.bak"
    assert json.loads(fresh.archived_backup.read_text(encoding="utf-8"))[
        "schema_version"
    ] == 1
    assert not fresh.output_path.exists()

    untouched = GradingSession("bob", descriptions, tmp_path, reset=True)
    assert untouched.archived_backup is None
