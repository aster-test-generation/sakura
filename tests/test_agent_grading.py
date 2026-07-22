from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from sakura.dataset_creation.agent_grading.docker_runner import (
    DockerSandboxError,
    DockerSandboxRunner,
)
from sakura.dataset_creation.agent_grading.file_paths import derive_java_file_path
from sakura.dataset_creation.agent_grading.orchestrator import (
    AggregateGradeFile,
    build_task_spec,
)
from sakura.dataset_creation.agent_grading.schema import (
    STRUCTURED_OUTPUT_FORMAT,
    validate_structured_grades,
)
from sakura.dataset_creation.description_grading import grading_criteria, web_ui
from sakura.dataset_creation.description_grading.session import (
    Grades,
    GradingSession,
    load_grader_order,
)
from sakura.test2nl.model.models import Test2NLEntry


FIELDNAMES = [
    "abstraction_level",
    "description",
    "id",
    "is_bdd",
    "method_signature",
    "project_name",
    "qualified_class_name",
]

AUTH_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
)


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


def _entry(**overrides) -> Test2NLEntry:
    values = {
        "id": 7,
        "description": "A description.",
        "project_name": "example",
        "qualified_class_name": "org.example.FooTest",
        "method_signature": "testFoo()",
        "abstraction_level": "medium",
        "is_bdd": False,
    }
    values.update(overrides)
    return Test2NLEntry(**values)


def test_load_grader_order_matches_grading_session(tmp_path: Path) -> None:
    descriptions = _write_sample(tmp_path, (3, 1, 2))
    session = GradingSession("alice", descriptions, tmp_path)
    order = load_grader_order(descriptions)
    assert [entry.id for entry in order] == [entry.id for entry in session.entries]


def test_docker_command_hardening_mounts_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in AUTH_ENV_NAMES + ("ANTHROPIC_BASE_URL",):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy:4000")

    runner = DockerSandboxRunner(
        image="test-image:latest",
        dockerfile=tmp_path / "Dockerfile",
        memory="2g",
        cpus=1.5,
        pids_limit=64,
    )
    command = runner._docker_command(
        output_root=tmp_path / "out",
        container_spec_path="/out/specs/entry-000007.json",
        repo_source=tmp_path / "mirror.git",
    )

    assert command[:3] == ["docker", "run", "--rm"]
    pairs = [command[i : i + 2] for i in range(len(command) - 1)]
    for flag_pair in (
        ["--cap-drop", "ALL"],
        ["--security-opt", "no-new-privileges"],
        ["--network", "bridge"],
        ["--pids-limit", "64"],
        ["--memory", "2g"],
        ["--cpus", "1.5"],
        ["-e", "ANTHROPIC_API_KEY"],
        ["-e", "ANTHROPIC_BASE_URL"],
        ["-v", f"{(tmp_path / 'out').resolve()}:/out:rw"],
        ["-v", f"{(tmp_path / 'mirror.git').resolve()}:/repo-src:ro"],
    ):
        assert flag_pair in pairs
    assert ["-e", "ANTHROPIC_AUTH_TOKEN"] not in pairs
    assert command[-3:] == [
        "test-image:latest",
        "--task-spec",
        "/out/specs/entry-000007.json",
    ]


def test_ensure_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in AUTH_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    runner = DockerSandboxRunner(dockerfile=tmp_path / "Dockerfile")
    with pytest.raises(DockerSandboxError, match="--allow-missing-auth"):
        runner.ensure_auth()

    DockerSandboxRunner(
        dockerfile=tmp_path / "Dockerfile", require_auth_env=False
    ).ensure_auth()

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "token")
    runner.ensure_auth()


def test_build_task_spec_hides_abstraction_level_and_carries_criteria() -> None:
    spec = build_task_spec(
        _entry(),
        project_description="Example project.",
        file_path="src/test/java/org/example/FooTest.java",
        pinned_commit="abc123",
        model="claude-sonnet-5",
        thinking="high",
        result_relpath="details/entry-000007.json",
    )

    assert "abstraction_level" not in json.dumps(spec)
    assert spec["entry"]["id"] == 7
    assert spec["entry"]["qualified_class_name"] == "org.example.FooTest"
    assert spec["entry"]["method_signature"] == "testFoo()"
    assert spec["entry"]["description"] == "A description."
    assert spec["pinned_commit"] == "abc123"
    assert spec["agent"] == {"model": "claude-sonnet-5", "thinking": "high"}
    criteria = spec["criteria"]
    assert criteria["fidelity_help"] == grading_criteria.FIDELITY_HELP
    assert criteria["fidelity_options"] == grading_criteria.FIDELITY_OPTIONS
    assert criteria["perceived_options"] == grading_criteria.PERCEIVED_OPTIONS
    assert criteria["level_definitions"] == grading_criteria.LEVEL_DEFINITIONS
    assert criteria["invariant_note"] == grading_criteria.INVARIANT_NOTE
    json.dumps(spec)  # must be JSON-serializable


def test_derive_java_file_path(tmp_path: Path) -> None:
    test_file = tmp_path / "src/test/java/org/foo/BarTest.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("class BarTest {}", encoding="utf-8")
    decoy = tmp_path / "target/generated/org/foo/BarTest.java"
    decoy.parent.mkdir(parents=True)
    decoy.write_text("class BarTest {}", encoding="utf-8")

    expected = "src/test/java/org/foo/BarTest.java"
    assert derive_java_file_path(tmp_path, "org.foo.BarTest") == expected
    assert derive_java_file_path(tmp_path, "org.foo.BarTest$Inner") == expected
    assert derive_java_file_path(tmp_path, "org.foo.BarTest.Inner") == expected
    assert derive_java_file_path(tmp_path, "org.foo.MissingTest") is None


def test_derive_java_file_path_prefers_test_root_and_falls_back(
    tmp_path: Path,
) -> None:
    main_file = tmp_path / "src/main/java/org/foo/Bar.java"
    main_file.parent.mkdir(parents=True)
    main_file.write_text("class Bar {}", encoding="utf-8")
    test_file = tmp_path / "src/test/java/org/foo/Bar.java"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("class Bar {}", encoding="utf-8")

    assert (
        derive_java_file_path(tmp_path, "org.foo.Bar")
        == "src/test/java/org/foo/Bar.java"
    )
    # Package mismatch falls back to any file named after the outer class.
    assert (
        derive_java_file_path(tmp_path, "com.other.Bar")
        == "src/test/java/org/foo/Bar.java"
    )


def test_aggregate_grade_file_init_resume_and_reset(tmp_path: Path) -> None:
    path = tmp_path / "agent.json"
    aggregate = AggregateGradeFile(path, "agent", [5, 3, 9])
    aggregate.save()

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 3
    assert saved["user"] == "agent"
    assert saved["selection"] == {"kind": "full", "path": None}
    assert [entry["id"] for entry in saved["entries"]] == [5, 3, 9]
    assert all(entry["grades"] is None for entry in saved["entries"])

    aggregate.set_grade(3, Grades(fidelity=4, perceived_level="medium_high"))
    aggregate.save()

    resumed = AggregateGradeFile(path, "agent", [5, 3, 9])
    assert resumed.grade_for(3) == Grades(fidelity=4, perceived_level="medium_high")
    assert resumed.grade_for(5) is None

    with pytest.raises(ValueError, match="--reset"):
        AggregateGradeFile(path, "agent", [5, 3])
    with pytest.raises(ValueError, match="--reset"):
        AggregateGradeFile(path, "other-label", [5, 3, 9])
    with pytest.raises(ValueError, match="grader order"):
        resumed.set_grade(999, Grades(fidelity=1, perceived_level="low"))

    fresh = AggregateGradeFile(path, "agent", [5, 3, 9], reset=True)
    assert fresh.grade_for(3) is None
    assert fresh.archived_backup is not None
    assert fresh.archived_backup.name == "agent.json.bak"
    assert not path.exists()


def test_validate_structured_grades() -> None:
    payload = {
        "fidelity": 3,
        "fidelity_rationale": "Matches the assertions.",
        "perceived_level": "medium",
        "perceived_rationale": "Architectural phrasing.",
    }
    assert validate_structured_grades(payload) == {
        "fidelity": 3,
        "perceived_level": "medium",
    }

    for bad in (
        None,
        "text",
        {**payload, "fidelity": 5},
        {**payload, "fidelity": True},
        {**payload, "fidelity": "3"},
        {**payload, "perceived_level": "medium-high"},
        {**payload, "fidelity_rationale": ""},
        {k: v for k, v in payload.items() if k != "perceived_rationale"},
    ):
        with pytest.raises(ValueError):
            validate_structured_grades(bad)


def test_structured_output_schema_matches_grades_model() -> None:
    schema = STRUCTURED_OUTPUT_FORMAT["schema"]
    assert schema["properties"]["fidelity"]["minimum"] == 1
    assert schema["properties"]["fidelity"]["maximum"] == 4
    assert schema["properties"]["perceived_level"]["enum"] == list(
        grading_criteria.PERCEIVED_VALUES
    )
    Grades(fidelity=2, perceived_level=schema["properties"]["perceived_level"]["enum"][1])


def test_web_ui_html_embeds_shared_criteria() -> None:
    assert "__" not in _leftover_tokens(web_ui.HTML)
    assert json.dumps(grading_criteria.FIDELITY_OPTIONS) in web_ui.HTML
    assert json.dumps(grading_criteria.PERCEIVED_OPTIONS) in web_ui.HTML
    assert json.dumps(grading_criteria.LEVEL_DEFINITIONS) in web_ui.HTML
    assert json.dumps(grading_criteria.FIDELITY_HELP) in web_ui.HTML
    assert json.dumps(grading_criteria.PERCEIVED_HELP) in web_ui.HTML
    assert grading_criteria.INVARIANT_NOTE in web_ui.HTML


def _leftover_tokens(html: str) -> str:
    tokens = (
        "__FIDELITY_OPTIONS__",
        "__PERCEIVED_OPTIONS__",
        "__FIDELITY_HELP__",
        "__PERCEIVED_HELP__",
        "__LEVEL_INFO__",
        "__INVARIANT_NOTE__",
    )
    return "".join(token for token in tokens if token in html)
