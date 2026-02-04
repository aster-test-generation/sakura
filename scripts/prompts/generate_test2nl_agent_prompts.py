from __future__ import annotations

import sys
from pathlib import Path

from jinja2 import Template

from sakura.test2nl.model.models import Test2NLEntry
from sakura.utils.file_io.structured_data_manager import StructuredDataManager

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


PROMPT_FILE = "scripts/prompts/out_of_box_prompt.jinja2"
TEST2NL_FILE = "resources/test2nl/filtered_dataset/test2nl.csv"
OUTPUT_DIR = "resources/out_of_box/"
OUTPUT_FILE_NAME = "test2nl_agent_prompts.csv"


class Test2NLAgentEntry(Test2NLEntry):
    agent_prompt: str


def load_prompt_template(prompt_path: Path) -> Template:
    prompt_text = prompt_path.read_text(encoding="utf-8")
    return Template(prompt_text)


def build_agent_entries(
    entries: list[Test2NLEntry], prompt_template: Template
) -> list[Test2NLAgentEntry]:
    agent_entries: list[Test2NLAgentEntry] = []
    for entry in entries:
        if not isinstance(entry, Test2NLEntry):
            continue
        agent_prompt = prompt_template.render(test_description=entry.description)
        agent_entries.append(
            Test2NLAgentEntry(**entry.model_dump(), agent_prompt=agent_prompt)
        )
    return agent_entries


def main() -> None:
    prompt_path = (ROOT_DIR / PROMPT_FILE).resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

    test2nl_path = (ROOT_DIR / TEST2NL_FILE).resolve()
    if not test2nl_path.is_file():
        raise FileNotFoundError(f"Test2NL CSV not found: {test2nl_path}")

    output_dir = (ROOT_DIR / OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = load_prompt_template(prompt_path)

    data_manager = StructuredDataManager(test2nl_path.parent)
    test2nl_entries = data_manager.load(test2nl_path.name, Test2NLEntry, format="csv")

    agent_entries = build_agent_entries(test2nl_entries, prompt_template)

    output_path = output_dir / OUTPUT_FILE_NAME
    output_manager = StructuredDataManager(output_dir)
    output_manager.save(output_path.name, agent_entries, format="csv")

    print("Agent prompt generation complete.")
    print(f"Entries written: {len(agent_entries)}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
