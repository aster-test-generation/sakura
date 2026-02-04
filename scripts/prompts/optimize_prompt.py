import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === CONFIGURATION CONSTANTS ===
# Root directory (project root, two levels up from this script)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Prompt file names (located in src/sakura/nl2test/prompts/templates/{system,chat}/)
SYSTEM_PROMPT_FILE = "composition_agent_gherkin.jinja2"
CHAT_PROMPT_FILE = "composition_agent_gherkin.jinja2"

# Output directory relative to ROOT_DIR
OUTPUT_DIR = "tests/output/optimized_prompts"

# GCP Configuration for Vertex AI Prompt Optimizer
# Can be set via environment variables or hardcoded here
GCP_PROJECT = os.getenv("GCP_PROJECT", "your-gcp-project-id")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")


def main() -> None:
    # Add src to path for imports
    src_dir = ROOT_DIR / "src"
    sys.path.insert(0, str(src_dir))

    # Import after path setup
    from sakura.nl2test.prompts.load_prompt import LoadPrompt, PromptFormat
    from sakura.utils.llm import PromptFormatter

    # Resolve output directory
    output_dir = (ROOT_DIR / OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompt templates
    print(f"Loading system prompt: {SYSTEM_PROMPT_FILE}")
    system_template = LoadPrompt.load_prompt(
        file_name=SYSTEM_PROMPT_FILE,
        prompt_format=PromptFormat.JINJA2,
        prompt_type="system",
    )

    print(f"Loading chat prompt: {CHAT_PROMPT_FILE}")
    chat_template = LoadPrompt.load_prompt(
        file_name=CHAT_PROMPT_FILE,
        prompt_format=PromptFormat.JINJA2,
        prompt_type="chat",
    )

    # Extract raw template strings
    system_prompt_str = system_template.template
    chat_prompt_str = chat_template.template

    # Initialize the prompt optimizer
    print(f"Initializing PromptFormatter with project={GCP_PROJECT}, location={GCP_LOCATION}")
    formatter = PromptFormatter(project=GCP_PROJECT, location=GCP_LOCATION)

    # Optimize prompts
    print("Optimizing prompts via Vertex AI...")
    optimized = formatter.optimize(
        system_prompt=system_prompt_str,
        chat_prompt=chat_prompt_str,
    )

    # Generate output filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(SYSTEM_PROMPT_FILE).stem

    system_output_file = output_dir / f"{base_name}_system_optimized_{timestamp}.jinja2"
    chat_output_file = output_dir / f"{base_name}_chat_optimized_{timestamp}.jinja2"

    # Save optimized prompts
    system_output_file.write_text(optimized.system_prompt)
    print(f"Saved optimized system prompt to: {system_output_file}")

    chat_output_file.write_text(optimized.chat_prompt)
    print(f"Saved optimized chat prompt to: {chat_output_file}")

    print("Prompt optimization completed successfully!")


if __name__ == "__main__":
    main()
