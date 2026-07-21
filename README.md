[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
<h3 align="center">ASTER-Sakura: [A]utomated Natural Language to Te[s][t] Cas[e] Generato[r]</h3>
<p align="center">
<img src="./doc/aster.png" width="400" height="400">
</p>

## Overview

ASTER-Sakura is a research project investigating the automated generation of structurally complex Java test code from abstract natural language test descriptions. It consists of two complementary pipelines:

- **Test2NL** -- generates natural language descriptions from existing Java test methods at three abstraction levels (high, medium, low).
- **NL2Test** -- generates compilable Java test code from natural language descriptions using a trio of cooperating LangGraph ReAct agents (Supervisor, Localization, Composition).

The evaluation dataset is built from 27 Apache Commons projects. Each project is included as a git submodule under `resources/datasets/`.

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Java 11+ and Maven (for compiling and executing generated tests)
- An LLM API key for at least one supported provider (see [LLM Providers](#llm-providers))

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd sakura

# Or initialize submodules after cloning
git submodule update --init --recursive

# Install Python and synchronize the project environment
uv sync
```

Create a `.env` file in the project root with your API keys:

```
LLM_API_KEY=<your-llm-api-key>
EMB_API_KEY=<your-embedding-api-key>
```

`LLM_API_KEY` is used by the LLM client for both pipelines. `EMB_API_KEY` is used by the embedding client for FAISS vector search in the NL2Test pipeline.

## Project Structure

```
sakura/
|-- src/sakura/              # Core Python package
|   |-- cli.py               # Typer CLI (entry point for both pipelines)
|   |-- nl2test/             # NL-to-test generation pipeline
|   |   |-- pipeline.py      #   NL2Test pipeline entry point (used by CLI)
|   |   |-- core/            #   Base ReAct agent architecture
|   |   |-- generation/      #   Supervisor, Localization, Composition agents
|   |   |-- models/          #   Input/output data models
|   |   |-- preprocessing/   #   FAISS indexing, embeddings, vector stores
|   |   +-- prompts/         #   Jinja2 prompt templates
|   |-- test2nl/             # Test-to-NL description pipeline
|   |   |-- pipeline.py      #   Test2NL pipeline entry point (used by CLI)
|   |   |-- extractors/      #   Test method extraction
|   |   |-- generation/      #   Description generation
|   |   |-- model/           #   Data models
|   |   +-- prompts/         #   Prompt templates
|   |-- dataset_creation/    # Dataset bucketing and model definitions
|   |-- ray_utils/           # Ray actors for distributed processing
|   +-- utils/               # LLM clients, Maven compilation, git, file I/O
|
|-- scripts/
|   |-- runners/             # Wrapper scripts that invoke the CLI
|   |-- evaluation/          # Result aggregation, statistics, diagrams
|   |-- data/                # Dataset creation and subsetting utilities
|   |-- diagrams/            # Research question visualization generators
|   |-- prompts/             # Prompt generation helpers
|   +-- utilities/           # Maintenance helpers (compile, reset, etc.)
|
|-- resources/
|   |-- datasets/            # 27 Apache Commons projects (git submodules)
|   |-- analysis/            # CLDK analysis.json per project
|   |-- filtered_bucketed_tests/  # Tests bucketed by focal method count
|   +-- test2nl/
|       +-- filtered_dataset/     # Master test2nl.csv and spanning subsets
|
+-- outputs/
    |-- raw_outputs/         # Per-model NL2Test evaluation results
    |-- evaluation_stats/    # Aggregated statistics (JSON)
    +-- diagrams/            # Comparative charts
```

## Prompt Mapping

The prompt files are split by pipeline:

- `src/sakura/test2nl/prompts/templates/system/` and `src/sakura/test2nl/prompts/templates/chat/` drive Test2NL description generation.
- `src/sakura/nl2test/prompts/templates/system/` and `src/sakura/nl2test/prompts/templates/chat/` drive NL2Test decomposition and agent orchestration.

### Test2NL prompt mapping

The `generate-descriptions` CLI path goes through `Pipeline.run_descriptions_*()`, `DescriptionGenerator.generate_for_method()`, and then `Test2NLPrompt.generate()`. For each abstraction level, the system prompt is loaded directly and the chat prompt is rendered after the method, setup/teardown, helper, and application context have been assembled.

| Abstraction level | System prompt | Chat prompt | Loaded / rendered in | Used from |
|--------|-------------|-------------|-------------|-------------|
| High | `system/high_abs.jinja2` | `chat/high_abs.jinja2` | `src/sakura/test2nl/prompts/test2nl_prompt.py` (`generate()` and `format()`) | `src/sakura/test2nl/generation/description_generator.py` |
| Medium | `system/medium_abs.jinja2` | `chat/medium_abs.jinja2` | `src/sakura/test2nl/prompts/test2nl_prompt.py` (`generate()` and `format()`) | `src/sakura/test2nl/generation/description_generator.py` |
| Low | `system/low_abs.jinja2` | `chat/low_abs.jinja2` | `src/sakura/test2nl/prompts/test2nl_prompt.py` (`generate()` and `format()`) | `src/sakura/test2nl/generation/description_generator.py` |

### NL2Test prompt mapping

The `run-nl2test` CLI path enters `Pipeline.run_nl2test()`, which first decomposes the input description (Given-When-Then) and then constructs a Supervisor orchestrator. The Supervisor coordinates Localization and Composition sub-agents with their own prompt pairs.

| Stage | System prompt | Chat prompt | Loaded / rendered in | Used from |
|--------|-------------|-------------|-------------|-------------|
| NL decomposition | `system/gherkin_decomposition.jinja2` | `chat/gherkin_decomposition.jinja2` | `src/sakura/nl2test/preprocessing/decomposers/gherkin.py` | `src/sakura/nl2test/preprocessing/nl_decomposer.py` -> `src/sakura/nl2test/pipeline.py` |
| Localization agent | `system/localization_agent_gherkin.jinja2` | `chat/localization_agent_gherkin.jinja2` | `src/sakura/nl2test/generation/localization/orchestrators/base.py` and `src/sakura/nl2test/generation/localization/orchestrators/gherkin.py` | `src/sakura/nl2test/generation/supervisor/agent.py` and `src/sakura/nl2test/pipeline.py` |
| Composition agent | `system/composition_agent_gherkin.jinja2` | `chat/composition_agent_gherkin.jinja2` | `src/sakura/nl2test/generation/composition/orchestrators/base.py` and `src/sakura/nl2test/generation/composition/orchestrators/gherkin.py` | `src/sakura/nl2test/generation/supervisor/agent.py` |
| Supervisor agent | `system/supervisor_agent_gherkin.jinja2` | `chat/supervisor_agent_gherkin.jinja2` | `src/sakura/nl2test/generation/supervisor/orchestrators/base.py` and `src/sakura/nl2test/generation/supervisor/orchestrators/gherkin.py` | `src/sakura/nl2test/pipeline.py` |

| NL2Test finalize / fallback path | Prompt file(s) | Where it is used | Notes |
|--------|-------------|-------------|-------------|
| Localization force-finalize | `system/localization_agent_gherkin_finalize.jinja2` and `chat/localization_agent_gherkin_finalize.jinja2` | `src/sakura/nl2test/generation/localization/agent.py` | Used only when the localization ReAct loop must force a final structured output |
| Composition force-finalize | `system/composition_agent_gherkin_finalize.jinja2` and `chat/composition_agent_gherkin_finalize.jinja2` | `src/sakura/nl2test/generation/composition/agent.py` | Used only when the composition ReAct loop must force a final response |
| Supervisor force-finalize | No template file; `_execute_force_end()` returns a fixed comment | `src/sakura/nl2test/generation/supervisor/agent.py` | Supervisor does not use an LLM finalize prompt |

## CLI Reference

The CLI is exposed as the `sakura` command (or `python -m sakura.cli`). It has two subcommands.

### `generate-descriptions` (Test2NL)

Generates natural language descriptions for test methods. Reads bucketed test datasets, sends each method to an LLM, and writes `test2nl.csv` and `descriptions.json`.

```bash
uv run sakura generate-descriptions \
    --analysis-dir resources/analysis \
    --output-dir <output-path> \
    --organized-methods-dir resources/filtered_bucketed_tests \
    --organized-methods-file-name nl2test.json \
    --llm-model "x-ai/grok-4-fast" \
    --llm-provider openrouter \
    --num-proj-parallel 2 \
    --per-proj-concurrency 12
```

Key options:

| Option | Description |
|--------|-------------|
| `--analysis-dir` | Directory containing per-project `analysis.json` files |
| `--output-dir` | Where to write `test2nl.csv` and `descriptions.json` |
| `--organized-methods-dir` | Root of per-project directories with `nl2test.json` |
| `--llm-model` | Model identifier (e.g. `x-ai/grok-4-fast`) |
| `--llm-provider` | Provider name (see [LLM Providers](#llm-providers)) |
| `--clear-dataset` | Clear existing output before writing (default: true) |
| `--max-methods` | Cap on total methods across all projects (0 = unlimited) |
| `--num-proj-parallel` | Concurrent project count |
| `--per-proj-concurrency` | Concurrent LLM calls per project |
| `--exclude-groups` | Skip specific focal-method buckets (repeatable) |

### `run-nl2test` (NL2Test)

Runs the multi-agent test generation pipeline. Reads a Test2NL CSV, processes each entry through the Supervisor/Localization/Composition agents, compiles the generated test, executes it, and records evaluation metrics.

```bash
uv run sakura run-nl2test \
    --base-project-dir resources/datasets \
    --base-analysis-dir resources/analysis \
    --output-dir <output-path> \
    --test2nl-file resources/test2nl/filtered_dataset/test2nl.csv \
    --llm-model "google/gemini-2.5-flash" \
    --llm-provider openrouter \
    --emb-model "qwen/qwen3-embedding-8b" \
    --emb-provider openrouter \
    --num-proj-parallel 10 \
    --debug
```

Key options:

| Option | Description |
|--------|-------------|
| `--base-project-dir` | Root directory containing all project source trees |
| `--base-analysis-dir` | Root directory containing per-project `analysis.json` |
| `--test2nl-file` | Path to the Test2NL CSV with NL descriptions |
| `--llm-model` / `--llm-provider` | LLM for agent reasoning |
| `--emb-model` / `--emb-provider` | Embedding model for FAISS vector search |
| `--decomposition-mode` | `gherkin` (default, Given-When-Then) |
| `--supervisor-max-iters` | Max agent turns for Supervisor (default: 10) |
| `--localization-max-iters` | Max agent turns for Localization (default: 40) |
| `--composition-max-iters` | Max agent turns for Composition (default: 30) |
| `--num-proj-parallel` | Concurrent projects via Ray |
| `--max-entries` | Limit number of CSV entries to process |
| `--target-ids` | Process only specific entry IDs (repeatable) |
| `--exclude-test-dirs` | Skip test directories when building indexes |
| `--use-stored-index` | Reuse cached FAISS indexes if available |
| `--debug` | Verbose logging |
| `--log-file` | Log file name (saved under `--output-dir`) |
| `--save-localized-scenarios` | Persist localized scenarios to JSON |
| `--store-code-iteration` | Save each code generation iteration |

## Running the Pipelines via Scripts

The `scripts/runners/` directory contains pre-configured wrapper scripts. Edit the configuration constants at the top of each file to match your environment, then run them.

### Test2NL (description generation)

```bash
# Edit scripts/runners/generate_test2nl.py to set:
#   LLM_MODEL, LLM_PROVIDER, OUTPUT_DIR, etc.
uv run python scripts/runners/generate_test2nl.py
```

This script invokes `sakura generate-descriptions` with paths resolved relative to the project root. It reads bucketed test methods from `resources/filtered_bucketed_tests/` and writes the output CSV and JSON to the configured `OUTPUT_DIR`.

### NL2Test (test code generation)

```bash
# Edit scripts/runners/run_nl2test_on_test2nl.py to set:
#   LLM_MODEL, LLM_PROVIDER, EMB_MODEL, EMB_PROVIDER, OUTPUT_DIR, etc.
uv run python scripts/runners/run_nl2test_on_test2nl.py
```

This script invokes `sakura run-nl2test`. It reads the Test2NL CSV from `resources/test2nl/filtered_dataset/test2nl.csv`, runs the multi-agent pipeline against the dataset projects in `resources/datasets/`, and writes per-project evaluation results (`nl2test_evaluation_results.json`) under the configured `OUTPUT_DIR`.

## Evaluation and Replication

### Step 1: Verify Project Compilation

Confirm that all dataset projects compile before running any pipelines:

```bash
uv run python scripts/utilities/compile_dataset_projects.py
```

### Step 2: Generate CLDK Analysis (if not present)

The analysis files under `resources/analysis/` are required by both pipelines. To regenerate them:

```bash
uv run python scripts/data/generate_analysis.py
```

This uses Ray to run CLDK symbol-table analysis on all 27 projects in parallel.

### Step 3: Run the Pipelines

Run Test2NL and/or NL2Test as described above. Results land in `outputs/raw_outputs/<your-output-dir>/` with per-project subdirectories containing:

- `nl2test_evaluation_results.json` -- per-entry evaluation (compilation, coverage, structural metrics)
- `nl2test_failures.json` -- entries that failed during generation
- `nl2test_localized_scenarios.json` -- localized scenarios (if `--save-localized-scenarios`)

### Step 4: Aggregate Statistics

After running NL2Test, aggregate the per-project results into summary statistics:

```bash
uv run python scripts/evaluation/generate_statistics.py
```

This reads from directories listed in its `EVAL_DIRS` constant (edit the script to point at your output directories). It writes JSON evaluation summaries to `outputs/evaluation_stats/`.

Metrics computed include:
- **Compilation rate** -- percentage of generated tests that compile
- **Structural fidelity** -- focal precision/recall, assertion precision/recall, callable precision/recall, object creation precision/recall (with F1)
- **Code coverage** -- class, method, line, and branch coverage
- **Localization recall** -- accuracy of code context retrieval
- **Cost** -- token usage and estimated API cost per entry

All metrics are broken down by abstraction level (high/medium/low) and focal method bucket.

### Step 5: Generate Comparison Diagrams

To produce comparative bar charts across multiple evaluation runs:

```bash
uv run python scripts/evaluation/perform_comparison.py
```

Edit `INPUT_FILES` in the script to list the evaluation JSON files to compare. Diagrams are saved to `outputs/diagrams/`.

### Additional Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `scripts/evaluation/generate_tool_analysis.py` | Analyze tool usage patterns (retrieval, generation, validation) |
| `scripts/evaluation/generate_tool_statistics.py` | Compute tool call statistics |
| `scripts/evaluation/generate_localization_stats.py` | Localization-specific metrics |
| `scripts/evaluation/identify_motivation_examples.py` | Find compile-gap and quality-gap examples |
| `scripts/evaluation/make_other_agent_eval.py` | Evaluate baseline agent outputs |
| `scripts/evaluation/calculate_worst_case_cost.py` | Token usage and API cost estimates |
| `scripts/diagrams/rq1_compilation.py` | RQ1: Compilation success charts |
| `scripts/diagrams/rq2_coverage.py` | RQ2: Code coverage charts |
| `scripts/diagrams/rq3_structural_fidelity.py` | RQ3: Structural fidelity charts |
| `scripts/diagrams/rq4_abstraction_sensitivity.py` | RQ4: Abstraction level sensitivity |
| `scripts/diagrams/rq5_complexity_sensitivity.py` | RQ5: Complexity sensitivity |
| `scripts/diagrams/rq6_tool_calls.py` | RQ6: Tool usage analysis |

## Dataset Files

### Test2NL CSV (`resources/test2nl/filtered_dataset/`)

| File | Description |
|------|-------------|
| `test2nl.csv` | Full dataset -- all projects, all abstraction levels |
| `spanning_subset_20.csv` | 20-entry spanning subset (covers all bucket/level combinations) |
| `spanning_subset_30.csv` | 30-entry spanning subset |
| `spanning_subset_40.csv` | 40-entry spanning subset |

Each row contains: `id`, `project_name`, `qualified_class_name`, `method_signature`, `description`, `abstraction_level`, `is_bdd`.

### Bucketed Tests (`resources/filtered_bucketed_tests/`)

Per-project `nl2test.json` files that organize test methods into five buckets by focal method count:

- `tests_with_one_focal_methods`
- `tests_with_two_focal_methods`
- `tests_with_more_than_two_to_five_focal_methods`
- `tests_with_more_than_five_to_ten_focal_methods`
- `tests_with_more_than_ten_focal_methods`

### Analysis (`resources/analysis/`)

Per-project directories each containing an `analysis.json` produced by CLDK. These cache the symbol-table analysis used by the Localization agent for code retrieval.

### Evaluation Outputs (`outputs/raw_outputs/`)

Pre-computed results from different model configurations:

| Directory | Model |
|-----------|-------|
| `gemini_cli_flash_output/` | Gemini CLI baseline (Flash 2.5) |
| `gemini_cli_pro_output/` | Gemini CLI baseline (Pro 2.5) |
| `nl2test_gemini_flash_output/` | Sakura with Gemini Flash 2.5 |
| `nl2test_gemini_pro_output/` | Sakura with Gemini Pro 2.5 |
| `nl2test_devstral_output/` | Sakura with Devstral Small |
| `nl2test_qwen3_output/` | Sakura with Qwen3 Coder |

### Dataset Utility Scripts

| Script | Purpose |
|--------|---------|
| `scripts/data/create_spanning_subset.py` | Create minimal spanning subsets from the full CSV |
| `scripts/data/create_subset_dataset.py` | Trim bucketed JSON files to N entries per group |
| `scripts/data/subset_test2nl_csv.py` | Filter CSV entries to match a subset JSON |
| `scripts/data/get_dataset_statistics.py` | Compute per-project NCLOC and focal method stats |

## Apache Commons Dataset

27 projects included as git submodules under `resources/datasets/`:

| Project | Multi-module |
|---------|:------------:|
| commons-bcel | - |
| commons-beanutils | - |
| commons-bsf | - |
| commons-cli | - |
| commons-codec | - |
| commons-collections | - |
| commons-configuration | - |
| commons-crypto | - |
| commons-csv | - |
| commons-dbcp | - |
| commons-dbutils | - |
| commons-email | :white_check_mark: |
| commons-exec | - |
| commons-fileupload | :white_check_mark: |
| commons-imaging | - |
| commons-io | - |
| commons-jcs | :white_check_mark: |
| commons-jexl | - |
| commons-lang | - |
| commons-logging | :white_check_mark: |
| commons-math | :white_check_mark: |
| commons-net | - |
| commons-numbers | :white_check_mark: |
| commons-pool | - |
| commons-text | - |
| commons-validator | - |
| commons-vfs | :white_check_mark: |

## LLM Providers

Supported values for `--llm-provider`:

| Provider | Notes |
|----------|-------|
| `openrouter` | Proxy supporting many models (Gemini, Grok, Qwen, Devstral, etc.) |
| `openai` | OpenAI API directly |
| `gcp` | Google Cloud AI Platform (Gemini models) |
| `ollama` | Local inference via Ollama |
| `vllm` | Local inference via vLLM server |
| `mistral` | Mistral API |

Supported values for `--emb-provider`:

| Provider | Notes |
|----------|-------|
| `openrouter` | Proxy supporting many embedding models |
| `openai` | OpenAI API directly |
| `gcp` | Google Cloud AI Platform |
| `ollama` | Local inference via Ollama (uses dedicated embedder) |
| `vllm` | Local inference via vLLM server |

Either `--llm-provider` or `--llm-api-url` must be provided. The same applies to `--emb-provider` / `--emb-api-url` for embeddings.

## Architecture

The NL2Test pipeline coordinates three LangGraph ReAct agents:

1. **Supervisor** (`src/sakura/nl2test/generation/supervisor/`) -- orchestrates the end-to-end flow, delegates to sub-agents, and validates compilation/execution.
2. **Localization** (`src/sakura/nl2test/generation/localization/`) -- maps natural language descriptions to relevant Java source code using FAISS vector search with embeddings.
3. **Composition** (`src/sakura/nl2test/generation/composition/`) -- generates compilable Java test code from the localized scenario context.

Ray is used for between-project parallelism. Each project runs as an independent Ray actor.

## Development

```bash
# Type checking
uv run pyright src/sakura/path/to/file.py

# Formatting
uv run ruff format src/sakura/path/to/file.py
uv run ruff check --fix src/sakura/path/to/file.py
```

Note: Do not run `pytest` or the CLI pipelines casually -- they make LLM API calls that incur costs.
