from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Linux Libertine"

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
STATS_DIR = ROOT_DIR / "outputs" / "evaluation_stats"
OUTPUT_DIR = ROOT_DIR / "outputs" / "diagrams"
DIAGRAM_STATS_DIR = ROOT_DIR / "outputs" / "diagram_stats"

INPUT_FILES: dict[str, str] = {
    "gemini_cli_flash_eval.json": "Gemini CLI (Gemini 2.5 Flash)",
    "gemini_cli_pro_eval.json": "Gemini CLI (Gemini 2.5 Pro)",
    "nl2test_gemini_flash_eval.json": "NL2Test (Gemini 2.5 Flash)",
    "nl2test_gemini_pro_eval.json": "NL2Test (Gemini 2.5 Pro)",
    "nl2test_qwen3_eval.json": "NL2Test (Qwen3-Coder)",
    "nl2test_devstral_eval.json": "NL2Test (Devstral Small 2)",
}

# IBM Design Language color palette (levels 20-60)
IBM_COLORS: dict[str, dict[str, str]] = {
    "red": {
        "20": "#ffd7d9",
        "30": "#ffb3b8",
        "40": "#ff8389",
        "50": "#fa4d56",
        "60": "#da1e28",
    },
    "magenta": {
        "20": "#ffd6e8",
        "30": "#ffafd2",
        "40": "#ff7eb6",
        "50": "#ee5396",
        "60": "#d02670",
    },
    "purple": {
        "20": "#e8daff",
        "30": "#d4bbff",
        "40": "#be95ff",
        "50": "#a56eff",
        "60": "#8a3ffc",
    },
    "blue": {
        "20": "#d0e2ff",
        "30": "#a6c8ff",
        "40": "#78a9ff",
        "50": "#4589ff",
        "60": "#0f62fe",
    },
    "cyan": {
        "20": "#bae6ff",
        "30": "#82cfff",
        "40": "#33b1ff",
        "50": "#1192e8",
        "60": "#0072c3",
    },
    "teal": {
        "20": "#9ef0f0",
        "30": "#3ddbd9",
        "40": "#08bdba",
        "50": "#009d9a",
        "60": "#007d79",
    },
    "cool_gray": {
        "20": "#dde1e6",
        "30": "#c1c7cd",
        "40": "#a2a9b0",
        "50": "#878d96",
        "60": "#697077",
    },
    "gray": {
        "20": "#e0e0e0",
        "30": "#c6c6c6",
        "40": "#a8a8a8",
        "50": "#8d8d8d",
        "60": "#6f6f6f",
    },
    "warm_gray": {
        "20": "#e5e0df",
        "30": "#cac5c4",
        "40": "#ada8a8",
        "50": "#8f8b8b",
        "60": "#726e6e",
    },
    "green": {
        "20": "#a7f0ba",
        "30": "#6fdc8c",
        "40": "#42be65",
        "50": "#24a148",
        "60": "#198038",
    },
}

CUSTOM_COLORS: dict[str, str] = {
    "dark_gray": "#4D5358",
    "light_gray": "#C1C7CD",
    "dark_blue": "#002D9C",
    "light_blue": "#78A9FF",
}

MODEL_COLORS: dict[str, str] = {
    "gemini_cli_flash_eval.json": CUSTOM_COLORS["light_gray"],
    "gemini_cli_pro_eval.json": CUSTOM_COLORS["dark_gray"],
    "nl2test_gemini_flash_eval.json": CUSTOM_COLORS["light_blue"],
    "nl2test_gemini_pro_eval.json": CUSTOM_COLORS["dark_blue"],
    "nl2test_qwen3_eval.json": IBM_COLORS["purple"]["30"],
    "nl2test_devstral_eval.json": IBM_COLORS["magenta"]["30"],
}


def get_color(filename: str) -> str:
    """Get hex color for a model based on filename."""
    return MODEL_COLORS[filename]
