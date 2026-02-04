from pathlib import Path

from sakura.utils.pretty.color_logger import RichLog
from sakura.utils.vcs.git_utils import GitUtilities

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SUBMODULES_DIR = ROOT_DIR / "resources" / "datasets"


def main() -> None:
    """I am too lazy to run the command..."""
    RichLog.info(f"Resetting submodules under {SUBMODULES_DIR}")
    GitUtilities.reset_submodules_in_dir(SUBMODULES_DIR)


if __name__ == "__main__":
    main()
