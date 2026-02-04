"""
RichLog is a thin wrapper around the Python logging module that
adds convenient helpers and a Rich console handler by default.

Notes
- By default, logs go to the console only (via Rich).
- Use add_file_handler() to also write logs to a file.
"""

import logging
from rich.logging import RichHandler
from pathlib import Path

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(show_path=False)]
)

# Disable HTTP request logging from underlying libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)


class RichLog:
    """Common Logger that uses Rich"""

    log = logging.getLogger("rich")

    @staticmethod
    def info(msg: str):
        """Log level INFO"""
        RichLog.log.info(msg, extra={"markup": True})

    @staticmethod
    def warn(msg: str):
        """Log level WARNING"""
        RichLog.log.warning(msg, extra={"markup": True})

    @staticmethod
    def debug(msg: str):
        """Log level DEBUG"""
        RichLog.log.debug(msg, extra={"markup": True})

    @staticmethod
    def error(msg: str):
        """Log level ERROR"""
        RichLog.log.error(msg, extra={"markup": True})

    @staticmethod
    def activate_debug():
        """Sets the logging level to DEBUG"""
        RichLog.log.setLevel(logging.DEBUG)

    @staticmethod
    def set_level(level: int | str) -> None:
        """Set log level on both this logger and the root logger.

        This helps when third-party libraries emit useful debug logs.
        """
        logging.getLogger().setLevel(level)
        RichLog.log.setLevel(level)

    @staticmethod
    def add_file_handler(file_path: str, *, overwrite: bool = False, level: int | str | None = None) -> None:
        """Add a file handler so logs are also written to disk.

        Parameters
        - file_path: destination log file path (created if missing).
        - overwrite: when True, truncates the file; otherwise appends.
        - level: optional log level for this handler (defaults to logger level).
        """
        # Ensure parent directory exists
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Do not fail logging setup if directory creation fails
            pass

        mode = "w" if overwrite else "a"
        handler = logging.FileHandler(file_path, mode=mode, encoding="utf-8")
        # Use a simple timestamped format for files; avoid Rich formatting
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        if level is not None:
            handler.setLevel(level)
        RichLog.log.addHandler(handler)
