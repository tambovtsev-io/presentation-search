import logging
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler

from src.config import Config


def setup_logging(
    logger: logging.Logger, log_dir: Path = Config().navigator.log
) -> None:
    """Setup logging to both file and console with rich formatting

    Args:
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter("%(message)s")

    # Setup file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Setup console handler with rich formatting
    console_handler = RichHandler(rich_tracebacks=True, markup=True, show_time=False)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger.info(f"Logging to file: {log_file}")
