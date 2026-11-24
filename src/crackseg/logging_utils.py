import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Configures root logger once, at the program entry point (scripts, train.py).
    Modules only call logging.getLogger(__name__) and never configure anything themselves."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))   # utf-8: polskie znaki na Windowsie

    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=handlers, force=True)
    logging.captureWarnings(True)   # ostrzeżenia z modułu warnings (torch, albumentations) też trafiają do loga
