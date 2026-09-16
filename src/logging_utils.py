from __future__ import annotations
import logging
import os
import sys
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def get_logger(name: str, log_dir: str = LOG_DIR) -> tuple[logging.Logger, str]:
    """Create (or fetch) a logger that writes to both console and a
    timestamped file under logs/. Returns (logger, log_file_path).
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers = []  # avoid duplicate handlers if called twice

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"=== {name} started | log file: {log_path} ===")
    return logger, log_path


def log_exception_and_exit(logger: logging.Logger, message: str, exc: Exception, code: int = 1):
    logger.error(f"{message}: {exc}")
    logger.exception("Full traceback:")
    sys.exit(code)


def log_exceptions(logger: logging.Logger):
    """Decorator: log (with full traceback) and re-raise any exception
    raised inside the wrapped function, so failures are always captured
    in the log file even if the caller's try/except only prints a short
    message.
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                logger.exception(f"Unhandled exception in {fn.__name__}")
                raise
        return wrapper
    return decorator


if __name__ == "__main__":
    logger, path = get_logger("logging_utils_selftest")
    logger.info("logging_utils import + self-test OK")
    print(f"Wrote test log to: {path}")
