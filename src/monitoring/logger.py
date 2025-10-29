import json
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from log_deduplicator import should_log

_LOGGER_NAME = "hunter"
_LOG_DIR = "logs"
_LOG_FILE = os.path.join(_LOG_DIR, "hunter.log")

def _ensure_log_dir():
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
    except Exception:
        pass

def _get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    _ensure_log_dir()
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(_LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    # Keep formatter simple; we pre-serialize JSON in log_event
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

    # Also log to stdout minimally (optional)
    try:
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream)
    except Exception:
        pass

    return logger

def set_log_level(level_name: str):
    level = getattr(logging, (level_name or "INFO").upper(), logging.INFO)
    _get_logger().setLevel(level)

def log_event(event: str, level: str = "INFO", log_type: str = "general", **context):
    try:
        # Create log message for deduplication check
        log_message = f"{event}: {json.dumps(context) if context else ''}"
        
        # Check if we should log this message
        if not should_log(log_message, level, log_type):
            return  # Skip duplicate or rate-limited log
        
        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": level.upper(),
            "event": event,
            **({"context": context} if context else {}),
        }
        msg = json.dumps(record, ensure_ascii=False)
        logger = _get_logger()
        if record["level"] == "ERROR":
            logger.error(msg)
        elif record["level"] == "WARNING":
            logger.warning(msg)
        else:
            logger.info(msg)
    except Exception:
        # Never fail business logic due to logging
        pass


