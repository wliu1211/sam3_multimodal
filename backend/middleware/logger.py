import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR     = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE     = LOGS_DIR / "app.log"

# ── Log format ────────────────────────────────────────────────────────────────

LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── Setup ─────────────────────────────────────────────────────────────────────

_configured = False

def _configure_root_logger() -> None:
    """
    Configures the root logger once on first import.
    Sets up two handlers:
      - StreamHandler → stdout (visible in the terminal while running)
      - FileHandler   → logs/app.log (persistent audit trail)

    Uses a guard flag so calling get_logger() multiple times
    doesn't add duplicate handlers to the root logger.
    """
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Terminal output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # File output — append mode so logs persist across restarts
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    # Quieten noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    _configured = True
    root.info(f"[logger] Logging to {LOG_FILE}")


# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger for the given module.
    Configures the root logger on first call.

    Usage in any backend file:
        from middleware.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    _configure_root_logger()
    return logging.getLogger(name)


# ── FastAPI request logging middleware ────────────────────────────────────────

async def log_requests(request, call_next):
    """
    Starlette middleware function that logs every incoming HTTP request
    and its response status + duration.

    Registered in main.py with:
        from middleware.logger import log_requests
        app.middleware("http")(log_requests)
    """
    _configure_root_logger()
    logger = logging.getLogger("http")

    start  = datetime.now(timezone.utc)
    method = request.method
    path   = request.url.path

    response = await call_next(request)

    duration_ms = int(
        (datetime.now(timezone.utc) - start).total_seconds() * 1000
    )
    logger.info(
        f"{method} {path} → {response.status_code} ({duration_ms}ms)"
    )

    return response