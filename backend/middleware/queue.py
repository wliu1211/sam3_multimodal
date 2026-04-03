import asyncio
from functools import wraps
from middleware.logger import get_logger

logger = get_logger(__name__)

# ── Global GPU lock ───────────────────────────────────────────────────────────

# A single asyncio.Lock that every route acquires before touching the GPU.
# This serializes all vision pipeline calls so SAM 3 and CLIP are never
# called concurrently — concurrent GPU calls would either crash with an
# out-of-memory error or produce corrupted results.
#
# For a single-user demo this is the right primitive.
# For multi-user production, replace with a proper task queue (Celery + Redis).
_gpu_lock = asyncio.Lock()


# ── Context manager ───────────────────────────────────────────────────────────

class GpuQueue:
    """
    Async context manager that wraps the global GPU lock.
    Logs wait time so you can see in the console when a request
    had to queue behind another.

    Usage in a route:
        async with GpuQueue():
            instances = await run_vision_pipeline(...)
    """

    async def __aenter__(self):
        if _gpu_lock.locked():
            logger.info("[queue] GPU is busy — request queued, waiting...")

        await _gpu_lock.acquire()
        logger.debug("[queue] GPU lock acquired")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        _gpu_lock.release()
        logger.debug("[queue] GPU lock released")
        # Don't suppress exceptions — let them propagate normally
        return False


# ── Decorator (optional convenience) ─────────────────────────────────────────

def gpu_queued(func):
    """
    Decorator version of GpuQueue for async functions that should be
    serialized through the GPU lock.

    Usage:
        @gpu_queued
        async def run_vision_pipeline(...):
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        async with GpuQueue():
            return await func(*args, **kwargs)
    return wrapper