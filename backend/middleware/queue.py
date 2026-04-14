import asyncio
import time
import torch
from fastapi import HTTPException
from functools import wraps
from middleware.logger import get_logger

logger = get_logger(__name__)

                                                                                

                                                                          
                                                                              
                                                            
QUEUE_WAIT_TIMEOUT_S = 100

                                                                                

                                                                          
                                                                       
                                                                       
                                                   
 
                                                     
                                                                               
_gpu_lock = asyncio.Lock()


                                                                                

class GpuQueue:
    """
    Async context manager that wraps the global GPU lock.

    - Waits at most QUEUE_WAIT_TIMEOUT_S seconds to acquire the lock;
      raises HTTP 503 if the queue is too backed-up so the client gets a
      useful message instead of a raw connection-aborted timeout.
    - Flushes the CUDA memory cache on every release so that cached-but-freed
      GPU memory is returned promptly, keeping per-inference allocation time
      stable across many successive calls.

    Usage in a route:
        async with GpuQueue():
            instances = await run_vision_pipeline(...)
    """

    async def __aenter__(self):
        if _gpu_lock.locked():
            logger.info("[queue] GPU is busy — request queued, waiting...")

        t0 = time.monotonic()
        try:
            await asyncio.wait_for(
                _gpu_lock.acquire(),
                timeout=QUEUE_WAIT_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            waited = time.monotonic() - t0
            logger.warning(
                f"[queue] Request rejected after waiting {waited:.1f}s — "
                "GPU queue is overloaded."
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "The server is busy processing another request. "
                    "Please wait a moment and try again."
                ),
            )

        waited = time.monotonic() - t0
        if waited > 1.0:
            logger.info(f"[queue] GPU lock acquired after {waited:.1f}s wait")
        else:
            logger.debug("[queue] GPU lock acquired")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        _gpu_lock.release()
        logger.debug("[queue] GPU lock released")

                                                                             
                                                                             
                                                                               
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

                                                                 
        return False


                                                                               

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