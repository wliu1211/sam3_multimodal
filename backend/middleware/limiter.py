import time
from collections import defaultdict
from fastapi import HTTPException, Request
from middleware.logger import get_logger

logger = get_logger(__name__)

                                                                                

                                                                     
MAX_REQUESTS    = 10
WINDOW_SECONDS  = 60                           

                                                                          
                                                                              
MIN_INTERVAL_SECONDS = 3.0

                                                                                

                                                                        
                                                                      
_request_log: dict[str, list[float]] = defaultdict(list)

                                                          
_last_request: dict[str, float] = {}


                                                                                

async def rate_limit(request: Request) -> None:
    """
    FastAPI dependency that enforces two rate-limiting rules:

    1. Minimum interval — rejects requests that arrive too quickly
       after the previous one from the same IP. Prevents double-submits
       and accidental hammering.

    2. Rolling window limit — rejects requests if the client has already
       sent MAX_REQUESTS within the last WINDOW_SECONDS.

    Usage in a route:
        @router.post("/query", dependencies=[Depends(rate_limit)])
        async def query(...):
            ...
    """
    client_ip = _get_client_ip(request)
    now       = time.time()

                                                
    last = _last_request.get(client_ip)
    if last is not None:
        elapsed = now - last
        if elapsed < MIN_INTERVAL_SECONDS:
            wait = round(MIN_INTERVAL_SECONDS - elapsed, 1)
            logger.warning(f"[limiter] {client_ip} — too fast, wait {wait}s")
            raise HTTPException(
                status_code=429,
                detail=f"Please wait {wait}s before submitting another query.",
                headers={"Retry-After": str(wait)},
            )

                             
    window_start = now - WINDOW_SECONDS
    timestamps   = _request_log[client_ip]

                                        
    _request_log[client_ip] = [t for t in timestamps if t > window_start]

    if len(_request_log[client_ip]) >= MAX_REQUESTS:
        oldest    = min(_request_log[client_ip])
        retry_in  = round(WINDOW_SECONDS - (now - oldest), 1)
        logger.warning(
            f"[limiter] {client_ip} — rate limit hit "
            f"({MAX_REQUESTS} requests in {WINDOW_SECONDS}s)"
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached ({MAX_REQUESTS} queries per minute). "
                   f"Try again in {retry_in}s.",
            headers={"Retry-After": str(retry_in)},
        )

                         
    _request_log[client_ip].append(now)
    _last_request[client_ip] = now
    logger.debug(
        f"[limiter] {client_ip} — "
        f"{len(_request_log[client_ip])}/{MAX_REQUESTS} requests this window"
    )


                                                                                

def _get_client_ip(request: Request) -> str:
    """
    Returns the client IP address.
    Checks X-Forwarded-For first in case the app is behind a proxy or ngrok.
    Falls back to the direct connection IP.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"