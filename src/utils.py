from functools import wraps
import asyncio
import logging
import random

logger = logging.getLogger(__name__)

def is_hard_quota_error(e: Exception) -> bool:
    """Errors that retrying will never fix — stop immediately, don't waste attempts."""
    s = str(e).lower()
    return "insufficient_quota" in s or "monthly limit" in s or "quota exceeded" in s


def retry_with_backoff(max_attempts=3, base_delay=2, max_delay=60):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args,**kwargs):
            for attempts in range(max_attempts):
                try:
                    return await func(*args,**kwargs)
                except Exception as e:
                    if is_hard_quota_error(e):
                        logger.error(f"{func.__name__}: hard quota hit, not retrying: {e}")
                        raise
                    if attempts == max_attempts-1:
                        logger.error(f"{func.__name__}failed after {max_attempts} attempts: {e}")
                    wait = min(max_delay,base_delay*(2**attempts)+random.uniform(0,1))
                    logger.warning(f"{func.__name__} failed {e} , retry {attempts+1}/{max_attempts} in {wait:.1f}s")
                    await asyncio.sleep(wait)
        return wrapper
    return decorator
                    



    