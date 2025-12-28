import datetime as dt
import os
import time
from typing import Any, Optional

import requests


def utc_day_bounds(date_utc: dt.date) -> tuple[int, int]:
    """Return begin/end unix seconds for a UTC date [00:00, 23:59:59]."""
    begin = int(dt.datetime(date_utc.year, date_utc.month, date_utc.day, 0, 0, 0, tzinfo=dt.timezone.utc).timestamp())
    end = int(dt.datetime(date_utc.year, date_utc.month, date_utc.day, 23, 59, 59, tzinfo=dt.timezone.utc).timestamp())
    return begin, end


def ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def safe_sleep(seconds: float) -> None:
    time.sleep(max(0.0, seconds))


def retry_request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    auth: Optional[tuple[str, str]] = None,
    data: Optional[Any] = None,
    timeout: int = 30,
    max_retries: int = 5,
    retry_backoff_base: float = 1.5,
) -> requests.Response:
    """Perform an HTTP request with retries for transient errors."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.request(
                method,
                url,
                params=params,
                headers=headers,
                auth=auth,
                timeout=timeout,
                data=data,
            )
            if resp.status_code in (429, 500, 502, 503, 504):
                backoff = (retry_backoff_base ** attempt) + (0.1 * attempt)
                print(f"[WARN] {resp.status_code} from {url}. Retry {attempt}/{max_retries} in {backoff:.1f}s")
                safe_sleep(backoff)
                continue
            resp.raise_for_status()
            return resp
        except Exception as exc:  # broad by design to retry network hiccups
            last_exc = exc
            backoff = (retry_backoff_base ** attempt) + (0.1 * attempt)
            print(f"[WARN] Request error: {exc}. Retry {attempt}/{max_retries} in {backoff:.1f}s")
            safe_sleep(backoff)
    raise RuntimeError(f"Failed request after {max_retries} retries: {url}") from last_exc


__all__ = [
    "ensure_dir",
    "retry_request",
    "safe_sleep",
    "utc_day_bounds",
]
