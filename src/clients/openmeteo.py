import json
import os
from typing import Any

import requests

from src.utils.helpers import ensure_dir, retry_request
from src.utils.weather import weather_cache_key


class OpenMeteoClient:
    def __init__(
        self,
        base_url: str,
        *,
        hourly_vars: str,
        cache_dir: str,
        request_timeout: int = 30,
        max_retries: int = 5,
        retry_backoff_base: float = 1.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.hourly_vars = hourly_vars
        self.cache_dir = cache_dir
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.session = requests.Session()

    def fetch_hourly(self, lat: float, lon: float, start_date: str, end_date: str) -> dict[str, Any]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": self.hourly_vars,
            "timeformat": "unixtime",
            "timezone": "UTC",
        }
        resp = retry_request(
            self.session,
            "GET",
            self.base_url,
            params=params,
            timeout=self.request_timeout,
            max_retries=self.max_retries,
            retry_backoff_base=self.retry_backoff_base,
        )
        return resp.json()

    def ensure_weather_for_day(self, icao: str, lat: float, lon: float, date_iso: str) -> dict[str, Any]:
        weather_dir = os.path.join(self.cache_dir, "weather")
        ensure_dir(weather_dir)
        path = weather_cache_key(self.cache_dir, icao, date_iso)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)

        payload = self.fetch_hourly(lat, lon, date_iso, date_iso)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return payload


__all__ = ["OpenMeteoClient"]
