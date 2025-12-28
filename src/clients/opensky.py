import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from src.utils.helpers import retry_request


@dataclass
class OpenSkyAuth:
    mode: str  # "oauth2", "basic", or "none"
    token: str = ""
    token_expires_at: float = 0.0
    username: str = ""
    password: str = ""


class OpenSkyClient:
    def __init__(
        self,
        api_base: str,
        *,
        client_id: str = "",
        client_secret: str = "",
        token_url: str = "",
        username: str = "",
        password: str = "",
        request_timeout: int = 30,
        max_retries: int = 5,
        retry_backoff_base: float = 1.5,
        access_token: str = "",
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base
        self.manual_token = access_token.strip()
        self.session = requests.Session()
        self.auth = self._init_auth(username=username, password=password)

    def _init_auth(self, *, username: str, password: str) -> OpenSkyAuth:
        if self.manual_token:
            return OpenSkyAuth(mode="token", token=self.manual_token, token_expires_at=float("inf"))
        if self.client_id and self.client_secret and self.token_url:
            return OpenSkyAuth(mode="oauth2")
        if username and password:
            return OpenSkyAuth(mode="basic", username=username, password=password)
        return OpenSkyAuth(mode="none")

    def _ensure_token(self) -> None:
        if self.auth.mode != "oauth2":
            return
        now = time.time()
        if self.auth.token and now < (self.auth.token_expires_at - 60):
            return

        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        resp = retry_request(
            self.session,
            "POST",
            self.token_url,
            params=None,
            headers=headers,
            auth=None,
            data=data,
            timeout=self.request_timeout,
            max_retries=self.max_retries,
            retry_backoff_base=self.retry_backoff_base,
        )
        try:
            token_json = resp.json()
        except Exception:
            token_resp = self.session.post(
                self.token_url,
                data=data,
                headers=headers,
                timeout=self.request_timeout,
            )
            token_resp.raise_for_status()
            token_json = token_resp.json()

        access_token = token_json.get("access_token", "")
        expires_in = float(token_json.get("expires_in", 3600))
        if not access_token:
            raise RuntimeError(f"OAuth2 token response missing access_token: {token_json}")

        self.auth.token = access_token
        self.auth.token_expires_at = time.time() + expires_in
        print("[INFO] OpenSky OAuth2 token obtained/refreshed.")

    def _headers(self) -> dict[str, str]:
        if self.auth.mode == "oauth2":
            self._ensure_token()
            return {"Authorization": f"Bearer {self.auth.token}"}
        if self.auth.mode == "token":
            return {"Authorization": f"Bearer {self.auth.token}"}
        return {}

    def _basic_auth(self) -> Optional[tuple[str, str]]:
        if self.auth.mode == "basic":
            return (self.auth.username, self.auth.password)
        return None

    def get_departures(self, airport_icao: str, begin_ts: int, end_ts: int) -> list[dict[str, Any]]:
        url = f"{self.api_base}/flights/departure"
        params = {"airport": airport_icao, "begin": begin_ts, "end": end_ts}
        resp = retry_request(
            self.session,
            "GET",
            url,
            params=params,
            headers=self._headers(),
            auth=self._basic_auth(),
            timeout=self.request_timeout,
            max_retries=self.max_retries,
            retry_backoff_base=self.retry_backoff_base,
        )
        return resp.json()

    def get_arrivals(self, airport_icao: str, begin_ts: int, end_ts: int) -> list[dict[str, Any]]:
        url = f"{self.api_base}/flights/arrival"
        params = {"airport": airport_icao, "begin": begin_ts, "end": end_ts}
        resp = retry_request(
            self.session,
            "GET",
            url,
            params=params,
            headers=self._headers(),
            auth=self._basic_auth(),
            timeout=self.request_timeout,
            max_retries=self.max_retries,
            retry_backoff_base=self.retry_backoff_base,
        )
        return resp.json()


__all__ = ["OpenSkyAuth", "OpenSkyClient"]
