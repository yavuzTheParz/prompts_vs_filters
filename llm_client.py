from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import json
import time
from urllib import error, request

try:
    import requests
except Exception:
    requests = None


@dataclass
class LocalLLMClient:
    base_url: str
    api_key: Optional[str] = None
    timeout_sec: int = 300
    max_connect_retries: int = 3

    _session: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if requests is not None:
            self._session = requests.Session()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/generate"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        }

        if requests is not None:
            last_error = None

            for attempt in range(self.max_connect_retries):
                try:
                    response = self._session.post(
                        url,
                        json=payload,
                        headers=headers,
                        # 10 saniye bağlantı, timeout_sec üretim bekleme süresi
                        timeout=(10, self.timeout_sec),
                    )
                    response.raise_for_status()
                    return response.json()

                except (
                    requests.exceptions.ConnectTimeout,
                    requests.exceptions.ConnectionError,
                ) as exc:
                    last_error = exc

                    if attempt + 1 >= self.max_connect_retries:
                        raise

                    wait_seconds = 2 ** attempt
                    print(
                        f"[LLM CONNECTION RETRY] "
                        f"attempt={attempt + 1}/{self.max_connect_retries}, "
                        f"waiting={wait_seconds}s, error={exc}"
                    )
                    time.sleep(wait_seconds)

            raise RuntimeError(
                f"Local LLM connection failed: {last_error}"
            )

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(
                req,
                timeout=self.timeout_sec,
            ) as response:
                return json.loads(response.read().decode("utf-8"))

        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Local LLM request failed with HTTP {exc.code}: {body}"
            ) from exc