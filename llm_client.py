from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import requests

@dataclass
class LocalLLMClient:
    base_url: str                    # e.g. "http://100.91.151.105:8000"
    api_key: Optional[str] = None    # X-API-Key
    timeout_sec: int = 180

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

        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_sec)
        r.raise_for_status()
        return r.json()