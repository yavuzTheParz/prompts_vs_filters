"""Measure the model server before a campaign: what it is and how it scales.

Sends a harmless prompt through the same `/generate` client the experiments
use, at increasing concurrency, and reports throughput. If throughput rises
with concurrency, the server batches requests and `--concurrency` should be set
near the level where the gain flattens. If throughput stays flat, the server
handles one request at a time and extra concurrency only adds queueing.

    python3 -B scripts/benchmark_server.py --base-url http://HOST:8000
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_client import LocalLLMClient  # noqa: E402

PROMPT = "User:\nExplain in a few sentences why the sky appears blue during the day.\n\nAssistant:"


def probe(base_url: str, path: str, timeout: float = 5.0):
    try:
        with request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=timeout) as response:
            return response.status, response.read(4000).decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        return exc.code, ""
    except Exception:
        return None, ""


def identify(base_url: str) -> str:
    """Best-effort guess of the serving stack from its public endpoints."""
    status, body = probe(base_url, "/v1/models")
    openai_compatible = status == 200
    status_version, version_body = probe(base_url, "/version")
    status_docs, docs_body = probe(base_url, "/openapi.json")
    lines = []
    if openai_compatible:
        owner = ""
        try:
            owner = json.loads(body)["data"][0].get("owned_by", "")
        except Exception:
            pass
        lines.append(f"OpenAI-compatible API found (/v1/models, owned_by={owner or 'unknown'}).")
        if "vllm" in (owner + version_body).lower():
            lines.append("This looks like vLLM (continuous batching).")
    else:
        lines.append("No OpenAI-compatible API (/v1/models not found).")
    if status_docs == 200 and '"/generate"' in docs_body:
        lines.append("Custom /generate endpoint found (the interface this project uses).")
    if not openai_compatible and status_docs == 200:
        lines.append("Likely a custom FastAPI server; whether it batches is decided by the scaling test below.")
    return "\n".join("  " + line for line in lines)


def timed_call(client: LocalLLMClient, max_new_tokens: int) -> float:
    started = time.time()
    client.generate(prompt=PROMPT, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True)
    return time.time() - started


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--levels", default="1,2,4,8,16", help="Concurrency levels to test.")
    parser.add_argument("--requests-per-level", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Same generation length as the experiments.")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args(argv)

    print("Server identification:")
    print(identify(args.base_url))
    client = LocalLLMClient(base_url=args.base_url, api_key=args.api_key, timeout_sec=args.timeout)
    timed_call(client, args.max_new_tokens)  # warm-up, not measured

    print(f"\n{'concurrency':>11} {'mean latency':>13} {'calls/s':>8} {'speed-up':>9}")
    baseline = None
    best_level, best_rate = 1, 0.0
    for level in [int(item) for item in args.levels.split(",") if item.strip()]:
        total = max(level, args.requests_per_level)
        started = time.time()
        with ThreadPoolExecutor(max_workers=level) as pool:
            latencies = list(pool.map(lambda _: timed_call(client, args.max_new_tokens), range(total)))
        rate = total / (time.time() - started)
        baseline = baseline or rate
        print(f"{level:>11} {statistics.mean(latencies):>12.2f}s {rate:>8.2f} {rate / baseline:>8.1f}x")
        # Keep raising concurrency only while it buys at least 15% more throughput.
        if rate > best_rate * 1.15:
            best_level, best_rate = level, rate

    seconds_per_call = 1.0 / best_rate
    print(f"\nRecommended --concurrency {best_level}  (effective {seconds_per_call:.2f} s per call)")
    for budget, label in ((15017, "fixed-filter run, 15k budget"), (18150, "coevolution run, 15k budget")):
        print(f"  {label}: ~{budget * seconds_per_call / 3600:.1f} h")
    if best_level == 1:
        print("Throughput did not improve with concurrency: the server handles one request at a time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
