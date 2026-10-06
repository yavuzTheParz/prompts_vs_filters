"""Local Dolphin LLM server with dynamic request batching.

Drop-in replacement for llm_server.py: same `/generate` request and response,
same model loading (CUDA 4-bit, FP16, or CPU fallback), same raw-prompt
handling. The difference is throughput. llm_server.py holds a lock and
generates one request at a time; this server collects the requests that are
waiting, groups the ones with identical generation settings, and generates
each group in a single batched `model.generate` call.

Run:
    uvicorn llm_server_batched:app --host 0.0.0.0 --port 8000

Tuning (environment variables):
    MAX_BATCH_SIZE    largest batch per generate call (default 16)
    BATCH_WAIT_MS     how long to wait for more requests before starting (default 25)
    MAX_BATCH_TOKENS  cap on batch_size * (longest prompt + max_new_tokens),
                      which bounds GPU memory for the attention cache (default 32000)

Notes on equivalence with the unbatched server:
- Sampled requests (do_sample=true) draw from the same distribution; the
  random stream differs, as it already did between any two runs.
- Greedy requests can differ in rare near-ties because padded batches change
  floating-point rounding.
"""

from __future__ import annotations

import asyncio
import gc
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_NAME = os.getenv("MODEL_NAME", "dphn/Dolphin3.0-Llama3.1-8B")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "4096"))
MAX_NEW_TOKENS_LIMIT = int(os.getenv("MAX_NEW_TOKENS_LIMIT", "2048"))
USE_4BIT = os.getenv("USE_4BIT", "1").strip().lower() not in {"0", "false", "no"}

MAX_BATCH_SIZE = max(1, int(os.getenv("MAX_BATCH_SIZE", "16")))
BATCH_WAIT_MS = max(0.0, float(os.getenv("BATCH_WAIT_MS", "25")))
MAX_BATCH_TOKENS = max(1, int(os.getenv("MAX_BATCH_TOKENS", "32000")))
# Requests taken off the queue per scheduling round; they are then split into
# batches of at most MAX_BATCH_SIZE.
MAX_ROUND_REQUESTS = 4 * MAX_BATCH_SIZE

# Global model state. A single model instance is shared by all requests.
tokenizer: Optional[Any] = None
model: Optional[Any] = None
model_device: torch.device = torch.device("cpu")
load_mode = "not-loaded"

# Created inside the server's event loop (see lifespan).
request_queue: Optional[asyncio.Queue] = None
stats = {"batches": 0, "requests": 0, "generated_tokens": 0, "busy_seconds": 0.0}


# -----------------------------------------------------------------------------
# Request models
# -----------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    do_sample: bool = False
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=5.0)


@dataclass
class PendingRequest:
    request: GenerateRequest
    future: "asyncio.Future[Dict[str, Any]]"
    prompt_tokens: int


def generation_key(request: GenerateRequest) -> Tuple[Any, ...]:
    """Requests can share a batch only if every generation setting matches."""
    if request.do_sample:
        return (
            request.max_new_tokens, True, round(request.temperature, 6),
            round(request.top_p, 6), round(request.repetition_penalty, 6),
        )
    return (request.max_new_tokens, False, round(request.repetition_penalty, 6))


# -----------------------------------------------------------------------------
# Model loading and cleanup
# -----------------------------------------------------------------------------

def _select_compute_dtype() -> torch.dtype:
    """Select a stable compute dtype for the active CUDA device."""
    if not torch.cuda.is_available():
        return torch.float32

    # RTX 40-series supports BF16, but FP16 is broadly compatible with
    # transformers/bitsandbytes versions on Windows.
    return torch.float16


def _load_model() -> None:
    global tokenizer, model, model_device, load_mode

    cuda_available = torch.cuda.is_available()
    compute_dtype = _select_compute_dtype()

    print(f"Loading model: {MODEL_NAME}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models continue from the last token, so batches are padded
    # on the left to keep every prompt's final token adjacent to its output.
    tokenizer.padding_side = "left"

    common_kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
    }

    if cuda_available and USE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=compute_dtype,
            **common_kwargs,
        )
        load_mode = "cuda-4bit"

    elif cuda_available:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=compute_dtype,
            **common_kwargs,
        )
        load_mode = "cuda-fp16"

    else:
        print(
            "WARNING: CUDA is unavailable. The model will run on CPU and may be "
            "very slow. Install a CUDA-enabled PyTorch build for GPU inference."
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map={"": "cpu"},
            torch_dtype=torch.float32,
            **common_kwargs,
        )
        load_mode = "cpu-fp32"

    model.eval()

    # Place input tensors on the device hosting the input embedding layer.
    try:
        model_device = model.get_input_embeddings().weight.device
    except Exception:
        model_device = next(model.parameters()).device

    print(f"Load mode: {load_mode}")
    print(f"Input device: {model_device}")
    if hasattr(model, "hf_device_map"):
        print(f"Model device map: {model.hf_device_map}")

    if cuda_available:
        allocated_gib = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_gib = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"CUDA memory allocated: {allocated_gib:.2f} GiB")
        print(f"CUDA memory reserved: {reserved_gib:.2f} GiB")

    print(
        f"Batching: up to {MAX_BATCH_SIZE} requests, wait {BATCH_WAIT_MS:g} ms, "
        f"token cap {MAX_BATCH_TOKENS}"
    )
    print("Model loaded and ready.")


def _free_model() -> None:
    global tokenizer, model, model_device, load_mode

    print("Shutting down... freeing model memory.")

    model = None
    tokenizer = None
    model_device = torch.device("cpu")
    load_mode = "not-loaded"

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


# -----------------------------------------------------------------------------
# Batched inference
# -----------------------------------------------------------------------------

def _eos_token_ids() -> List[int]:
    eos = tokenizer.eos_token_id
    ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    return [int(token) for token in ids if token is not None]


def _generate_batch_sync(requests: List[GenerateRequest]) -> List[Dict[str, Any]]:
    """Generate one batch. All requests must share the same generation key."""
    if model is None or tokenizer is None:
        raise RuntimeError("Model is not loaded.")

    started_at = time.perf_counter()
    first = requests[0]

    encoded = tokenizer(
        [request.prompt for request in requests],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        add_special_tokens=True,
    )
    prompt_token_counts = encoded["attention_mask"].sum(dim=1).tolist()
    padded_length = int(encoded["input_ids"].shape[-1])
    # Only the two tensors generation needs; some tokenizers add fields the
    # model would reject.
    encoded = {
        name: encoded[name].to(model_device) for name in ("input_ids", "attention_mask")
    }

    # Sampling-only parameters are added only when sampling is enabled,
    # preventing the "temperature may be ignored" warning in deterministic mode.
    config_kwargs: dict[str, Any] = {
        "max_new_tokens": first.max_new_tokens,
        "do_sample": first.do_sample,
        "repetition_penalty": first.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if first.do_sample:
        config_kwargs["temperature"] = first.temperature
        config_kwargs["top_p"] = first.top_p

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            generation_config=GenerationConfig(**config_kwargs),
        )

    continuations = output_ids[:, padded_length:].tolist()
    elapsed_seconds = time.perf_counter() - started_at
    stop_ids = set(_eos_token_ids())
    pad_id = tokenizer.pad_token_id

    results: List[Dict[str, Any]] = []
    total_generated = 0
    for row, prompt_tokens in zip(continuations, prompt_token_counts):
        # A row that finished early is padded up to the longest row; its own
        # output ends at its first end-of-sequence token.
        generated = len(row)
        for position, token in enumerate(row):
            if token in stop_ids:
                generated = position + 1
                break
            if pad_id is not None and token == pad_id and pad_id not in stop_ids:
                generated = position
                break
        text = tokenizer.decode(
            row[:generated],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        total_generated += generated
        result: Dict[str, Any] = {
            "generated_text": text,
            # Compatibility alias for clients that expect response["response"].
            "response": text,
            "model": MODEL_NAME,
            "load_mode": load_mode,
            "prompt_tokens": int(prompt_tokens),
            "generated_tokens": int(generated),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "batch_size": len(requests),
        }
        if elapsed_seconds > 0:
            result["tokens_per_second"] = round(generated / elapsed_seconds, 3)
        results.append(result)

    stats["batches"] += 1
    stats["requests"] += len(requests)
    stats["generated_tokens"] += total_generated
    stats["busy_seconds"] += elapsed_seconds
    if stats["batches"] % 50 == 0:
        print(
            f"[batching] {stats['batches']} batches, "
            f"mean size {stats['requests'] / stats['batches']:.1f}, "
            f"{stats['generated_tokens'] / max(stats['busy_seconds'], 1e-9):.0f} tokens/s, "
            f"{stats['busy_seconds'] / stats['requests']:.2f} s per request"
        )
    return results


def _split_into_batches(items: List[PendingRequest]) -> List[List[PendingRequest]]:
    """Split one generation group into batches bounded by size and token count."""
    ordered = sorted(items, key=lambda item: item.prompt_tokens)
    batches: List[List[PendingRequest]] = []
    current: List[PendingRequest] = []
    for item in ordered:
        candidate = current + [item]
        # Sorted ascending, so the newest item is the longest prompt.
        tokens = len(candidate) * (item.prompt_tokens + item.request.max_new_tokens)
        if current and (len(candidate) > MAX_BATCH_SIZE or tokens > MAX_BATCH_TOKENS):
            batches.append(current)
            current = [item]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _is_out_of_memory(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


async def _run_batch(batch: List[PendingRequest]) -> None:
    try:
        results = await asyncio.to_thread(
            _generate_batch_sync, [item.request for item in batch]
        )
    except Exception as exc:
        if _is_out_of_memory(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(batch) > 1:
                # Retry in halves; a batch that does not fit is not a reason
                # to fail every request in it.
                middle = len(batch) // 2
                print(f"[batching] out of memory at batch size {len(batch)}; retrying in halves")
                await _run_batch(batch[:middle])
                await _run_batch(batch[middle:])
                return
        for item in batch:
            if not item.future.done():
                item.future.set_exception(exc)
        return
    for item, result in zip(batch, results):
        if not item.future.done():
            item.future.set_result(result)


async def _batch_worker() -> None:
    assert request_queue is not None
    loop = asyncio.get_running_loop()
    while True:
        pending = [await request_queue.get()]
        # Give concurrent clients a moment to arrive, then take everything
        # that queued up (typically while the previous batch was generating).
        deadline = loop.time() + BATCH_WAIT_MS / 1000.0
        while len(pending) < MAX_ROUND_REQUESTS:
            timeout = deadline - loop.time()
            try:
                if timeout > 0:
                    pending.append(await asyncio.wait_for(request_queue.get(), timeout))
                else:
                    pending.append(request_queue.get_nowait())
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                break

        groups: Dict[Tuple[Any, ...], List[PendingRequest]] = {}
        for item in pending:
            groups.setdefault(generation_key(item.request), []).append(item)
        for items in groups.values():
            for batch in _split_into_batches(items):
                await _run_batch(batch)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global request_queue
    _load_model()
    request_queue = asyncio.Queue()
    worker = asyncio.create_task(_batch_worker())
    try:
        yield
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass
        _free_model()


app = FastAPI(
    title="Local Dolphin LLM Server (batched)",
    version="3.0.0",
    lifespan=lifespan,
)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Local Dolphin LLM Server (batched)",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()

    response: dict[str, Any] = {
        "status": "ready" if model is not None else "not-ready",
        "model": MODEL_NAME,
        "load_mode": load_mode,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "input_device": str(model_device),
        "batching": {
            "max_batch_size": MAX_BATCH_SIZE,
            "batch_wait_ms": BATCH_WAIT_MS,
            "max_batch_tokens": MAX_BATCH_TOKENS,
            "batches": stats["batches"],
            "requests": stats["requests"],
            "mean_batch_size": (
                round(stats["requests"] / stats["batches"], 2) if stats["batches"] else 0.0
            ),
        },
    }

    if cuda_available:
        response.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "cuda_memory_allocated_gib": round(
                    torch.cuda.memory_allocated(0) / (1024**3),
                    3,
                ),
                "cuda_memory_reserved_gib": round(
                    torch.cuda.memory_reserved(0) / (1024**3),
                    3,
                ),
            }
        )

    return response


@app.post("/generate")
async def generate(request: GenerateRequest) -> dict[str, Any]:
    if model is None or tokenizer is None or request_queue is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    if request.max_new_tokens > MAX_NEW_TOKENS_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"max_new_tokens cannot exceed {MAX_NEW_TOKENS_LIMIT}.",
        )
    if request.do_sample and request.temperature <= 0:
        raise HTTPException(
            status_code=400,
            detail="temperature must be greater than 0 when do_sample=true.",
        )

    prompt_tokens = min(
        MAX_INPUT_TOKENS,
        len(tokenizer(request.prompt, add_special_tokens=True)["input_ids"]),
    )
    future: "asyncio.Future[Dict[str, Any]]" = asyncio.get_running_loop().create_future()
    await request_queue.put(PendingRequest(request, future, prompt_tokens))

    try:
        return await future

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except RuntimeError as exc:
        message = str(exc)
        if _is_out_of_memory(exc):
            raise HTTPException(
                status_code=507,
                detail=(
                    "GPU/CPU memory was exhausted. Reduce max_new_tokens, "
                    "MAX_BATCH_SIZE or MAX_BATCH_TOKENS and retry."
                ),
            ) from exc

        raise HTTPException(status_code=500, detail=message) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {exc}",
        ) from exc
