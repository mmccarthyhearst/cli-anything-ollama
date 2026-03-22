"""
Ollama REST API backend wrapper.

Wraps http://localhost:11434 (or OLLAMA_HOST) using the requests library.
Streaming responses use NDJSON (one JSON object per line).
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import re
from typing import Any, Callable, Generator, Optional

import requests

_OLLAMA_REGISTRY_URL = "https://ollama.com/api/tags"
_OLLAMA_SEARCH_URL = "https://ollama.com/search"


def _get_host() -> str:
    """Return the Ollama host URL from environment or default."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    return host


def _url(path: str) -> str:
    return _get_host() + path


def is_available() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        r = requests.get(_url("/api/version"), timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _require_server() -> None:
    """Raise RuntimeError with install guidance if server is not running."""
    if not is_available():
        host = _get_host()
        raise RuntimeError(
            f"Ollama server not reachable at {host}.\n"
            "Start it with: ollama serve\n"
            "Or install Ollama from: https://ollama.com"
        )


# ---------------------------------------------------------------------------
# Server info
# ---------------------------------------------------------------------------

def version() -> str:
    """Return the Ollama server version string."""
    _require_server()
    r = requests.get(_url("/api/version"), timeout=5)
    r.raise_for_status()
    return r.json().get("version", "unknown")


def health() -> dict[str, Any]:
    """Return server health info (version + reachability)."""
    ver = version()
    return {"status": "ok", "version": ver, "host": _get_host()}


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def list_models() -> list[dict[str, Any]]:
    """Return list of locally available models."""
    _require_server()
    r = requests.get(_url("/api/tags"), timeout=10)
    r.raise_for_status()
    return r.json().get("models", [])


def list_running() -> list[dict[str, Any]]:
    """Return list of currently loaded/running models."""
    _require_server()
    r = requests.get(_url("/api/ps"), timeout=10)
    r.raise_for_status()
    return r.json().get("models", [])


def show(model: str, verbose: bool = False) -> dict[str, Any]:
    """Return model metadata, template, parameters, and system prompt."""
    _require_server()
    payload: dict[str, Any] = {"name": model}
    if verbose:
        payload["verbose"] = True
    r = requests.post(_url("/api/show"), json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def pull(
    model: str,
    insecure: bool = False,
    progress_fn: Optional[Callable[[dict[str, Any]], None]] = None,
) -> None:
    """Pull a model from the Ollama registry with streaming progress."""
    _require_server()
    payload = {"name": model, "insecure": insecure, "stream": True}
    with requests.post(_url("/api/pull"), json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                obj = json.loads(line)
                if progress_fn:
                    progress_fn(obj)
                if obj.get("error"):
                    raise RuntimeError(f"Pull error: {obj['error']}")


def push(
    model: str,
    insecure: bool = False,
    progress_fn: Optional[Callable[[dict[str, Any]], None]] = None,
) -> None:
    """Push a model to the Ollama registry."""
    _require_server()
    payload = {"name": model, "insecure": insecure, "stream": True}
    with requests.post(_url("/api/push"), json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                obj = json.loads(line)
                if progress_fn:
                    progress_fn(obj)
                if obj.get("error"):
                    raise RuntimeError(f"Push error: {obj['error']}")


def delete(model: str) -> None:
    """Delete a local model."""
    _require_server()
    r = requests.delete(_url("/api/delete"), json={"name": model}, timeout=10)
    r.raise_for_status()


def copy(source: str, destination: str) -> None:
    """Copy (duplicate) a model under a new name."""
    _require_server()
    r = requests.post(_url("/api/copy"), json={"source": source, "destination": destination}, timeout=10)
    r.raise_for_status()


def stop(model: str) -> None:
    """Unload a model from memory by setting keep_alive=0."""
    _require_server()
    # Ollama unloads by sending a generate/chat with keep_alive=0
    r = requests.post(
        _url("/api/generate"),
        json={"model": model, "keep_alive": 0, "stream": False},
        timeout=15,
    )
    r.raise_for_status()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    options: Optional[dict[str, Any]] = None,
    stream: bool = True,
    keep_alive: Optional[str] = None,
    format: Optional[str] = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Raw text completion (single-turn). Yields response chunks when stream=True.
    Each chunk: {"response": "...", "done": bool, ...metrics...}
    """
    _require_server()
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
    }
    if system:
        payload["system"] = system
    if options:
        payload["options"] = options
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if format:
        payload["format"] = format

    with requests.post(_url("/api/generate"), json=payload, stream=stream, timeout=300) as r:
        r.raise_for_status()
        if stream:
            for line in r.iter_lines():
                if line:
                    obj = json.loads(line)
                    if obj.get("error"):
                        raise RuntimeError(f"Generate error: {obj['error']}")
                    yield obj
        else:
            obj = r.json()
            if obj.get("error"):
                raise RuntimeError(f"Generate error: {obj['error']}")
            yield obj


def chat(
    model: str,
    messages: list[dict[str, Any]],
    options: Optional[dict[str, Any]] = None,
    stream: bool = True,
    keep_alive: Optional[str] = None,
    format: Optional[str] = None,
    tools: Optional[list[dict[str, Any]]] = None,
) -> Generator[dict[str, Any], None, None]:
    """
    Multi-turn chat completion. Yields chunks when stream=True.
    Each chunk: {"message": {"role": "assistant", "content": "..."}, "done": bool}
    Final chunk includes timing metrics.
    """
    _require_server()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if options:
        payload["options"] = options
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if format:
        payload["format"] = format
    if tools:
        payload["tools"] = tools

    with requests.post(_url("/api/chat"), json=payload, stream=stream, timeout=300) as r:
        r.raise_for_status()
        if stream:
            for line in r.iter_lines():
                if line:
                    obj = json.loads(line)
                    if obj.get("error"):
                        raise RuntimeError(f"Chat error: {obj['error']}")
                    yield obj
        else:
            obj = r.json()
            if obj.get("error"):
                raise RuntimeError(f"Chat error: {obj['error']}")
            yield obj


def embed(
    model: str,
    input: str | list[str],
    options: Optional[dict[str, Any]] = None,
    truncate: bool = True,
    keep_alive: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate embeddings for a string or list of strings.
    Returns: {"embeddings": [[float, ...], ...], "model": str, ...}
    """
    _require_server()
    payload: dict[str, Any] = {
        "model": model,
        "input": input,
        "truncate": truncate,
    }
    if options:
        payload["options"] = options
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    r = requests.post(_url("/api/embed"), json=payload, timeout=60)
    r.raise_for_status()
    result = r.json()
    if result.get("error"):
        raise RuntimeError(f"Embed error: {result['error']}")
    return result


# ---------------------------------------------------------------------------
# Model discovery (ollama.com registry)
# ---------------------------------------------------------------------------

def search_models(
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Search the Ollama model library at ollama.com/search.

    Scrapes the HTML search page (no public JSON API exists) and returns
    list of {name, url} dicts. Does NOT require the local Ollama server.
    """
    try:
        r = requests.get(
            _OLLAMA_SEARCH_URL,
            params={"q": query},
            headers={"User-Agent": "cli-anything-ollama/1.0"},
            timeout=10,
        )
        r.raise_for_status()
        # Extract model names from href="/library/<name>" links
        names = re.findall(r'href="/library/([a-z0-9_.-][a-z0-9_.:-]*)"', r.text)
        # Deduplicate while preserving order
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for name in names:
            if name not in seen:
                seen.add(name)
                results.append({"name": name, "url": f"https://ollama.com/library/{name}"})
        return results[:limit]
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach ollama.com. Check your internet connection."
        )


def capabilities(model: str) -> list[str]:
    """
    Return the capability tags for a model: completion, tools, vision,
    embedding, thinking, image, insert.

    Calls /api/show and extracts the capabilities field.
    """
    info = show(model)
    caps = info.get("capabilities", [])
    # Normalize — may come back as strings or dicts
    return [c if isinstance(c, str) else str(c) for c in caps]


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare(
    models: list[str],
    prompt: str,
    system: Optional[str] = None,
    options: Optional[dict[str, Any]] = None,
    timeout_per_model: int = 120,
) -> list[dict[str, Any]]:
    """
    Run the same prompt across multiple models in parallel (thread pool).

    Returns list of:
      {"model": str, "response": str, "error": str|None,
       "eval_duration_ms": int|None}
    """
    _require_server()

    def _run_one(model: str) -> dict[str, Any]:
        try:
            chunks = list(generate(
                model=model,
                prompt=prompt,
                system=system,
                options=options,
                stream=False,
            ))
            final = chunks[-1] if chunks else {}
            duration_ns = final.get("eval_duration")
            return {
                "model": model,
                "response": final.get("response", ""),
                "error": None,
                "eval_duration_ms": round(duration_ns / 1_000_000) if duration_ns else None,
            }
        except Exception as e:
            return {"model": model, "response": "", "error": str(e), "eval_duration_ms": None}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = {pool.submit(_run_one, m): m for m in models}
        results = []
        for future in concurrent.futures.as_completed(futures, timeout=timeout_per_model * len(models)):
            results.append(future.result())

    # Return in original model order
    order = {m: i for i, m in enumerate(models)}
    return sorted(results, key=lambda r: order.get(r["model"], 999))


# ---------------------------------------------------------------------------
# Vision / multimodal helpers
# ---------------------------------------------------------------------------

def encode_image(path: str) -> str:
    """Read an image file and return a base64-encoded string for the API."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def chat_with_images(
    model: str,
    messages: list[dict[str, Any]],
    image_paths: Optional[list[str]] = None,
    options: Optional[dict[str, Any]] = None,
    stream: bool = True,
) -> Generator[dict[str, Any], None, None]:
    """
    Send a chat message with attached images (for vision-capable models).

    image_paths: list of local file paths to encode and attach to the
    last user message. Encodes each image as base64.
    """
    _require_server()
    msg_list = [dict(m) for m in messages]  # shallow copy

    if image_paths:
        encoded = [encode_image(p) for p in image_paths]
        # Attach images to the last user message
        for i in range(len(msg_list) - 1, -1, -1):
            if msg_list[i].get("role") == "user":
                msg_list[i] = dict(msg_list[i])
                msg_list[i]["images"] = encoded
                break

    yield from chat(model=model, messages=msg_list, options=options, stream=stream)


# ---------------------------------------------------------------------------
# Tool calling helpers
# ---------------------------------------------------------------------------

def chat_with_tools(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Run a non-streaming chat with tools defined. Returns the full final
    response including any tool_calls the model emitted.

    tools format (OpenAI-compatible, which Ollama accepts):
    [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather",
          "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
          }
        }
      }
    ]

    Returns:
      {
        "content": str,
        "tool_calls": [{"function": {"name": str, "arguments": {...}}}],
        "model": str,
        "done": bool
      }
    """
    _require_server()
    chunks = list(chat(
        model=model,
        messages=messages,
        tools=tools,
        options=options,
        stream=False,
    ))
    final = chunks[-1] if chunks else {}
    msg = final.get("message", {})
    return {
        "model": final.get("model", model),
        "content": msg.get("content", ""),
        "tool_calls": msg.get("tool_calls", []),
        "done": final.get("done", True),
    }


def create(
    model: str,
    modelfile: Optional[str] = None,
    path: Optional[str] = None,
    quantize: Optional[str] = None,
    progress_fn: Optional[Callable[[dict[str, Any]], None]] = None,
) -> None:
    """Create a custom model from a Modelfile string or path."""
    _require_server()
    payload: dict[str, Any] = {"name": model, "stream": True}
    if modelfile:
        payload["modelfile"] = modelfile
    if path:
        payload["path"] = path
    if quantize:
        payload["quantize"] = quantize
    with requests.post(_url("/api/create"), json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                obj = json.loads(line)
                if progress_fn:
                    progress_fn(obj)
                if obj.get("error"):
                    raise RuntimeError(f"Create error: {obj['error']}")
