"""
Ollama REST API backend wrapper.

Wraps http://localhost:11434 (or OLLAMA_HOST) using the requests library.
Streaming responses use NDJSON (one JSON object per line).
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Generator, Optional

import requests


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
