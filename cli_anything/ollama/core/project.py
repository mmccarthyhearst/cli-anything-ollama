"""
project.py — Ollama conversation session management.

A "project" in cli-anything-ollama is a persistent conversation session:
  - the model being used
  - full message history
  - model options (temperature, num_ctx, etc.)
  - the Ollama host URL

Session files are JSON stored wherever the user specifies.
"""

from __future__ import annotations

import json
import os
import fcntl
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Session schema
# ---------------------------------------------------------------------------

def _default_session(
    model: str = "llama3.2",
    host: str = "http://localhost:11434",
    system: Optional[str] = None,
) -> dict[str, Any]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    return {
        "model": model,
        "host": host,
        "messages": messages,
        "options": {},
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "snapshots": [],
    }


# ---------------------------------------------------------------------------
# File I/O with locking
# ---------------------------------------------------------------------------

def _locked_save_json(path: str, data: dict[str, Any]) -> None:
    """Atomically write JSON with exclusive file locking."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        f = open(path, "r+")
    except FileNotFoundError:
        f = open(path, "w")
    with f:
        locked = False
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            locked = True
        except (ImportError, OSError):
            pass
        try:
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
            f.flush()
        finally:
            if locked:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def create(
    output_path: str,
    model: str = "llama3.2",
    host: str = "http://localhost:11434",
    system: Optional[str] = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Create a new conversation session file.

    Returns the session dict. Raises FileExistsError if file exists and
    overwrite=False.
    """
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Session file already exists: {output_path}\n"
            "Use overwrite=True to replace it."
        )
    host = os.environ.get("OLLAMA_HOST", host).rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host

    session = _default_session(model=model, host=host, system=system)
    _locked_save_json(output_path, session)
    return session


def open_session(path: str) -> dict[str, Any]:
    """Load and return a session from a JSON file."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Session file not found: {path}")
    with open(path) as f:
        return json.load(f)


def save(session: dict[str, Any], path: str) -> None:
    """Save session dict to a JSON file."""
    session["updated_at"] = datetime.utcnow().isoformat() + "Z"
    _locked_save_json(path, session)


def info(session: dict[str, Any]) -> dict[str, Any]:
    """Return summary info about the current session."""
    messages = session.get("messages", [])
    user_msgs = sum(1 for m in messages if m.get("role") == "user")
    asst_msgs = sum(1 for m in messages if m.get("role") == "assistant")
    sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), None)
    return {
        "model": session.get("model"),
        "host": session.get("host"),
        "total_messages": len(messages),
        "user_messages": user_msgs,
        "assistant_messages": asst_msgs,
        "system_prompt": sys_msg,
        "options": session.get("options", {}),
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "snapshots": len(session.get("snapshots", [])),
    }


def list_sessions(directory: str = ".") -> list[dict[str, Any]]:
    """
    Scan a directory for session files (JSON with 'model' and 'messages' keys).
    Returns a list of {path, model, messages, updated_at}.
    """
    results = []
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return results
    for entry in sorted(Path(directory).glob("*.json")):
        try:
            with open(entry) as f:
                data = json.load(f)
            if "model" in data and "messages" in data:
                results.append({
                    "path": str(entry),
                    "model": data.get("model"),
                    "messages": len(data.get("messages", [])),
                    "updated_at": data.get("updated_at"),
                })
        except Exception:
            continue
    return results


# ---------------------------------------------------------------------------
# Message management
# ---------------------------------------------------------------------------

def add_message(
    session: dict[str, Any],
    role: str,
    content: str,
    images: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Append a message to the session's message history.

    Args:
        session: Session dict (mutated in place)
        role: "user", "assistant", or "system"
        content: Text content
        images: Optional list of base64-encoded image strings

    Returns the new message dict.
    """
    msg: dict[str, Any] = {"role": role, "content": content}
    if images:
        msg["images"] = images
    session.setdefault("messages", []).append(msg)
    return msg


def set_model(session: dict[str, Any], model: str) -> None:
    """Change the model in the session."""
    session["model"] = model


def set_system(session: dict[str, Any], system: str) -> None:
    """
    Set or replace the system prompt message.
    Places system message at index 0, removing any existing system message.
    """
    messages = [m for m in session.get("messages", []) if m.get("role") != "system"]
    session["messages"] = [{"role": "system", "content": system}] + messages


def set_option(session: dict[str, Any], key: str, value: Any) -> None:
    """Set a model option (e.g. temperature=0.9, num_ctx=8192)."""
    session.setdefault("options", {})[key] = value


def reset_history(session: dict[str, Any], keep_system: bool = True) -> None:
    """
    Clear message history. If keep_system=True, preserves the system prompt.
    """
    if keep_system:
        sys_msgs = [m for m in session.get("messages", []) if m.get("role") == "system"]
        session["messages"] = sys_msgs
    else:
        session["messages"] = []
