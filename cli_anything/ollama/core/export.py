"""
export.py — Export conversation sessions to various formats.

Supported formats:
  - json   : Raw session JSON (full fidelity, machine-readable)
  - md     : Markdown conversation log (human-readable)
  - txt    : Plain text log
  - jsonl  : One message per line (for fine-tuning datasets)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

# Export format presets
EXPORT_PRESETS: dict[str, dict[str, Any]] = {
    "json": {
        "description": "Full session JSON (complete metadata)",
        "extension": ".json",
    },
    "md": {
        "description": "Markdown conversation log",
        "extension": ".md",
    },
    "txt": {
        "description": "Plain text conversation log",
        "extension": ".txt",
    },
    "jsonl": {
        "description": "One message per line (for fine-tuning)",
        "extension": ".jsonl",
    },
}


def render(
    session: dict[str, Any],
    output_path: str,
    preset: str = "md",
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Export a session to a file.

    Args:
        session:     The session dict (from project.open_session)
        output_path: Destination file path
        preset:      One of: json, md, txt, jsonl
        overwrite:   Replace if file exists

    Returns dict with output path and metadata.
    """
    output_path = os.path.abspath(output_path)

    if preset not in EXPORT_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {list(EXPORT_PRESETS.keys())}"
        )

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}\nUse overwrite=True."
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    content = _render_content(session, preset)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    file_size = os.path.getsize(output_path)
    return {
        "output": output_path,
        "preset": preset,
        "format": EXPORT_PRESETS[preset]["description"],
        "file_size": file_size,
        "messages_exported": len(
            [m for m in session.get("messages", []) if m.get("role") != "system"]
        ),
    }


def _render_content(session: dict[str, Any], preset: str) -> str:
    """Render session content as a string for the given preset."""
    messages = session.get("messages", [])
    model = session.get("model", "unknown")
    created = session.get("created_at", "")
    updated = session.get("updated_at", "")

    if preset == "json":
        return json.dumps(session, indent=2)

    elif preset == "md":
        lines = [
            f"# Conversation — {model}",
            f"",
            f"**Model**: `{model}`  ",
            f"**Created**: {created}  ",
            f"**Updated**: {updated}  ",
            f"",
            "---",
            "",
        ]
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                lines.append(f"> **System**: {content}")
                lines.append("")
            elif role == "user":
                lines.append(f"### User")
                lines.append(content)
                lines.append("")
            elif role == "assistant":
                lines.append(f"### Assistant")
                lines.append(content)
                lines.append("")
            else:
                lines.append(f"### {role.capitalize()}")
                lines.append(content)
                lines.append("")
        return "\n".join(lines)

    elif preset == "txt":
        lines = [
            f"Conversation with {model}",
            f"Created: {created}",
            f"Updated: {updated}",
            "=" * 60,
            "",
        ]
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"[{role}]")
            lines.append(content)
            lines.append("")
        return "\n".join(lines)

    elif preset == "jsonl":
        # One message per line — useful for fine-tuning datasets
        lines = []
        for msg in messages:
            if msg.get("role") != "system":
                lines.append(json.dumps(msg))
        return "\n".join(lines) + "\n"

    raise ValueError(f"Unhandled preset: {preset}")
