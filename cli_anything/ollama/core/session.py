"""
session.py — Undo/redo/snapshot management for Ollama conversation sessions.

Snapshots capture the full message history at a point in time. Undo/redo
navigate through that snapshot stack.
"""

from __future__ import annotations

import copy
from typing import Any, Optional


class Session:
    """
    Wraps a session dict with undo/redo capability via snapshots.

    Usage:
        sess = Session(session_dict)
        sess.snapshot()          # Save current state
        # ... modify session_dict externally ...
        sess.undo()              # Restore to last snapshot
        sess.redo()              # Re-apply undone change
    """

    def __init__(self, session: dict[str, Any]) -> None:
        self._session = session
        self._history: list[list[dict[str, Any]]] = []
        self._future: list[list[dict[str, Any]]] = []

        # Re-hydrate from any serialized snapshots
        for snap in session.get("snapshots", []):
            self._history.append(snap)

    @property
    def session(self) -> dict[str, Any]:
        return self._session

    def snapshot(self) -> None:
        """
        Save a copy of the current message history onto the undo stack.
        Clears the redo stack (new action invalidates future states).
        """
        self._history.append(copy.deepcopy(self._session.get("messages", [])))
        self._future.clear()
        # Persist snapshots in the session dict so they survive serialization
        self._session["snapshots"] = self._history[-10:]  # keep last 10

    def undo(self) -> Optional[list[dict[str, Any]]]:
        """
        Restore the previous message history snapshot.
        Returns the restored messages, or None if nothing to undo.
        """
        if not self._history:
            return None
        current = copy.deepcopy(self._session.get("messages", []))
        self._future.append(current)
        previous = self._history.pop()
        self._session["messages"] = previous
        self._session["snapshots"] = self._history[-10:]
        return previous

    def redo(self) -> Optional[list[dict[str, Any]]]:
        """
        Re-apply the most recently undone change.
        Returns the restored messages, or None if nothing to redo.
        """
        if not self._future:
            return None
        current = copy.deepcopy(self._session.get("messages", []))
        self._history.append(current)
        future = self._future.pop()
        self._session["messages"] = future
        self._session["snapshots"] = self._history[-10:]
        return future

    def can_undo(self) -> bool:
        return len(self._history) > 0

    def can_redo(self) -> bool:
        return len(self._future) > 0

    def history_depth(self) -> int:
        """Number of undo steps available."""
        return len(self._history)

    def status(self) -> dict[str, Any]:
        """Return a summary of undo/redo state."""
        return {
            "can_undo": self.can_undo(),
            "can_redo": self.can_redo(),
            "undo_steps": self.history_depth(),
            "redo_steps": len(self._future),
            "current_messages": len(self._session.get("messages", [])),
        }
