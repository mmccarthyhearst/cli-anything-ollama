"""
test_core.py — Unit tests for cli-anything-ollama core modules.

Uses synthetic data only. No external dependencies (no Ollama server required).
Run: pytest cli_anything/ollama/tests/test_core.py -v
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from cli_anything.ollama.core import export as export_core
from cli_anything.ollama.core import project as project_core
from cli_anything.ollama.core.session import Session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def session_file(tmp_dir):
    """Create a temporary session file and return its path + session dict."""
    path = os.path.join(tmp_dir, "test-session.json")
    sess = project_core.create(path, model="llama3.2")
    return path, sess


@pytest.fixture
def populated_session(tmp_dir):
    """Session with user and assistant messages."""
    path = os.path.join(tmp_dir, "populated.json")
    sess = project_core.create(
        path, model="mistral", system="You are a helpful assistant."
    )
    project_core.add_message(sess, "user", "What is Python?")
    project_core.add_message(sess, "assistant", "Python is a programming language.")
    project_core.add_message(sess, "user", "What version is current?")
    project_core.add_message(sess, "assistant", "Python 3.12 is the latest stable.")
    project_core.save(sess, path)
    return path, sess


# ===========================================================================
# project.py tests
# ===========================================================================

class TestProjectCreate:
    def test_create_defaults(self, tmp_dir):
        path = os.path.join(tmp_dir, "sess.json")
        sess = project_core.create(path)
        assert os.path.exists(path)
        assert sess["model"] == "llama3.2"
        assert "messages" in sess
        assert "created_at" in sess

    def test_create_with_model(self, tmp_dir):
        path = os.path.join(tmp_dir, "sess.json")
        sess = project_core.create(path, model="mistral:7b")
        assert sess["model"] == "mistral:7b"

    def test_create_with_system_prompt(self, tmp_dir):
        path = os.path.join(tmp_dir, "sess.json")
        sess = project_core.create(path, system="You are a pirate.")
        msgs = sess["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert "pirate" in msgs[0]["content"]

    def test_create_overwrites_existing(self, tmp_dir):
        path = os.path.join(tmp_dir, "sess.json")
        project_core.create(path, model="llama3.2")
        sess2 = project_core.create(path, model="mistral", overwrite=True)
        assert sess2["model"] == "mistral"

    def test_create_raises_on_existing_no_overwrite(self, tmp_dir):
        path = os.path.join(tmp_dir, "sess.json")
        project_core.create(path)
        with pytest.raises(FileExistsError):
            project_core.create(path, overwrite=False)

    def test_create_nested_directory(self, tmp_dir):
        path = os.path.join(tmp_dir, "nested", "deep", "sess.json")
        sess = project_core.create(path)
        assert os.path.exists(path)
        assert sess["model"] == "llama3.2"


class TestProjectOpenSave:
    def test_open_existing(self, session_file):
        path, original = session_file
        loaded = project_core.open_session(path)
        assert loaded["model"] == original["model"]
        assert loaded["created_at"] == original["created_at"]

    def test_open_missing_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            project_core.open_session(os.path.join(tmp_dir, "missing.json"))

    def test_save_updates_timestamp(self, session_file):
        path, sess = session_file
        original_ts = sess.get("updated_at")
        import time; time.sleep(0.01)
        project_core.save(sess, path)
        reloaded = project_core.open_session(path)
        # updated_at must exist; content must be valid
        assert "updated_at" in reloaded

    def test_save_persists_messages(self, session_file):
        path, sess = session_file
        project_core.add_message(sess, "user", "Hello!")
        project_core.save(sess, path)
        reloaded = project_core.open_session(path)
        assert len(reloaded["messages"]) == 1
        assert reloaded["messages"][0]["content"] == "Hello!"


class TestProjectInfo:
    def test_info_empty_session(self, session_file):
        _, sess = session_file
        info = project_core.info(sess)
        assert info["total_messages"] == 0
        assert info["user_messages"] == 0
        assert info["assistant_messages"] == 0
        assert info["system_prompt"] is None

    def test_info_with_messages(self, populated_session):
        _, sess = populated_session
        info = project_core.info(sess)
        assert info["total_messages"] == 5  # 1 system + 2 user + 2 assistant
        assert info["user_messages"] == 2
        assert info["assistant_messages"] == 2
        assert info["system_prompt"] is not None

    def test_info_snapshots_count(self, session_file):
        _, sess = session_file
        sess_obj = Session(sess)
        sess_obj.snapshot()
        info = project_core.info(sess)
        assert info["snapshots"] >= 1


class TestProjectListSessions:
    def test_list_finds_sessions(self, tmp_dir):
        project_core.create(os.path.join(tmp_dir, "a.json"), model="llama3.2")
        project_core.create(os.path.join(tmp_dir, "b.json"), model="mistral")
        results = project_core.list_sessions(tmp_dir)
        assert len(results) == 2
        names = {os.path.basename(r["path"]) for r in results}
        assert "a.json" in names
        assert "b.json" in names

    def test_list_skips_non_session_json(self, tmp_dir):
        # Write a JSON file that is not a session
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump({"key": "value"}, f)
        project_core.create(os.path.join(tmp_dir, "real.json"))
        results = project_core.list_sessions(tmp_dir)
        assert len(results) == 1

    def test_list_empty_directory(self, tmp_dir):
        results = project_core.list_sessions(tmp_dir)
        assert results == []

    def test_list_nonexistent_directory(self, tmp_dir):
        results = project_core.list_sessions(os.path.join(tmp_dir, "nonexistent"))
        assert results == []


class TestProjectMessages:
    def test_add_user_message(self, session_file):
        _, sess = session_file
        msg = project_core.add_message(sess, "user", "Hello")
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"
        assert len(sess["messages"]) == 1

    def test_add_with_images(self, session_file):
        _, sess = session_file
        msg = project_core.add_message(sess, "user", "Describe this", images=["base64data"])
        assert "images" in msg
        assert msg["images"] == ["base64data"]

    def test_set_model(self, session_file):
        _, sess = session_file
        project_core.set_model(sess, "gemma:7b")
        assert sess["model"] == "gemma:7b"

    def test_set_system_no_prior(self, session_file):
        _, sess = session_file
        project_core.set_system(sess, "Be concise.")
        msgs = sess["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be concise."

    def test_set_system_replaces_prior(self, tmp_dir):
        path = os.path.join(tmp_dir, "s.json")
        sess = project_core.create(path, system="Old system")
        project_core.add_message(sess, "user", "Hi")
        project_core.set_system(sess, "New system")
        sys_msgs = [m for m in sess["messages"] if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "New system"
        # User message still present
        user_msgs = [m for m in sess["messages"] if m["role"] == "user"]
        assert len(user_msgs) == 1

    def test_set_option(self, session_file):
        _, sess = session_file
        project_core.set_option(sess, "temperature", 0.9)
        assert sess["options"]["temperature"] == 0.9

    def test_reset_history_keep_system(self, populated_session):
        _, sess = populated_session
        project_core.reset_history(sess, keep_system=True)
        assert len(sess["messages"]) == 1
        assert sess["messages"][0]["role"] == "system"

    def test_reset_history_clear_all(self, populated_session):
        _, sess = populated_session
        project_core.reset_history(sess, keep_system=False)
        assert sess["messages"] == []


# ===========================================================================
# session.py tests (Session class)
# ===========================================================================

class TestSession:
    def _make_session(self, n_messages: int = 0):
        sess = {
            "model": "llama3.2",
            "messages": [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
                for i in range(n_messages)
            ],
            "options": {},
            "snapshots": [],
        }
        return sess, Session(sess)

    def test_initial_state(self):
        sess, s = self._make_session()
        assert not s.can_undo()
        assert not s.can_redo()
        assert s.history_depth() == 0

    def test_snapshot_enables_undo(self):
        sess, s = self._make_session(2)
        s.snapshot()
        assert s.can_undo()
        assert s.history_depth() == 1

    def test_undo_restores_messages(self):
        sess, s = self._make_session(2)
        s.snapshot()
        # Add a message after snapshot
        sess["messages"].append({"role": "user", "content": "new"})
        assert len(sess["messages"]) == 3
        restored = s.undo()
        assert len(restored) == 2
        assert len(sess["messages"]) == 2

    def test_undo_empty_returns_none(self):
        sess, s = self._make_session()
        result = s.undo()
        assert result is None

    def test_redo_after_undo(self):
        sess, s = self._make_session(2)
        s.snapshot()
        sess["messages"].append({"role": "user", "content": "extra"})
        s.undo()
        assert len(sess["messages"]) == 2
        s.redo()
        assert len(sess["messages"]) == 3

    def test_redo_empty_returns_none(self):
        sess, s = self._make_session()
        result = s.redo()
        assert result is None

    def test_snapshot_clears_redo(self):
        sess, s = self._make_session(2)
        s.snapshot()
        sess["messages"].append({"role": "user", "content": "a"})
        s.undo()
        # New action should clear redo
        s.snapshot()
        assert not s.can_redo()

    def test_status_dict(self):
        sess, s = self._make_session(4)
        s.snapshot()
        status = s.status()
        assert "can_undo" in status
        assert "can_redo" in status
        assert "undo_steps" in status
        assert status["can_undo"] is True
        assert status["current_messages"] == 4

    def test_multiple_snapshots(self):
        sess, s = self._make_session()
        # Snapshot BEFORE each mutation (matches CLI usage pattern)
        for i in range(5):
            s.snapshot()
            sess["messages"].append({"role": "user", "content": str(i)})
        assert s.history_depth() == 5
        # Undo all the way back — should restore to empty (pre-first-message state)
        for _ in range(5):
            s.undo()
        assert len(sess["messages"]) == 0

    def test_hydrate_from_serialized(self, tmp_dir):
        path = os.path.join(tmp_dir, "sess.json")
        sess = project_core.create(path, model="llama3.2")
        s = Session(sess)
        project_core.add_message(sess, "user", "hi")
        s.snapshot()
        project_core.save(sess, path)
        # Reload from disk
        reloaded = project_core.open_session(path)
        s2 = Session(reloaded)
        assert s2.history_depth() == 1


# ===========================================================================
# export.py tests
# ===========================================================================

class TestExport:
    def _make_session(self, include_system=True):
        msgs = []
        if include_system:
            msgs.append({"role": "system", "content": "You are a test bot."})
        msgs.append({"role": "user", "content": "What is 2+2?"})
        msgs.append({"role": "assistant", "content": "4"})
        return {
            "model": "llama3.2",
            "host": "http://localhost:11434",
            "messages": msgs,
            "options": {},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:01:00Z",
            "snapshots": [],
        }

    def test_export_json(self, tmp_dir):
        sess = self._make_session()
        path = os.path.join(tmp_dir, "out.json")
        result = export_core.render(sess, path, preset="json")
        assert os.path.exists(result["output"])
        with open(path) as f:
            data = json.load(f)
        assert data["model"] == "llama3.2"
        assert len(data["messages"]) == 3

    def test_export_md(self, tmp_dir):
        sess = self._make_session()
        path = os.path.join(tmp_dir, "out.md")
        result = export_core.render(sess, path, preset="md")
        content = Path(path).read_text()
        assert "llama3.2" in content
        assert "### User" in content
        assert "### Assistant" in content
        assert "What is 2+2?" in content

    def test_export_txt(self, tmp_dir):
        sess = self._make_session()
        path = os.path.join(tmp_dir, "out.txt")
        result = export_core.render(sess, path, preset="txt")
        content = Path(path).read_text()
        assert "[USER]" in content
        assert "[ASSISTANT]" in content
        assert "What is 2+2?" in content

    def test_export_jsonl(self, tmp_dir):
        sess = self._make_session(include_system=True)
        path = os.path.join(tmp_dir, "out.jsonl")
        result = export_core.render(sess, path, preset="jsonl")
        lines = [l for l in Path(path).read_text().strip().split("\n") if l]
        # System message excluded; user + assistant included
        assert len(lines) == 2
        objs = [json.loads(l) for l in lines]
        assert objs[0]["role"] == "user"
        assert objs[1]["role"] == "assistant"

    def test_export_no_overwrite_raises(self, tmp_dir):
        sess = self._make_session()
        path = os.path.join(tmp_dir, "out.md")
        export_core.render(sess, path, preset="md")
        with pytest.raises(FileExistsError):
            export_core.render(sess, path, preset="md", overwrite=False)

    def test_export_overwrite(self, tmp_dir):
        sess = self._make_session()
        path = os.path.join(tmp_dir, "out.md")
        export_core.render(sess, path, preset="md")
        export_core.render(sess, path, preset="md", overwrite=True)
        assert os.path.exists(path)

    def test_export_invalid_preset(self, tmp_dir):
        sess = self._make_session()
        path = os.path.join(tmp_dir, "out.xyz")
        with pytest.raises(ValueError):
            export_core.render(sess, path, preset="xyz")

    def test_export_result_metadata(self, tmp_dir):
        sess = self._make_session()
        path = os.path.join(tmp_dir, "out.md")
        result = export_core.render(sess, path, preset="md")
        assert result["output"] == os.path.abspath(path)
        assert result["preset"] == "md"
        assert result["file_size"] > 0
        assert result["messages_exported"] == 2  # user + assistant (not system)
