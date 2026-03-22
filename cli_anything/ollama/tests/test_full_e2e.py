"""
test_full_e2e.py — E2E and subprocess tests for cli-anything-ollama.

Requires:
  - Ollama server running (ollama serve)
  - At least one model pulled (e.g. ollama pull llama3.2)
  - cli-anything-ollama installed (pip install -e .)

Run:
  pytest cli_anything/ollama/tests/test_full_e2e.py -v -s

Force installed command:
  CLI_ANYTHING_FORCE_INSTALLED=1 pytest cli_anything/ollama/tests/test_full_e2e.py -v -s
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from cli_anything.ollama.utils import ollama_backend as backend
from cli_anything.ollama.core import project as project_core
from cli_anything.ollama.core import export as export_core


# ---------------------------------------------------------------------------
# _resolve_cli helper
# ---------------------------------------------------------------------------

def _resolve_cli(name: str) -> list[str]:
    """
    Resolve the installed CLI command; falls back to python -m for development.

    Set CLI_ANYTHING_FORCE_INSTALLED=1 to require the installed command.
    Prints which backend is being used so -s output confirms the mode.
    """
    force = os.environ.get("CLI_ANYTHING_FORCE_INSTALLED", "").strip() == "1"
    path = shutil.which(name)
    if path:
        print(f"[_resolve_cli] Using installed command: {path}")
        return [path]
    if force:
        raise RuntimeError(
            f"{name} not found in PATH. Install with: pip install -e ."
        )
    module = name.replace("cli-anything-", "cli_anything.") + "." + name.split("-")[-1] + "_cli"
    print(f"[_resolve_cli] Falling back to: {sys.executable} -m {module}")
    return [sys.executable, "-m", module]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ollama_model() -> str:
    """Return the first available model name, or skip if none."""
    if not backend.is_available():
        pytest.skip("Ollama server not running. Start with: ollama serve")
    models = backend.list_models()
    if not models:
        pytest.skip("No models available. Pull one: ollama pull llama3.2")
    model = models[0]["name"]
    print(f"\n[ollama_model] Using model: {model}")
    return model


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ===========================================================================
# E2E — Backend API tests (use real Ollama server)
# ===========================================================================

class TestBackendAPI:
    def test_server_health(self):
        """Server must be reachable and return version."""
        if not backend.is_available():
            pytest.skip("Ollama not running")
        info = backend.health()
        assert info["status"] == "ok"
        assert "version" in info
        assert len(info["version"]) > 0
        print(f"\n  Ollama version: {info['version']} at {info['host']}")

    def test_version(self):
        """version() returns a non-empty string."""
        if not backend.is_available():
            pytest.skip("Ollama not running")
        v = backend.version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_list_models(self, ollama_model):
        """list_models() returns a list with at least one model."""
        models = backend.list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        for m in models:
            assert "name" in m
            assert "size" in m
        print(f"\n  Models available: {[m['name'] for m in models]}")

    def test_list_running(self):
        """list_running() returns a list (may be empty)."""
        if not backend.is_available():
            pytest.skip("Ollama not running")
        running = backend.list_running()
        assert isinstance(running, list)
        print(f"\n  Running models: {[m.get('name') for m in running]}")

    def test_show_model(self, ollama_model):
        """show() returns model details dict."""
        info = backend.show(ollama_model)
        assert isinstance(info, dict)
        print(f"\n  Model info keys: {list(info.keys())}")

    def test_generate_streaming(self, ollama_model):
        """generate() yields chunks; final chunk has done=True."""
        chunks = list(backend.generate(
            model=ollama_model,
            prompt="Say the word 'hello' and nothing else.",
            stream=True,
        ))
        assert len(chunks) > 0
        final = chunks[-1]
        assert final.get("done") is True
        full_text = "".join(c.get("response", "") for c in chunks)
        assert len(full_text) > 0
        print(f"\n  Generate response: '{full_text.strip()}'")

    def test_chat_streaming(self, ollama_model):
        """chat() with streaming yields chunks with message content."""
        messages = [{"role": "user", "content": "Reply with only the number 42."}]
        chunks = list(backend.chat(
            model=ollama_model,
            messages=messages,
            stream=True,
        ))
        assert len(chunks) > 0
        full_text = "".join(
            c.get("message", {}).get("content", "") for c in chunks
        )
        assert len(full_text) > 0
        print(f"\n  Chat response: '{full_text.strip()}'")

    def test_embed(self):
        """embed() returns a non-empty float vector."""
        if not backend.is_available():
            pytest.skip("Ollama not running")
        # Try common embedding models; skip if none available
        models = [m["name"] for m in backend.list_models()]
        embed_model = next(
            (m for m in models if "embed" in m.lower() or "nomic" in m.lower()),
            models[0] if models else None,
        )
        if not embed_model:
            pytest.skip("No models available for embedding test")
        result = backend.embed(model=embed_model, input="Hello world")
        assert "embeddings" in result
        embeddings = result["embeddings"]
        assert len(embeddings) > 0
        assert len(embeddings[0]) > 0
        assert isinstance(embeddings[0][0], float)
        print(f"\n  Embedding model: {embed_model}, dimensions: {len(embeddings[0])}")


# ===========================================================================
# E2E — CLI Subprocess tests
# ===========================================================================

class TestCLISubprocess:
    CLI_BASE = _resolve_cli("cli-anything-ollama")

    def _run(self, args: list[str], check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        return subprocess.run(
            self.CLI_BASE + args,
            capture_output=True,
            text=True,
            check=check,
            env=full_env,
            # Do NOT set cwd — installed commands work from any directory
        )

    def test_help(self):
        """--help exits 0 and mentions command groups."""
        result = self._run(["--help"])
        assert result.returncode == 0
        assert "model" in result.stdout
        assert "chat" in result.stdout
        assert "session" in result.stdout

    def test_model_help(self):
        result = self._run(["model", "--help"])
        assert result.returncode == 0
        assert "pull" in result.stdout or "list" in result.stdout

    def test_server_status_json(self):
        """server status --json returns valid JSON with version."""
        if not backend.is_available():
            pytest.skip("Ollama not running")
        result = self._run(["--json", "server", "status"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "ok"
        assert "version" in data["data"]
        print(f"\n  Server version: {data['data']['version']}")

    def test_model_list_json(self, ollama_model):
        """model list --json returns array of models."""
        result = self._run(["--json", "model", "list"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "ok"
        assert isinstance(data["data"], list)
        assert len(data["data"]) > 0
        print(f"\n  Models: {[m['name'] for m in data['data']]}")

    def test_session_new_json(self, tmp_dir, ollama_model):
        """session new creates a valid session file."""
        path = os.path.join(tmp_dir, "test.json")
        result = self._run(["--json", "session", "new", path, "--model", ollama_model])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "ok"
        assert os.path.exists(path)
        # Verify the file content
        sess = project_core.open_session(path)
        assert sess["model"] == ollama_model
        print(f"\n  Session created: {path}")

    def test_generate_json(self, ollama_model, tmp_dir):
        """generate run returns a non-empty response."""
        result = self._run([
            "--json", "generate", "run",
            "Say only the word YES",
            "--model", ollama_model,
        ])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "ok"
        assert len(data["data"]["response"]) > 0
        print(f"\n  Generate response: '{data['data']['response'].strip()}'")

    def test_full_session_workflow(self, tmp_dir, ollama_model):
        """
        Full workflow: create session → send message → check history →
        export to md → export to jsonl → undo → reset → verify.
        """
        session_path = os.path.join(tmp_dir, "workflow.json")
        md_path = os.path.join(tmp_dir, "chat.md")
        jsonl_path = os.path.join(tmp_dir, "chat.jsonl")

        # 1. Create session
        r = self._run(["--json", "session", "new", session_path, "--model", ollama_model])
        assert r.returncode == 0
        assert os.path.exists(session_path)

        # 2. Send a message
        r = self._run([
            "--project", session_path,
            "--json", "chat", "send",
            "Reply with only the word HELLO",
        ])
        assert r.returncode == 0
        data = json.loads(r.stdout)
        assert data["status"] == "ok"
        response = data["data"]["response"]
        assert len(response) > 0
        print(f"\n  Chat response: '{response.strip()}'")

        # 3. Check session info
        r = self._run(["--project", session_path, "--json", "session", "info"])
        assert r.returncode == 0
        info = json.loads(r.stdout)["data"]
        assert info["user_messages"] == 1
        assert info["assistant_messages"] == 1

        # 4. Export to markdown
        r = self._run([
            "--project", session_path,
            "--json", "session", "export", md_path, "--preset", "md",
        ])
        assert r.returncode == 0
        assert os.path.exists(md_path)
        md_content = Path(md_path).read_text()
        assert "### User" in md_content
        assert "### Assistant" in md_content
        assert "HELLO" in md_content.upper() or len(md_content) > 100
        print(f"\n  MD export: {md_path} ({os.path.getsize(md_path):,} bytes)")
        print(f"  MD content preview: {md_content[:200]}")

        # 5. Export to jsonl
        r = self._run([
            "--project", session_path,
            "--json", "session", "export", jsonl_path, "--preset", "jsonl",
        ])
        assert r.returncode == 0
        assert os.path.exists(jsonl_path)
        lines = [l for l in Path(jsonl_path).read_text().strip().split("\n") if l]
        assert len(lines) == 2  # user + assistant (no system)
        objs = [json.loads(l) for l in lines]
        assert objs[0]["role"] == "user"
        assert objs[1]["role"] == "assistant"
        print(f"\n  JSONL export: {jsonl_path} ({len(lines)} messages)")

        # 6. Undo the message
        r = self._run(["--project", session_path, "--json", "session", "undo"])
        assert r.returncode == 0
        undo_data = json.loads(r.stdout)
        assert undo_data["status"] == "ok"

        # 7. Reset history
        r = self._run(["--project", session_path, "--json", "session", "reset"])
        assert r.returncode == 0
        reset_sess = project_core.open_session(session_path)
        assert len([m for m in reset_sess["messages"] if m["role"] != "system"]) == 0

    def test_session_info_no_project(self):
        """session info without --project exits with error (not crash)."""
        result = self._run(["--json", "session", "info"], check=False)
        assert result.returncode != 0

    def test_model_show_json(self, ollama_model):
        """model show --json returns model details."""
        result = self._run(["--json", "model", "show", ollama_model])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "ok"

    def test_model_ps_json(self):
        """model ps --json returns list (may be empty)."""
        if not backend.is_available():
            pytest.skip("Ollama not running")
        result = self._run(["--json", "model", "ps"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "ok"
        assert isinstance(data["data"], list)

    def test_embed_json(self):
        """embed run returns dimensions > 0."""
        if not backend.is_available():
            pytest.skip("Ollama not running")
        models = [m["name"] for m in backend.list_models()]
        if not models:
            pytest.skip("No models available")
        # Use any available model for embedding
        model = models[0]
        result = self._run(["--json", "embed", "run", "Hello world", "--model", model])
        if result.returncode != 0:
            pytest.skip(f"Model {model} may not support embeddings")
        data = json.loads(result.stdout)
        assert data["status"] == "ok"
        assert data["data"]["dimensions"] > 0
        print(f"\n  Embedding dimensions: {data['data']['dimensions']}")

    # -----------------------------------------------------------------------
    # New: multi-model support tests
    # -----------------------------------------------------------------------

    def test_model_search_json(self):
        """model search hits ollama.com and returns results."""
        result = self._run(["--json", "model", "search", "llama"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "ok"
        results = data["data"]
        assert isinstance(results, list)
        assert len(results) > 0
        # Each result should have a name field
        assert all("name" in m for m in results)
        print(f"\n  Search 'llama': {len(results)} results, first={results[0].get('name')}")

    def test_model_search_no_results(self):
        """model search with obscure query returns empty list gracefully."""
        result = self._run(["--json", "model", "search", "xyzzy_no_such_model_abc123"])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "ok"
        assert isinstance(data["data"], list)

    def test_model_capabilities_json(self, ollama_model):
        """model capabilities returns capability_map with known keys."""
        result = self._run(["--json", "model", "capabilities", ollama_model])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "ok"
        cap_data = data["data"]
        assert cap_data["model"] == ollama_model
        assert isinstance(cap_data["capabilities"], list)
        assert "capability_map" in cap_data
        known = {"completion", "tools", "vision", "embedding", "thinking", "image", "insert"}
        assert set(cap_data["capability_map"].keys()) == known
        print(f"\n  Capabilities: {cap_data['capabilities']}")

    def test_compare_run_json(self, ollama_model):
        """compare run with a single model returns one result entry."""
        result = self._run([
            "--json", "compare", "run",
            "Reply with only the number 1.",
            "--models", ollama_model,
        ])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "ok"
        results = data["data"]
        assert len(results) == 1
        assert results[0]["model"] == ollama_model
        assert len(results[0]["response"]) > 0
        assert results[0]["error"] is None
        print(f"\n  Compare response: '{results[0]['response'].strip()[:80]}'")

    def test_chat_send_with_image(self, tmp_dir, ollama_model):
        """chat send --image encodes image and sends to model (vision models only)."""
        # Create a minimal valid PNG (1x1 white pixel)
        import struct, zlib
        def make_tiny_png():
            def chunk(name, data):
                c = name + data
                return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
            ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            raw_row = b'\x00\xff\xff\xff'
            idat = zlib.compress(raw_row)
            return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', b'')

        img_path = os.path.join(tmp_dir, "test.png")
        with open(img_path, "wb") as f:
            f.write(make_tiny_png())

        session_path = os.path.join(tmp_dir, "vision.json")
        self._run(["--json", "session", "new", session_path, "--model", ollama_model])

        result = self._run([
            "--project", session_path,
            "--json", "chat", "send",
            "What color is this image?",
            "--image", img_path,
        ])
        # Vision may or may not be supported on the test model;
        # success means no crash — accept either a response or a model error
        assert result.returncode == 0 or "vision" in result.stderr.lower() or "image" in result.stderr.lower()
        print(f"\n  Vision test returncode: {result.returncode}")
        if result.returncode == 0:
            data = json.loads(result.stdout)
            print(f"  Response: '{data['data']['response'][:80]}'")

    def test_chat_tools_json(self, tmp_dir, ollama_model):
        """chat tools sends tool definition and returns tool_calls if supported."""
        caps = backend.capabilities(ollama_model)
        if "tools" not in caps:
            pytest.skip(f"{ollama_model} does not support tools (capabilities={caps})")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Returns the current time",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
        ]
        tools_path = os.path.join(tmp_dir, "tools.json")
        with open(tools_path, "w") as f:
            json.dump(tools, f)

        session_path = os.path.join(tmp_dir, "tools-session.json")
        self._run(["--json", "session", "new", session_path, "--model", ollama_model])

        result = self._run([
            "--project", session_path,
            "--json", "chat", "tools",
            "What time is it right now?",
            "--tools-file", tools_path,
        ])
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["status"] == "ok"
        resp = data["data"]
        assert "content" in resp
        assert "tool_calls" in resp
        assert isinstance(resp["tool_calls"], list)
        print(f"\n  Tool calls: {resp['tool_calls']}")
        print(f"  Content: '{resp['content'][:80]}'")

    def test_help_includes_new_commands(self):
        """--help output includes all new command groups."""
        result = self._run(["--help"])
        assert "compare" in result.stdout
        result2 = self._run(["model", "--help"])
        assert "search" in result2.stdout
        assert "capabilities" in result2.stdout
        result3 = self._run(["chat", "--help"])
        assert "tools" in result3.stdout
