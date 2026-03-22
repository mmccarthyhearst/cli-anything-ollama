"""
ollama_cli.py — CLI-Anything harness for Ollama.

Entry point: cli-anything-ollama

Command groups:
  model    — Model lifecycle (pull, list, show, copy, rm, ps)
  chat     — Multi-turn conversation with session persistence
  generate — Single-turn text completion
  embed    — Vector embeddings
  server   — Server management (status, version)
  session  — Session file management (info, history, undo, redo, reset, export)
"""

from __future__ import annotations

import json
import os
import sys
from functools import wraps
from typing import Any, Callable, Optional

import click

from cli_anything.ollama.core import export as export_core
from cli_anything.ollama.core import project as project_core
from cli_anything.ollama.core.session import Session
from cli_anything.ollama.utils import ollama_backend as backend
from cli_anything.ollama.utils.repl_skin import ReplSkin


# ---------------------------------------------------------------------------
# Global session state
# ---------------------------------------------------------------------------

_skin = ReplSkin("ollama", version="1.0.0")
_active_session: Optional[dict[str, Any]] = None
_active_session_path: Optional[str] = None
_session_obj: Optional[Session] = None


def _get_session() -> tuple[dict[str, Any], Session]:
    """Return active session and Session wrapper; raise if none loaded."""
    if _active_session is None:
        raise click.ClickException(
            "No session loaded. Use --project <file> or run: session new <path>"
        )
    return _active_session, _session_obj  # type: ignore[return-value]


def _load_session(path: str) -> None:
    global _active_session, _active_session_path, _session_obj
    _active_session = project_core.open_session(path)
    _active_session_path = os.path.abspath(path)
    _session_obj = Session(_active_session)


def _save_session_if_loaded() -> None:
    if _active_session is not None and _active_session_path is not None:
        project_core.save(_active_session, _active_session_path)


# ---------------------------------------------------------------------------
# Error handling decorator
# ---------------------------------------------------------------------------

def handle_error(fn: Callable) -> Callable:
    """Decorator: catch exceptions and format as CLI errors (or JSON errors)."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        ctx = click.get_current_context()
        use_json = ctx.find_root().params.get("json_output", False)
        try:
            return fn(*args, **kwargs)
        except click.ClickException:
            raise
        except Exception as e:
            if use_json:
                click.echo(json.dumps({"status": "error", "message": str(e)}))
                sys.exit(1)
            else:
                raise click.ClickException(str(e))
    return wrapper


def _out(ctx_or_json_flag: bool, data: Any) -> None:
    """Output data as JSON or human-readable based on --json flag."""
    if ctx_or_json_flag:
        click.echo(json.dumps({"status": "ok", "data": data}))
    else:
        if isinstance(data, str):
            click.echo(data)
        elif isinstance(data, dict):
            for k, v in data.items():
                click.echo(f"  {k}: {v}")
        elif isinstance(data, list):
            for item in data:
                click.echo(str(item))
        else:
            click.echo(str(data))


# ---------------------------------------------------------------------------
# Root CLI group
# ---------------------------------------------------------------------------

@click.group(invoke_without_command=True)
@click.option("--project", "-p", "project_path", default=None,
              help="Path to session JSON file")
@click.option("--json", "json_output", is_flag=True, default=False,
              help="Output as machine-readable JSON")
@click.option("--host", default=None, envvar="OLLAMA_HOST",
              help="Ollama server URL (default: http://localhost:11434)")
@click.version_option("1.0.0", prog_name="cli-anything-ollama")
@click.pass_context
def cli(ctx: click.Context, project_path: Optional[str], json_output: bool,
        host: Optional[str]) -> None:
    """CLI-Anything harness for Ollama — run, manage, and script LLMs."""
    ctx.ensure_object(dict)
    ctx.obj["json"] = json_output

    if host:
        os.environ["OLLAMA_HOST"] = host

    if project_path:
        try:
            _load_session(project_path)
        except FileNotFoundError:
            if not json_output:
                _skin.warning(f"Session file not found: {project_path}")

    if ctx.invoked_subcommand is None:
        ctx.invoke(repl)


# ---------------------------------------------------------------------------
# REPL command
# ---------------------------------------------------------------------------

@cli.command("repl")
@click.pass_context
def repl(ctx: click.Context) -> None:
    """Start the interactive REPL (default when no subcommand given)."""
    json_output = ctx.find_root().params.get("json_output", False)
    if json_output:
        click.echo(json.dumps({"status": "error", "message": "REPL not available in --json mode"}))
        return

    _skin.print_banner()

    if not backend.is_available():
        _skin.warning("Ollama server is not running. Start it with: ollama serve")
        _skin.warning(f"Expected at: {os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}")

    commands_dict = {
        "model list": "List local models",
        "model pull <name>": "Download a model",
        "model rm <name>": "Delete a model",
        "model show <name>": "Show model info",
        "model ps": "Show running models",
        "chat send <message>": "Send a chat message",
        "chat history": "Show conversation history",
        "generate <prompt>": "Single-turn completion",
        "embed <text>": "Generate embeddings",
        "session new <path>": "Create a new session file",
        "session info": "Show current session info",
        "session reset": "Clear conversation history",
        "session undo": "Undo last message",
        "session redo": "Redo last undo",
        "session export <path>": "Export conversation",
        "server status": "Check Ollama server status",
        "quit / exit": "Exit the REPL",
    }

    pt_session = _skin.create_prompt_session()

    while True:
        model_name = _active_session.get("model") if _active_session else None
        try:
            line = _skin.get_input(
                pt_session,
                project_name=model_name,
                modified=_active_session is not None,
            )
        except (KeyboardInterrupt, EOFError):
            break

        if not line:
            continue

        tokens = line.strip().split(None, 1)
        cmd = tokens[0].lower()
        rest = tokens[1] if len(tokens) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "help":
            _skin.help(commands_dict)
        elif cmd == "chat" and rest:
            _repl_chat(rest)
        elif cmd == "model":
            _repl_model(rest)
        elif cmd == "session":
            _repl_session(rest)
        elif cmd == "server":
            _repl_server(rest)
        elif cmd == "generate":
            _repl_generate(rest)
        else:
            # Default: treat as chat message if session is loaded
            if _active_session is not None:
                _repl_chat(line)
            else:
                _skin.error(f"Unknown command: {cmd}. Type 'help' for commands.")

    _save_session_if_loaded()
    _skin.print_goodbye()


def _repl_chat(message: str) -> None:
    if _active_session is None:
        _skin.error("No session loaded. Use: session new <path>")
        return
    _session_obj.snapshot()  # type: ignore[union-attr]
    project_core.add_message(_active_session, "user", message)
    _skin.info("Thinking...")
    full_response = ""
    try:
        for chunk in backend.chat(
            model=_active_session["model"],
            messages=_active_session["messages"],
            options=_active_session.get("options"),
            stream=True,
        ):
            part = chunk.get("message", {}).get("content", "")
            if part:
                full_response += part
                click.echo(part, nl=False)
        click.echo()
        project_core.add_message(_active_session, "assistant", full_response)
        _save_session_if_loaded()
    except Exception as e:
        _skin.error(str(e))


def _repl_model(args: str) -> None:
    parts = args.strip().split()
    sub = parts[0] if parts else ""
    arg = parts[1] if len(parts) > 1 else ""
    if sub == "list":
        try:
            models = backend.list_models()
            if models:
                _skin.table(
                    ["Name", "Size", "Modified"],
                    [[m.get("name"), _fmt_size(m.get("size", 0)), m.get("modified_at", "")[:10]]
                     for m in models]
                )
            else:
                _skin.info("No models found. Pull one with: model pull llama3.2")
        except Exception as e:
            _skin.error(str(e))
    elif sub == "pull" and arg:
        _skin.info(f"Pulling {arg}...")
        try:
            def on_progress(obj):
                status = obj.get("status", "")
                total = obj.get("total", 0)
                completed = obj.get("completed", 0)
                if total and completed:
                    _skin.progress(completed, total, status)
                else:
                    click.echo(f"  {status}", nl=False)
                    click.echo("\r", nl=False)
            backend.pull(arg, progress_fn=on_progress)
            click.echo()
            _skin.success(f"Pulled {arg}")
        except Exception as e:
            _skin.error(str(e))
    elif sub == "rm" and arg:
        try:
            backend.delete(arg)
            _skin.success(f"Deleted {arg}")
        except Exception as e:
            _skin.error(str(e))
    elif sub == "show" and arg:
        try:
            info = backend.show(arg)
            _skin.status("Model", arg)
            if "details" in info:
                for k, v in info["details"].items():
                    _skin.status(k, str(v))
        except Exception as e:
            _skin.error(str(e))
    elif sub == "ps":
        try:
            running = backend.list_running()
            if running:
                _skin.table(
                    ["Name", "Size", "Processor"],
                    [[m.get("name"), _fmt_size(m.get("size", 0)), m.get("details", {}).get("family", "")]
                     for m in running]
                )
            else:
                _skin.info("No models currently loaded.")
        except Exception as e:
            _skin.error(str(e))
    else:
        _skin.error(f"Unknown model subcommand: {sub}")


def _repl_session(args: str) -> None:
    global _active_session, _active_session_path, _session_obj
    parts = args.strip().split()
    sub = parts[0] if parts else ""
    arg = parts[1] if len(parts) > 1 else ""

    if sub == "new" and arg:
        model = parts[2] if len(parts) > 2 else "llama3.2"
        try:
            sess = project_core.create(arg, model=model)
            _active_session = sess
            _active_session_path = os.path.abspath(arg)
            _session_obj = Session(sess)
            _skin.success(f"Created session: {arg} (model: {model})")
        except Exception as e:
            _skin.error(str(e))
    elif sub == "load" and arg:
        try:
            _load_session(arg)
            _skin.success(f"Loaded: {arg}")
        except Exception as e:
            _skin.error(str(e))
    elif sub == "save":
        if _active_session is not None:
            _save_session_if_loaded()
            _skin.success(f"Saved: {_active_session_path}")
        else:
            _skin.error("No session loaded.")
    elif sub == "info":
        if _active_session is not None:
            info = project_core.info(_active_session)
            for k, v in info.items():
                _skin.status(k, str(v))
        else:
            _skin.error("No session loaded.")
    elif sub == "history":
        if _active_session is not None:
            msgs = _active_session.get("messages", [])
            for i, m in enumerate(msgs):
                role = m.get("role", "?").upper()
                content = m.get("content", "")[:120]
                click.echo(f"  [{i}] {role}: {content}")
        else:
            _skin.error("No session loaded.")
    elif sub == "reset":
        if _active_session is not None:
            _session_obj.snapshot()  # type: ignore[union-attr]
            project_core.reset_history(_active_session)
            _save_session_if_loaded()
            _skin.success("History cleared.")
        else:
            _skin.error("No session loaded.")
    elif sub == "undo":
        if _active_session is not None:
            result = _session_obj.undo()  # type: ignore[union-attr]
            if result is not None:
                _save_session_if_loaded()
                _skin.success(f"Undone. {len(result)} messages remain.")
            else:
                _skin.info("Nothing to undo.")
        else:
            _skin.error("No session loaded.")
    elif sub == "redo":
        if _active_session is not None:
            result = _session_obj.redo()  # type: ignore[union-attr]
            if result is not None:
                _save_session_if_loaded()
                _skin.success(f"Redone. {len(result)} messages.")
            else:
                _skin.info("Nothing to redo.")
        else:
            _skin.error("No session loaded.")
    elif sub == "export" and arg:
        if _active_session is not None:
            preset = parts[2] if len(parts) > 2 else "md"
            try:
                result = export_core.render(
                    _active_session, arg, preset=preset, overwrite=True
                )
                _skin.success(f"Exported to {result['output']} ({result['file_size']:,} bytes)")
            except Exception as e:
                _skin.error(str(e))
        else:
            _skin.error("No session loaded.")
    else:
        _skin.error(f"Unknown session subcommand: {sub}")


def _repl_server(args: str) -> None:
    parts = args.strip().split()
    sub = parts[0] if parts else "status"
    if sub == "status":
        try:
            info = backend.health()
            _skin.status("Status", "running")
            _skin.status("Version", info.get("version", "unknown"))
            _skin.status("Host", info.get("host", ""))
        except Exception:
            _skin.error("Server not reachable. Run: ollama serve")
    elif sub == "version":
        try:
            v = backend.version()
            _skin.status("Version", v)
        except Exception as e:
            _skin.error(str(e))
    else:
        _skin.error(f"Unknown server subcommand: {sub}")


def _repl_generate(prompt: str) -> None:
    if not prompt:
        _skin.error("No prompt provided.")
        return
    model = _active_session["model"] if _active_session else "llama3.2"
    try:
        for chunk in backend.generate(model=model, prompt=prompt, stream=True):
            click.echo(chunk.get("response", ""), nl=False)
        click.echo()
    except Exception as e:
        _skin.error(str(e))


def _fmt_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# model group
# ---------------------------------------------------------------------------

@cli.group("model")
def model_group() -> None:
    """Model lifecycle: pull, list, show, copy, rm, create, ps."""


@model_group.command("list")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def model_list(ctx: click.Context, json_output: bool) -> None:
    """List locally available models."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    models = backend.list_models()
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": models}))
    else:
        if not models:
            _skin.info("No models found. Run: cli-anything-ollama model pull llama3.2")
            return
        _skin.table(
            ["Name", "Size", "Modified"],
            [[m.get("name", ""), _fmt_size(m.get("size", 0)), m.get("modified_at", "")[:10]]
             for m in models]
        )


@model_group.command("pull")
@click.argument("model_name")
@click.option("--insecure", is_flag=True, default=False)
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def model_pull(ctx: click.Context, model_name: str, insecure: bool, json_output: bool) -> None:
    """Pull a model from the Ollama registry."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    progress_events = []

    def on_progress(obj):
        progress_events.append(obj)
        if not use_json:
            status = obj.get("status", "")
            total = obj.get("total", 0)
            completed = obj.get("completed", 0)
            if total and completed:
                _skin.progress(completed, total, status)
            else:
                click.echo(f"\r  {status}          ", nl=False)

    backend.pull(model_name, insecure=insecure, progress_fn=on_progress)
    if not use_json:
        click.echo()
        _skin.success(f"Pulled: {model_name}")
    else:
        click.echo(json.dumps({"status": "ok", "data": {"model": model_name, "events": len(progress_events)}}))


@model_group.command("rm")
@click.argument("model_name")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def model_rm(ctx: click.Context, model_name: str, json_output: bool) -> None:
    """Delete a local model."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    backend.delete(model_name)
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": {"deleted": model_name}}))
    else:
        _skin.success(f"Deleted: {model_name}")


@model_group.command("show")
@click.argument("model_name")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def model_show(ctx: click.Context, model_name: str, verbose: bool, json_output: bool) -> None:
    """Show model metadata, template, and parameters."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    info = backend.show(model_name, verbose=verbose)
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": info}))
    else:
        _skin.status("Model", model_name)
        details = info.get("details", {})
        for k, v in details.items():
            _skin.status(k, str(v))
        if verbose and info.get("parameters"):
            click.echo("\nParameters:")
            click.echo(info["parameters"])


@model_group.command("copy")
@click.argument("source")
@click.argument("destination")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def model_copy(ctx: click.Context, source: str, destination: str, json_output: bool) -> None:
    """Copy (duplicate) a model under a new name."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    backend.copy(source, destination)
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": {"source": source, "destination": destination}}))
    else:
        _skin.success(f"Copied {source} → {destination}")


@model_group.command("ps")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def model_ps(ctx: click.Context, json_output: bool) -> None:
    """List models currently loaded in memory."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    running = backend.list_running()
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": running}))
    else:
        if not running:
            _skin.info("No models currently loaded.")
        else:
            _skin.table(
                ["Name", "Size", "Processor"],
                [[m.get("name", ""), _fmt_size(m.get("size", 0)),
                  m.get("details", {}).get("family", "unknown")]
                 for m in running]
            )


@model_group.command("stop")
@click.argument("model_name")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def model_stop(ctx: click.Context, model_name: str, json_output: bool) -> None:
    """Unload a model from memory."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    backend.stop(model_name)
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": {"stopped": model_name}}))
    else:
        _skin.success(f"Unloaded: {model_name}")


@model_group.command("search")
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Max results to return")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def model_search(ctx: click.Context, query: str, limit: int, json_output: bool) -> None:
    """Search the Ollama model registry at ollama.com."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    results = backend.search_models(query, limit=limit)
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": results}))
    else:
        if not results:
            _skin.info(f"No models found for: {query}")
            return
        rows = []
        for m in results:
            name = m.get("name", "")
            desc = (m.get("description") or "")[:60]
            pulls = m.get("pulls", m.get("pull_count", ""))
            rows.append([name, str(pulls), desc])
        _skin.table(["Name", "Pulls", "Description"], rows)


@model_group.command("capabilities")
@click.argument("model_name")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def model_capabilities(ctx: click.Context, model_name: str, json_output: bool) -> None:
    """Show what a model can do: completion, tools, vision, embedding, thinking."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    caps = backend.capabilities(model_name)
    all_caps = ["completion", "tools", "vision", "embedding", "thinking", "image", "insert"]
    cap_map = {c: (c in caps) for c in all_caps}
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": {
            "model": model_name,
            "capabilities": caps,
            "capability_map": cap_map,
        }}))
    else:
        _skin.status("Model", model_name)
        for cap, supported in cap_map.items():
            mark = "✓" if supported else "✗"
            click.echo(f"  {mark}  {cap}")


@model_group.command("create")
@click.argument("model_name")
@click.option("--file", "-f", "modelfile_path", default=None,
              help="Path to Modelfile (default: ./Modelfile)")
@click.option("--quantize", "-q", default=None, help="Quantization level (e.g. q4_K_M)")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def model_create(ctx: click.Context, model_name: str, modelfile_path: Optional[str],
                 quantize: Optional[str], json_output: bool) -> None:
    """Create a custom model from a Modelfile."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    mf_path = modelfile_path or "Modelfile"
    if not os.path.exists(mf_path):
        raise click.ClickException(f"Modelfile not found: {mf_path}")
    with open(mf_path) as f:
        modelfile_content = f.read()

    events = []

    def on_progress(obj):
        events.append(obj)
        if not use_json:
            click.echo(f"  {obj.get('status', '')}", nl=False)
            click.echo("\r", nl=False)

    backend.create(model_name, modelfile=modelfile_content, quantize=quantize, progress_fn=on_progress)
    if not use_json:
        click.echo()
        _skin.success(f"Created: {model_name}")
    else:
        click.echo(json.dumps({"status": "ok", "data": {"model": model_name}}))


# ---------------------------------------------------------------------------
# chat group
# ---------------------------------------------------------------------------

@cli.group("chat")
def chat_group() -> None:
    """Multi-turn conversation with session persistence."""


@chat_group.command("send")
@click.argument("message")
@click.option("--model", "-m", default=None, help="Override session model")
@click.option("--image", "-i", "image_paths", multiple=True,
              help="Image file(s) to attach (vision models). Repeatable.")
@click.option("--json", "json_output", is_flag=True)
@click.option("--no-stream", is_flag=True, default=False)
@click.pass_context
@handle_error
def chat_send(ctx: click.Context, message: str, model: Optional[str],
              image_paths: tuple, json_output: bool, no_stream: bool) -> None:
    """Send a message and get a response (requires --project).

    For vision models, attach images with --image <path> (repeatable).
    """
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    sess, sess_obj = _get_session()

    effective_model = model or sess["model"]
    sess_obj.snapshot()

    if image_paths:
        encoded = [backend.encode_image(p) for p in image_paths]
        project_core.add_message(sess, "user", message, images=encoded)
    else:
        project_core.add_message(sess, "user", message)

    stream = not no_stream and not use_json
    full_response = ""

    for chunk in backend.chat(
        model=effective_model,
        messages=sess["messages"],
        options=sess.get("options"),
        stream=stream,
    ):
        part = chunk.get("message", {}).get("content", "")
        if part:
            full_response += part
            if stream:
                click.echo(part, nl=False)

    if stream:
        click.echo()

    project_core.add_message(sess, "assistant", full_response)
    _save_session_if_loaded()

    if use_json:
        click.echo(json.dumps({"status": "ok", "data": {
            "model": effective_model,
            "response": full_response,
            "messages": len(sess.get("messages", [])),
        }}))


@chat_group.command("history")
@click.option("--json", "json_output", is_flag=True)
@click.option("--limit", "-n", default=20, help="Max messages to show")
@click.pass_context
@handle_error
def chat_history(ctx: click.Context, json_output: bool, limit: int) -> None:
    """Show conversation history."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    sess, _ = _get_session()
    messages = sess.get("messages", [])[-limit:]
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": messages}))
    else:
        for i, m in enumerate(messages):
            role = m.get("role", "?").upper()
            content = m.get("content", "")
            click.echo(f"\n[{role}]")
            click.echo(content)


@chat_group.command("tools")
@click.argument("message")
@click.option("--model", "-m", default=None, help="Override session model")
@click.option("--tools-file", "-t", "tools_file", default=None,
              help="JSON file defining tools (OpenAI function-calling format)")
@click.option("--tools-json", default=None,
              help="Inline JSON string defining tools")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def chat_tools(ctx: click.Context, message: str, model: Optional[str],
               tools_file: Optional[str], tools_json: Optional[str],
               json_output: bool) -> None:
    """Send a message with tool definitions (function calling).

    Requires a tools-capable model (check with: model capabilities <name>).
    Returns the response content and any tool_calls the model emits.

    Example tools JSON:
      [{"type":"function","function":{"name":"get_weather",
        "description":"Get weather","parameters":{"type":"object",
        "properties":{"location":{"type":"string"}},"required":["location"]}}}]
    """
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    sess, sess_obj = _get_session()
    effective_model = model or sess["model"]

    # Load tools definition
    if tools_file:
        with open(tools_file) as f:
            tools = json.load(f)
    elif tools_json:
        tools = json.loads(tools_json)
    else:
        raise click.ClickException(
            "Provide tools with --tools-file <path> or --tools-json '<json>'"
        )

    sess_obj.snapshot()
    project_core.add_message(sess, "user", message)

    result = backend.chat_with_tools(
        model=effective_model,
        messages=sess["messages"],
        tools=tools,
        options=sess.get("options"),
    )

    project_core.add_message(sess, "assistant", result["content"])
    _save_session_if_loaded()

    if use_json:
        click.echo(json.dumps({"status": "ok", "data": result}))
    else:
        if result["content"]:
            click.echo(result["content"])
        if result["tool_calls"]:
            click.echo("\n[Tool Calls]")
            for tc in result["tool_calls"]:
                fn = tc.get("function", {})
                click.echo(f"  {fn.get('name')}({json.dumps(fn.get('arguments', {}))})")


# ---------------------------------------------------------------------------
# generate group
# ---------------------------------------------------------------------------

@cli.group("generate")
def generate_group() -> None:
    """Single-turn raw text completion."""


@generate_group.command("run")
@click.argument("prompt")
@click.option("--model", "-m", default="llama3.2")
@click.option("--system", "-s", default=None)
@click.option("--temperature", "-t", type=float, default=None)
@click.option("--format", "fmt", default=None, help="Response format: json")
@click.option("--json", "json_output", is_flag=True)
@click.option("--no-stream", is_flag=True, default=False)
@click.pass_context
@handle_error
def generate_run(ctx: click.Context, prompt: str, model: str, system: Optional[str],
                 temperature: Optional[float], fmt: Optional[str],
                 json_output: bool, no_stream: bool) -> None:
    """Run a single-turn completion (no conversation history)."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)

    # Use session model if --project was given and no --model override
    if _active_session and model == "llama3.2":
        model = _active_session.get("model", model)

    options = {}
    if temperature is not None:
        options["temperature"] = temperature

    stream = not no_stream and not use_json
    full_response = ""

    for chunk in backend.generate(
        model=model, prompt=prompt, system=system,
        options=options or None, stream=stream, format=fmt,
    ):
        part = chunk.get("response", "")
        if part:
            full_response += part
            if stream:
                click.echo(part, nl=False)

    if stream:
        click.echo()

    if use_json:
        click.echo(json.dumps({"status": "ok", "data": {
            "model": model,
            "response": full_response,
        }}))


# ---------------------------------------------------------------------------
# embed group
# ---------------------------------------------------------------------------

@cli.group("embed")
def embed_group() -> None:
    """Generate vector embeddings from text."""


@embed_group.command("run")
@click.argument("text")
@click.option("--model", "-m", default="nomic-embed-text")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def embed_run(ctx: click.Context, text: str, model: str, json_output: bool) -> None:
    """Embed text and return the vector."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    result = backend.embed(model=model, input=text)
    embeddings = result.get("embeddings", [[]])[0]
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": {
            "model": model,
            "dimensions": len(embeddings),
            "embedding": embeddings,
        }}))
    else:
        _skin.status("Model", model)
        _skin.status("Dimensions", str(len(embeddings)))
        _skin.status("Preview", str(embeddings[:5]) + "...")


# ---------------------------------------------------------------------------
# compare group
# ---------------------------------------------------------------------------

@cli.group("compare")
def compare_group() -> None:
    """Run the same prompt across multiple models and compare results."""


@compare_group.command("run")
@click.argument("prompt")
@click.option("--models", "-m", required=True,
              help="Comma-separated list of models, e.g. llama3.2,mistral,gemma")
@click.option("--system", "-s", default=None, help="System prompt")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def compare_run(ctx: click.Context, prompt: str, models: str, system: Optional[str],
                json_output: bool) -> None:
    """Run a prompt across multiple models in parallel and compare responses."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        raise click.ClickException("No models specified. Use --models llama3.2,mistral")

    if not use_json:
        _skin.info(f"Comparing {len(model_list)} models...")

    results = backend.compare(model_list, prompt=prompt, system=system)

    if use_json:
        click.echo(json.dumps({"status": "ok", "data": results}))
    else:
        for r in results:
            ms = f"  ({r['eval_duration_ms']}ms)" if r.get("eval_duration_ms") else ""
            click.echo(f"\n{'─'*60}")
            click.echo(f"  Model: {r['model']}{ms}")
            click.echo(f"{'─'*60}")
            if r.get("error"):
                _skin.error(r["error"])
            else:
                click.echo(r["response"])


# ---------------------------------------------------------------------------
# server group
# ---------------------------------------------------------------------------

@cli.group("server")
def server_group() -> None:
    """Ollama server management."""


@server_group.command("status")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def server_status(ctx: click.Context, json_output: bool) -> None:
    """Check if Ollama server is running and get version."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    info = backend.health()
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": info}))
    else:
        _skin.status("Status", "running")
        _skin.status("Version", info.get("version", "unknown"))
        _skin.status("Host", info.get("host", ""))


@server_group.command("version")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def server_version(ctx: click.Context, json_output: bool) -> None:
    """Print Ollama server version."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    v = backend.version()
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": {"version": v}}))
    else:
        click.echo(v)


# ---------------------------------------------------------------------------
# session group
# ---------------------------------------------------------------------------

@cli.group("session")
def session_group() -> None:
    """Conversation session file management."""


@session_group.command("new")
@click.argument("output_path")
@click.option("--model", "-m", default="llama3.2", help="Model to use")
@click.option("--system", "-s", default=None, help="System prompt")
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def session_new(ctx: click.Context, output_path: str, model: str,
                system: Optional[str], overwrite: bool, json_output: bool) -> None:
    """Create a new conversation session file."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    sess = project_core.create(output_path, model=model, system=system, overwrite=overwrite)
    info = project_core.info(sess)
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": {
            "path": os.path.abspath(output_path),
            **info,
        }}))
    else:
        _skin.success(f"Created session: {output_path}")
        _skin.status("Model", model)


@session_group.command("info")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def session_info(ctx: click.Context, json_output: bool) -> None:
    """Show info about the current session."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    sess, _ = _get_session()
    info = project_core.info(sess)
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": info}))
    else:
        for k, v in info.items():
            _skin.status(k, str(v))


@session_group.command("reset")
@click.option("--keep-system/--no-system", default=True)
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def session_reset(ctx: click.Context, keep_system: bool, json_output: bool) -> None:
    """Clear conversation history (optionally keep system prompt)."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    sess, sess_obj = _get_session()
    sess_obj.snapshot()
    project_core.reset_history(sess, keep_system=keep_system)
    _save_session_if_loaded()
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": {"messages": len(sess.get("messages", []))}}))
    else:
        _skin.success("History cleared.")


@session_group.command("undo")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def session_undo(ctx: click.Context, json_output: bool) -> None:
    """Undo the last message exchange."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    sess, sess_obj = _get_session()
    result = sess_obj.undo()
    if result is None:
        if use_json:
            click.echo(json.dumps({"status": "ok", "data": {"undone": False}}))
        else:
            _skin.info("Nothing to undo.")
        return
    _save_session_if_loaded()
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": {"undone": True, "messages": len(result)}}))
    else:
        _skin.success(f"Undone. {len(result)} messages remain.")


@session_group.command("redo")
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def session_redo(ctx: click.Context, json_output: bool) -> None:
    """Redo the last undone message exchange."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    sess, sess_obj = _get_session()
    result = sess_obj.redo()
    if result is None:
        if use_json:
            click.echo(json.dumps({"status": "ok", "data": {"redone": False}}))
        else:
            _skin.info("Nothing to redo.")
        return
    _save_session_if_loaded()
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": {"redone": True, "messages": len(result)}}))
    else:
        _skin.success(f"Redone. {len(result)} messages.")


@session_group.command("export")
@click.argument("output_path")
@click.option("--preset", "-p", default="md",
              type=click.Choice(["json", "md", "txt", "jsonl"]),
              help="Export format")
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--json", "json_output", is_flag=True)
@click.pass_context
@handle_error
def session_export(ctx: click.Context, output_path: str, preset: str,
                   overwrite: bool, json_output: bool) -> None:
    """Export conversation to a file (md, txt, json, jsonl)."""
    use_json = json_output or ctx.find_root().params.get("json_output", False)
    sess, _ = _get_session()
    result = export_core.render(sess, output_path, preset=preset, overwrite=overwrite)
    if use_json:
        click.echo(json.dumps({"status": "ok", "data": result}))
    else:
        _skin.success(f"Exported: {result['output']} ({result['file_size']:,} bytes)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli(obj={})


if __name__ == "__main__":
    main()
