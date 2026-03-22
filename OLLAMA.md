# OLLAMA.md — CLI-Anything SOP for Ollama

## Architecture Analysis

Ollama is a local LLM server written in Go. Unlike traditional GUI apps, Ollama
exposes a REST API at `http://localhost:11434` (configurable via `OLLAMA_HOST`).

### Backend Engine

- **Server**: Ollama daemon (`ollama serve`) — Go HTTP server wrapping llama.cpp
- **API base**: `http://localhost:11434` (default)
- **Protocol**: HTTP/1.1 with streaming NDJSON responses for generation
- **Authentication**: None (local-only by default; token-based via `OLLAMA_HOST`)

### Data Model

| Entity | Description |
|--------|-------------|
| Model | Named LLM (e.g. `llama3.2`, `mistral:7b`). Stored in `~/.ollama/models/` |
| Message | `{role: user/assistant/system, content: str, images?: [base64]}` |
| Conversation | Ordered list of messages = the full context sent to the model |
| Options | Model hyperparams: `temperature`, `num_ctx`, `top_p`, `seed`, etc. |
| Session file | JSON file persisting conversation + model + options for REPL use |

### Existing CLI Commands (Ollama native)

| Command | Purpose |
|---------|---------|
| `ollama run MODEL` | Interactive REPL chat |
| `ollama pull MODEL` | Download model from registry |
| `ollama push MODEL` | Upload model to registry |
| `ollama list` | List local models |
| `ollama ps` | List running models |
| `ollama show MODEL` | Model info, parameters, template |
| `ollama copy SRC DST` | Duplicate a model |
| `ollama rm MODEL` | Delete model |
| `ollama create MODEL -f Modelfile` | Build custom model |
| `ollama stop MODEL` | Unload from memory |

### Command Map (GUI → API)

| GUI / User Action | API Endpoint | Python Call |
|-------------------|-------------|-------------|
| Type message, press Enter | `POST /api/chat` | `backend.chat()` |
| Start new conversation | (stateless — clear history) | `project.reset_history()` |
| Switch model | Update session model | `session.set_model()` |
| Download model | `POST /api/pull` | `backend.pull()` |
| Delete model | `DELETE /api/delete` | `backend.delete()` |
| List models | `GET /api/tags` | `backend.list_models()` |
| View running models | `GET /api/ps` | `backend.list_running()` |
| Get embeddings | `POST /api/embed` | `backend.embed()` |
| Show model info | `POST /api/show` | `backend.show()` |

### Rendering Gap Assessment

Ollama has **no rendering gap** in the traditional sense. There are no project files
to render via a separate engine. The "rendering" is the LLM inference itself, which
always happens through the Ollama server. Our CLI calls the server directly.

The only export concern is **conversation logging**: we export the message history
to JSON, Markdown, or plain text for archival.

## CLI Architecture

### Command Groups

```
cli-anything-ollama
├── model           Model lifecycle (pull, list, show, copy, rm, create, ps)
├── chat            Send messages and manage conversations
├── generate        Raw text completion (single-turn, no history)
├── embed           Generate vector embeddings
├── server          Server management (status, version, health)
└── session         Session file management (save, load, history, undo, redo)
```

### State Model

```json
{
  "model": "llama3.2",
  "host": "http://localhost:11434",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help you?"}
  ],
  "options": {
    "temperature": 0.7,
    "num_ctx": 4096
  },
  "snapshots": []
}
```

### Output Format

All commands support `--json` flag for machine-readable output:
- Success: `{"status": "ok", "data": {...}}`
- Error: `{"status": "error", "message": "...", "code": "..."}`
