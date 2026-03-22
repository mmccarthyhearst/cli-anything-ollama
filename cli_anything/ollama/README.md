# cli-anything-ollama

Agent-native CLI harness for [Ollama](https://ollama.com) — run, manage, and script
local LLMs with full session persistence and JSON output for AI agent consumption.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (`ollama serve`)

## Installation

```bash
cd cli-anything/ollama/agent-harness
pip install -e .
```

Verify:
```bash
which cli-anything-ollama
cli-anything-ollama --help
```

## Usage

### Interactive REPL (default)

```bash
cli-anything-ollama
# or with a session file:
cli-anything-ollama --project my-session.json
```

### One-shot commands

```bash
# List models
cli-anything-ollama model list

# Pull a model
cli-anything-ollama model pull llama3.2

# Single-turn generation
cli-anything-ollama generate run "Summarize the Rust borrow checker in one sentence"

# Create a session and chat
cli-anything-ollama session new chat.json --model llama3.2
cli-anything-ollama --project chat.json chat send "Hello, who are you?"

# Embeddings
cli-anything-ollama embed run "The quick brown fox" --model nomic-embed-text

# Server status
cli-anything-ollama server status
```

### JSON output (for agents)

Every command supports `--json` for machine-readable output:

```bash
cli-anything-ollama --json model list
# {"status": "ok", "data": [...]}

cli-anything-ollama --project chat.json --json chat send "What is 2+2?"
# {"status": "ok", "data": {"model": "llama3.2", "response": "4", "messages": 3}}
```

## Command Reference

### `model`
| Command | Description |
|---------|-------------|
| `model list` | List locally downloaded models |
| `model pull <name>` | Download a model from registry |
| `model rm <name>` | Delete a local model |
| `model show <name>` | Show model metadata |
| `model copy <src> <dst>` | Duplicate a model |
| `model ps` | Show models loaded in memory |
| `model stop <name>` | Unload a model from memory |
| `model create <name> -f Modelfile` | Build a custom model |
| `model search <query>` | Search Ollama registry for models |
| `model capabilities <name>` | Show model capability tags (completion/tools/vision/embedding) |

### `chat`
| Command | Description |
|---------|-------------|
| `chat send <message>` | Send message (requires `--project`) |
| `chat send <message> -i <image>` | Send message with attached image(s) for vision models |
| `chat tools <message> --tools-file <file>` | Send message with tool definitions (function calling) |
| `chat history` | View conversation history |

### `compare`
| Command | Description |
|---------|-------------|
| `compare run <prompt> --models m1,m2` | Run prompt across multiple models in parallel |

### `generate`
| Command | Description |
|---------|-------------|
| `generate run <prompt>` | Single-turn text completion |

### `embed`
| Command | Description |
|---------|-------------|
| `embed run <text>` | Generate embedding vector |

### `server`
| Command | Description |
|---------|-------------|
| `server status` | Check server health |
| `server version` | Print server version |

### `session`
| Command | Description |
|---------|-------------|
| `session new <path>` | Create a new session file |
| `session info` | Show current session info |
| `session reset` | Clear conversation history |
| `session undo` | Undo last message exchange |
| `session redo` | Redo last undone exchange |
| `session export <path>` | Export to md/txt/json/jsonl |

## Examples

### Agent workflow (JSON mode)
```bash
# Create session
cli-anything-ollama --json session new /tmp/analysis.json --model llama3.2

# Ask a question
cli-anything-ollama --project /tmp/analysis.json --json chat send "List 5 Python best practices"

# Export for archival
cli-anything-ollama --project /tmp/analysis.json --json session export /tmp/chat.md -p md
```

### Multi-turn conversation
```bash
SESSION=/tmp/code-review.json
cli-anything-ollama session new $SESSION --model deepseek-coder --system "You are an expert code reviewer."
cli-anything-ollama --project $SESSION chat send "Review this: def add(a,b): return a+b"
cli-anything-ollama --project $SESSION chat send "What tests would you write for it?"
cli-anything-ollama --project $SESSION session export /tmp/review.md
```

### Multi-model comparison
```bash
# Compare two models on the same prompt
cli-anything-ollama --json compare run "Explain recursion in one sentence" --models llama3.2,mistral
```

### Vision / multimodal
```bash
# Send an image to a vision-capable model (e.g., llama3.2-vision)
SESSION=/tmp/vision.json
cli-anything-ollama session new $SESSION --model llama3.2-vision
cli-anything-ollama --project $SESSION chat send "What's in this image?" -i /path/to/photo.jpg
```

### Model discovery
```bash
# Search the Ollama registry
cli-anything-ollama model search llama
cli-anything-ollama --json model search mistral

# Check what capabilities a model supports
cli-anything-ollama model capabilities llama3.2
# output: completion, tools (if supported), vision (if supported)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_HOST` | Ollama server URL (default: `http://localhost:11434`) |
