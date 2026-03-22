---
name: "cli-anything-ollama"
description: "CLI harness for Ollama — run, manage, and script local LLMs. Supports multi-turn chat sessions, model lifecycle, embeddings, and JSON output for agent consumption."
---

# cli-anything-ollama

Agent-native CLI for [Ollama](https://ollama.com). Wraps the Ollama REST API with
session persistence, undo/redo, and machine-readable JSON output.

## Prerequisites

- Ollama running: `ollama serve`
- Python package installed: `pip install -e /path/to/ollama/agent-harness`

## Installation

```bash
pip install -e ~/cli-anything/ollama/agent-harness
```

## Basic Syntax

```
cli-anything-ollama [--project <session.json>] [--json] [--host <url>] <group> <command> [args]
```

## Command Groups

| Group | Purpose |
|-------|---------|
| `model` | Model lifecycle (pull, list, show, copy, rm, ps, stop, create) |
| `chat` | Multi-turn conversation (requires --project) |
| `generate` | Single-turn text completion |
| `embed` | Vector embeddings |
| `server` | Server health and version |
| `session` | Session file management (new, info, reset, undo, redo, export) |

## Agent Usage (JSON mode)

All commands return `{"status": "ok", "data": {...}}` or `{"status": "error", "message": "..."}`.

```bash
# Check server
cli-anything-ollama --json server status

# List models
cli-anything-ollama --json model list

# Create a session
cli-anything-ollama --json session new /tmp/chat.json --model llama3.2

# Send a message
cli-anything-ollama --project /tmp/chat.json --json chat send "Explain recursion in one line"

# View conversation
cli-anything-ollama --project /tmp/chat.json --json chat history

# Export conversation
cli-anything-ollama --project /tmp/chat.json --json session export /tmp/chat.md --preset md

# Single-turn generation (no history)
cli-anything-ollama --json generate run "What is 2+2?" --model llama3.2

# Embeddings
cli-anything-ollama --json embed run "The quick brown fox" --model nomic-embed-text

# Undo last message
cli-anything-ollama --project /tmp/chat.json --json session undo
```

## Session File Format

Sessions are JSON files persisting conversation state:

```json
{
  "model": "llama3.2",
  "host": "http://localhost:11434",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ],
  "options": {"temperature": 0.7},
  "created_at": "...",
  "updated_at": "...",
  "snapshots": []
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

## Common Workflows

### Multi-turn code review
```bash
SESSION=/tmp/code-review.json
cli-anything-ollama --json session new $SESSION --model deepseek-coder \
  --system "You are an expert code reviewer."
cli-anything-ollama --project $SESSION --json chat send "Review: def fib(n): return fib(n-1)+fib(n-2)"
cli-anything-ollama --project $SESSION --json session export /tmp/review.md --preset md
```

### Build a fine-tuning dataset
```bash
# Export conversation as JSONL (one message per line, system excluded)
cli-anything-ollama --project /tmp/chat.json --json session export /tmp/dataset.jsonl --preset jsonl
```

### Semantic search setup
```bash
# Generate embeddings for indexing
cli-anything-ollama --json embed run "machine learning fundamentals" --model nomic-embed-text
```
