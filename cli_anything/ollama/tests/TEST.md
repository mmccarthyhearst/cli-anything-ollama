# TEST.md — CLI-Anything Ollama Test Plan

## Part 1: Test Plan

### Test Inventory Plan

| File | Planned Tests |
|------|--------------|
| `test_core.py` | 28 unit tests |
| `test_full_e2e.py` | 22 E2E + subprocess tests |
| **Total** | **50 tests** |

---

### Unit Test Plan (`test_core.py`)

#### `core/project.py` — Session Management (12 tests)

| Function | Tests | Edge Cases |
|----------|-------|-----------|
| `create()` | Create with defaults; create with model/system; overwrite=False raises FileExistsError | Empty system string, non-existent parent dir |
| `open_session()` | Load valid JSON; raise FileNotFoundError for missing | Malformed JSON (not tested here — stdlib behavior) |
| `save()` | Write + update `updated_at`; verify on-disk content | Concurrent save (locking) |
| `info()` | Correct counts for user/assistant/system messages | All-system, empty history |
| `list_sessions()` | Find JSON files with correct schema; skip non-session JSONs | Empty directory, non-existent directory |
| `add_message()` | Append user/assistant/system; with images | Empty content string |
| `set_model()` | Updates model key | Same model |
| `set_system()` | Replaces existing system msg; inserts at position 0 | No prior system message |
| `set_option()` | Adds to options dict | Overwrite existing key |
| `reset_history()` | keep_system=True preserves system; False clears all | Empty history |

#### `core/session.py` — Undo/Redo (10 tests)

| Function | Tests | Edge Cases |
|----------|-------|-----------|
| `snapshot()` | Push to history; clear future on snapshot | Multiple snapshots |
| `undo()` | Restore previous messages; returns None when empty | Undo after zero snapshots |
| `redo()` | Re-applies undone state; returns None when empty | Redo after no undo |
| `can_undo()` / `can_redo()` | Correct booleans after each operation | Initial state |
| `history_depth()` | Counts undo steps | Deep stack |
| `status()` | Returns correct dict with all fields | — |

#### `core/export.py` — Export (6 tests)

| Function | Tests | Edge Cases |
|----------|-------|-----------|
| `render()` json preset | Valid JSON output; roundtrip-readable | — |
| `render()` md preset | Contains model name, role headers | Empty history |
| `render()` txt preset | Contains role labels | — |
| `render()` jsonl preset | One line per non-system message | System-only history |
| `render()` overwrite=False | Raises FileExistsError | — |
| `render()` invalid preset | Raises ValueError | — |

---

### E2E Test Plan (`test_full_e2e.py`)

**Prerequisite**: Ollama server running at `localhost:11434` with at least one model.
Tests will use `llama3.2` or whatever model is first in `ollama list`.

#### E2E — Backend API (8 tests)

| Test | What it verifies |
|------|-----------------|
| `test_server_health` | `is_available()` returns True; `health()` has version |
| `test_version` | `version()` returns non-empty string |
| `test_list_models` | Returns list; each item has `name` and `size` |
| `test_list_running` | Returns list (may be empty) |
| `test_show_model` | Returns dict with `details` key |
| `test_generate_streaming` | Yields chunks with `response` key; final chunk has `done=True` |
| `test_chat_streaming` | Yields chunks with `message.content`; accumulates to non-empty string |
| `test_embed` | Returns dict with `embeddings`; first embedding is a non-empty float list |

#### E2E — Full Workflow via CLI Subprocess (14 tests)

Uses `_resolve_cli("cli-anything-ollama")` — never hardcoded paths, no `cwd`.

| Test class | Tests | Verifies |
|-----------|-------|---------|
| `TestCLIHelp` | `--help`; `model --help`; `chat --help` | Return code 0; output contains group names |
| `TestCLIServerStatus` | `--json server status` | JSON with `status=ok`; has `version` key |
| `TestCLIModelList` | `--json model list` | JSON array of models |
| `TestCLISessionWorkflow` | new → send → info → history → export-md → export-jsonl → undo → redo → reset | Full session file lifecycle; file content verification |
| `TestCLIGenerate` | `--json generate run <prompt>` | JSON with `response` key; non-empty |
| `TestCLIEmbed` | `--json embed run <text>` | JSON with `dimensions > 0` |

---

### Realistic Workflow Scenarios

#### Scenario 1: Code Review Pipeline
**Simulates**: Developer using Ollama for automated code review
**Operations**:
1. `session new code-review.json --model deepseek-coder`
2. `chat send "Review this Python function: def fib(n): return fib(n-1)+fib(n-2)"`
3. `chat send "What's the time complexity?"`
4. `session export code-review.md --preset md`
**Verified**: Session file exists; MD export contains both user and assistant turns

#### Scenario 2: Embedding + Similarity (unit test)
**Simulates**: Building a semantic search pipeline
**Operations**:
1. `embed run "machine learning"` (model: nomic-embed-text)
2. `embed run "artificial intelligence"`
**Verified**: Both return 768-dimensional vectors; cosine similarity > 0.7

#### Scenario 3: Session Undo/Redo Round-trip
**Simulates**: Agent correcting a bad prompt
**Operations**:
1. Create session
2. Send message A → snapshot taken automatically
3. Undo → message A gone
4. Redo → message A restored
**Verified**: Message count correct at each step; JSONL export matches expected

---

## Part 2: Test Results

Last run: 2026-03-22

```
============================= test session starts ==============================
platform darwin -- Python 3.13.5, pytest-8.3.4

[_resolve_cli] Using installed command: /opt/anaconda3/bin/cli-anything-ollama

test_core.py::TestProjectCreate::test_create_defaults PASSED
test_core.py::TestProjectCreate::test_create_with_model PASSED
test_core.py::TestProjectCreate::test_create_with_system_prompt PASSED
test_core.py::TestProjectCreate::test_create_overwrites_existing PASSED
test_core.py::TestProjectCreate::test_create_raises_on_existing_no_overwrite PASSED
test_core.py::TestProjectCreate::test_create_nested_directory PASSED
test_core.py::TestProjectOpenSave::test_open_existing PASSED
test_core.py::TestProjectOpenSave::test_open_missing_raises PASSED
test_core.py::TestProjectOpenSave::test_save_updates_timestamp PASSED
test_core.py::TestProjectOpenSave::test_save_persists_messages PASSED
test_core.py::TestProjectInfo::test_info_empty_session PASSED
test_core.py::TestProjectInfo::test_info_with_messages PASSED
test_core.py::TestProjectInfo::test_info_snapshots_count PASSED
test_core.py::TestProjectListSessions::test_list_finds_sessions PASSED
test_core.py::TestProjectListSessions::test_list_skips_non_session_json PASSED
test_core.py::TestProjectListSessions::test_list_empty_directory PASSED
test_core.py::TestProjectListSessions::test_list_nonexistent_directory PASSED
test_core.py::TestProjectMessages::test_add_user_message PASSED
test_core.py::TestProjectMessages::test_add_with_images PASSED
test_core.py::TestProjectMessages::test_set_model PASSED
test_core.py::TestProjectMessages::test_set_system_no_prior PASSED
test_core.py::TestProjectMessages::test_set_system_replaces_prior PASSED
test_core.py::TestProjectMessages::test_set_option PASSED
test_core.py::TestProjectMessages::test_reset_history_keep_system PASSED
test_core.py::TestProjectMessages::test_reset_history_clear_all PASSED
test_core.py::TestSession::test_initial_state PASSED
test_core.py::TestSession::test_snapshot_enables_undo PASSED
test_core.py::TestSession::test_undo_restores_messages PASSED
test_core.py::TestSession::test_undo_empty_returns_none PASSED
test_core.py::TestSession::test_redo_after_undo PASSED
test_core.py::TestSession::test_redo_empty_returns_none PASSED
test_core.py::TestSession::test_snapshot_clears_redo PASSED
test_core.py::TestSession::test_status_dict PASSED
test_core.py::TestSession::test_multiple_snapshots PASSED
test_core.py::TestSession::test_hydrate_from_serialized PASSED
test_core.py::TestExport::test_export_json PASSED
test_core.py::TestExport::test_export_md PASSED
test_core.py::TestExport::test_export_txt PASSED
test_core.py::TestExport::test_export_jsonl PASSED
test_core.py::TestExport::test_export_no_overwrite_raises PASSED
test_core.py::TestExport::test_export_overwrite PASSED
test_core.py::TestExport::test_export_invalid_preset PASSED
test_core.py::TestExport::test_export_result_metadata PASSED
test_full_e2e.py::TestBackendAPI::test_server_health PASSED    [Ollama 0.18.2]
test_full_e2e.py::TestBackendAPI::test_version PASSED
test_full_e2e.py::TestBackendAPI::test_list_models PASSED      [llama3:latest]
test_full_e2e.py::TestBackendAPI::test_list_running PASSED
test_full_e2e.py::TestBackendAPI::test_show_model PASSED
test_full_e2e.py::TestBackendAPI::test_generate_streaming PASSED
test_full_e2e.py::TestBackendAPI::test_chat_streaming PASSED
test_full_e2e.py::TestBackendAPI::test_embed PASSED            [4096 dimensions]
test_full_e2e.py::TestCLISubprocess::test_help PASSED
test_full_e2e.py::TestCLISubprocess::test_model_help PASSED
test_full_e2e.py::TestCLISubprocess::test_server_status_json PASSED
test_full_e2e.py::TestCLISubprocess::test_model_list_json PASSED
test_full_e2e.py::TestCLISubprocess::test_session_new_json PASSED
test_full_e2e.py::TestCLISubprocess::test_generate_json PASSED
test_full_e2e.py::TestCLISubprocess::test_full_session_workflow PASSED
test_full_e2e.py::TestCLISubprocess::test_session_info_no_project PASSED
test_full_e2e.py::TestCLISubprocess::test_model_show_json PASSED
test_full_e2e.py::TestCLISubprocess::test_model_ps_json PASSED
test_full_e2e.py::TestCLISubprocess::test_embed_json PASSED
======================= 62 passed in 10.90s ============================
```

**Summary**: 62 passed, 0 failed, 100% pass rate, 10.90s
- Unit tests (test_core.py): 43/43
- E2E + subprocess tests (test_full_e2e.py): 19/19
- Backend: Ollama v0.18.2, model: llama3:latest
- Subprocess: confirmed `[_resolve_cli] Using installed command: /opt/anaconda3/bin/cli-anything-ollama`

**Coverage Notes**:
- All 3 core modules fully covered (project, session, export)
- Real LLM inference verified: generate, chat, embed
- Full session lifecycle tested end-to-end via subprocess
- `CLI_ANYTHING_FORCE_INSTALLED=1` verified the installed binary is used

---

## Test Results — Refine Run 1 (Multi-Model Support)

Last run: 2026-03-22

```
test_core.py::TestProjectCreate::test_create_defaults PASSED
test_core.py::TestProjectCreate::test_create_with_model PASSED
test_core.py::TestProjectCreate::test_create_with_system_prompt PASSED
test_core.py::TestProjectCreate::test_create_overwrites_existing PASSED
test_core.py::TestProjectCreate::test_create_raises_on_existing_no_overwrite PASSED
test_core.py::TestProjectCreate::test_create_nested_directory PASSED
test_core.py::TestProjectOpenSave::test_open_existing PASSED
test_core.py::TestProjectOpenSave::test_open_missing_raises PASSED
test_core.py::TestProjectOpenSave::test_save_updates_timestamp PASSED
test_core.py::TestProjectOpenSave::test_save_persists_messages PASSED
test_core.py::TestProjectInfo::test_info_empty_session PASSED
test_core.py::TestProjectInfo::test_info_with_messages PASSED
test_core.py::TestProjectInfo::test_info_snapshots_count PASSED
test_core.py::TestProjectListSessions::test_list_finds_sessions PASSED
test_core.py::TestProjectListSessions::test_list_skips_non_session_json PASSED
test_core.py::TestProjectListSessions::test_list_empty_directory PASSED
test_core.py::TestProjectListSessions::test_list_nonexistent_directory PASSED
test_core.py::TestProjectMessages::test_add_user_message PASSED
test_core.py::TestProjectMessages::test_add_with_images PASSED
test_core.py::TestProjectMessages::test_set_model PASSED
test_core.py::TestProjectMessages::test_set_system_no_prior PASSED
test_core.py::TestProjectMessages::test_set_system_replaces_prior PASSED
test_core.py::TestProjectMessages::test_set_option PASSED
test_core.py::TestProjectMessages::test_reset_history_keep_system PASSED
test_core.py::TestProjectMessages::test_reset_history_clear_all PASSED
test_core.py::TestSession::test_initial_state PASSED
test_core.py::TestSession::test_snapshot_enables_undo PASSED
test_core.py::TestSession::test_undo_restores_messages PASSED
test_core.py::TestSession::test_undo_empty_returns_none PASSED
test_core.py::TestSession::test_redo_after_undo PASSED
test_core.py::TestSession::test_redo_empty_returns_none PASSED
test_core.py::TestSession::test_snapshot_clears_redo PASSED
test_core.py::TestSession::test_status_dict PASSED
test_core.py::TestSession::test_multiple_snapshots PASSED
test_core.py::TestSession::test_hydrate_from_serialized PASSED
test_core.py::TestExport::test_export_json PASSED
test_core.py::TestExport::test_export_md PASSED
test_core.py::TestExport::test_export_txt PASSED
test_core.py::TestExport::test_export_jsonl PASSED
test_core.py::TestExport::test_export_no_overwrite_raises PASSED
test_core.py::TestExport::test_export_overwrite PASSED
test_core.py::TestExport::test_export_invalid_preset PASSED
test_core.py::TestExport::test_export_result_metadata PASSED
test_core.py::TestBackendHelpers::test_encode_image_roundtrip PASSED
test_core.py::TestBackendHelpers::test_encode_image_missing_file PASSED
test_core.py::TestBackendHelpers::test_compare_result_structure PASSED
test_core.py::TestBackendHelpers::test_compare_handles_model_error PASSED
test_core.py::TestBackendHelpers::test_capabilities_parses_list PASSED
test_core.py::TestBackendHelpers::test_capabilities_empty_model PASSED
test_full_e2e.py::TestBackendAPI::test_server_health PASSED
test_full_e2e.py::TestBackendAPI::test_version PASSED
test_full_e2e.py::TestBackendAPI::test_list_models PASSED
test_full_e2e.py::TestBackendAPI::test_list_running PASSED
test_full_e2e.py::TestBackendAPI::test_show_model PASSED
test_full_e2e.py::TestBackendAPI::test_generate_streaming PASSED
test_full_e2e.py::TestBackendAPI::test_chat_streaming PASSED
test_full_e2e.py::TestBackendAPI::test_embed PASSED
test_full_e2e.py::TestCLISubprocess::test_help PASSED
test_full_e2e.py::TestCLISubprocess::test_model_help PASSED
test_full_e2e.py::TestCLISubprocess::test_server_status_json PASSED
test_full_e2e.py::TestCLISubprocess::test_model_list_json PASSED
test_full_e2e.py::TestCLISubprocess::test_session_new_json PASSED
test_full_e2e.py::TestCLISubprocess::test_generate_json PASSED
test_full_e2e.py::TestCLISubprocess::test_full_session_workflow PASSED
test_full_e2e.py::TestCLISubprocess::test_session_info_no_project PASSED
test_full_e2e.py::TestCLISubprocess::test_model_show_json PASSED
test_full_e2e.py::TestCLISubprocess::test_model_ps_json PASSED
test_full_e2e.py::TestCLISubprocess::test_embed_json PASSED
test_full_e2e.py::TestCLISubprocess::test_model_search_json PASSED
test_full_e2e.py::TestCLISubprocess::test_model_search_no_results PASSED
test_full_e2e.py::TestCLISubprocess::test_model_capabilities_json PASSED
test_full_e2e.py::TestCLISubprocess::test_compare_run_json PASSED
test_full_e2e.py::TestCLISubprocess::test_chat_send_with_image PASSED
test_full_e2e.py::TestCLISubprocess::test_chat_tools_json SKIPPED (llama3:latest lacks tools cap)
test_full_e2e.py::TestCLISubprocess::test_help_includes_new_commands PASSED
```

**Summary**: 74 passed, 1 skipped, 0 failed — 100% pass rate, 22.35s
- Unit tests (test_core.py): 49/49 (6 new `TestBackendHelpers` tests added)
- E2E + subprocess tests (test_full_e2e.py): 25/25 passed, 1 skipped
- New commands tested: `model search`, `model capabilities`, `compare run`, `chat send --image`, `chat tools`
- tools test skipped: `llama3:latest` only has `completion` capability; skip logic is correct behavior
- Backend: Ollama v0.18.2, model: llama3:latest
- Search: scrapes `ollama.com/search?q=<query>` HTML (no public JSON API available)
