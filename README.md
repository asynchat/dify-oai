# Dify OpenAI Compatibility Proxy (`dify_oai.py`)

[FastAPI](https://fastapi.tiangolo.com/) router that exposes your [Dify](https://dify.ai/) applications as an **OpenAI Chat Completions**-compatible API. Use it with the OpenAI Python SDK, Cursor, or any client that speaks the OpenAI `/v1` protocol.

Supports both **agent-chat** and **advanced-chat (Chatflow)** apps, including Agent nodes with MCP/tool workflows.

## Features

- **OpenAI-compatible endpoints**
  - `POST /v1/chat/completions` — chat (streaming and non-streaming)
  - `GET /v1/models` — list models (model name = Dify app name)
- **Tool calling**
  - Strict OpenAI `tool_calls` format (`id`, `type`, `function.name`, `function.arguments`)
  - Streaming tool-call deltas (incremental `arguments`, standard OpenAI semantics)
  - Optional `include_tool_extensions` for non-standard `tool_results` (input/output)
- **Dify app types**
  - `agent-chat` — parses `agent_thought` events
  - `advanced-chat` / Chatflow — parses `agent_log` (ROUND / Thought / CALL) and `node_finished`
- **Streaming behavior aligned with Chatflow**
  - Forwards Answer-node `message` content (including `<think>` blocks)
  - Skips duplicate `agent_message` mid-stream to avoid repeated thinking output
  - Ends cleanly on `message_end` / `workflow_finished` with `[DONE]`
- **Non-streaming mode**
  - Internally uses Dify streaming to collect full answers and tool events (required for advanced-chat)
- **Conversation memory modes**
  - `history_message` (default) — embeds history in the user query
  - Zero-width character mode — encodes `conversation_id` in assistant output
- **Debugging**
  - Optional raw Dify SSE event logging to console and JSONL file

## Quick Start

### 1. Mount the router

```python
from app.routers.dify_oai import router as dify_oai_router

app.include_router(dify_oai_router)
```

### 2. Configure environment variables

```env
# Required
DIFY_API_BASE=http://localhost/v1
DIFY_API_KEYS=app-xxxxxxxx
VALID_API_KEYS=sk-your-client-key

# Optional
DIFY_DEFAULT_MODEL=MyApp
CONVERSATION_MEMORY_MODE=1
DIFY_RAW_EVENT_LOG=0
DIFY_RAW_EVENT_LOG_FILE=logs/dify_raw_events.jsonl
```

| Variable | Description |
|----------|-------------|
| `DIFY_API_BASE` | Dify API base URL |
| `DIFY_API_KEYS` | Comma-separated Dify app API keys |
| `VALID_API_KEYS` | Comma-separated keys clients use in `Authorization: Bearer ...` |
| `DIFY_DEFAULT_MODEL` | Fallback model name when `/info` is unavailable (single-key setups) |
| `CONVERSATION_MEMORY_MODE` | `1` = history in query (default), `2` = zero-width `conversation_id` |
| `DIFY_RAW_EVENT_LOG` | `1` to log raw Dify SSE events for debugging |

`VALID_API_KEYS` and `DIFY_API_KEYS` are intentionally separate: one authenticates **clients**, the other calls **Dify**.

### 3. Call with the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-client-key",
    base_url="http://localhost:8000/v1",
)

# Strict OpenAI mode (default)
response = client.chat.completions.create(
    model="MyApp",  # must match Dify app name
    messages=[{"role": "user", "content": "Hello"}],
)

# With tool result extensions (non-standard)
response = client.chat.completions.create(
    model="MyApp",
    messages=[{"role": "user", "content": "Analyze this."}],
    extra_body={"include_tool_extensions": True},
)

# Streaming
stream = client.chat.completions.create(
    model="MyApp",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
    if delta.tool_calls:
        print(delta.tool_calls)
```

## Authentication

Clients must send:

```http
Authorization: Bearer <one of VALID_API_KEYS>
```

## `include_tool_extensions`

Request body field (not forwarded to Dify):

```json
{ "include_tool_extensions": false }
```

| Mode | Behavior |
|------|----------|
| `false` (default) | Strict OpenAI compatibility — `tool_calls` only |
| `true` | Also returns `tool_results` with `tool_call_id`, `name`, `input`, `output` |

- **Non-streaming:** `choices[].tool_results[]`
- **Streaming:** `delta.tool_results[]`

## Model mapping

On startup, the router calls Dify `/info` to map app names to API keys.

- If `/info` fails and only one `DIFY_API_KEYS` is configured, any `model` name falls back to that key.
- With multiple keys, set `DIFY_DEFAULT_MODEL` or call `GET /v1/models`.

## Streaming vs blocking

| Client `stream` | Behavior |
|-----------------|----------|
| `true` | Proxies Dify SSE → OpenAI chunks; tool calls from `agent_log`; content from Answer `message` events |
| `false` | Collects via internal streaming, then returns one JSON response |

advanced-chat blocking responses do not include full agent/tool metadata, so non-streaming mode always collects over SSE internally.

## Tool event sources

| Dify event | Use case |
|------------|----------|
| `agent_thought` | agent-chat apps |
| `agent_log` | Chatflow Agent node (ROUND, Thought, CALL) |
| `node_finished` | Tool / MCP / HTTP workflow nodes |

## Debugging

Enable raw event logging:

```env
DIFY_RAW_EVENT_LOG=1
```

Logs appear in the **server** terminal (not the test client) and in:

```
logs/dify_raw_events.jsonl
```

Useful events: `agent_log`, `agent_thought`, `node_finished`.

## Requirements

- Python 3.10+
- FastAPI
- httpx
- python-dotenv
- openai (for client tests)

## License

MIT.
