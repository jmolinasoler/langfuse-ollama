# Langfuse × Ollama Tracer

> Full observability for local [Ollama](https://ollama.com) models via [Langfuse](https://langfuse.com) using the OpenAI-compatible drop-in wrapper.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Langfuse](https://img.shields.io/badge/Langfuse-v4-blueviolet)](https://langfuse.com)

---

## Overview

This project wires **Ollama** (local LLM runtime) to **Langfuse** (LLM observability platform) with minimal overhead. Every chat turn — whether streaming or non-streaming — is automatically traced: tokens, latency, session grouping, custom tags, and full message history.

Two interfaces are provided:

| Interface | File | Use case |
|-----------|------|----------|
| **Streamlit UI** | `app.py` | Interactive multi-turn chat + JSONL batch runner with live sidebar controls |
| **CLI** | `trace_cli.py` | Single prompt or JSONL batch tracing from the terminal |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  app.py  (Streamlit UI)   /   trace_cli.py  (CLI)   │
└───────────────────────┬─────────────────────────────┘
                        │ calls
                        ▼
              ollama_client.py
          ┌───────────────────────┐
          │  chat_complete()      │  ──► langfuse.openai.OpenAI (wrapper)
          │  chat_stream()        │  ──► propagate_attributes() [OTel ctx]
          │  list_models()        │  ──► httpx → Ollama /api/tags
          │  new_session_id()     │
          └───────────────────────┘
                        │
                    config.py
          (env vars via python-dotenv)
```

**Key design decisions:**

- **Lazy imports** — `langfuse.openai.OpenAI` and `propagate_attributes` are imported only when a real request is made, so the module can be imported safely in tests without side effects.
- **Dependency injection** — both `chat_complete` and `chat_stream` accept an optional `client` argument; tests pass a fake client instead of mocking the module.
- **Langfuse v4 OTel context** — `session_id`, `user_id`, and `tags` are propagated via `propagate_attributes()` (OpenTelemetry context manager). Only `name` is passed directly to `.create()` as a Langfuse-specific kwarg.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (`ollama serve`)
- A [Langfuse](https://langfuse.com) account (Cloud or self-hosted)

---

## Installation

```bash
git clone https://github.com/<your-username>/langfuse-ollama.git
cd langfuse-ollama

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Configuration

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LANGFUSE_PUBLIC_KEY` | ✅ | — | Langfuse project public key (`pk-lf-…`) |
| `LANGFUSE_SECRET_KEY` | ✅ | — | Langfuse project secret key (`sk-lf-…`) |
| `LANGFUSE_BASE_URL` | | `https://cloud.langfuse.com` | Region endpoint (see below) |
| `OLLAMA_BASE_URL` | | `http://localhost:11434` | Ollama server URL |
| `DEFAULT_MODEL` | | `llama3.1` | Pre-selected model in the UI |
| `DEFAULT_SYSTEM_PROMPT` | | `You are a helpful assistant.` | Default system prompt |
| `DEFAULT_USER_ID` | | `user` | Default user ID for traces |

**Langfuse regions:**

| Region | URL |
|--------|-----|
| EU Cloud (default) | `https://cloud.langfuse.com` |
| US Cloud | `https://us.cloud.langfuse.com` |
| JP Cloud | `https://jp.cloud.langfuse.com` |
| HIPAA | `https://hipaa.cloud.langfuse.com` |
| Self-hosted | `http://localhost:3000` |

---

## Usage

### Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501`. The sidebar lets you override credentials, switch models, set system prompt, tags, temperature, and max tokens — all without restarting.

The UI has two tabs:

- **💬 Chat** — interactive multi-turn conversation with streaming output.
- **📋 Batch** — upload a JSONL file, preview parsed prompts, run them sequentially with a live progress bar, inspect each result in expandable cards, and download `results.jsonl`.

### CLI — single prompt

```bash
# Streaming (default)
python trace_cli.py --prompt "Explain MiCA Article 16"

# Non-streaming, full options
python trace_cli.py \
  --model llama3.1 \
  --prompt "Summarize the ERC-4626 vault standard" \
  --system "You are a DeFi expert." \
  --user-id alice \
  --trace-name "defi-research" \
  --tags "defi,erc4626" \
  --temperature 0.5 \
  --max-tokens 1024 \
  --no-stream
```

### CLI — batch mode

Feed a JSONL file where each line is a prompt object. The `prompt` key is required; all others are optional and override the CLI defaults for that entry.

```jsonc
// prompts.jsonl
{"prompt": "What is the capital of France?"}
{"prompt": "Compare REST vs GraphQL", "model": "mistral", "temperature": 0.2}
{"prompt": "Translate 'hello' to Spanish", "system": "Reply only with the translation.", "tags": ["test", "i18n"]}
```

```bash
# Run and print to stdout
python trace_cli.py --batch-file prompts.jsonl

# Run and save results
python trace_cli.py --batch-file prompts.jsonl --output results.jsonl

# Override default model for entries that don't specify one
python trace_cli.py --batch-file prompts.jsonl --model mistral --no-stream --output results.jsonl
```

Each line in `results.jsonl` contains `line`, `session_id`, `model`, `prompt`, `response` (and `error` if the request failed). Invalid or malformed lines are written as error records and execution continues.

Supported per-entry keys: `prompt`, `model`, `system`, `user_id`, `trace_name`, `tags` (list or comma-separated string), `temperature`, `max_tokens`.

---

## What Gets Traced

Every request sends the following to Langfuse automatically:

| Field | Source |
|-------|--------|
| `session_id` | UUID per chat session (groups multi-turn conversations) |
| `user_id` | Configurable per request |
| `trace_name` | Configurable (default: `ollama-chat` / `ollama-cli`) |
| `tags` | Model name + custom tags |
| Input tokens | Counted by Langfuse SDK |
| Output tokens | Counted by Langfuse SDK |
| Latency | Automatic (wall-clock) |
| Stream chunks | Full reconstruction into single trace |

---

## Project Structure

```
langfuse-ollama/
├── app.py                    # Streamlit UI — chat tab + batch tab
├── trace_cli.py              # CLI tracer — single prompt or JSONL batch
├── ollama_client.py          # Core client: chat_complete, chat_stream, list_models
├── config.py                 # Env-var loader (python-dotenv)
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── prompts.example.jsonl     # Sample batch file showing all supported keys
└── tests/
    ├── test_config.py          # Unit tests for config module
    ├── test_ollama_client.py   # Unit tests for ollama_client (DI-based fakes)
    └── test_trace_cli.py       # Unit tests for CLI argument parsing and dispatch
```

---

## Running Tests

Tests use Python's built-in `unittest` — no external test framework required.

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run a specific module
python -m unittest tests.test_ollama_client -v
```

The test suite uses **dependency injection** (fake client objects) instead of heavy mocking, keeping tests fast and dependency-free.

---

## Self-Hosted Langfuse

```bash
# docker-compose.yml from Langfuse docs
docker compose up -d
```

Then set:

```env
LANGFUSE_BASE_URL=http://localhost:3000
```

Full guide: [langfuse.com/self-hosting/deployment/docker-compose](https://langfuse.com/self-hosting/deployment/docker-compose)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `langfuse>=2.0.0` | LLM observability SDK (provides `langfuse.openai.OpenAI` wrapper and `propagate_attributes`) |
| `openai>=1.0.0` | OpenAI-compatible client (used by Langfuse wrapper) |
| `streamlit>=1.35.0` | Web UI framework |
| `python-dotenv>=1.0.0` | `.env` file loading |
| `httpx>=0.27.0` | HTTP client for Ollama model listing |

---

## License

[MIT](LICENSE) © 2026 — see [LICENSE](LICENSE) for full text.
