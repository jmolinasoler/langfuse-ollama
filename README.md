# Langfuse × Ollama Tracer

> Full observability for local [Ollama](https://ollama.com) models via [Langfuse](https://langfuse.com) using the OpenAI-compatible drop-in wrapper.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Langfuse](https://img.shields.io/badge/Langfuse-v3-blueviolet)](https://langfuse.com)

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

The codebase follows [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html): source-code dependencies only point inward, and the inner rings know nothing about Streamlit, httpx, OpenAI or Langfuse.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Frameworks & Drivers                                               │
│  app.py + langfuse_ollama/ui/* (Streamlit) · trace_cli.py (CLI)     │
│  config.py (env defaults) — composition roots build the adapters    │
├─────────────────────────────────────────────────────────────────────┤
│  Interface Adapters            langfuse_ollama/adapters/            │
│  LangfuseOllamaGateway  ──►  langfuse.openai wrapper + root span    │
│  OllamaApi              ──►  httpx → Ollama REST API                │
├─────────────────────────────────────────────────────────────────────┤
│  Use Cases                     langfuse_ollama/use_cases/           │
│  chat.make_chat_request / complete_turn / stream_turn               │
│  batch.parse_jsonl / resolve_params / run_entry                     │
├─────────────────────────────────────────────────────────────────────┤
│  Entities + Ports              langfuse_ollama/domain/              │
│  ChatMessage · ChatRequest · BatchDefaults · new_session_id         │
│  ChatGateway · ModelCatalog  (typing.Protocol boundaries)           │
└─────────────────────────────────────────────────────────────────────┘
        dependencies point downward (inward) only — the Dependency Rule
```

**Key design decisions:**

- **The Dependency Rule** — `domain/` is stdlib-only; `use_cases/` imports only the domain; `adapters/` implement the domain ports (`ChatGateway`, `ModelCatalog`) with the real SDKs; the UI and CLI are composition roots that wire adapters into use cases. Use cases are tested with an in-memory `FakeGateway`, no mocking of SDK modules.
- **Session-scoped config, no global mutation** — the sidebar builds an immutable `SidebarConfig` per rerun; neither `config.*` module globals nor `os.environ` are mutated, so concurrent Streamlit sessions can't leak credentials into each other. Langfuse credentials are registered by instantiating `Langfuse(...)` explicitly and each call is routed with the `langfuse_public_key` kwarg.
- **Cached gateways** — `adapters.langfuse_ollama.get_gateway()` caches one gateway per (Ollama URL, Langfuse credentials) combination, preserving connection pooling across reruns and batch entries.
- **Langfuse v3 trace attributes** — the OpenAI wrapper only extracts `name`, `metadata` and `langfuse_public_key` from `.create()`. `session_id`, `user_id` and `tags` are set with the v3 pattern: a root span via `start_as_current_span()` + `update_trace()` (see `LangfuseOllamaGateway._trace_root()`); the wrapper's generation nests under it.
- **Shared batch use case** — `use_cases/batch.py` holds JSONL parsing and per-entry default merging, used by both the Streamlit batch tab and the CLI; the CLI streams by passing an `on_chunk` callback.

---

## Requirements

- Python 3.9+
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

### Quick Start with Makefile

Alternatively, use the provided Makefile for a streamlined workflow:

```bash
make install    # Create virtualenv and install dependencies
source .venv/bin/activate  # Load virtualenv (or: eval $(make load))
make run        # Start the Streamlit app
make clean      # Remove virtualenv and build artifacts
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
- **📋 Batch** — upload a JSONL file, preview parsed prompts, run them sequentially with a live progress bar (cancellable mid-run, partial results are kept), inspect each result in expandable cards, and download `results.jsonl`.

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
├── app.py                        # Entry point: Streamlit — layout, session state, tabs
├── trace_cli.py                  # Entry point: CLI — single prompt or JSONL batch
├── langfuse_ollama/              # Application package (Clean Architecture rings)
│   ├── config.py                 # Env-var defaults (python-dotenv)
│   ├── domain/                   # Entities + ports — stdlib only
│   │   ├── entities.py           # ChatMessage, ChatRequest, BatchDefaults, new_session_id
│   │   └── ports.py              # ChatGateway, ModelCatalog (typing.Protocol)
│   ├── use_cases/                # Application rules — depend on domain only
│   │   ├── chat.py               # make_chat_request, complete_turn, stream_turn
│   │   └── batch.py              # parse_jsonl, resolve_params, run_entry
│   ├── adapters/                 # Port implementations over real SDKs
│   │   ├── langfuse_ollama.py    # LangfuseOllamaGateway + cache + flush
│   │   └── ollama_api.py         # OllamaApi: ping, list_models (httpx)
│   └── ui/                       # Frameworks & drivers (Streamlit)
│       ├── sidebar.py            # Sidebar controls → immutable SidebarConfig
│       ├── chat_tab.py           # Interactive chat tab (streaming + non-streaming)
│       ├── batch_tab.py          # JSONL batch tab (incremental, cancellable runs)
│       ├── feedback_widget.py    # Featurebase feedback widget (anonymous boot)
│       └── styles.py             # Custom CSS
├── pyproject.toml                # Project metadata + pinned dependencies
├── requirements.txt              # Python dependencies (mirror of pyproject)
├── .env.example                  # Environment variable template
├── prompts.example.jsonl         # Sample batch file showing all supported keys
└── tests/
    ├── test_domain.py            # Entities (session ids)
    ├── test_use_cases_chat.py    # ChatRequest construction
    ├── test_use_cases_batch.py   # Batch parsing/merging/running (FakeGateway)
    ├── test_adapters.py          # OllamaApi + LangfuseOllamaGateway (DI fakes)
    ├── test_config.py            # Env-var defaults
    ├── test_feedback_widget.py   # Featurebase boot snippet
    └── test_trace_cli.py         # CLI wiring (real main(), FakeGateway)
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
| `langfuse>=3.7,<4` | LLM observability SDK (provides the `langfuse.openai.OpenAI` wrapper) |
| `openai>=1.0,<3` | OpenAI-compatible client (used by Langfuse wrapper) |
| `streamlit>=1.45,<2` | Web UI framework |
| `python-dotenv>=1.0,<2` | `.env` file loading |
| `httpx>=0.27,<1` | HTTP client for Ollama health check and model listing |

---

## License

[MIT](LICENSE) © 2026 — see [LICENSE](LICENSE) for full text.
