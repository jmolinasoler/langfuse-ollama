# Langfuse × Ollama Tracer

Full observability for local Ollama models via Langfuse using the OpenAI-compatible drop-in.

## Stack

- `langfuse.openai.OpenAI` → drop-in replacement, zero-overhead tracing
- Ollama OpenAI-compatible API at `localhost:11434/v1`
- Streamlit UI + CLI mode

## Setup

```bash
cp .env.example .env
# Fill: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

pip install -r requirements.txt
ollama serve  # ensure running
```

## Run

```bash
# UI
streamlit run app.py

# CLI
python trace_cli.py --model llama3.1 --prompt "Explain MiCA Article 16"
python trace_cli.py --model mistral  --prompt "What is a CASP?" --tags "mica,compliance"
```

## What Gets Traced

| Field | Source |
|-------|--------|
| `session_id` | UUID per chat session (groups multi-turn) |
| `user_id` | Configurable per request |
| `trace_name` | Configurable (default: `ollama-chat`) |
| `tags` | Model name + custom |
| Input tokens | Counted by Langfuse SDK |
| Output tokens | Counted by Langfuse SDK |
| Latency | Automatic |
| Stream chunks | Full reconstruction |

## Langfuse Self-Hosted

Set `LANGFUSE_BASE_URL=http://localhost:3000` (Docker Compose default).  
Guide: https://langfuse.com/self-hosting/deployment/docker-compose
