#!/usr/bin/env python3
"""
CLI tracer — non-UI mode for scripted/batch tracing.
Usage:
  Single: python trace_cli.py --prompt "Explain MiCA regulation"
  Batch:  python trace_cli.py --batch-file prompts.jsonl [--output results.jsonl]

JSONL batch format (one JSON object per line):
  {"prompt": "What is X?"}
  {"prompt": "Compare A and B", "model": "mistral", "temperature": 0.2}
  {"prompt": "Summarize this", "system": "You are a concise summarizer.", "tags": ["test", "summary"]}

Supported per-entry keys: prompt (required), model, system, user_id, trace_name,
                           tags (list or comma-separated string), temperature, max_tokens
"""
import argparse
import json
import sys

import config
import batch_runner as br
import ollama_client as oc


def _streaming_chat(**kwargs):
    """chat_fn para batch_runner: imprime chunks a stdout y retorna el texto completo."""
    chunks = []
    for chunk in oc.chat_stream(**kwargs):
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print()
    return "".join(chunks)


def _run_entry(args, line_num, entry, defaults, client):
    print(f"\n[{entry.get('model', defaults.model)}]\n{'─'*60}")
    result = br.run_entry(
        line_num, entry, defaults,
        client=client,
        lf_public_key=config.LANGFUSE_PUBLIC_KEY or None,
        chat_fn=_streaming_chat if args.stream else None,
    )
    if result["error"]:
        print(f"Error: {result['error']}", file=sys.stderr)
    elif not args.stream:
        print(result["response"])
    print(f"{'─'*60}\n[session: {result['session_id']}]")
    return result


def main():
    parser = argparse.ArgumentParser(description="Langfuse × Ollama CLI Tracer")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt",     help="Single prompt string")
    group.add_argument("--batch-file", help="JSONL file — one prompt object per line")

    parser.add_argument("--output",      help="Write batch results to this JSONL file")
    parser.add_argument("--model",       default=config.DEFAULT_MODEL)
    parser.add_argument("--system",      default=config.DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-id",     default="cli-user")
    parser.add_argument("--trace-name",  default="ollama-cli")
    parser.add_argument("--tags",        default="cli,ollama")
    parser.add_argument("--stream",      action=argparse.BooleanOptionalAction, default=True,
                        help="Stream tokens to stdout (default: on)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens",  type=int,   default=2048)
    args = parser.parse_args()

    defaults = br.BatchDefaults(
        model=args.model,
        system=args.system,
        user_id=args.user_id,
        trace_name=args.trace_name,
        tags=br.normalize_tags(args.tags, ["cli", "ollama"]),
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    client = oc.get_chat_client(
        ollama_url=config.OLLAMA_BASE_URL,
        lf_public_key=config.LANGFUSE_PUBLIC_KEY,
        lf_secret_key=config.LANGFUSE_SECRET_KEY,
        lf_host=config.LANGFUSE_BASE_URL,
    )

    # ── Single mode ──────────────────────────────────────────────────────────
    if args.prompt:
        _run_entry(args, 1, {"prompt": args.prompt}, defaults, client)
        _finish()
        return

    # ── Batch mode ───────────────────────────────────────────────────────────
    try:
        with open(args.batch_file, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Error: file not found: {args.batch_file}", file=sys.stderr)
        sys.exit(1)

    entries, parse_errors = br.parse_jsonl(text)
    total = len(entries)
    print(f"[batch] {total} prompt(s) — default model: {args.model}")

    out_fh = open(args.output, "w", encoding="utf-8") if args.output else None

    try:
        for line_num, msg in parse_errors:
            print(f"[batch] line {line_num}: {msg} — skipping", file=sys.stderr)
            if out_fh:
                out_fh.write(json.dumps({"line": line_num, "error": msg}) + "\n")
                out_fh.flush()

        for i, (line_num, entry) in enumerate(entries, 1):
            print(f"\n[batch {i}/{total}]", end="")
            result = _run_entry(args, line_num, entry, defaults, client)
            if out_fh:
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_fh.flush()
    finally:
        if out_fh:
            out_fh.close()
            print(f"\n[batch] results written → {args.output}")

    print(f"\n[batch] done — {total} prompt(s) processed")
    _finish()


def _finish():
    """Garantiza que los traces bufferizados salen antes de terminar el proceso."""
    if config.langfuse_configured():
        oc.flush(config.LANGFUSE_PUBLIC_KEY)
        print("[traces flushed → Langfuse]")


if __name__ == "__main__":
    main()
