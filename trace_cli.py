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
import ollama_client as oc


def _run_one(args, prompt, system, model, user_id, trace_name, tags, temperature, max_tokens):
    """Send a single prompt and return the full response string."""
    session_id = oc.new_session_id()
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ]

    print(f"\n[session: {session_id}] [{model}]\n{'─'*60}")

    if args.stream:
        chunks = []
        for chunk in oc.chat_stream(
            messages=messages,
            model=model,
            session_id=session_id,
            user_id=user_id,
            trace_name=trace_name,
            tags=tags,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        print(f"\n{'─'*60}\n[trace sent → Langfuse]")
        return session_id, "".join(chunks)
    else:
        reply = oc.chat_complete(
            messages=messages,
            model=model,
            session_id=session_id,
            user_id=user_id,
            trace_name=trace_name,
            tags=tags,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        print(reply)
        print(f"\n{'─'*60}\n[trace sent → Langfuse]")
        return session_id, reply


def _parse_tags(value, fallback):
    """Accept a list or comma-separated string; return a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [t.strip() for t in value.split(",")]
    return fallback


def main():
    oc.init_langfuse_env()
    parser = argparse.ArgumentParser(description="Langfuse × Ollama CLI Tracer")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt",     help="Single prompt string")
    group.add_argument("--batch-file", help="JSONL file — one prompt object per line")

    parser.add_argument("--output",      help="Write batch results to this JSONL file")
    parser.add_argument("--model",       default="llama3.1")
    parser.add_argument("--system",      default="You are a helpful assistant.")
    parser.add_argument("--user-id",     default="cli-user")
    parser.add_argument("--trace-name",  default="ollama-cli")
    parser.add_argument("--tags",        default="cli,ollama")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens",  type=int,   default=2048)
    parser.add_argument("--stream",      action="store_true",  default=True)
    parser.add_argument("--no-stream",   action="store_false", dest="stream")
    args = parser.parse_args()

    default_tags = _parse_tags(args.tags, ["cli", "ollama"])

    # ── Single mode ──────────────────────────────────────────────────────────
    if args.prompt:
        _run_one(
            args,
            prompt=args.prompt,
            system=args.system,
            model=args.model,
            user_id=args.user_id,
            trace_name=args.trace_name,
            tags=default_tags,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        return

    # ── Batch mode ───────────────────────────────────────────────────────────
    try:
        with open(args.batch_file, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except FileNotFoundError:
        print(f"Error: file not found: {args.batch_file}", file=sys.stderr)
        sys.exit(1)

    total = len(lines)
    print(f"[batch] {total} prompt(s) — default model: {args.model}")

    out_fh = open(args.output, "w", encoding="utf-8") if args.output else None

    try:
        for i, line in enumerate(lines, 1):
            print(f"\n[batch {i}/{total}]", end="")

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"line {i}: invalid JSON — {exc}"
                print(f"\n[batch] {msg}", file=sys.stderr)
                if out_fh:
                    out_fh.write(json.dumps({"line": i, "error": str(exc)}) + "\n")
                    out_fh.flush()
                continue

            if "prompt" not in entry:
                print(f"\n[batch] line {i}: missing 'prompt' key — skipping", file=sys.stderr)
                continue

            model       = entry.get("model",       args.model)
            system      = entry.get("system",      args.system)
            user_id     = entry.get("user_id",     args.user_id)
            trace_name  = entry.get("trace_name",  args.trace_name)
            tags        = _parse_tags(entry.get("tags", args.tags), default_tags)
            temperature = entry.get("temperature", args.temperature)
            max_tokens  = entry.get("max_tokens",  args.max_tokens)

            session_id, response = _run_one(
                args,
                prompt=entry["prompt"],
                system=system,
                model=model,
                user_id=user_id,
                trace_name=trace_name,
                tags=tags,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if out_fh:
                out_fh.write(json.dumps({
                    "line":       i,
                    "session_id": session_id,
                    "model":      model,
                    "prompt":     entry["prompt"],
                    "response":   response,
                }, ensure_ascii=False) + "\n")
                out_fh.flush()

    finally:
        if out_fh:
            out_fh.close()
            print(f"\n[batch] results written → {args.output}")

    print(f"\n[batch] done — {total} prompt(s) processed")


if __name__ == "__main__":
    main()
