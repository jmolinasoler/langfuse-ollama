#!/usr/bin/env python3
"""
CLI tracer — non-UI mode for scripted/batch tracing.
Usage: python trace_cli.py --model llama3.1 --prompt "Explain MiCA regulation"
"""
import argparse
import sys
import ollama_client as oc


def main():
    oc.init_langfuse_env()
    parser = argparse.ArgumentParser(description="Langfuse × Ollama CLI Tracer")
    parser.add_argument("--model",       default="llama3.1")
    parser.add_argument("--prompt",      required=True)
    parser.add_argument("--system",      default="You are a helpful assistant.")
    parser.add_argument("--user-id",     default="cli-user")
    parser.add_argument("--trace-name",  default="ollama-cli")
    parser.add_argument("--tags",        default="cli,ollama")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens",  type=int, default=2048)
    parser.add_argument("--stream",      action="store_true", default=True)
    parser.add_argument("--no-stream",   action="store_false", dest="stream")
    args = parser.parse_args()

    session_id = oc.new_session_id()
    tags = [t.strip() for t in args.tags.split(",")]

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user",   "content": args.prompt},
    ]

    print(f"\n[session: {session_id}] [{args.model}]\n{'─'*60}")

    if args.stream:
        for chunk in oc.chat_stream(
            messages=messages,
            model=args.model,
            session_id=session_id,
            user_id=args.user_id,
            trace_name=args.trace_name,
            tags=tags,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ):
            print(chunk, end="", flush=True)
        print(f"\n{'─'*60}\n[trace sent → Langfuse]")
    else:
        reply = oc.chat_complete(
            messages=messages,
            model=args.model,
            session_id=session_id,
            user_id=args.user_id,
            trace_name=args.trace_name,
            tags=tags,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(reply)
        print(f"\n{'─'*60}\n[trace sent → Langfuse]")


if __name__ == "__main__":
    main()
