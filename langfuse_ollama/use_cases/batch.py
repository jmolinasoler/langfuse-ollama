"""
Caso de uso: batch de prompts JSONL — compartido por la UI (batch_tab) y el
CLI (trace_cli). Depende solo del dominio y del puerto ChatGateway.

Formato por línea: objeto JSON con clave obligatoria `prompt`; el resto de
claves (model, system, user_id, trace_name, tags, temperature, max_tokens)
sobreescriben los defaults para esa entrada.
"""

import json
from dataclasses import replace
from typing import Callable, List, Optional, Tuple

from langfuse_ollama.domain.entities import BatchDefaults, ChatMessage, new_session_id
from langfuse_ollama.domain.ports import ChatGateway
from langfuse_ollama.use_cases.chat import make_chat_request

OVERRIDE_KEYS = ("model", "system", "user_id", "trace_name", "temperature", "max_tokens")


def parse_jsonl(text: str) -> Tuple[List[Tuple[int, dict]], List[Tuple[int, str]]]:
    """
    Parsea el contenido JSONL. Retorna (entries, errors):
    entries = [(line_num, obj)], errors = [(line_num, mensaje)].
    """
    entries: List[Tuple[int, dict]] = []
    errors: List[Tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append((i, f"invalid JSON — {exc}"))
            continue
        if not isinstance(obj, dict) or "prompt" not in obj:
            errors.append((i, "missing 'prompt' key"))
            continue
        entries.append((i, obj))
    return entries, errors


def normalize_tags(value, fallback: List[str]) -> List[str]:
    """Acepta lista o string separado por comas; descarta vacíos."""
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    return fallback


def resolve_params(entry: dict, defaults: BatchDefaults) -> BatchDefaults:
    """Mergea las claves de la entrada sobre los defaults."""
    overrides = {k: entry[k] for k in OVERRIDE_KEYS if k in entry}
    if "tags" in entry:
        overrides["tags"] = normalize_tags(entry["tags"], defaults.tags)
    return replace(defaults, **overrides)


def run_entry(
    gateway: ChatGateway,
    line_num: int,
    entry: dict,
    defaults: BatchDefaults,
    on_chunk: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Ejecuta una entrada del batch en su propia sesión de trace.
    Con `on_chunk` la respuesta se consume en streaming, notificando cada
    chunk (p. ej. para imprimirlo en el CLI). Retorna un registro de
    resultado serializable (con `error` si la llamada falló).
    """
    p = resolve_params(entry, defaults)
    session_id = new_session_id()
    request = make_chat_request(
        history=[ChatMessage(role="user", content=entry["prompt"])],
        system_prompt=p.system,
        model=p.model,
        session_id=session_id,
        user_id=p.user_id,
        trace_name=p.trace_name,
        tags=p.tags,
        temperature=p.temperature,
        max_tokens=p.max_tokens,
    )

    result = {
        "line":       line_num,
        "session_id": session_id,
        "model":      p.model,
        "prompt":     entry["prompt"],
        "response":   None,
        "error":      None,
    }
    try:
        if on_chunk is not None:
            chunks = []
            for chunk in gateway.stream(request):
                on_chunk(chunk)
                chunks.append(chunk)
            result["response"] = "".join(chunks)
        else:
            result["response"] = gateway.complete(request)
    except Exception as exc:
        result["error"] = str(exc)
    return result
