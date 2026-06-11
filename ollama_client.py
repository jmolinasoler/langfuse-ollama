"""
Langfuse-instrumented Ollama client via OpenAI-compatible API.

Langfuse v3: el wrapper `langfuse.openai.OpenAI` solo extrae `name`,
`metadata` y `langfuse_public_key` de cada `.create()`. Los atributos de
trace (session_id, user_id, tags) se fijan con el patrón v3: un span raíz
vía `start_as_current_span()` + `update_trace()` — ver _trace_root().
No se mutan variables de entorno: las credenciales se registran
instanciando `Langfuse(...)` y cada llamada se enruta con
`langfuse_public_key`.
"""

import threading
import uuid
import logging
from contextlib import contextmanager
from typing import Generator, Optional

import httpx

import config

logger = logging.getLogger(__name__)

# Cache de clientes por (ollama_url, credenciales Langfuse). Streamlit re-ejecuta
# el script en cada interacción y el CLI procesa lotes: un cliente por combinación
# preserva el pool de conexiones y evita re-inicializar la instrumentación.
_clients: dict = {}
_clients_lock = threading.Lock()


def get_chat_client(
    ollama_url: Optional[str] = None,
    lf_public_key: str = "",
    lf_secret_key: str = "",
    lf_host: str = "",
):
    """Retorna un cliente OpenAI (Langfuse-wrapped) cacheado para esa configuración."""
    key = (
        ollama_url or config.OLLAMA_BASE_URL,
        lf_public_key,
        lf_secret_key,
        lf_host,
    )
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            client = _build_client(*key)
            _clients[key] = client
    return client


def _build_client(ollama_url: str, lf_public_key: str, lf_secret_key: str, lf_host: str):
    from langfuse import Langfuse
    from langfuse.openai import OpenAI

    if lf_public_key and lf_secret_key:
        # Registra (o recupera) la instancia Langfuse para esta public_key.
        # El wrapper la resuelve después vía get_client(public_key=...).
        Langfuse(
            public_key=lf_public_key,
            secret_key=lf_secret_key,
            host=lf_host or config.LANGFUSE_BASE_URL,
        )
    return OpenAI(base_url=f"{ollama_url}/v1", api_key="ollama")


def ping(base_url: Optional[str] = None) -> bool:
    """Comprueba si Ollama responde. No confundir con list_models()."""
    url = base_url or config.OLLAMA_BASE_URL
    try:
        r = httpx.get(f"{url}/api/version", timeout=3)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Ollama unreachable at %s: %s", url, e)
        return False


def list_models(base_url: Optional[str] = None) -> list:
    """Lista modelos disponibles en Ollama. Retorna [] si no responde."""
    url = base_url or config.OLLAMA_BASE_URL
    try:
        r = httpx.get(f"{url}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        logger.warning("Failed to list Ollama models: %s", e)
        return []


@contextmanager
def _trace_root(
    trace_name: str,
    session_id: str,
    user_id: str,
    tags: list,
    lf_public_key: Optional[str],
):
    """
    Span raíz que agrupa la generación y fija los atributos de trace.
    Patrón Langfuse v3: el wrapper OpenAI anida su observación bajo el
    span OTel activo; session_id/user_id/tags se fijan con update_trace().
    Sin credenciales no hay tracing → no-op.
    """
    if not lf_public_key:
        yield
        return
    from langfuse import get_client
    lf = get_client(public_key=lf_public_key)
    with lf.start_as_current_span(name=trace_name) as span:
        span.update_trace(
            name=trace_name,
            session_id=session_id,
            user_id=user_id,
            tags=tags,
        )
        yield


def _chat_kwargs(
    messages: list,
    model: str,
    trace_name: str,
    temperature: float,
    max_tokens: int,
    lf_public_key: Optional[str],
) -> dict:
    kwargs: dict = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        # `name` y `langfuse_public_key` son los únicos kwargs Langfuse que
        # el wrapper v3 extrae antes de reenviar a OpenAI.
        name=trace_name,
    )
    if lf_public_key:
        kwargs["langfuse_public_key"] = lf_public_key
    return kwargs


def chat_complete(
    messages: list,
    model: str,
    session_id: str,
    user_id: str,
    trace_name: str = "ollama-chat",
    tags: Optional[list] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    client=None,
    lf_public_key: Optional[str] = None,
) -> str:
    c = client if client is not None else get_chat_client()
    kwargs = _chat_kwargs(messages, model, trace_name, temperature, max_tokens, lf_public_key)
    with _trace_root(trace_name, session_id, user_id,
                     tags or ["ollama", model], lf_public_key):
        response = c.chat.completions.create(**kwargs)
        return response.choices[0].message.content


def chat_stream(
    messages: list,
    model: str,
    session_id: str,
    user_id: str,
    trace_name: str = "ollama-chat",
    tags: Optional[list] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    client=None,
    lf_public_key: Optional[str] = None,
) -> Generator[str, None, None]:
    c = client if client is not None else get_chat_client()
    kwargs = _chat_kwargs(messages, model, trace_name, temperature, max_tokens, lf_public_key)
    kwargs["stream"] = True
    with _trace_root(trace_name, session_id, user_id,
                     tags or ["ollama", model, "stream"], lf_public_key):
        stream = c.chat.completions.create(**kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield delta


def flush(lf_public_key: Optional[str] = None) -> None:
    """Fuerza el envío de traces bufferizados. Necesario antes de salir en CLI."""
    try:
        from langfuse import get_client
        get_client(public_key=lf_public_key or None).flush()
    except Exception as e:
        logger.warning("Langfuse flush failed: %s", e)


def new_session_id() -> str:
    return str(uuid.uuid4())
