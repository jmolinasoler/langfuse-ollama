"""
Langfuse-instrumented Ollama client via OpenAI-compatible API.
Langfuse v4: usa propagate_attributes() (OTel context manager) para session_id,
user_id y tags. Solo `name` se pasa directamente a .create() — es el único
kwarg Langfuse que el wrapper extrae antes de reenviar a OpenAI.
"""

import os
import uuid
import logging
from typing import Generator, Optional
import httpx

import config

logger = logging.getLogger(__name__)


def init_langfuse_env():
    """
    Configura las variables de entorno que Langfuse SDK necesita.
    Debe llamarse explícitamente antes de usar el client — no como side effect de import.
    """
    os.environ["LANGFUSE_PUBLIC_KEY"] = config.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = config.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_BASE_URL"]   = config.LANGFUSE_BASE_URL
    os.environ["OPENAI_API_KEY"]      = "ollama"


def _get_openai_cls():
    """Import lazy del wrapper Langfuse — evita side effects al importar el módulo."""
    from langfuse.openai import OpenAI
    return OpenAI


def _get_propagate_attributes():
    """Import lazy de propagate_attributes."""
    from langfuse import propagate_attributes
    return propagate_attributes


def _make_client(client=None):
    """
    Retorna un OpenAI client. Acepta inyección para testing.
    Si no se pasa client, crea uno nuevo con Langfuse wrapping.
    """
    if client is not None:
        return client
    OpenAI = _get_openai_cls()
    return OpenAI(
        base_url=f"{config.OLLAMA_BASE_URL}/v1",
        api_key="ollama",
    )


def list_models() -> list:
    """Lista modelos disponibles en Ollama. Retorna fallback con logging si falla."""
    try:
        r = httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        logger.warning("Failed to list Ollama models: %s", e)
        return [config.DEFAULT_MODEL]


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
) -> str:
    c = _make_client(client)

    openai_kwargs: dict = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if client is not None:
        response = c.chat.completions.create(**openai_kwargs)
        return response.choices[0].message.content

    # `name` es el único kwarg Langfuse-específico que el wrapper extrae antes de
    # reenviar a OpenAI. session_id/user_id/tags van por OTel context.
    openai_kwargs["name"] = trace_name
    propagate_attributes = _get_propagate_attributes()
    with propagate_attributes(
        session_id=session_id,
        user_id=user_id,
        tags=tags or ["ollama", model],
        trace_name=trace_name,
    ):
        response = c.chat.completions.create(**openai_kwargs)
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
) -> Generator[str, None, None]:
    c = _make_client(client)

    openai_kwargs: dict = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    if client is not None:
        stream = c.chat.completions.create(**openai_kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield delta
        return

    openai_kwargs["name"] = trace_name
    propagate_attributes = _get_propagate_attributes()
    with propagate_attributes(
        session_id=session_id,
        user_id=user_id,
        tags=tags or ["ollama", model, "stream"],
        trace_name=trace_name,
    ):
        stream = c.chat.completions.create(**openai_kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                yield delta


def new_session_id() -> str:
    return str(uuid.uuid4())
