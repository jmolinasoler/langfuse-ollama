"""
Adapter del puerto ChatGateway: Ollama vía API OpenAI-compatible, con
instrumentación Langfuse.

Langfuse v3: el wrapper `langfuse.openai.OpenAI` solo extrae `name`,
`metadata` y `langfuse_public_key` de cada `.create()`. Los atributos de
trace (session_id, user_id, tags) se fijan con el patrón v3: un span raíz
vía `start_as_current_span()` + `update_trace()` — ver _trace_root().
No se mutan variables de entorno: las credenciales se registran
instanciando `Langfuse(...)` y cada llamada se enruta con
`langfuse_public_key`.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

from langfuse_ollama.domain.entities import ChatRequest

logger = logging.getLogger(__name__)


class LangfuseOllamaGateway:
    """Implementación de ChatGateway sobre langfuse.openai → Ollama."""

    def __init__(
        self,
        ollama_url: str,
        lf_public_key: str = "",
        lf_secret_key: str = "",
        lf_host: str = "",
        client=None,
    ):
        self._lf_public_key = lf_public_key
        self._client = client if client is not None else _build_client(
            ollama_url, lf_public_key, lf_secret_key, lf_host
        )

    def complete(self, request: ChatRequest) -> str:
        with self._trace_root(request):
            response = self._client.chat.completions.create(**self._kwargs(request))
            if not response.choices:
                return ""
            return response.choices[0].message.content or ""

    def stream(self, request: ChatRequest) -> Iterator[str]:
        kwargs = self._kwargs(request)
        kwargs["stream"] = True
        with self._trace_root(request):
            stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                # Las APIs compatibles con OpenAI (incl. Ollama) pueden emitir
                # chunks sin `choices` — p. ej. el chunk final con usage cuando
                # se activa stream_options. Ignorarlos evita un IndexError.
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    yield delta

    def _kwargs(self, request: ChatRequest) -> dict:
        kwargs: dict = dict(
            model=request.model,
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            # `name` y `langfuse_public_key` son los únicos kwargs Langfuse que
            # el wrapper v3 extrae antes de reenviar a OpenAI.
            name=request.trace_name,
        )
        if self._lf_public_key:
            kwargs["langfuse_public_key"] = self._lf_public_key
        return kwargs

    @contextmanager
    def _trace_root(self, request: ChatRequest):
        """
        Span raíz que agrupa la generación y fija los atributos de trace.
        Patrón Langfuse v3: el wrapper OpenAI anida su observación bajo el
        span OTel activo; session_id/user_id/tags se fijan con update_trace().
        Sin credenciales no hay tracing → no-op.
        """
        if not self._lf_public_key:
            yield
            return
        from langfuse import get_client
        lf = get_client(public_key=self._lf_public_key)
        with lf.start_as_current_span(name=request.trace_name) as span:
            span.update_trace(
                name=request.trace_name,
                session_id=request.session_id,
                user_id=request.user_id,
                tags=request.tags,
            )
            yield


# Cache de gateways por (ollama_url, credenciales Langfuse). Streamlit
# re-ejecuta el script en cada interacción y el CLI procesa lotes: un gateway
# por combinación preserva el pool de conexiones y la instrumentación.
_gateways: dict = {}
_gateways_lock = threading.Lock()


def get_gateway(
    ollama_url: str,
    lf_public_key: str = "",
    lf_secret_key: str = "",
    lf_host: str = "",
) -> LangfuseOllamaGateway:
    """Retorna un gateway cacheado para esa configuración."""
    key = (ollama_url, lf_public_key, lf_secret_key, lf_host)
    with _gateways_lock:
        gateway = _gateways.get(key)
        if gateway is None:
            gateway = LangfuseOllamaGateway(*key)
            _gateways[key] = gateway
    return gateway


def _build_client(ollama_url: str, lf_public_key: str, lf_secret_key: str, lf_host: str):
    from langfuse import Langfuse
    from langfuse.openai import OpenAI

    if lf_public_key and lf_secret_key:
        # Registra (o recupera) la instancia Langfuse para esta public_key.
        # El wrapper la resuelve después vía get_client(public_key=...).
        Langfuse(
            public_key=lf_public_key,
            secret_key=lf_secret_key,
            host=lf_host or "https://cloud.langfuse.com",
        )
    return OpenAI(base_url=f"{ollama_url}/v1", api_key="ollama")


def flush(lf_public_key: Optional[str] = None) -> None:
    """Fuerza el envío de traces bufferizados. Necesario antes de salir en CLI."""
    try:
        from langfuse import get_client
        get_client(public_key=lf_public_key or None).flush()
    except Exception as e:
        logger.warning("Langfuse flush failed: %s", e)
