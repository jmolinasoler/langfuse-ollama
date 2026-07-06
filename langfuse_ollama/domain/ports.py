"""
Puertos (boundaries) — interfaces que el dominio y los casos de uso consumen.
Los adapters del anillo exterior las implementan; la regla de dependencia
apunta hacia dentro: aquí no se importa ningún SDK.
"""

from typing import Iterator, List, Protocol

from .entities import ChatRequest


class ChatGateway(Protocol):
    """Salida hacia un motor de chat (LLM) con tracing incluido."""

    def complete(self, request: ChatRequest) -> str:
        """Genera la respuesta completa."""
        ...

    def stream(self, request: ChatRequest) -> Iterator[str]:
        """Genera la respuesta en chunks de texto."""
        ...


class ModelCatalog(Protocol):
    """Consulta de disponibilidad y modelos del runtime LLM."""

    def ping(self) -> bool:
        ...

    def list_models(self) -> List[str]:
        ...
