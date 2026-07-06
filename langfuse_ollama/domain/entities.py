"""
Entidades del dominio — anillo interior de la Clean Architecture.
Solo stdlib: este módulo no conoce Streamlit, httpx, OpenAI ni Langfuse.
"""

import uuid
from dataclasses import dataclass, field
from typing import List


def new_session_id() -> str:
    """Identificador de sesión de conversación (agrupa turnos en un trace)."""
    return str(uuid.uuid4())


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatRequest:
    """Petición completa de generación, incluida la metadata de observabilidad."""
    messages: List[ChatMessage]
    model: str
    session_id: str
    user_id: str
    trace_name: str = "ollama-chat"
    tags: List[str] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass(frozen=True)
class BatchDefaults:
    """Valores por defecto de un lote; cada entrada JSONL puede sobreescribirlos."""
    model: str
    system: str
    user_id: str
    trace_name: str
    tags: List[str]
    temperature: float
    max_tokens: int
