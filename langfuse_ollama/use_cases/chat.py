"""
Caso de uso: turno de chat. Construye la ChatRequest (system prompt +
historial + defaults) y delega en el ChatGateway. No conoce Streamlit,
OpenAI ni Langfuse — solo el dominio y sus puertos.
"""

from typing import Iterator, List, Optional

from langfuse_ollama.domain.entities import ChatMessage, ChatRequest
from langfuse_ollama.domain.ports import ChatGateway


def make_chat_request(
    history: List[ChatMessage],
    system_prompt: str,
    model: str,
    session_id: str,
    user_id: str,
    trace_name: str = "ollama-chat",
    tags: Optional[List[str]] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> ChatRequest:
    return ChatRequest(
        messages=[ChatMessage(role="system", content=system_prompt), *history],
        model=model,
        session_id=session_id,
        user_id=user_id,
        trace_name=trace_name,
        tags=list(tags) if tags else ["ollama", model],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def complete_turn(gateway: ChatGateway, request: ChatRequest) -> str:
    return gateway.complete(request)


def stream_turn(gateway: ChatGateway, request: ChatRequest) -> Iterator[str]:
    return gateway.stream(request)
