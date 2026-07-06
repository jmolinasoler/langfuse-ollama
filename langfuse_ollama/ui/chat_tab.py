import html as _html
from typing import Optional

import streamlit as st
from langfuse_ollama.adapters.langfuse_ollama import get_gateway
from langfuse_ollama.domain.entities import ChatMessage, ChatRequest
from langfuse_ollama.use_cases import chat as chat_uc
from langfuse_ollama.ui.sidebar import SidebarConfig


def render(cfg: SidebarConfig) -> None:
    chat_area = st.container()

    with chat_area:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">{_html.escape(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )
            elif msg["role"] == "assistant":
                model = msg.get("model", cfg.model)
                st.markdown(
                    f'<div class="chat-ai" data-model="{_html.escape(model)}">{_html.escape(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )

    user_input = st.chat_input("Message…", disabled=not st.session_state.ollama_status)

    if not (user_input and user_input.strip()):
        return

    st.session_state.messages.append({"role": "user", "content": user_input})

    with chat_area:
        st.markdown(
            f'<div class="chat-user">{_html.escape(user_input)}</div>',
            unsafe_allow_html=True,
        )

    request = chat_uc.make_chat_request(
        history=[ChatMessage(role=m["role"], content=m["content"])
                 for m in st.session_state.messages],
        system_prompt=cfg.system_prompt,
        model=cfg.model,
        session_id=st.session_state.session_id,
        user_id=cfg.user_id,
        trace_name=cfg.trace_name,
        tags=cfg.tags,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    with st.spinner(f"⏳ {cfg.model} thinking…"):
        reply = _call_model(cfg, request)

    if reply:
        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "model": cfg.model}
        )
        st.rerun()


def _gateway(cfg: SidebarConfig):
    """Composition root del chat: resuelve el adapter para la config actual."""
    return get_gateway(
        ollama_url=cfg.ollama_url,
        lf_public_key=cfg.lf_public_key,
        lf_secret_key=cfg.lf_secret_key,
        lf_host=cfg.lf_url,
    )


def _call_model(cfg: SidebarConfig, request: ChatRequest) -> Optional[str]:
    if cfg.use_streaming:
        return _stream(cfg, request)
    return _complete(cfg, request)


def _stream(cfg: SidebarConfig, request: ChatRequest) -> Optional[str]:
    partial = ""
    placeholder = st.empty()
    try:
        for chunk in chat_uc.stream_turn(_gateway(cfg), request):
            partial += chunk
            placeholder.markdown(
                f'<div class="chat-ai" data-model="{_html.escape(cfg.model)}">{_html.escape(partial)}▌</div>',
                unsafe_allow_html=True,
            )
        placeholder.empty()
        return partial
    except Exception as e:
        st.error(f"Stream error: {e}")
        return None


def _complete(cfg: SidebarConfig, request: ChatRequest) -> Optional[str]:
    try:
        return chat_uc.complete_turn(_gateway(cfg), request)
    except Exception as e:
        st.error(f"Completion error: {e}")
        return None
