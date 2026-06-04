import html as _html

import streamlit as st
import ollama_client as oc
from sidebar import SidebarConfig


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
                st.markdown(
                    f'<div class="chat-ai" data-model="{_html.escape(cfg.model)}">{_html.escape(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )

    user_input = st.chat_input("Message…", disabled=not st.session_state.ollama_status)

    if not (user_input and user_input.strip()):
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    full_messages = [{"role": "system", "content": cfg.system_prompt}] + st.session_state.messages

    with st.spinner(f"⏳ {cfg.model} thinking…"):
        reply = _call_model(cfg, full_messages)

    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()


def _call_model(cfg: SidebarConfig, full_messages: list) -> str | None:
    if cfg.use_streaming:
        return _stream(cfg, full_messages)
    return _complete(cfg, full_messages)


def _stream(cfg: SidebarConfig, full_messages: list) -> str | None:
    partial = ""
    placeholder = st.empty()
    try:
        for chunk in oc.chat_stream(
            messages=full_messages,
            model=cfg.model,
            session_id=st.session_state.session_id,
            user_id=cfg.user_id,
            trace_name=cfg.trace_name,
            tags=cfg.tags,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        ):
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


def _complete(cfg: SidebarConfig, full_messages: list) -> str | None:
    try:
        return oc.chat_complete(
            messages=full_messages,
            model=cfg.model,
            session_id=st.session_state.session_id,
            user_id=cfg.user_id,
            trace_name=cfg.trace_name,
            tags=cfg.tags,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    except Exception as e:
        st.error(f"Completion error: {e}")
        return None
