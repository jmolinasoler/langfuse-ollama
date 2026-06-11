import html as _html
from typing import Optional

import streamlit as st
from langfuse_ollama.core import ollama_client as oc
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

    full_messages = [{"role": "system", "content": cfg.system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    with st.spinner(f"⏳ {cfg.model} thinking…"):
        reply = _call_model(cfg, full_messages)

    if reply:
        st.session_state.messages.append(
            {"role": "assistant", "content": reply, "model": cfg.model}
        )
        st.rerun()


def _client(cfg: SidebarConfig):
    return oc.get_chat_client(
        ollama_url=cfg.ollama_url,
        lf_public_key=cfg.lf_public_key,
        lf_secret_key=cfg.lf_secret_key,
        lf_host=cfg.lf_url,
    )


def _call_model(cfg: SidebarConfig, full_messages: list) -> Optional[str]:
    if cfg.use_streaming:
        return _stream(cfg, full_messages)
    return _complete(cfg, full_messages)


def _stream(cfg: SidebarConfig, full_messages: list) -> Optional[str]:
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
            client=_client(cfg),
            lf_public_key=cfg.lf_public_key,
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


def _complete(cfg: SidebarConfig, full_messages: list) -> Optional[str]:
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
            client=_client(cfg),
            lf_public_key=cfg.lf_public_key,
        )
    except Exception as e:
        st.error(f"Completion error: {e}")
        return None
