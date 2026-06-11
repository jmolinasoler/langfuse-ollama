from dataclasses import dataclass
from typing import List

import streamlit as st
from langfuse_ollama import config
from langfuse_ollama.core import ollama_client as oc

LANGFUSE_REGIONS = [
    "https://cloud.langfuse.com",
    "https://us.cloud.langfuse.com",
    "https://jp.cloud.langfuse.com",
    "http://localhost:3000",
]


@dataclass
class SidebarConfig:
    lf_ok: bool
    lf_public_key: str
    lf_secret_key: str
    lf_url: str
    ollama_url: str
    model: str
    system_prompt: str
    user_id: str
    tags: List[str]
    tags_raw: str
    trace_name: str
    temperature: float
    max_tokens: int
    use_streaming: bool


def _refresh_ollama(ollama_url: str) -> None:
    st.session_state.ollama_status = oc.ping(ollama_url)
    st.session_state.models = oc.list_models(ollama_url) if st.session_state.ollama_status else []
    st.session_state.ollama_checked_url = ollama_url


def render() -> SidebarConfig:
    with st.sidebar:
        st.markdown("### ⚙️ CONFIG")
        st.markdown("---")

        lf_pub = st.text_input("LANGFUSE_PUBLIC_KEY", value=config.LANGFUSE_PUBLIC_KEY,
                               type="password", placeholder="pk-lf-...")
        lf_sec = st.text_input("LANGFUSE_SECRET_KEY", value=config.LANGFUSE_SECRET_KEY,
                               type="password", placeholder="sk-lf-...")

        regions = list(LANGFUSE_REGIONS)
        if config.LANGFUSE_BASE_URL not in regions:
            regions.insert(0, config.LANGFUSE_BASE_URL)
        lf_url = st.selectbox("Langfuse Host", regions,
                              index=regions.index(config.LANGFUSE_BASE_URL),
                              accept_new_options=True)

        lf_ok = config.langfuse_configured(lf_pub, lf_sec)
        st.markdown(
            f'<span class="badge {"badge-ok" if lf_ok else "badge-err"}">{"● LANGFUSE OK" if lf_ok else "● LANGFUSE NOT SET"}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        ollama_url = st.text_input("OLLAMA_BASE_URL", value=config.OLLAMA_BASE_URL)

        if st.button("🔄 Refresh Models"):
            _refresh_ollama(ollama_url)

        if (st.session_state.ollama_status is None
                or st.session_state.get("ollama_checked_url") != ollama_url):
            _refresh_ollama(ollama_url)

        n_models = len(st.session_state.models)
        ollama_badge = "badge-ok" if st.session_state.ollama_status else "badge-err"
        ollama_label = f"● OLLAMA ({n_models} models)" if st.session_state.ollama_status else "● OLLAMA OFFLINE"
        st.markdown(f'<span class="badge {ollama_badge}">{ollama_label}</span>', unsafe_allow_html=True)

        st.markdown("---")

        model_options = st.session_state.models or [config.DEFAULT_MODEL]
        default_index = model_options.index(config.DEFAULT_MODEL) if config.DEFAULT_MODEL in model_options else 0
        model = st.selectbox("Model", model_options, index=default_index)

        system_prompt = st.text_area(
            "System Prompt",
            value=config.DEFAULT_SYSTEM_PROMPT,
            height=80,
        )

        user_id = st.text_input("User ID (Langfuse)", value=config.DEFAULT_USER_ID)

        tags_raw = st.text_input(
            "Tags (comma-separated)",
            value=f"ollama,{model}",
            placeholder="prod,test,experiment",
        )
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

        trace_name = st.text_input("Trace Name", value="ollama-chat")

        temperature  = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        max_tokens   = st.slider("Max Tokens", 128, 8192, 2048, 128)
        use_streaming = st.checkbox("Streaming", value=True)

        st.markdown("---")
        st.markdown("**Session**")
        st.markdown(f"""
<div class="session-box">
  ID: {st.session_state.session_id[:8]}…{st.session_state.session_id[-8:]}<br>
  Turns: {len([m for m in st.session_state.messages if m["role"] == "user"])}<br>
  Model: {model}
</div>
""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔁 New Session"):
                st.session_state.session_id = oc.new_session_id()
                st.session_state.messages   = []
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()

    return SidebarConfig(
        lf_ok=lf_ok,
        lf_public_key=lf_pub,
        lf_secret_key=lf_sec,
        lf_url=lf_url,
        ollama_url=ollama_url,
        model=model,
        system_prompt=system_prompt,
        user_id=user_id,
        tags=tags,
        tags_raw=tags_raw,
        trace_name=trace_name,
        temperature=temperature,
        max_tokens=max_tokens,
        use_streaming=use_streaming,
    )
