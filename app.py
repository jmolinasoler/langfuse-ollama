"""
Langfuse × Ollama Tracer
Streamlit UI — multi-model, session-scoped, streaming chat with full Langfuse observability.
"""

import html
import json
import streamlit as st
import config
import ollama_client as oc

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Langfuse × Ollama",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #161921;
    --surface2: #1e2330;
    --border: #2a2f3d;
    --accent: #7c6af7;
    --accent2: #4ade80;
    --warn: #f59e0b;
    --text: #e2e8f0;
    --muted: #64748b;
    --user-bubble: #1a2035;
    --ai-bubble: #13192a;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--bg) !important;
    color: var(--text);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Sidebar labels */
.stSidebar label, .stSidebar .stTextInput label, .stSidebar .stSelectbox label,
.stSidebar .stSlider label, .stSidebar .stTextArea label {
    color: var(--muted) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px);
}

/* Chat messages */
.chat-user {
    background: var(--user-bubble);
    border: 1px solid var(--border);
    border-radius: 10px 10px 2px 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    margin-left: 20%;
    font-size: 0.9rem;
    line-height: 1.6;
    position: relative;
}
.chat-user::before {
    content: "YOU";
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--accent);
    font-weight: 700;
    letter-spacing: 0.1em;
    display: block;
    margin-bottom: 0.3rem;
}

.chat-ai {
    background: var(--ai-bubble);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 2px 10px 10px 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    margin-right: 10%;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-ai::before {
    content: attr(data-model);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--accent2);
    font-weight: 700;
    letter-spacing: 0.1em;
    display: block;
    margin-bottom: 0.3rem;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-ok   { background: #14532d; color: #4ade80; border: 1px solid #166534; }
.badge-err  { background: #450a0a; color: #f87171; border: 1px solid #7f1d1d; }
.badge-warn { background: #451a03; color: #fb923c; border: 1px solid #7c2d12; }

/* Session info */
.session-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    word-break: break-all;
    line-height: 1.8;
}

/* Main header */
.main-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.02em;
    margin-bottom: 0;
}
.main-sub {
    font-size: 0.78rem;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1rem;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

/* Slider */
.stSlider > div > div { background: var(--accent) !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Code blocks in chat */
code {
    background: #0a0c10 !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.1em 0.4em !important;
}
pre code {
    display: block !important;
    padding: 0.8rem !important;
    overflow-x: auto !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────
if "session_id"    not in st.session_state: st.session_state.session_id    = oc.new_session_id()
if "messages"      not in st.session_state: st.session_state.messages      = []
if "models"        not in st.session_state: st.session_state.models        = []
if "ollama_status" not in st.session_state: st.session_state.ollama_status = None
if "batch_entries" not in st.session_state: st.session_state.batch_entries = []
if "batch_parse_errors" not in st.session_state: st.session_state.batch_parse_errors = []
if "batch_results" not in st.session_state: st.session_state.batch_results = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ CONFIG")
    st.markdown("---")

    # Langfuse keys override
    lf_pub = st.text_input("LANGFUSE_PUBLIC_KEY", value=config.LANGFUSE_PUBLIC_KEY,
                           type="password", placeholder="pk-lf-...")
    lf_sec = st.text_input("LANGFUSE_SECRET_KEY", value=config.LANGFUSE_SECRET_KEY,
                           type="password", placeholder="sk-lf-...")
    lf_url = st.selectbox("Langfuse Region", [
        "https://cloud.langfuse.com",
        "https://us.cloud.langfuse.com",
        "https://jp.cloud.langfuse.com",
        "http://localhost:3000",  # self-hosted
    ], index=0)

    # Apply overrides to env and re-sync os.environ so Langfuse SDK picks up changes
    if lf_pub: config.LANGFUSE_PUBLIC_KEY = lf_pub
    if lf_sec: config.LANGFUSE_SECRET_KEY = lf_sec
    config.LANGFUSE_BASE_URL = lf_url
    oc.init_langfuse_env()

    lf_ok = bool(lf_pub and lf_sec)
    st.markdown(
        f'<span class="badge {"badge-ok" if lf_ok else "badge-err"}">{"● LANGFUSE OK" if lf_ok else "● LANGFUSE NOT SET"}</span>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Ollama config
    ollama_url = st.text_input("OLLAMA_BASE_URL", value=config.OLLAMA_BASE_URL)
    config.OLLAMA_BASE_URL = ollama_url

    if st.button("🔄 Refresh Models"):
        st.session_state.models = oc.list_models()
        st.session_state.ollama_status = len(st.session_state.models) > 0

    if not st.session_state.models:
        st.session_state.models = oc.list_models()
        st.session_state.ollama_status = len(st.session_state.models) > 0

    ollama_badge = "badge-ok" if st.session_state.ollama_status else "badge-err"
    ollama_label = f"● OLLAMA ({len(st.session_state.models)} models)" if st.session_state.ollama_status else "● OLLAMA OFFLINE"
    st.markdown(f'<span class="badge {ollama_badge}">{ollama_label}</span>', unsafe_allow_html=True)

    st.markdown("---")

    model = st.selectbox(
        "Model",
        st.session_state.models or [config.DEFAULT_MODEL],
        index=0,
    )

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

    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    max_tokens  = st.slider("Max Tokens", 128, 8192, 2048, 128)

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


# ── Main Panel ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🦙 Langfuse × Ollama Tracer</div>', unsafe_allow_html=True)
st.markdown(f'<div class="main-sub">session · {st.session_state.session_id} · traces → <a href="{lf_url}" target="_blank" style="color:#7c6af7">{lf_url}</a></div>', unsafe_allow_html=True)

# ── Validation Guard ──────────────────────────────────────────────────────────
if not lf_ok:
    st.warning("⚠️ Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in the sidebar (or .env) to enable tracing.")

if not st.session_state.ollama_status:
    st.error("❌ Ollama not reachable. Ensure `ollama serve` is running.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_batch = st.tabs(["💬 Chat", "📋 Batch"])


# ── Chat Tab ──────────────────────────────────────────────────────────────────
with tab_chat:
    chat_area = st.container()

    with chat_area:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">{html.escape(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )
            elif msg["role"] == "assistant":
                st.markdown(
                    f'<div class="chat-ai" data-model="{html.escape(model)}">{html.escape(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )

    user_input = st.chat_input("Message…", disabled=not st.session_state.ollama_status)

    if user_input and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})

        full_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

        with st.spinner(f"⏳ {model} thinking…"):
            if use_streaming:
                partial = ""
                placeholder = st.empty()
                try:
                    for chunk in oc.chat_stream(
                        messages=full_messages,
                        model=model,
                        session_id=st.session_state.session_id,
                        user_id=user_id,
                        trace_name=trace_name,
                        tags=tags,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ):
                        partial += chunk
                        placeholder.markdown(
                            f'<div class="chat-ai" data-model="{html.escape(model)}">{html.escape(partial)}▌</div>',
                            unsafe_allow_html=True,
                        )
                    placeholder.empty()
                    reply = partial
                except Exception as e:
                    st.error(f"Stream error: {e}")
                    reply = None
            else:
                try:
                    reply = oc.chat_complete(
                        messages=full_messages,
                        model=model,
                        session_id=st.session_state.session_id,
                        user_id=user_id,
                        trace_name=trace_name,
                        tags=tags,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as e:
                    st.error(f"Completion error: {e}")
                    reply = None

        if reply:
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()


# ── Batch Tab ─────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("Upload a JSONL file to run prompts sequentially. Each line is a JSON object with a required `prompt` key. All other keys are optional and override the sidebar defaults for that entry.")
    st.code('{"prompt": "What is X?"}\n{"prompt": "Compare A vs B", "model": "mistral", "temperature": 0.2}\n{"prompt": "Summarize this", "system": "Be concise.", "tags": ["test"]}', language="json")

    uploaded = st.file_uploader("Batch file (.jsonl)", type=["jsonl", "json", "txt"], label_visibility="collapsed")

    if uploaded is not None:
        content = uploaded.read().decode("utf-8")
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

        entries, parse_errors = [], []
        for i, line in enumerate(lines, 1):
            try:
                obj = json.loads(line)
                if "prompt" not in obj:
                    parse_errors.append(f"Line {i}: missing 'prompt' key")
                else:
                    entries.append((i, obj))
            except json.JSONDecodeError as exc:
                parse_errors.append(f"Line {i}: invalid JSON — {exc}")

        st.session_state.batch_entries      = entries
        st.session_state.batch_parse_errors = parse_errors
        st.session_state.batch_results      = []

    if st.session_state.batch_parse_errors:
        with st.expander(f"⚠️ {len(st.session_state.batch_parse_errors)} parse warning(s)"):
            for err in st.session_state.batch_parse_errors:
                st.warning(err)

    if st.session_state.batch_entries:
        n = len(st.session_state.batch_entries)
        st.markdown(f"**{n} prompt(s) loaded**")

        with st.expander("Preview", expanded=False):
            for line_num, entry in st.session_state.batch_entries:
                preview = entry["prompt"][:120] + ("…" if len(entry["prompt"]) > 120 else "")
                extras = [f"{k}={entry[k]}" for k in ("model", "temperature", "max_tokens") if k in entry]
                suffix = f"  ·  {', '.join(extras)}" if extras else ""
                st.code(f"[{line_num}]{suffix}\n{preview}", language=None)

        col_run, col_clear = st.columns([1, 5])
        with col_run:
            run_clicked = st.button("▶ Run Batch", type="primary", disabled=not st.session_state.ollama_status)
        with col_clear:
            if st.session_state.batch_results and st.button("🗑 Clear Results"):
                st.session_state.batch_results = []
                st.rerun()

        if run_clicked:
            st.session_state.batch_results = []
            total        = len(st.session_state.batch_entries)
            progress_bar = st.progress(0, text="Starting…")
            status_slot  = st.empty()

            for idx, (line_num, entry) in enumerate(st.session_state.batch_entries):
                m_model       = entry.get("model",       model)
                m_system      = entry.get("system",      system_prompt)
                m_user_id     = entry.get("user_id",     user_id)
                m_trace_name  = entry.get("trace_name",  trace_name)
                raw           = entry.get("tags",        tags_raw)
                m_tags        = [t.strip() for t in raw.split(",")] if isinstance(raw, str) else raw
                m_temperature = entry.get("temperature", temperature)
                m_max_tokens  = entry.get("max_tokens",  max_tokens)
                session_id_b  = oc.new_session_id()

                short = entry["prompt"][:70] + ("…" if len(entry["prompt"]) > 70 else "")
                progress_bar.progress(idx / total, text=f"{idx + 1}/{total}: {short}")
                status_slot.info(f"⏳ Running prompt {idx + 1} of {total}")

                messages_b = [
                    {"role": "system", "content": m_system},
                    {"role": "user",   "content": entry["prompt"]},
                ]

                try:
                    response = oc.chat_complete(
                        messages=messages_b,
                        model=m_model,
                        session_id=session_id_b,
                        user_id=m_user_id,
                        trace_name=m_trace_name,
                        tags=m_tags,
                        temperature=m_temperature,
                        max_tokens=m_max_tokens,
                    )
                    st.session_state.batch_results.append({
                        "line":       line_num,
                        "session_id": session_id_b,
                        "model":      m_model,
                        "prompt":     entry["prompt"],
                        "response":   response,
                        "error":      None,
                    })
                except Exception as exc:
                    st.session_state.batch_results.append({
                        "line":       line_num,
                        "session_id": session_id_b,
                        "model":      m_model,
                        "prompt":     entry["prompt"],
                        "response":   None,
                        "error":      str(exc),
                    })

                progress_bar.progress((idx + 1) / total)

            progress_bar.progress(1.0, text=f"✅ {total} prompt(s) complete")
            status_slot.empty()
            st.rerun()

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.batch_results:
        results   = st.session_state.batch_results
        ok_count  = sum(1 for r in results if r["error"] is None)
        err_count = len(results) - ok_count

        st.markdown(f"**Results — {ok_count} OK · {err_count} error(s)**")

        jsonl_out = "\n".join(json.dumps(r, ensure_ascii=False) for r in results)
        st.download_button(
            "⬇ Download results.jsonl",
            data=jsonl_out,
            file_name="results.jsonl",
            mime="application/jsonlines",
        )

        st.markdown("---")

        for r in results:
            icon  = "✅" if r["error"] is None else "❌"
            label = f"{icon} [{r['line']}] {r['prompt'][:80]}{'…' if len(r['prompt']) > 80 else ''}"
            with st.expander(label, expanded=False):
                st.markdown(f"`{r['model']}` · session `{r['session_id'][:8]}…`")
                st.markdown("**Prompt**")
                st.text(r["prompt"])
                if r["error"]:
                    st.error(r["error"])
                else:
                    st.markdown("**Response**")
                    st.markdown(r["response"])
