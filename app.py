"""
Langfuse × Ollama Tracer
Streamlit UI — multi-model, session-scoped, streaming chat with full Langfuse observability.
"""

import streamlit as st
import config
import ollama_client as oc
import styles
import sidebar
import chat_tab
import batch_tab

st.set_page_config(
    page_title="Langfuse × Ollama",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

styles.inject()

# ── Session State Init ─────────────────────────────────────────────────────────
if "session_id"         not in st.session_state: st.session_state.session_id         = oc.new_session_id()
if "messages"           not in st.session_state: st.session_state.messages           = []
if "models"             not in st.session_state: st.session_state.models             = []
if "ollama_status"      not in st.session_state: st.session_state.ollama_status      = None
if "batch_entries"      not in st.session_state: st.session_state.batch_entries      = []
if "batch_parse_errors" not in st.session_state: st.session_state.batch_parse_errors = []
if "batch_results"      not in st.session_state: st.session_state.batch_results      = []
if "show_wizard" not in st.session_state:
    st.session_state.show_wizard = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
cfg = sidebar.render()

# ── Header ─────────────────────────────────────────────────────────────────
col_header, col_feedback = st.columns([0.8, 0.2])
with col_header:
    st.markdown('<div class="main-header">🦙 Langfuse × Ollama Tracer</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="main-sub">session · {st.session_state.session_id} · traces → '
        f'<a href="{cfg.lf_url}" target="_blank" style="color:#7c6af7">{cfg.lf_url}</a></div>',
        unsafe_allow_html=True,
    )
with col_feedback:
    if st.button("💬 Send Feedback", help="Send anonymous feedback via Featurebase"):
        st.session_state.show_feedback = True
        st.rerun()

# ── Show Feedback or Main App ──────────────────────────────────────────────
if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False

if st.session_state.show_feedback:
    import feedback_widget
    feedback_widget.render_feedback(disabled=config.FEEDBACK_DISABLED)
    st.stop()

# ── Validation Guards ──────────────────────────────────────────────────────────
if not cfg.lf_ok:
    st.warning("⚠️ Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in the sidebar (or .env) to enable tracing.")

if not st.session_state.ollama_status:
    st.error("❌ Ollama not reachable. Ensure `ollama serve` is running.")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_chat, tab_batch = st.tabs(["💬 Chat", "📋 Batch"])

with tab_chat:
    chat_tab.render(cfg)

with tab_batch:
    batch_tab.render(cfg)
