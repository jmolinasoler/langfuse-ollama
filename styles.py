import streamlit as st


def inject() -> None:
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

/* Hide Streamlit default elements for minimal toolbar */
#MainMenu, footer, header { visibility: hidden; }
/* Restore sidebar collapse/expand button (stSidebarCollapseButton in Streamlit 1.50) */
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
}
.block-container { padding-top: 1.5rem !important; }
/* Minimal toolbar styling */
.stToolbar { 
    min-height: 0 !important;
    padding: 0.25rem 0.5rem !important;
}
.stToolbar > div > div > div { 
    gap: 0.25rem !important;
}

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
