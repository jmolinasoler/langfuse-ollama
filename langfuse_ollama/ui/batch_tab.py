import json

import streamlit as st
from langfuse_ollama.adapters.langfuse_ollama import get_gateway
from langfuse_ollama.domain.entities import BatchDefaults
from langfuse_ollama.use_cases import batch as batch_uc
from langfuse_ollama.ui.sidebar import SidebarConfig


def render(cfg: SidebarConfig) -> None:
    st.markdown(
        "Upload a JSONL file to run prompts sequentially. Each line is a JSON object "
        "with a required `prompt` key. All other keys are optional and override the "
        "sidebar defaults for that entry."
    )
    st.code(
        '{"prompt": "What is X?"}\n'
        '{"prompt": "Compare A vs B", "model": "mistral", "temperature": 0.2}\n'
        '{"prompt": "Summarize this", "system": "Be concise.", "tags": ["test"]}',
        language="json",
    )

    uploaded = st.file_uploader(
        "Batch file (.jsonl)", type=["jsonl", "json", "txt"], label_visibility="collapsed"
    )

    if uploaded is not None:
        _parse_if_new(uploaded)

    if st.session_state.batch_parse_errors:
        with st.expander(f"⚠️ {len(st.session_state.batch_parse_errors)} parse warning(s)"):
            for line_num, msg in st.session_state.batch_parse_errors:
                st.warning(f"Line {line_num}: {msg}")

    if st.session_state.batch_entries:
        _render_controls(cfg)

    if st.session_state.batch_running:
        _run_step(cfg)

    if st.session_state.batch_results:
        _render_results()


# ── helpers ───────────────────────────────────────────────────────────────────

def _file_key(uploaded) -> str:
    return getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}"


def _parse_if_new(uploaded) -> None:
    """Parsea solo cuando cambia el archivo — un rerun no debe borrar resultados."""
    key = _file_key(uploaded)
    if st.session_state.get("batch_file_key") == key:
        return
    entries, errors = batch_uc.parse_jsonl(uploaded.read().decode("utf-8"))
    st.session_state.batch_file_key     = key
    st.session_state.batch_entries      = entries
    st.session_state.batch_parse_errors = errors
    st.session_state.batch_results      = []
    st.session_state.batch_running      = False
    st.session_state.batch_pos          = 0


def _render_controls(cfg: SidebarConfig) -> None:
    n = len(st.session_state.batch_entries)
    st.markdown(f"**{n} prompt(s) loaded**")

    with st.expander("Preview", expanded=False):
        for line_num, entry in st.session_state.batch_entries:
            preview = entry["prompt"][:120] + ("…" if len(entry["prompt"]) > 120 else "")
            extras = [f"{k}={entry[k]}" for k in ("model", "temperature", "max_tokens") if k in entry]
            suffix = f"  ·  {', '.join(extras)}" if extras else ""
            st.code(f"[{line_num}]{suffix}\n{preview}", language=None)

    running = st.session_state.batch_running
    col_run, col_stop, col_clear = st.columns([1, 1, 4])
    with col_run:
        if st.button("▶ Run Batch", type="primary",
                     disabled=running or not st.session_state.ollama_status):
            st.session_state.batch_running = True
            st.session_state.batch_pos     = 0
            st.session_state.batch_results = []
            st.rerun()
    with col_stop:
        if running and st.button("⏹ Cancel"):
            st.session_state.batch_running = False
            st.rerun()
    with col_clear:
        if not running and st.session_state.batch_results and st.button("🗑 Clear Results"):
            st.session_state.batch_results = []
            st.rerun()


def _run_step(cfg: SidebarConfig) -> None:
    """Procesa UNA entrada por rerun: progreso real, cancelable, resultados parciales."""
    entries = st.session_state.batch_entries
    pos     = st.session_state.batch_pos
    total   = len(entries)

    if pos >= total:
        st.session_state.batch_running = False
        st.success(f"✅ {total} prompt(s) complete")
        return

    line_num, entry = entries[pos]
    short = entry["prompt"][:70] + ("…" if len(entry["prompt"]) > 70 else "")
    st.progress(pos / total, text=f"{pos + 1}/{total}: {short}")

    defaults = BatchDefaults(
        model=cfg.model,
        system=cfg.system_prompt,
        user_id=cfg.user_id,
        trace_name=cfg.trace_name,
        tags=cfg.tags,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    gateway = get_gateway(
        ollama_url=cfg.ollama_url,
        lf_public_key=cfg.lf_public_key,
        lf_secret_key=cfg.lf_secret_key,
        lf_host=cfg.lf_url,
    )

    result = batch_uc.run_entry(gateway, line_num, entry, defaults)
    st.session_state.batch_results.append(result)
    st.session_state.batch_pos = pos + 1
    st.rerun()


def _render_results() -> None:
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
