import json

import streamlit as st
import ollama_client as oc
from sidebar import SidebarConfig


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
        _parse_upload(uploaded)

    if st.session_state.batch_parse_errors:
        with st.expander(f"⚠️ {len(st.session_state.batch_parse_errors)} parse warning(s)"):
            for err in st.session_state.batch_parse_errors:
                st.warning(err)

    if st.session_state.batch_entries:
        _render_entries(cfg)

    if st.session_state.batch_results:
        _render_results()


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_upload(uploaded) -> None:
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


def _render_entries(cfg: SidebarConfig) -> None:
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
        _run_batch(cfg)


def _run_batch(cfg: SidebarConfig) -> None:
    st.session_state.batch_results = []
    total        = len(st.session_state.batch_entries)
    progress_bar = st.progress(0, text="Starting…")
    status_slot  = st.empty()

    for idx, (line_num, entry) in enumerate(st.session_state.batch_entries):
        m_model       = entry.get("model",       cfg.model)
        m_system      = entry.get("system",      cfg.system_prompt)
        m_user_id     = entry.get("user_id",     cfg.user_id)
        m_trace_name  = entry.get("trace_name",  cfg.trace_name)
        raw           = entry.get("tags",        cfg.tags_raw)
        m_tags        = [t.strip() for t in raw.split(",")] if isinstance(raw, str) else raw
        m_temperature = entry.get("temperature", cfg.temperature)
        m_max_tokens  = entry.get("max_tokens",  cfg.max_tokens)
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
