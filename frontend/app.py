"""
frontend/app.py
───────────────
Streamlit frontend — DocuMind AI.
Talks to FastAPI backend via HTTP.

Deploy backend on Render, then set BACKEND_URL in Streamlit Cloud secrets:
    BACKEND_URL = "https://your-render-app.onrender.com/api/v1"
"""

import streamlit as st
import httpx
import os
import time

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# FIX: safe secrets access — works locally (no secrets.toml) AND on
# Streamlit Cloud (has secrets). Wrapping in try/except prevents the app
# from crashing with "StreamlitSecretNotFoundError" when key is missing.
def _get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

BACKEND_URL = _get_secret("BACKEND_URL", "http://localhost:8000/api/v1")

LLM_MODEL   = "llama-3.3-70b-versatile"
TOP_K       = 5
TEMPERATURE = 0.2
USE_MMR     = True

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 780px;}
    .doc-pill {
        background: linear-gradient(90deg, #1f2937, #111827);
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .stat-strip {
        display: flex;
        gap: 14px;
        font-size: 0.78rem;
        color: #9ca3af;
        margin-top: 4px;
    }
    .stat-strip span {
        background: #1f2937;
        padding: 3px 9px;
        border-radius: 6px;
    }
    h1 {
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)


def _backend_status() -> dict:
    try:
        r = httpx.get(f"{BACKEND_URL}/status", timeout=5)
        return r.json()
    except Exception:
        return {"status": "unreachable"}


def _upload(file) -> dict:
    try:
        r = httpx.post(
            f"{BACKEND_URL}/upload",
            files={"file": (file.name, file.read(), "application/octet-stream")},
            timeout=360,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"error": detail}
    except Exception as e:
        return {"error": str(e)}


def _ask(query: str, history: list) -> dict:
    try:
        r = httpx.post(
            f"{BACKEND_URL}/ask",
            json={
                "query": query, "history": history, "model": LLM_MODEL,
                "top_k": TOP_K, "temperature": TEMPERATURE, "use_mmr": USE_MMR,
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"error": detail}
    except Exception as e:
        return {"error": str(e)}


# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "messages": [], "doc_loaded": False, "doc_name": None,
    "doc_chunks": 0, "doc_words": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar — just status + clear ────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocuMind AI")
    st.divider()
    status = _backend_status()
    if status.get("status") == "unreachable":
        st.error("❌ Backend offline")
    elif status.get("status") == "ready":
        st.success(f"✅ Document loaded · {status.get('total_chunks', 0)} chunks")
    else:
        st.info("ℹ️ Ready — upload a document to begin")
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.doc_loaded = False
        st.session_state.doc_name = None
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 DocuMind AI")
st.caption("Upload a document · Ask anything · Get cited answers")
st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your document", type=["pdf", "docx", "txt"],
    label_visibility="collapsed",
)

if uploaded and (
    not st.session_state.doc_loaded
    or st.session_state.doc_name != uploaded.name
):
    with st.status(f"Processing **{uploaded.name}**...", expanded=True) as s:
        st.write("Uploading and extracting text...")
        result = _upload(uploaded)
        if "error" in result:
            s.update(label="❌ Upload failed", state="error")
            st.error(result["error"])
        else:
            st.write(f"✅ {result['total_chunks']} chunks indexed")
            st.session_state.doc_loaded = True
            st.session_state.doc_name = result["filename"]
            st.session_state.doc_chunks = result["total_chunks"]
            st.session_state.doc_words = result["total_words"]
            st.session_state.messages = []
            s.update(
                label=f"✅ **{result['filename']}** ready",
                state="complete", expanded=False,
            )

if st.session_state.doc_loaded:
    st.markdown(
        f'<div class="doc-pill">📄 <b>{st.session_state.doc_name}</b></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Document details", expanded=False):
        st.markdown(
            f"**Chunks:** {st.session_state.doc_chunks}  \n"
            f"**Words:** {st.session_state.doc_words:,}"
        )

st.divider()

# ── Chat ──────────────────────────────────────────────────────────────────────
if not st.session_state.doc_loaded:
    st.info("👆 Upload a document above to start asking questions.", icon="📄")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(
                    f"📄 {len(msg['sources'])} source excerpts", expanded=False
                ):
                    for s in msg["sources"]:
                        page_str = f" · page {s['page']}" if s.get("page") else ""
                        st.markdown(
                            f"**Excerpt {s['chunk_id']+1}** · "
                            f"relevance `{s['score']:.2f}`{page_str}\n\n"
                            f"> {s['text'][:300]}"
                            f"{'…' if len(s['text']) > 300 else ''}"
                        )
            if msg.get("latency") is not None:
                st.markdown(
                    f'<div class="stat-strip">'
                    f'<span>⏱ {msg["latency"]:.1f}s</span>'
                    f'<span>🔎 {msg.get("chunks_retrieved", 0)} chunks</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    query = st.chat_input(f"Ask about {st.session_state.doc_name}…")

    if query:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                t0 = time.time()
                result = _ask(query, history)
                latency = time.time() - t0

            if "error" in result:
                answer = f"⚠️ {result['error']}"
                sources = []
            else:
                answer = result["answer"]
                sources = result["sources"]

            st.markdown(answer)

            if sources:
                with st.expander(
                    f"📄 {len(sources)} source excerpts", expanded=False
                ):
                    for s in sources:
                        page_str = f" · page {s['page']}" if s.get("page") else ""
                        st.markdown(
                            f"**Excerpt {s['chunk_id']+1}** · "
                            f"relevance `{s['score']:.2f}`{page_str}\n\n"
                            f"> {s['text'][:300]}"
                            f"{'…' if len(s['text']) > 300 else ''}"
                        )

            st.markdown(
                f'<div class="stat-strip">'
                f'<span>⏱ {latency:.1f}s</span>'
                f'<span>🔎 {len(sources)} chunks</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.session_state.messages.append({
            "role": "assistant", "content": answer,
            "sources": sources, "latency": latency,
            "chunks_retrieved": len(sources),
        })