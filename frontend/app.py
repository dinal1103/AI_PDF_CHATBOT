"""
frontend/app.py
───────────────
Streamlit frontend that talks to the FastAPI backend via HTTP.
Run this AFTER the backend is running on port 8000.

Start backend:   uvicorn backend.main:app --port 8000
Start frontend:  streamlit run frontend/app.py
"""

import streamlit as st
import httpx
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api/v1")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _backend_status() -> dict:
    try:
        r = httpx.get(f"{BACKEND_URL}/status", timeout=3)
        return r.json()
    except Exception:
        return {"status": "unreachable"}


def _upload(file) -> dict:
    try:
        r = httpx.post(
            f"{BACKEND_URL}/upload",
            files={"file": (file.name, file.read(), "application/octet-stream")},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": e.response.json().get("detail", str(e))}
    except Exception as e:
        return {"error": str(e)}


def _ask(query: str, history: list) -> dict:
    try:
        r = httpx.post(
            f"{BACKEND_URL}/ask",
            json={
                "query": query,
                "history": history,
                "model": st.session_state.get("llm_model", "llama3-8b-8192"),
                "top_k": st.session_state.get("top_k", 5),
                "temperature": st.session_state.get("temperature", 0.2),
                "use_mmr": st.session_state.get("use_mmr", True),
            },
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": e.response.json().get("detail", str(e))}
    except Exception as e:
        return {"error": str(e)}


# ── Session state ─────────────────────────────────────────────────────────────

defaults = {
    "messages": [],
    "doc_loaded": False,
    "doc_name": None,
    "llm_model": "llama3-8b-8192",
    "top_k": 5,
    "temperature": 0.2,
    "use_mmr": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    status = _backend_status()
    if status.get("status") == "unreachable":
        st.error("❌ Backend unreachable. Run: `uvicorn backend.main:app --port 8000`")
    elif status.get("status") == "ready":
        st.success(f"✅ Backend ready · {status.get('total_chunks', 0)} chunks loaded")
    else:
        st.info("ℹ️ Backend online. No document loaded yet.")

    st.divider()
    st.markdown("### 🤖 Model Settings")

    st.session_state.llm_model = st.selectbox(
        "LLM Model",
         ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
        index=0,
    )
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    st.session_state.top_k = st.slider("Top-K chunks", 1, 10, 5)
    st.session_state.use_mmr = st.toggle("MMR re-ranking", value=True)

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main layout ───────────────────────────────────────────────────────────────

st.markdown("# 🧠 DocuMind AI")
st.caption("Upload any document · Ask anything · Get cited answers")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📁 Upload Document")
    uploaded = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

    if uploaded and (not st.session_state.doc_loaded or st.session_state.doc_name != uploaded.name):
        with st.spinner("Uploading and indexing…"):
            result = _upload(uploaded)

        if "error" in result:
            st.error(f"❌ {result['error']}")
        else:
            st.session_state.doc_loaded = True
            st.session_state.doc_name = result["filename"]
            st.session_state.messages = []
            st.success(
                f"✅ **{result['filename']}** — "
                f"{result['total_chunks']} chunks, {result['total_words']:,} words"
            )

            # Stats
            cols = st.columns(4)
            cols[0].metric("Chunks", result["total_chunks"])
            cols[1].metric("Words", f"{result['total_words']:,}")
            cols[2].metric("Pages", result["total_pages"] or "—")
            cols[3].metric("Type", result["file_type"])

with col_right:
    st.markdown("### 💬 Ask Your Document")

    if not st.session_state.doc_loaded:
        st.info("👈 Upload a document first.", icon="📄")
    else:
        # Display history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("sources"):
                    with st.expander(f"📄 {len(msg['sources'])} sources used", expanded=False):
                        for s in msg["sources"]:
                            st.markdown(
                                f"**Excerpt {s['chunk_id']+1}** · score `{s['score']:.3f}`\n\n"
                                f"> {s['text'][:300]}…"
                            )

        # Chat input
        query = st.chat_input("Ask a question…")
        if query:
            st.session_state.messages.append({"role": "user", "content": query})

            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            with st.spinner("Thinking…"):
                result = _ask(query, history)

            if "error" in result:
                answer = f"⚠️ {result['error']}"
                sources = []
            else:
                answer = result["answer"]
                sources = result["sources"]

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
            st.rerun()