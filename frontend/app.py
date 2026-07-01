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

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/api/v1")

LLM_MODEL   = "llama-3.3-70b-versatile"
TOP_K       = 5
TEMPERATURE = 0.2
USE_MMR     = True

# ADDED: custom CSS — dark card styling, accent color, no default Streamlit look
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 780px;}
    .stChatMessage {
        border-radius: 14px;
        padding: 4px 10px;
    }
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
        r = httpx.get(f"{BACKEND_URL}/status", timeout=3)
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
        return {"error": e.response.json().get("detail", str(e))}
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
        return {"error": e.response.json().get("detail", str(e))}
    except Exception as e:
        return {"error": str(e)}


defaults = {
    "messages": [], "doc_loaded": False, "doc_name": None,
    "doc_chunks": 0, "doc_words": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# REMOVED: dead commented-out sidebar block entirely (was unused clutter)

st.markdown("# 🧠 DocuMind AI")
st.caption("Upload a document · Ask anything · Get cited answers")
st.divider()

uploaded = st.file_uploader(
    "Upload your document", type=["pdf", "docx", "txt"], label_visibility="collapsed",
)

if uploaded and (not st.session_state.doc_loaded or st.session_state.doc_name != uploaded.name):
    with st.status(f"Processing **{uploaded.name}**...", expanded=True) as s:
        st.write("Uploading file...")
        result = _upload(uploaded)
        if "error" in result:
            s.update(label="❌ Upload failed", state="error")
            st.error(result["error"])
        else:
            st.write(f"Chunking complete — {result['total_chunks']} chunks")
            st.write("Embedding and indexing...")
            st.session_state.doc_loaded = True
            st.session_state.doc_name = result["filename"]
            st.session_state.doc_chunks = result["total_chunks"]
            st.session_state.doc_words = result["total_words"]
            st.session_state.messages = []
            s.update(label=f"✅ **{result['filename']}** ready", state="complete", expanded=False)

if st.session_state.doc_loaded:
    # CHANGED: name stays visible, chunk/word stats moved into a collapsed
    # expander instead of being shown always — reduces clutter for end users
    st.markdown(
        f"""<div class="doc-pill">📄 <b>{st.session_state.doc_name}</b> ready</div>""",
        unsafe_allow_html=True,
    )
    with st.expander("Document details", expanded=False):
        st.markdown(
            f"**Chunks indexed:** {st.session_state.doc_chunks}  \n"
            f"**Total words:** {st.session_state.doc_words:,}"
        )

st.divider()

if not st.session_state.doc_loaded:
    st.info("👆 Upload a document above to start asking questions.", icon="📄")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} source excerpts", expanded=False):
                    for s in msg["sources"]:
                        page_str = f" · page {s['page']}" if s.get("page") else ""
                        st.markdown(
                            f"**Excerpt {s['chunk_id'] + 1}** · relevance `{s['score']:.2f}`{page_str}\n\n"
                            f"> {s['text'][:300]}{'…' if len(s['text']) > 300 else ''}"
                        )
            # ADDED: latency stat strip per assistant message
            if msg.get("latency") is not None:
                st.markdown(
                    f"""<div class="stat-strip">
                        <span>⏱ {msg['latency']:.1f}s</span>
                        <span>🔎 {msg.get('chunks_retrieved', 0)} chunks</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

    query = st.chat_input(f"Ask a question about {st.session_state.doc_name}…")

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
                t0 = time.time()  # ADDED: latency tracking
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
                with st.expander(f"📄 {len(sources)} source excerpts", expanded=False):
                    for s in sources:
                        page_str = f" · page {s['page']}" if s.get("page") else ""
                        st.markdown(
                            f"**Excerpt {s['chunk_id'] + 1}** · relevance `{s['score']:.2f}`{page_str}\n\n"
                            f"> {s['text'][:300]}{'…' if len(s['text']) > 300 else ''}"
                        )

            st.markdown(
                f"""<div class="stat-strip">
                    <span>⏱ {latency:.1f}s</span>
                    <span>🔎 {len(sources)} chunks</span>
                </div>""",
                unsafe_allow_html=True,
            )

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources,
            "latency": latency, "chunks_retrieved": len(sources),  # ADDED
        })