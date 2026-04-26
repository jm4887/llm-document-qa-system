"""
app.py — Streamlit web interface for the Document QA System.
Run with: streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from rag_pipeline import DocumentQASystem

st.set_page_config(page_title="Document QA System", page_icon="📄", layout="wide")

st.title("📄 LLM Document QA System")
st.caption("RAG pipeline · LangChain · HuggingFace Embeddings · Llama 3.3 via Groq (free)")

# ── Session state ─────────────────────────────────────────────────────────────
if "qa_system" not in st.session_state:
    st.session_state.qa_system = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "index_stats" not in st.session_state:
    st.session_state.index_stats = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    # Try secrets first (Streamlit Cloud), then env variable, then ask user
    api_key = ""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("✓ API key loaded", icon="🔑")
    except Exception:
        api_key = os.getenv("GROQ_API_KEY", "")

    # Only show input box if key not found anywhere
    if not api_key:
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Free key from https://console.groq.com — no credit card needed.",
        )

    model = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        help="llama-3.3-70b-versatile is the most capable free model.",
    )

    st.divider()
    st.header("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, or TXT files to index.",
    )

    index_btn = st.button(
    "🔍 Index Documents",
    disabled=not uploaded_files,
    use_container_width=True,
    type="primary",
)
    

    if index_btn:
        with st.spinner("Loading & indexing documents..."):
            try:
                tmp_dir = tempfile.mkdtemp()
                file_paths = []
                for uf in uploaded_files:
                    dest = Path(tmp_dir) / uf.name
                    dest.write_bytes(uf.read())
                    file_paths.append(str(dest))

                qa = DocumentQASystem(groq_api_key=api_key, model=model)
                stats = qa.index_documents(file_paths)

                st.session_state.qa_system = qa
                st.session_state.indexed = True
                st.session_state.index_stats = stats
                st.session_state.chat_history = []

                st.success(f"✓ Indexed {stats['total_chunks']} chunks from {len(stats['files'])} file(s)")

            except Exception as e:
                st.error(f"Indexing failed: {e}")

    if st.session_state.index_stats:
        st.divider()
        st.header("📊 Index Stats")
        for f in st.session_state.index_stats["files"]:
            st.metric(f['name'], f"{f['chunks']} chunks", f"{f['chars']:,} chars")

# ── Main chat area ────────────────────────────────────────────────────────────
if not st.session_state.indexed:
    st.info("👈 Upload documents and click **Index Documents** to get started.")

    with st.expander("💡 How does this work?"):
        st.markdown("""
**RAG (Retrieval-Augmented Generation)** in 4 steps:

1. **Load** — Your document is parsed into plain text
2. **Chunk** — Text is split into overlapping ~800-character segments  
3. **Embed** — Each chunk is converted to a vector using a local HuggingFace model (free, runs on your machine)
4. **Retrieve + Generate** — Your question finds the most similar chunks, which are sent to Llama 3.3 on Groq to write a grounded answer

The model can **only** answer from your documents — no hallucination.
        """)

    with st.expander("🔑 How to get a free Groq API key?"):
        st.markdown("""
1. Go to [console.groq.com](https://console.groq.com)
2. Sign in with Google
3. Click **API Keys** → **Create API Key**
4. Paste it in the sidebar
        """)
else:
    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📎 Sources · ⏱ {msg['latency_ms']} ms"):
                    for name, snippet in msg["sources"]:
                        st.markdown(f"**{name}**")
                        st.caption(snippet[:300] + "...")

    # Chat input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                try:
                    result = st.session_state.qa_system.query(question)
                    st.write(result["answer"])

                    if result["sources"]:
                        with st.expander(f"📎 Sources · ⏱ {result['latency_ms']} ms"):
                            for name, snippet in result["sources"]:
                                st.markdown(f"**{name}**")
                                st.caption(snippet[:300] + "...")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "latency_ms": result["latency_ms"],
                    })

                except Exception as e:
                    st.error(f"Query failed: {e}")