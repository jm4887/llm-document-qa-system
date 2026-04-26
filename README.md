# 📄 LLM Document QA System

A **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask natural-language
questions about your PDF, DOCX, and TXT documents — grounded entirely in their content.
No hallucination. Every answer is cited from the source document.

Built with **LangChain · HuggingFace Embeddings · Llama 3.3 via Groq · FAISS · Streamlit**

> 🚀 **Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app) ← update this after deploying

---

## ✨ Features

- 📁 Upload **PDF, DOCX, and TXT** files
- 🔍 **Semantic search** using local HuggingFace embeddings (no API cost)
- 🤖 **Llama 3.3 70B** via Groq for fast, accurate answers (free tier)
- 🛡️ **Anti-hallucination** — model only answers from your documents
- 📎 **Source citations** — every answer quotes the exact document text
- ⚡ **Auto context mode** — full document for small files, retrieval for large ones
- 🌐 **Web UI** via Streamlit + **CLI** for terminal use

---

## 🏗️ Architecture

```
Documents (PDF / DOCX / TXT)
       ↓
  [document_loader.py]   — parse into LangChain Document objects
       ↓
  RecursiveCharacterTextSplitter — 1500-char chunks, 200-char overlap
       ↓
  HuggingFace all-MiniLM-L6-v2  — local embeddings, no API needed
       ↓
  FAISS vector store             — in-memory similarity search
       ↓ (at query time)
  Auto k-selection:
    Small doc (< 15k chars) → send full document to LLM
    Large doc               → similarity search → top-k chunks
       ↓
  Llama 3.3 70B via Groq  — synthesize grounded answer
       ↓
  Answer + source citations
```

### Why RAG?

Standard LLMs hallucinate when asked about documents they haven't seen. RAG solves this
by retrieving relevant passages first, then having the LLM answer ONLY from those
passages. The model never has to guess — it just reads and synthesizes.

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/jm4887/llm-document-qa-system.git
cd llm-document-qa-system
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get a free Groq API key

1. Go to https://console.groq.com
2. Sign in with Google → API Keys → Create API Key
3. Copy the key (starts with gsk_...)

### 5. Set your API key

```bash
set GROQ_API_KEY=gsk_your_key_here      # Windows
export GROQ_API_KEY=gsk_your_key_here   # Mac/Linux
```

### 6a. Run the web UI

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### 6b. Run the CLI

```bash
python cli.py --files your_document.pdf

# Multiple files
python cli.py --files report.pdf contract.docx notes.txt

# Save index to avoid re-embedding next time
python cli.py --files report.pdf --save-index

# Load a saved index
python cli.py --load-index
```

---

## 📁 Project Structure

```
llm-document-qa-system/
├── rag_pipeline.py      # Core RAG engine (index + query)
├── document_loader.py   # PDF / DOCX / TXT parsers
├── cli.py               # Interactive command-line interface
├── app.py               # Streamlit web UI
├── benchmark.py         # Latency benchmarking script
├── requirements.txt
└── README.md
```

---

## 🧠 Key Concepts for Interviews

**Embeddings** — Dense vector representations of text. Semantically similar text maps
to nearby points in vector space. all-MiniLM-L6-v2 runs locally — no API cost.

**Chunk overlap** — 200-char overlap ensures context is not lost at chunk boundaries.

**FAISS** — Facebook AI Similarity Search. In-memory approximate nearest-neighbor index.
In production, swap for Pinecone, Weaviate, Qdrant, or pgvector.

**Auto k-selection** — Small documents get full context sent to the LLM.
Large documents use top-k retrieval to stay within context limits.

**Temperature=0** — Makes the LLM deterministic and factual.

**Anti-hallucination prompt** — System prompt explicitly forbids outside knowledge.
User prompt forces the model to cite exact quotes from the document.

---

## 📊 Benchmarking

```bash
python benchmark.py --files your_document.pdf --runs 3
```

---

## 🔧 Possible Extensions

- Conversation memory — Multi-turn Q&A with context from previous questions
- Streaming responses — Token-by-token output like ChatGPT
- Hybrid search — Combine vector similarity with BM25 keyword search
- Query rewriting — Rewrite vague questions for better retrieval accuracy
- Evaluation — RAGAs or TruLens for faithfulness and relevance scoring
- OCR support — Handle scanned PDFs using pytesseract

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq (free) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local) |
| Vector Store | FAISS |
| Framework | LangChain |
| Web UI | Streamlit |
| Document Parsing | PyPDF, Docx2txt |