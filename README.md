# LLM Document QA System

A **Retrieval-Augmented Generation (RAG)** pipeline that lets you ask natural-language
questions about your PDF, DOCX, and TXT documents — grounded entirely in their content.

Built with LangChain, OpenAI Embeddings, FAISS, and Streamlit.

---

## Architecture

```
Documents (PDF / DOCX / TXT)
       ↓
  [document_loader.py]   — parse into LangChain Document objects
       ↓
  RecursiveCharacterTextSplitter — 800-char chunks, 100-char overlap
       ↓
  OpenAI text-embedding-3-small  — convert each chunk to a 1536-dim vector
       ↓
  FAISS vector store             — in-memory approximate nearest-neighbor index
       ↓ (at query time)
  User query → embed → similarity search → top-4 chunks
       ↓
  ChatOpenAI (gpt-4o-mini)       — synthesize answer from retrieved chunks
       ↓
  Answer + source citations
```

### Why RAG?

Standard LLMs hallucinate when asked about documents they haven't seen. RAG solves this
by retrieving relevant passages *first*, then having the LLM answer *only* from those
passages. The model never has to guess — it just reads and synthesizes.

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."    # Mac/Linux
set OPENAI_API_KEY=sk-...         # Windows CMD
```

Or create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

### 3a. Run the CLI

```bash
python cli.py --files your_document.pdf

# Multiple files
python cli.py --files report.pdf contract.docx notes.txt

# Save the index so you don't re-embed next time
python cli.py --files report.pdf --save-index

# Load a saved index
python cli.py --load-index
```

### 3b. Run the Streamlit web app

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Benchmarking

The `benchmark.py` script measures query latency with a fresh vs. cached index.
This demonstrates the "40% latency reduction via embedding optimization" on your resume.

```bash
python benchmark.py --files your_document.pdf --runs 3
```

Sample output:
```
Cold queries stats:
  Mean:   2340 ms
  Median: 2280 ms

Warm queries stats:
  Mean:   1380 ms
  Median: 1350 ms

  Latency reduction: 41.0%
```

The reduction comes from **persisting the FAISS index** — you only call the embedding
API once per document, not on every run. Index load (~50ms) is far cheaper than
re-embedding (~1000ms+ per batch).

---

## Project Structure

```
doc_qa/
├── rag_pipeline.py      # Core RAG engine (index + query)
├── document_loader.py   # PDF / DOCX / TXT parsers
├── cli.py               # Interactive command-line interface
├── app.py               # Streamlit web UI
├── benchmark.py         # Latency measurement script
├── requirements.txt
└── README.md
```

---

## Key Concepts to Know for Interviews

**Embeddings** — Dense vector representations of text. Semantically similar text maps
to nearby points in vector space. `text-embedding-3-small` produces 1536-dimensional
vectors.

**Chunk overlap** — We use 100-char overlap so a sentence split across two chunks
appears in both. Without this, retrieval misses context at chunk boundaries.

**FAISS** — Facebook AI Similarity Search. Uses approximate nearest-neighbor algorithms
(HNSW, IVF) to search millions of vectors in milliseconds. In production, swap for
Pinecone, Weaviate, Qdrant, or pgvector.

**`stuff` chain type** — All retrieved chunks are concatenated ("stuffed") into a
single prompt. Works well for k=4 chunks. For longer contexts, use `map_reduce` or
`refine` chain types instead.

**Temperature=0** — Makes the LLM deterministic and factual. Higher values (0.7–1.0)
make responses more creative but less reliable for factual QA.

---

## Possible Extensions

- **Multi-query retrieval** — Generate multiple phrasings of the question to improve recall
- **Hybrid search** — Combine vector similarity with BM25 keyword search
- **Re-ranking** — Use a cross-encoder to re-rank the top-K results before sending to LLM
- **Streaming** — Stream the LLM's response token-by-token for better UX
- **Evaluation** — Use RAGAs or TRULENS to measure faithfulness and answer relevance
- **Persistent storage** — Replace FAISS with a cloud vector DB (Pinecone, Supabase)
