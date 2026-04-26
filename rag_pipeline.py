"""
rag_pipeline.py — RAG pipeline using Groq + local HuggingFace embeddings.
Anti-hallucination techniques:
  1. Strict system prompt
  2. Forced source citations  
  3. Full document context for small docs (< 10k chars) — no chunking issues
  4. Auto k selection for large docs
  5. Temperature = 0
"""

import re
import time
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

from document_loader import load_document


class DocumentQASystem:
    def __init__(self, groq_api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.model_name = model
        self.client = Groq(api_key=groq_api_key)

        print("  Loading embedding model (first run downloads ~90MB)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.vector_store = None
        self.total_chunks = 0
        self.full_text = ""  # Used for small documents

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, file_paths: List[str]) -> dict:
        all_chunks = []
        stats = {"files": [], "total_chunks": 0}

        for path in file_paths:
            print(f"  Loading: {path}")
            docs = load_document(path)
            chunks = self.splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["source"] = Path(path).name
            all_chunks.extend(chunks)
            stats["files"].append({
                "name": Path(path).name,
                "chunks": len(chunks),
                "chars": sum(len(c.page_content) for c in chunks),
            })
            print(f"    → {len(chunks)} chunks")

        if not all_chunks:
            raise ValueError("No content extracted from the provided files.")

        # For small documents store the full text — avoids ALL chunking boundary issues
        full_text = "\n\n".join([c.page_content for c in all_chunks])
        if len(full_text) < 15000:
            self.full_text = full_text
            print(f"\n  Small document detected ({len(full_text)} chars) — using full context mode.")
        else:
            self.full_text = ""
            print(f"\n  Large document detected — using retrieval mode.")

        print(f"  Embedding {len(all_chunks)} chunks locally...")
        self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
        self.total_chunks = len(all_chunks)
        stats["total_chunks"] = self.total_chunks
        print("  ✓ Index built successfully.\n")
        return stats

    def save_index(self, path: str = "faiss_index"):
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"Index saved to '{path}/'")

    def load_index(self, path: str = "faiss_index"):
        self.vector_store = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        print(f"Index loaded from '{path}/'")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _pick_k(self) -> int:
        if self.total_chunks <= 30:
            return self.total_chunks
        elif self.total_chunks <= 50:
            return 8
        else:
            return 5

    def query(self, question: str) -> dict:
        if not self.vector_store:
            raise RuntimeError("No documents indexed yet.")

        start = time.perf_counter()

        # Small doc: send entire document — no retrieval needed
        # Large doc: retrieve most relevant chunks only
        if self.full_text:
            context = self.full_text
            docs = self.vector_store.similarity_search(question, k=min(3, self.total_chunks))
        else:
            k = self._pick_k()
            docs = self.vector_store.similarity_search(question, k=k)
            context = "\n\n".join([d.page_content for d in docs])

        system_prompt = (
            "You are a precise document assistant. Follow these rules strictly:\n"
            "1. Answer ONLY from the provided document. Never use outside knowledge.\n"
            "2. If the answer is not in the document, say exactly: "
            "'This information is not in the document.'\n"
            "3. Keep sections separate — do not mix information from different parts.\n"
            "4. Never infer, assume, or guess. If unsure, say so.\n"
            "5. List ALL items when asked — never say 'and others' or summarize incompletely."
        )

        user_message = (
            f"Document:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            "1. List EVERY item that answers the question. Do not say 'and others' or skip any.\n"
            "2. Answer based only on the document above.\n"
            "3. After your answer, write 'Source:' and quote the exact text from the document."
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content
        latency = time.perf_counter() - start

        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        seen = set()
        sources = []
        for doc in docs:
            name = doc.metadata.get("source", "unknown")
            if name not in seen:
                seen.add(name)
                sources.append((name, doc.page_content[:200].strip()))

        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": round(latency * 1000),
            "chunks_used": len(docs),
        }