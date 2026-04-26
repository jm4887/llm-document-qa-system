"""
cli.py — Interactive command-line interface for the Document QA System.

Run with:
  python cli.py --files report.pdf notes.txt contract.docx
"""

import argparse
import os
import sys
from pathlib import Path

from rag_pipeline import DocumentQASystem


BANNER = """
╔══════════════════════════════════════════╗
║       LLM Document QA System            ║
║     RAG + DeepSeek-R1 via Groq          ║
╚══════════════════════════════════════════╝
Type your question, or:
  :sources  — toggle source snippets
  :stats    — show index stats
  :quit     — exit
"""


def get_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("ERROR: GROQ_API_KEY environment variable not set.")
        print("  Get a free key at: https://console.groq.com")
        print("  Then run: set GROQ_API_KEY=your_key_here")
        sys.exit(1)
    return key


def parse_args():
    parser = argparse.ArgumentParser(description="RAG-based Document QA System")
    parser.add_argument("--files", nargs="+", metavar="FILE",
                        help="Documents to index (PDF, DOCX, TXT)")
    parser.add_argument("--save-index", action="store_true",
                        help="Save the FAISS index to disk after indexing")
    parser.add_argument("--load-index", action="store_true",
                        help="Load a previously saved FAISS index")
    parser.add_argument("--model", default="llama-3.3-70b-versatile",
                        help="Groq model to use (default: llama-3.3-70b-versatile)")
    return parser.parse_args()


def validate_files(file_paths):
    valid = []
    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            print(f"  WARNING: File not found, skipping: {fp}")
        elif p.stat().st_size == 0:
            print(f"  WARNING: File is empty, skipping: {fp}")
        else:
            valid.append(fp)
    return valid


def print_result(result, show_sources):
    print(f"\n{'─'*60}")
    print(f"Answer:\n{result['answer']}")
    print(f"\n⏱  {result['latency_ms']} ms")
    if show_sources and result["sources"]:
        print("\nSources:")
        for name, snippet in result["sources"]:
            print(f"  [{name}]")
            print(f"    ...{snippet[:150]}...")
    print(f"{'─'*60}\n")


def main():
    args = parse_args()
    api_key = get_api_key()

    print(BANNER)

    qa = DocumentQASystem(groq_api_key=api_key, model=args.model)

    if args.load_index:
        print("Loading saved index...")
        qa.load_index()
        index_stats = None
    elif args.files:
        files = validate_files(args.files)
        if not files:
            print("No valid files to index. Exiting.")
            sys.exit(1)
        print(f"Indexing {len(files)} file(s)...\n")
        index_stats = qa.index_documents(files)
        if args.save_index:
            qa.save_index()
    else:
        print("ERROR: Provide --files or --load-index. See --help.")
        sys.exit(1)

    if index_stats:
        print("Index summary:")
        for f in index_stats["files"]:
            print(f"  {f['name']:30s} {f['chunks']:3d} chunks  ({f['chars']:,} chars)")
        print(f"  {'TOTAL':30s} {index_stats['total_chunks']:3d} chunks\n")

    show_sources = True
    print("Ready! Ask a question about your documents.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input == ":quit":
            print("Goodbye!")
            break
        elif user_input == ":sources":
            show_sources = not show_sources
            print(f"Source snippets: {'ON' if show_sources else 'OFF'}\n")
            continue
        elif user_input == ":stats":
            if index_stats:
                print(f"  Total chunks: {index_stats['total_chunks']}")
                print(f"  Files: {[f['name'] for f in index_stats['files']]}\n")
            continue

        try:
            result = qa.query(user_input)
            print_result(result, show_sources)
        except Exception as e:
            print(f"\nERROR: {e}\n")


if __name__ == "__main__":
    main()