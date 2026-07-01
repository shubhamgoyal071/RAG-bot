"""
ingest.py
Reads all PDFs from ./pdfs, splits them into chunks, embeds them locally
(free, no API needed for this step) and stores them in a local ChromaDB.

Usage:
    python ingest.py
Re-run this any time you add/remove PDFs from the pdfs/ folder.
"""

import argparse
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader


def extract_chunks_from_pdf(pdf_path: Path, chunk_size: int = 1000, overlap: int = 150):
    """Split a PDF into overlapping text chunks, keeping track of page numbers."""
    reader = PdfReader(str(pdf_path))
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "source": pdf_path.name, "page": page_num})
            start = end - overlap  # overlap so we don't cut ideas in half
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="pdfs", help="Folder containing the course PDFs")
    parser.add_argument("--db_dir", default="chroma_db", help="Where to store the vector database")
    parser.add_argument("--reset", action="store_true", help="Wipe and rebuild the database from scratch")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in '{pdf_dir}/'. Drop your course PDFs there and re-run this script.")
        return

    print(f"Found {len(pdf_files)} PDF(s). Loading local embedding model (first run downloads it, ~90MB)...")

    client = chromadb.PersistentClient(path=args.db_dir)

    if args.reset:
        try:
            client.delete_collection("course_material")
            print("Cleared existing database.")
        except Exception:
            pass

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name="course_material", embedding_function=embed_fn)

    existing_sources = set()
    try:
        existing = collection.get()
        existing_sources = {m["source"] for m in existing["metadatas"]} if existing["metadatas"] else set()
    except Exception:
        pass

    doc_id = collection.count()
    for pdf_path in pdf_files:
        if pdf_path.name in existing_sources:
            print(f"Skipping {pdf_path.name} (already in database). Use --reset to rebuild everything.")
            continue

        print(f"Processing {pdf_path.name} ...")
        chunks = extract_chunks_from_pdf(pdf_path)
        if not chunks:
            print(f"  WARNING: no extractable text found. This PDF may be scanned images "
                  f"(would need OCR, not handled by this script yet).")
            continue

        ids = [f"doc_{doc_id + i}" for i in range(len(chunks))]
        documents = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        doc_id += len(chunks)
        print(f"  Added {len(chunks)} chunks.")

    print(f"\nDone. Total chunks now in database: {collection.count()}")


if __name__ == "__main__":
    main()
