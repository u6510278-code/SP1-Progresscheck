import uuid
import sys
import os

from .pdf_utils import extract_text_from_pdf, chunk_text
from .embedding_model import embed_texts
from .db import insert_chunks
from .audio_utils import transcribe_audio  

MAX_CHUNKS_FOR_V0 = 80  
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}  


def _ingest_text_source(label: str, path: str, raw_text: str):
    print(f"[INFO] Extracted {len(raw_text)} characters of text from {label}: {path}")

    if not raw_text.strip():
        print(f"[ERROR] No text found in {label}: {path}")
        return

    
    chunks = chunk_text(raw_text)
    print(f"[INFO] Created {len(chunks)} chunks")

    if len(chunks) > MAX_CHUNKS_FOR_V0:
        print(
            f"[WARN] Too many chunks ({len(chunks)}) for this simple demo. "
            f"Refusing to index. You can increase MAX_CHUNKS_FOR_V0 if needed."
        )
        return

    print(f"[INFO] Embedding {len(chunks)} chunks (pure Python, very light)...")
    embeddings = embed_texts(chunks)

    doc_id = str(uuid.uuid4())
    insert_chunks(doc_id, chunks, embeddings)
    print(f"[SUCCESS] Ingested {label.lower()} {path} as doc_id={doc_id}, {len(chunks)} chunks")


def ingest_pdf(path: str):
    print(f"[INFO] Reading PDF: {path}")
    raw_text = extract_text_from_pdf(path)
    _ingest_text_source("PDF", path, raw_text)


def ingest_audio(path: str, model_name: str = "tiny"):
    print(f"[INFO] Reading audio file: {path}")
    raw_text = transcribe_audio(path, model_name=model_name)
    _ingest_text_source("audio", path, raw_text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.rag_index path/to/file.(pdf|mp3|wav|...)")
        sys.exit(1)

    path = sys.argv[1]
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        ingest_pdf(path)
    elif ext in AUDIO_EXTS:
        ingest_audio(path)
    else:
        print(f"[ERROR] Unsupported file type: {ext}")
        print("Supported: .pdf + audio: " + ", ".join(sorted(AUDIO_EXTS)))
        sys.exit(1)
