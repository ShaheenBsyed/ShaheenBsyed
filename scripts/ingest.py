from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from app.config import settings
from app.services.chunker import build_chunks_for_path, iter_supported_files, DocumentChunk


def ensure_dirs() -> None:
    settings.index_dir.mkdir(parents=True, exist_ok=True)


def load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name, device=device)
    return model


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    return np.asarray(model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True))


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def main() -> None:
    ensure_dirs()

    data_root = settings.data_dir
    files = list(iter_supported_files(data_root))
    if not files:
        print(f"No supported files found in {data_root}. Place .txt or .md files.")
        return

    print(f"Found {len(files)} files. Chunking...")
    all_chunks: List[DocumentChunk] = []
    for f in files:
        all_chunks.extend(build_chunks_for_path(f))

    texts = [c.content for c in all_chunks]
    meta = [asdict(c) for c in all_chunks]

    print(f"Embedding {len(texts)} chunks with {settings.embedding.model_name}...")
    model = load_embedding_model(settings.embedding.model_name, settings.embedding.device)
    embeddings = embed_texts(model, texts, batch_size=settings.embedding.batch_size)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    index_path = Path(settings.index_dir) / "chunks.index"
    meta_path = Path(settings.index_dir) / "chunks.meta.jsonl"

    print(f"Saving index to {index_path}")
    faiss.write_index(index, str(index_path))

    print(f"Saving metadata to {meta_path}")
    with meta_path.open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
