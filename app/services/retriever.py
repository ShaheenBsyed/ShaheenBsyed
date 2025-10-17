from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


@dataclass
class RetrievedChunk:
    content: str
    score: float
    source_path: str


class FaissRetriever:
    def __init__(self, index_path: Path | str | None = None, meta_path: Path | str | None = None):
        if index_path is None:
            index_path = settings.index_dir / "chunks.index"
        if meta_path is None:
            meta_path = settings.index_dir / "chunks.meta.jsonl"
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self._load()
        self.embedder = SentenceTransformer(settings.embedding.model_name, device=settings.embedding.device)

    def _load(self) -> None:
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("Missing FAISS index or metadata. Run scripts/ingest.py first.")
        self.index = faiss.read_index(str(self.index_path))
        self.meta: List[dict] = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))

    def search(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        query_vec = np.asarray(self.embedder.encode([query], convert_to_numpy=True))
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, k)
        hits: List[RetrievedChunk] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            m = self.meta[idx]
            hits.append(RetrievedChunk(content=m["content"], score=float(score), source_path=m["source_path"]))
        return hits

    def search_with_hyde(self, query: str, k: int = 5) -> List[RetrievedChunk]:
        # placeholder for HyDE; currently same as search
        return self.search(query, k=k)
