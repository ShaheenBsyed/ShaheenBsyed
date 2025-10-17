from pathlib import Path
import json
import numpy as np
import faiss

from app.services.retriever import FaissRetriever
from app.config import settings


def test_retriever_smoke(tmp_path: Path, monkeypatch):
    # build small temp index
    meta = []
    texts = ["hello world", "foo bar baz", "lorem ipsum"]
    for i, t in enumerate(texts):
        meta.append({"content": t, "source_path": f"doc{i}.txt"})
    dim = 8
    xb = np.random.RandomState(0).randn(len(texts), dim).astype('float32')
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(dim)
    index.add(xb)

    index_path = tmp_path / "chunks.index"
    meta_path = tmp_path / "chunks.meta.jsonl"
    faiss.write_index(index, str(index_path))
    with meta_path.open('w') as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")

    monkeypatch.setattr(settings, 'index_dir', tmp_path)

    r = FaissRetriever(index_path=index_path, meta_path=meta_path)
    hits = r.search("hello", k=2)
    assert len(hits) <= 2
