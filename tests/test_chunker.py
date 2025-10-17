from pathlib import Path

from app.services.chunker import chunk_text, build_chunks_for_path

def test_chunk_text_overlap():
    text = " ".join([f"w{i}" for i in range(100)])
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    assert chunks, "should create chunks"
    # ensure overlap by checking token reoccurrence
    assert any("w19" in c for c in chunks[1:])


def test_build_chunks_for_path(tmp_path: Path):
    p = tmp_path / "doc.txt"
    p.write_text("hello world " * 100)
    chunks = build_chunks_for_path(p, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 3
