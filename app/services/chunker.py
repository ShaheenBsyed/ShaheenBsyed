from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class DocumentChunk:
    doc_id: str
    chunk_id: str
    content: str
    source_path: str


def read_text_file(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    return text


def simple_markdown_text(md: str) -> str:
    # naive cleanup for markdown; can be improved later
    import re

    no_code = re.sub(r"```[\s\S]*?```", " ", md)
    no_links = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", no_code)
    no_images = re.sub(r"!\[(.*?)\]\((.*?)\)", " ", no_links)
    collapsed = re.sub(r"\s+", " ", no_images)
    return collapsed.strip()


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def iter_supported_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            yield path


def build_chunks_for_path(path: Path, chunk_size: int = 800, chunk_overlap: int = 150) -> List[DocumentChunk]:
    raw_text = read_text_file(path)
    if path.suffix.lower() == ".md":
        raw_text = simple_markdown_text(raw_text)
    chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    file_id = path.stem
    result: List[DocumentChunk] = []
    for idx, content in enumerate(chunks):
        result.append(
            DocumentChunk(
                doc_id=file_id,
                chunk_id=f"{file_id}-{idx}",
                content=content,
                source_path=str(path),
            )
        )
    return result
