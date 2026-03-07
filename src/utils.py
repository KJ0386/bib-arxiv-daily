from __future__ import annotations

import re
from collections.abc import Iterable


_ARXIV_PATTERNS = (
    re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\b([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)\b"),
)


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", value)
    text = text.replace("{", "").replace("}", "")
    return text.strip()


def normalize_title(title: str | None) -> str:
    cleaned = clean_text(title).lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def normalize_doi(doi: str | None) -> str | None:
    cleaned = clean_text(doi).lower()
    if not cleaned:
        return None
    cleaned = cleaned.removeprefix("https://doi.org/")
    cleaned = cleaned.removeprefix("http://doi.org/")
    cleaned = cleaned.removeprefix("doi:")
    return cleaned.strip() or None


def extract_arxiv_id(*values: str | None) -> str | None:
    for value in values:
        cleaned = clean_text(value)
        if not cleaned:
            continue
        for pattern in _ARXIV_PATTERNS:
            match = pattern.search(cleaned)
            if match is not None:
                return match.group(1).lower()
    return None


def canonical_identity(title: str | None, doi: str | None = None, arxiv_id: str | None = None) -> str | None:
    normalized_arxiv = extract_arxiv_id(arxiv_id)
    if normalized_arxiv:
        return f"arxiv:{normalized_arxiv}"
    normalized_doi = normalize_doi(doi)
    if normalized_doi:
        return f"doi:{normalized_doi}"
    normalized_title = normalize_title(title)
    if normalized_title:
        return f"title:{normalized_title}"
    return None


def chunked(items: Iterable[str], chunk_size: int) -> list[list[str]]:
    chunk: list[str] = []
    output: list[list[str]] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == chunk_size:
            output.append(chunk)
            chunk = []
    if chunk:
        output.append(chunk)
    return output

