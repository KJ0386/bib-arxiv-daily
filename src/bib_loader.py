from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import logging

from bibtexparser import load as load_bibtex

from models import LibraryLoadStats, LibraryPaper
from utils import canonical_identity, clean_text, extract_arxiv_id, normalize_doi


LOGGER = logging.getLogger(__name__)


def discover_bib_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(path for path in data_dir.rglob("*.bib") if path.is_file())


def _get_field(entry: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = entry.get(name)
        if value:
            return str(value)
    return None


def _build_library_paper(entry: dict[str, str], source_file: Path) -> LibraryPaper | None:
    title = clean_text(_get_field(entry, "title", "TITLE"))
    if not title:
        return None

    abstract = clean_text(_get_field(entry, "abstract", "ABSTRACT"))
    doi = normalize_doi(_get_field(entry, "doi", "DOI"))
    url = clean_text(_get_field(entry, "url", "URL")) or None
    arxiv_id = extract_arxiv_id(
        _get_field(entry, "eprint", "EPRINT"),
        _get_field(entry, "archiveprefix", "ARCHIVEPREFIX"),
        url,
        doi,
    )
    bib_key = clean_text(_get_field(entry, "ID")) or None

    return LibraryPaper(
        title=title,
        abstract=abstract,
        source_file=source_file.as_posix(),
        bib_key=bib_key,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
    )


def _prefer_record(current: LibraryPaper, incoming: LibraryPaper) -> LibraryPaper:
    # [EN] Prefer the record with more semantic signal because longer abstracts stabilize embedding similarity. / [CN] 优先保留语义信息更多的记录，因为更长摘要会让嵌入相似度更稳定。
    current_signal = len(current.abstract)
    incoming_signal = len(incoming.abstract)
    if incoming_signal > current_signal:
        return incoming
    if incoming_signal == current_signal and incoming.url and not current.url:
        return replace(current, url=incoming.url, doi=incoming.doi or current.doi, arxiv_id=incoming.arxiv_id or current.arxiv_id)
    return current


def load_library(data_dir: Path) -> tuple[list[LibraryPaper], LibraryLoadStats]:
    bib_files = discover_bib_files(data_dir)
    papers_by_identity: dict[str, LibraryPaper] = {}
    total_entries = 0
    skipped_missing_title = 0

    for bib_file in bib_files:
        with bib_file.open("r", encoding="utf-8") as handle:
            parsed = load_bibtex(handle)
        for entry in parsed.entries:
            total_entries += 1
            paper = _build_library_paper(entry, bib_file)
            if paper is None:
                skipped_missing_title += 1
                continue
            identity = canonical_identity(paper.title, paper.doi, paper.arxiv_id)
            if identity is None:
                skipped_missing_title += 1
                continue
            existing = papers_by_identity.get(identity)
            if existing is None:
                papers_by_identity[identity] = paper
            else:
                papers_by_identity[identity] = _prefer_record(existing, paper)

    papers = []
    skipped_missing_abstract = 0
    for paper in papers_by_identity.values():
        if not paper.abstract:
            skipped_missing_abstract += 1
            continue
        papers.append(paper)

    papers.sort(key=lambda item: item.title.lower())
    stats = LibraryLoadStats(
        files_scanned=len(bib_files),
        entries_total=total_entries,
        entries_with_abstract=len(papers),
        duplicates_removed=max(0, total_entries - skipped_missing_title - len(papers_by_identity)),
        skipped_missing_title=skipped_missing_title,
        skipped_missing_abstract=skipped_missing_abstract,
    )
    LOGGER.info("Loaded %s library papers with abstracts from %s bib files", len(papers), len(bib_files))
    return papers, stats


def build_library_identity_set(papers: list[LibraryPaper]) -> set[str]:
    identities = set()
    for paper in papers:
        identity = canonical_identity(paper.title, paper.doi, paper.arxiv_id)
        if identity is not None:
            identities.add(identity)
    return identities

