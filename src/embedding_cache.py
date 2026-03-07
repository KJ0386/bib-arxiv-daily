from __future__ import annotations

from hashlib import sha256
import json
import logging
from pathlib import Path

import numpy as np

from models import LibraryPaper
from utils import canonical_identity


LOGGER = logging.getLogger(__name__)


def build_library_fingerprint(model_name: str, papers: list[LibraryPaper]) -> str:
    digest = sha256()
    digest.update(model_name.encode("utf-8"))
    for paper in papers:
        identity = canonical_identity(paper.title, paper.doi, paper.arxiv_id) or ""
        payload = {
            "identity": identity,
            "title": paper.title,
            "abstract": paper.abstract,
        }
        digest.update(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


class LibraryEmbeddingCache:
    def __init__(self, cache_dir: Path, model_name: str):
        self.cache_dir = cache_dir
        self.model_name = model_name

    def _cache_path(self, fingerprint: str) -> Path:
        return self.cache_dir / f"library_embeddings_{fingerprint}.npz"

    def load_or_compute(self, papers: list[LibraryPaper], embedder) -> np.ndarray:
        fingerprint = build_library_fingerprint(self.model_name, papers)
        cache_path = self._cache_path(fingerprint)
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as data:
                embeddings = np.asarray(data["embeddings"], dtype=float)
            if embeddings.shape[0] != len(papers):
                raise ValueError(f"Cached library embeddings are stale or corrupted: {cache_path}")
            LOGGER.info("Loaded %s library embeddings from cache %s", embeddings.shape[0], cache_path)
            return embeddings

        embeddings = np.asarray(embedder.encode([paper.embedding_text for paper in papers]), dtype=np.float32)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # [EN] Persist only library vectors because the bib corpus changes slowly, while daily candidates are cheap enough to recompute each run. / [CN] 只持久化馆藏向量，因为 bib 语料变化慢，而每日候选重新计算的成本足够低。
        np.savez_compressed(cache_path, embeddings=embeddings)
        LOGGER.info("Saved %s library embeddings to cache %s", embeddings.shape[0], cache_path)
        self._prune(keep=4)
        return embeddings

    def _prune(self, keep: int) -> None:
        cache_files = sorted(self.cache_dir.glob("library_embeddings_*.npz"), key=lambda item: item.stat().st_mtime, reverse=True)
        for stale_file in cache_files[keep:]:
            stale_file.unlink(missing_ok=True)

