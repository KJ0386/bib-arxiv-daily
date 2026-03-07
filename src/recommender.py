from __future__ import annotations

import numpy as np

from models import CandidatePaper, LibraryPaper, NeighborMatch, Recommendation
from utils import canonical_identity


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class Recommender:
    def __init__(self, embedder, top_k_neighbors: int, max_results: int):
        self.embedder = embedder
        self.top_k_neighbors = top_k_neighbors
        self.max_results = max_results

    def recommend(
        self,
        library_papers: list[LibraryPaper],
        candidate_papers: list[CandidatePaper],
    ) -> list[Recommendation]:
        if not library_papers or not candidate_papers:
            return []

        library_identities = {
            identity
            for identity in (
                canonical_identity(paper.title, paper.doi, paper.arxiv_id) for paper in library_papers
            )
            if identity is not None
        }
        filtered_candidates = [
            paper
            for paper in candidate_papers
            if canonical_identity(paper.title, paper.doi, paper.arxiv_id) not in library_identities
        ]
        if not filtered_candidates:
            return []

        library_embeddings = _normalize_rows(self.embedder.encode([paper.embedding_text for paper in library_papers]))
        candidate_embeddings = _normalize_rows(self.embedder.encode([paper.embedding_text for paper in filtered_candidates]))
        # [EN] Each candidate is scored by its nearest library neighborhood, which keeps recommendations anchored to local research themes instead of global topic frequency. / [CN] 每篇候选论文按其在馆藏语料中的近邻得分，这样推荐更贴近个人研究主题，而不是被全局高频主题主导。
        similarity_matrix = candidate_embeddings @ library_embeddings.T

        neighbor_count = min(self.top_k_neighbors, similarity_matrix.shape[1])
        recommendations: list[Recommendation] = []
        preview_count = min(3, neighbor_count)

        for row_index, candidate in enumerate(filtered_candidates):
            similarities = similarity_matrix[row_index]
            neighbor_indices = np.argsort(similarities)[::-1][:neighbor_count]
            score = float(np.mean(similarities[neighbor_indices]))
            neighbors = tuple(
                NeighborMatch(
                    title=library_papers[index].title,
                    similarity=float(similarities[index]),
                )
                for index in neighbor_indices[:preview_count]
            )
            recommendations.append(Recommendation(candidate=candidate, score=score, neighbors=neighbors))

        recommendations.sort(key=lambda item: item.score, reverse=True)
        return recommendations[: self.max_results]

