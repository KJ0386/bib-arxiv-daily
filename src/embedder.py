from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=float)

        model = self._get_model()
        # [EN] Normalize embeddings once so cosine similarity reduces to a fast matrix multiply. / [CN] 先归一化嵌入，这样余弦相似度就能退化为高效的矩阵乘法。
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=float)

