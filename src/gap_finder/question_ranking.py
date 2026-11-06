import logging

import numpy as np
import pandas as pd

from gap_finder.embedding_model.embedding_model import EmbeddingModel
from gap_finder.vector_store.vector_store import VectorStore

logger = logging.getLogger(__name__)


class QuestionRanker:
    """Rank generated questions by their similarity to the indexed corpus."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        top_k: int,
        batch_size: int = 32,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.batch_size = batch_size

    def rank(self, questions: pd.DataFrame) -> pd.DataFrame:
        """Rank questions by corpus similarity.

        Args:
            questions: DataFrame with at least a ``question`` column.

        Returns:
            A copy of ``questions`` enriched with ``embedding`` and
            ``similarity_score`` columns.
        """
        if questions.empty:
            result = questions.copy()
            result["embedding"] = pd.Series(dtype=object)
            result["similarity_score"] = pd.Series(dtype=object)
            return result

        logger.info("Encoding %d questions for ranking.", len(questions))
        question_embeddings = self.embedding_model.encode(
            questions["question"].tolist(),
            show_progress_bar=True,
            batch_size=self.batch_size,
        )
        question_embeddings = np.asarray(question_embeddings, dtype=np.float32)

        result = questions.copy()
        result["embedding"] = list(question_embeddings)

        score_rows = self._score(result)
        result["similarity_score"] = score_rows
        return result

    def _score(self, result: pd.DataFrame) -> list[list[float]]:
        """Search the vector store for every embedding in ``result``."""
        score_rows = []
        for query_embedding in result["embedding"]:
            scores, _, _, _ = self.vector_store.search(
                np.expand_dims(query_embedding, axis=0),
                k=self.top_k,
            )
            score_rows.append(scores.flatten().tolist())
        return score_rows
