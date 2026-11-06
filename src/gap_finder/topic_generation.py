from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from gap_finder.io_helper import bertopic_model_path

if TYPE_CHECKING:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def release_topic_model_embedding(
    topic_model: BERTopic,
    *,
    owns_embedding_model: bool,
) -> None:
    """Detach BERTopic's embedding backend without moving caller-owned models."""
    embedding_backend = getattr(topic_model, "embedding_model", None)
    if embedding_backend is None:
        return

    owned_local_model = None
    if owns_embedding_model:
        owned_local_model = getattr(embedding_backend, "embedding_model", None)
        if owned_local_model is not None and hasattr(owned_local_model, "to"):
            try:
                owned_local_model.to("cpu")
            except Exception:
                logger.warning(
                    "Could not offload BERTopic's owned embedding model.",
                    exc_info=True,
                )

    topic_model.embedding_model = None
    del embedding_backend
    del owned_local_model
    gc.collect()

    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class TopicGenerator():
    """Class for generating topics from questions."""

    def __init__(self,
                 language: str,
                 embedding_model: str | SentenceTransformer,
                 ) -> None:
        self.language = language
        self.embedding_model = embedding_model

    def generate(
        self,
        questions: pd.DataFrame,
        cleanup_after_generation: bool = False,
        verbose: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, BERTopic]:
        """Generate topics from questions using BERTopic.

        Args:
            questions: DataFrame with one row per question and at least a
                ``question`` column.
            cleanup_after_generation: Whether to release the topic model after
                generation.
            verbose: Whether BERTopic should log detailed progress.

        Returns:
            A tuple containing a topics DataFrame with ``id`` and
            ``topic_words``, and a copy of ``questions`` enriched with
            ``topic_id`` and, when available, ``topic_prob``, followed by the
            fitted BERTopic model. When ``cleanup_after_generation`` is true,
            the returned model no longer retains its embedding model.
        """

        questions_copy = questions.copy()
        topic_model, topics, probs = self._fit_topic_model(
            questions_copy, verbose=verbose)
        try:
            questions_copy["topic_id"] = topics
            if probs is not None:
                questions_copy["topic_prob"] = [prob.max() if len(
                    prob) > 0 else 0 for prob in probs]

            topics = self._topic_words_by_topic(topic_model, topics)
            return topics, questions_copy, topic_model
        finally:
            if cleanup_after_generation:
                self.cleanup_topic_model()

    def _topic_words_by_topic(self, topic_model, topics: list[int]) -> pd.DataFrame:
        """Return BERTopic keywords for each topic id."""
        topic_ids = list(set(topics))
        return pd.DataFrame({
            "id": topic_ids,
            "topic_words": [topic_model.get_topic(tid) or [] for tid in topic_ids],
        })

    def save_model(self, base_path: str, id: str):
        model_path = bertopic_model_path(base_path, id)
        self.topic_model.save(model_path,
                              serialization="pytorch",
                              save_ctfidf=True)
        logger.info(f"Saved BERTopic model to: {model_path}")

    def _fit_topic_model(self, questions: pd.DataFrame, verbose: bool = True) -> tuple[BERTopic, list[int], Union[np.ndarray, None]]:
        """Instantiate and fit a BERTopic model on the preprocessed documents."""
        from bertopic import BERTopic

        logger.info("Initializing embedding model for BERTopic...")

        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            language=self.language,
            calculate_probabilities=True,
            verbose=verbose,
        )

        documents = questions["question"].tolist()

        topics, probs = topic_model.fit_transform(documents)

        # -1 to exclude outlier topic
        logger.info(f"Number of topics found: {len(set(topics)) - 1}")
        logger.info(
            f"Number of outliers (topic -1): {sum(1 for t in topics if t == -1)}")
        logger.info("Topic Information:")
        logger.info(f"\n{topic_model.get_topic_info()}")

        self.topic_model = topic_model
        self.topics = topics
        self.probs = probs

        return topic_model, topics, probs

    def cleanup_topic_model(self):
        """Free GPU/CUDA memory held by the BERTopic model."""
        logger.info("Cleaning up BERTopic model and releasing GPU memory...")

        release_topic_model_embedding(
            self.topic_model,
            owns_embedding_model=isinstance(self.embedding_model, str),
        )

        logger.info("BERTopic model cleaned up, GPU memory released.")
