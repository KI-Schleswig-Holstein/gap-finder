import logging

import pandas as pd

logger = logging.getLogger(__name__)


class TopicRanker:
    """Aggregate question similarity scores into topic-level rankings."""

    _STAT_COLUMNS = [
        "question_count",
        "avg_similarity",
        "std_similarity",
        "min_similarity",
        "max_similarity",
    ]

    def rank(
        self,
        questions_with_topics: pd.DataFrame,
        topics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate coverage statistics for each topic.

        Higher similarity means a topic is better covered by the indexed corpus;
        lower similarity points to a stronger potential gap.

        Args:
            questions_with_topics: DataFrame with one row per question. It must
                contain ``topic_id`` and ``similarity_score``. ``topic_id``
                links each question to a topic. ``similarity_score`` is the
                score calculated during question ranking.
            topics: DataFrame with one row per topic. It must contain ``id``,
                which is matched against ``questions_with_topics.topic_id``.

        Returns:
            The ``topics`` DataFrame enriched with ``question_count``,
            ``avg_similarity``, ``std_similarity``, ``min_similarity``, and
            ``max_similarity``.
        """
        missing_question_columns = {"topic_id", "similarity_score"} - set(
            questions_with_topics.columns
        )
        if missing_question_columns:
            raise ValueError(
                "questions_with_topics must contain 'topic_id' and "
                f"'similarity_score'. Missing: {sorted(missing_question_columns)}"
            )

        if "id" not in topics:
            raise ValueError("topics must contain an 'id' column.")

        enriched_topics = topics.drop(
            columns=[column for column in self._STAT_COLUMNS if column in topics],
        ).copy()
        if questions_with_topics.empty:
            for column in self._STAT_COLUMNS:
                enriched_topics[column] = 0 if column == "question_count" else pd.NA
            return enriched_topics

        questions = questions_with_topics.copy()
        questions["_similarity"] = questions["similarity_score"].apply(
            self._similarity_value
        )

        topic_stats = (
            questions.groupby("topic_id")
            .agg(
                question_count=("topic_id", "size"),
                avg_similarity=("_similarity", "mean"),
                std_similarity=("_similarity", "std"),
                min_similarity=("_similarity", "min"),
                max_similarity=("_similarity", "max"),
            )
            .rename_axis("id")
            .reset_index()
        )

        enriched_topics = enriched_topics.merge(topic_stats, on="id", how="left")
        enriched_topics["question_count"] = (
            enriched_topics["question_count"].fillna(0).astype("int64")
        )

        logger.info("Calculated rankings for %d topics.", len(enriched_topics))
        return enriched_topics

    @staticmethod
    def _similarity_value(similarity_score) -> float:
        if similarity_score is None:
            return float("nan")

        if isinstance(similarity_score, (list, tuple)):
            if not similarity_score:
                return float("nan")
            return TopicRanker._similarity_value(similarity_score[0])

        if hasattr(similarity_score, "size") and hasattr(similarity_score, "flat"):
            if similarity_score.size == 0:
                return float("nan")
            return TopicRanker._similarity_value(similarity_score.flat[0])

        try:
            if pd.isna(similarity_score):
                return float("nan")
        except (TypeError, ValueError):
            pass

        return float(similarity_score)
