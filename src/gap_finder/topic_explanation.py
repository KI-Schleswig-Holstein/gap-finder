import logging

import pandas as pd

from gap_finder.language_model.language_model import LanguageModel
from gap_finder.prompts import PromptLoader
from gap_finder.singleton.constants import PromptTemplates

logger = logging.getLogger(__name__)


class TopicExplainer:
    """Generate human-readable titles and explanations for ranked topics."""

    def __init__(
        self,
        language_model: LanguageModel,
        top_k: int,
        max_sample_size: int = 10,
        language: str = "english",
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if max_sample_size < 1:
            raise ValueError("max_sample_size must be at least 1.")

        self.language_model = language_model
        self.top_k = top_k
        self.max_sample_size = max_sample_size
        self._prompt_loader = PromptLoader(language=language)

    def explain(
        self,
        questions_with_topics: pd.DataFrame,
        topics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Explain the configured number of lowest-coverage topics.

        Args:
            questions_with_topics: DataFrame with one row per question. It must
                contain ``topic_id`` and ``question``.
            topics: DataFrame with one row per topic. It must contain ``id``,
                ``topic_words``, and ``avg_similarity``.

        Returns:
            The ``topics`` DataFrame enriched with ``title`` and
            ``explanation`` columns.
        """
        self._validate_inputs(questions_with_topics, topics)

        enriched_topics = topics.copy()
        enriched_topics["title"] = pd.Series(dtype="object")
        enriched_topics["explanation"] = pd.Series(dtype="object")

        top_topics = enriched_topics.nsmallest(self.top_k, "avg_similarity")
        if top_topics.empty:
            return enriched_topics

        prompts = [
            self._build_prompt(
                questions_with_topics=questions_with_topics,
                topic_id=int(row["id"]),
                topic_words=row["topic_words"],
            )
            for _, row in top_topics.iterrows()
        ]
        responses = self.language_model.invoke_batch(prompts)
        parsed = [self._parse_response(response) for response in responses]

        enriched_topics.loc[top_topics.index, "title"] = [title for title, _ in parsed]
        enriched_topics.loc[top_topics.index, "explanation"] = [
            explanation for _, explanation in parsed
        ]

        for topic_id, title in zip(
            top_topics["id"],
            enriched_topics.loc[top_topics.index, "title"],
        ):
            logger.info("Topic %s: %s was explained successfully.", topic_id, title)

        return enriched_topics

    def _validate_inputs(
        self,
        questions_with_topics: pd.DataFrame,
        topics: pd.DataFrame,
    ):
        missing_question_columns = {"topic_id", "question"} - set(
            questions_with_topics.columns
        )
        if missing_question_columns:
            raise ValueError(
                "questions_with_topics must contain 'topic_id' and 'question'. "
                f"Missing: {sorted(missing_question_columns)}"
            )

        missing_topic_columns = {"id", "topic_words", "avg_similarity"} - set(
            topics.columns
        )
        if missing_topic_columns:
            raise ValueError(
                "topics must contain 'id', 'topic_words', and 'avg_similarity'. "
                f"Missing: {sorted(missing_topic_columns)}"
            )

    def _build_prompt(
        self,
        questions_with_topics: pd.DataFrame,
        topic_id: int,
        topic_words: list,
    ) -> str:
        topic_questions = questions_with_topics[
            questions_with_topics["topic_id"] == topic_id
        ]
        sample_size = min(self.max_sample_size, len(topic_questions))
        sample = topic_questions.sample(n=sample_size, random_state=42)
        questions_list = "\n".join(f"- {q}" for q in sample["question"].tolist())

        return self._prompt_loader.render(
            PromptTemplates.TOPIC_EXPLANATION,
            topic_id=topic_id,
            topic_info_section=self._topic_info_section(topic_words),
            questions_list=questions_list,
        )

    def _topic_info_section(
        self,
        topic_words: list,
    ) -> str:
        if not topic_words:
            return ""

        top_words = ", ".join(
            f"{word} ({score:.3f})"
            for word, score in topic_words[:10]
        )
        return (
            "The topic model identified these representative keywords "
            f"(with relevance score):\n{top_words}\n"
        )

    @staticmethod
    def _parse_response(response: str) -> tuple[str, str]:
        title = ""
        explanation = ""
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]

        for index, line in enumerate(lines):
            label = line.split(":", 1)[0].strip().lower()
            if label in {"title", "titel"} and ":" in line:
                title = line.split(":", 1)[1].strip()
            elif label.startswith("explanation") and ":" in line:
                explanation = line.split(":", 1)[1].strip()
                rest = lines[index + 1:]
                if rest:
                    explanation = " ".join([explanation, *rest]).strip()
                break

        if not title and lines:
            title = lines[0]
        if not explanation and len(lines) > 1:
            explanation = " ".join(lines[1:])

        return title, explanation
