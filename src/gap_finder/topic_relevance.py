import logging
import re

import pandas as pd

from gap_finder.language_model.language_model import LanguageModel
from gap_finder.prompts import PromptLoader
from gap_finder.singleton.constants import PromptTemplates

logger = logging.getLogger(__name__)


class TopicRelevanceScorer:
    """Rate explained topics by their relevance for the configured domain."""

    _SCORE_PATTERN = re.compile(
        r"score\s*:\s*([01](?:\.\d+)?|\.\d+)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        language_model: LanguageModel,
        language: str = "english",
        domain_description: str | None = None,
    ):
        self.language_model = language_model
        self._prompt_loader = PromptLoader(language=language)
        self.domain_description = domain_description

    def score(self, topics: pd.DataFrame) -> pd.DataFrame:
        """Score explained topics for their relevance to the configured domain.

        Args:
            topics: Topic DataFrame with at least ``title`` and
                ``explanation`` columns. Only rows where both columns contain
                non-empty text are sent to the chat model for relevance
                scoring.

        Returns:
            A copy of ``topics`` enriched with ``relevance_score`` and
            ``relevance_reason`` columns. Topics without an explanation keep
            empty relevance values.
        """
        self._validate_inputs(topics)

        scored_topics = topics.copy()
        scored_topics["relevance_score"] = pd.NA
        scored_topics["relevance_reason"] = pd.NA
        if self.domain_description is None:
            logger.info("Skipping topic relevance scoring: no domain description set.")
            return scored_topics

        if scored_topics.empty:
            return scored_topics

        topics_with_explanation = scored_topics[
            scored_topics["title"].apply(self._has_text)
            & scored_topics["explanation"].apply(self._has_text)
        ]
        if topics_with_explanation.empty:
            logger.info("Skipping topic relevance scoring: no topic explanations found.")
            return scored_topics

        prompts = [
            self._build_prompt(row["title"], row["explanation"])
            for _, row in topics_with_explanation.iterrows()
        ]
        responses = self.language_model.invoke_batch(prompts)
        parsed = [self._parse_response(response) for response in responses]

        scored_topics.loc[topics_with_explanation.index, "relevance_score"] = [
            score for score, _ in parsed
        ]
        scored_topics.loc[topics_with_explanation.index, "relevance_reason"] = [
            reason for _, reason in parsed
        ]

        logger.info("Scored relevance for %d topics.", len(topics_with_explanation))
        return scored_topics

    def _validate_inputs(self, topics: pd.DataFrame):
        missing = {"title", "explanation"} - set(topics.columns)
        if missing:
            raise ValueError(
                "topics must contain 'title' and 'explanation'."
            )

    def _build_prompt(self, title: str, explanation: str) -> str:
        return self._prompt_loader.render(
            PromptTemplates.TOPIC_RELEVANCE,
            title=title,
            explanation=explanation,
            domain_description=self.domain_description,
        )

    @staticmethod
    def _has_text(value) -> bool:
        return isinstance(value, str) and bool(value.strip())

    @classmethod
    def _parse_response(cls, response: str) -> tuple[float | None, str]:
        score = cls._parse_score(response)
        reason = ""

        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        for index, line in enumerate(lines):
            label = line.split(":", 1)[0].strip().lower()
            if label in {"reason", "begruendung", "begründung"} and ":" in line:
                reason = line.split(":", 1)[1].strip()
                rest = lines[index + 1:]
                if rest:
                    reason = " ".join([reason, *rest]).strip()
                break

        if not reason:
            reason = response.strip()

        return score, reason

    @classmethod
    def _parse_score(cls, response: str) -> float | None:
        match = cls._SCORE_PATTERN.search(response)
        if match is None:
            return None

        score = float(match.group(1))
        return max(0.0, min(1.0, score))
