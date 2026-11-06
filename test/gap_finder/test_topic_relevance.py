import pandas as pd
import pytest

from gap_finder.language_model.language_model import LanguageModel
from gap_finder.topic_relevance import TopicRelevanceScorer


class DummyLanguageModel(LanguageModel):
    def __init__(self):
        self.prompts = []
        self.system_prompt = None

    def invoke(self, prompt: str, system_prompt: str | None = None) -> str:
        return "Score: 0.0\nReason: unused"

    def invoke_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
    ) -> list[str]:
        self.prompts = prompts
        self.system_prompt = system_prompt
        responses = [
            "Score: 0.9\nReason: This is central to the domain.",
            "Score: 0.4\nReason: This is related, but only indirectly.",
        ]
        return responses[: len(prompts)]


def test_scores_topic_relevance_in_one_batch():
    topics = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "title": ["Training data gaps", "Runtime details", pd.NA],
            "explanation": [
                "This topic is about missing training data and datasets.",
                "This topic is about implementation runtime.",
                pd.NA,
            ],
        }
    )
    language_model = DummyLanguageModel()

    relevance_scores = TopicRelevanceScorer(
        language_model=language_model,
        language="english",
        domain_description="The domain is machine learning research.",
    ).score(topics)

    assert relevance_scores["id"].tolist() == [0, 1, 2]
    assert relevance_scores["relevance_score"].iloc[:2].tolist() == [0.9, 0.4]
    assert pd.isna(relevance_scores.loc[2, "relevance_score"])
    assert relevance_scores["relevance_reason"].iloc[:2].tolist() == [
        "This is central to the domain.",
        "This is related, but only indirectly.",
    ]
    assert pd.isna(relevance_scores.loc[2, "relevance_reason"])
    assert len(language_model.prompts) == 2
    assert "Topic: Training data gaps" in language_model.prompts[0]
    assert "Explanation: This topic is about missing training data" in language_model.prompts[0]
    assert "machine learning research" in language_model.prompts[0]
    assert language_model.system_prompt is None


def test_score_preserves_empty_input_with_relevance_columns():
    language_model = DummyLanguageModel()
    topics = pd.DataFrame(columns=["id", "title", "explanation"])

    relevance_scores = TopicRelevanceScorer(
        language_model=language_model,
        domain_description="The domain is machine learning research.",
    ).score(topics)

    assert relevance_scores.empty
    assert set(["relevance_score", "relevance_reason"]).issubset(
        relevance_scores.columns
    )
    assert language_model.prompts == []


def test_score_requires_title_and_explanation_columns():
    topics = pd.DataFrame({"id": [0], "title": ["Missing data"]})

    with pytest.raises(
        ValueError,
        match="topics must contain 'title' and 'explanation'",
    ):
        TopicRelevanceScorer(
            language_model=DummyLanguageModel(),
            domain_description="The domain is machine learning research.",
        ).score(topics)


def test_score_skips_language_model_without_domain_description():
    language_model = DummyLanguageModel()
    topics = pd.DataFrame(
        {
            "id": [0],
            "title": ["Training data gaps"],
            "explanation": ["This topic is about missing training data."],
        }
    )

    relevance_scores = TopicRelevanceScorer(
        language_model=language_model,
        domain_description=None,
    ).score(topics)

    assert relevance_scores["id"].tolist() == [0]
    assert pd.isna(relevance_scores.loc[0, "relevance_score"])
    assert pd.isna(relevance_scores.loc[0, "relevance_reason"])
    assert language_model.prompts == []
    assert language_model.system_prompt is None


def test_score_skips_topics_without_explanation():
    language_model = DummyLanguageModel()
    topics = pd.DataFrame(
        {
            "id": [0, 1],
            "title": ["", "Runtime details"],
            "explanation": ["   ", "This topic is about implementation runtime."],
        }
    )

    relevance_scores = TopicRelevanceScorer(
        language_model=language_model,
        domain_description="The domain is machine learning research.",
    ).score(topics)

    assert pd.isna(relevance_scores.loc[0, "relevance_score"])
    assert relevance_scores.loc[1, "relevance_score"] == 0.9
    assert len(language_model.prompts) == 1


def test_parse_response_accepts_german_reason_label_and_clamps_score():
    score, reason = TopicRelevanceScorer._parse_response(
        "Score: 1.2\nBegruendung: Sehr wichtig fuer das Fachgebiet."
    )

    assert score == 1.0
    assert reason == "Sehr wichtig fuer das Fachgebiet."
