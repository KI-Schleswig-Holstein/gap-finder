"""Integration tests for real topic relevance scoring.

Skipped by default because they call a real chat model:

    GAP_FINDER_RUN_LLM_INTEGRATION=1 uv run pytest -s test/integration/test_topic_relevance_integration.py
"""

import json

import pandas as pd
import pytest

from gap_finder.topic_relevance import TopicRelevanceScorer
from dynamics import (
    build_language_model_or_skip,
    language_backend_params,
    language_model_name_for_backend,
    llm_pytestmark,
    llm_settings,
)


pytestmark = [
    *llm_pytestmark(
        "set GAP_FINDER_RUN_LLM_INTEGRATION=1 to run real LLM integration tests"
    ),
]


@pytest.mark.parametrize("language_backend", language_backend_params())
def test_topic_relevance_scorer_rates_explained_topics_with_real_model(
    language_backend,
):
    topics = pd.DataFrame(
        {
            "id": [0, 1],
            "question_count": [2, 1],
            "avg_similarity": [0.2, 0.7],
            "title": [
                "Trainingsdaten und Datensaetze",
                "Laufzeit der Implementierung",
            ],
            "explanation": [
                "Das Thema behandelt fehlende Trainingsdaten, Datensaetze und deren Einfluss auf die Aussagekraft der Evaluation.",
                "Das Thema behandelt offene Fragen zur Laufzeit einzelner Implementierungsdetails.",
            ],
        }
    )
    language_model_name = language_model_name_for_backend(language_backend)
    print(
        "\nSelected topic relevance integration language backend: "
        f"{language_backend}, model: {language_model_name}, device: {llm_settings().device}"
    )
    language_model = build_language_model_or_skip(language_backend)

    try:
        relevance_scores = TopicRelevanceScorer(
            language_model=language_model,
            language="german",
            domain_description=(
                "Die Domaene ist Forschung zu maschinellem Lernen und "
                "wissenschaftlicher Evaluation von KI-Systemen."
            ),
        ).score(topics)
    finally:
        language_model.cleanup()

    print(
        f"\nGenerated topic relevance scores with {language_backend} backend "
        f"({language_model_name}):\n"
        f"{json.dumps(relevance_scores.to_dict('records'), ensure_ascii=False, indent=2)}"
    )

    assert len(relevance_scores) == 2
    assert relevance_scores["relevance_score"].between(0.0, 1.0).all()
    assert relevance_scores["relevance_reason"].map(
        lambda reason: isinstance(reason, str) and reason.strip()
    ).all()
    assert set(["relevance_score", "relevance_reason"]).issubset(
        relevance_scores.columns
    )
