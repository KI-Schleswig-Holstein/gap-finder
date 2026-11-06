"""Integration tests for real topic explanation.

Skipped by default because they call a real chat model:

    GAP_FINDER_RUN_LLM_INTEGRATION=1 uv run pytest -s test/integration/test_topic_explanation_integration.py
"""

import json

import pandas as pd
import pytest

from gap_finder.topic_explanation import TopicExplainer
from dynamics import (
    build_language_model_or_skip,
    language_backend_params,
    language_model_name_for_backend,
    llm_pytestmark,
)


pytestmark = [
    *llm_pytestmark(
        "set GAP_FINDER_RUN_LLM_INTEGRATION=1 to run real LLM integration tests"
    ),
]


@pytest.mark.parametrize("language_backend", language_backend_params())
def test_topic_explainer_generates_titles_and_explanations_with_real_model(
    language_backend,
):
    questions_with_topics = pd.DataFrame(
        {
            "question": [
                "Welche Trainingsdaten fehlen fuer diese Methode?",
                "Welche Datensaetze wurden nicht untersucht?",
                "Welche Benchmark Evaluation fehlt noch?",
                "Welche Metriken wurden nicht verglichen?",
                "Welche Laufzeitfragen bleiben offen?",
            ],
            "topic_id": [0, 0, 1, 1, 2],
            "topic_prob": [0.91, 0.88, 0.93, 0.86, 0.72],
        }
    )
    topics = pd.DataFrame(
        {
            "id": [2, 0, 1],
            "topic_words": [
                [("laufzeit", 0.5)],
                [("trainingsdaten", 0.8), ("datensatz", 0.7)],
                [("evaluation", 0.9), ("benchmark", 0.6)],
            ],
            "question_count": [1, 2, 2],
            "avg_similarity": [0.7, 0.2, 0.3],
        }
    )
    language_model_name = language_model_name_for_backend(language_backend)
    print(
        "\nSelected topic explanation integration language backend: "
        f"{language_backend}, model: {language_model_name}"
    )
    language_model = build_language_model_or_skip(language_backend)
    try:
        explanations = TopicExplainer(
            language_model=language_model,
            top_k=2,
            max_sample_size=2,
            language="german",
        ).explain(
            questions_with_topics=questions_with_topics,
            topics=topics,
        )
    finally:
        language_model.cleanup()

    print(
        f"\nGenerated topic explanations with {language_backend} backend "
        f"({language_model_name}):\n"
        f"{json.dumps(explanations.to_dict('records'), ensure_ascii=False, indent=2)}"
    )

    explained = explanations.dropna(subset=["title", "explanation"])
    assert explained["id"].tolist() == [0, 1]
    assert explained["title"].map(lambda title: isinstance(title, str) and title.strip()).all()
    assert explanations["explanation"].map(
        lambda explanation: pd.isna(explanation)
        or isinstance(explanation, str)
        and explanation.strip()
    ).all()
    assert set(["title", "explanation"]).issubset(explanations.columns)
