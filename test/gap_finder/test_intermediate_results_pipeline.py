from unittest.mock import Mock

import pandas as pd
import pytest

from gap_finder.main import GapFinder
from gap_finder.singleton.constants import DataFileNames
from gap_finder.vector_store.numpy_vector_store import NumpyVectorStore


@pytest.mark.parametrize(
    ("domain_description", "expect_relevance"),
    [(None, False), ("Machine learning", True)],
)
def test_find_saves_each_pipeline_stage(
    tmp_path,
    monkeypatch,
    mock_embedding_model,
    mock_language_model,
    domain_description,
    expect_relevance,
):
    gap_finder = GapFinder(
        NumpyVectorStore(dim=2),
        mock_embedding_model,
        mock_language_model,
        domain_description=domain_description,
        save_intermediate_results=True,
        base_path=tmp_path,
    )
    questions = pd.DataFrame({"id": [0], "question": ["What is missing?"]})
    questions_with_topics = questions.assign(topic_id=3)
    ranked_questions = questions_with_topics.assign(similarity=0.1)
    ranked_topics = pd.DataFrame({"id": [3], "avg_similarity": [0.1]})
    explained_topics = ranked_topics.assign(
        title="Missing evidence", explanation="Evidence is incomplete."
    )
    relevant_topics = explained_topics.assign(
        relevance_score=0.8, relevance_reason="Important to the domain."
    )
    final_input = relevant_topics if expect_relevance else explained_topics
    final_topics = final_input.assign(report="## Missing evidence\n")
    topic_model = object()

    monkeypatch.setattr(
        gap_finder,
        "_load_chunks_and_embeddings",
        Mock(return_value=pd.DataFrame({"id": [0], "text": ["chunk"]})),
    )
    monkeypatch.setattr(
        gap_finder, "_get_or_create_questions", Mock(return_value=questions)
    )
    monkeypatch.setattr(
        gap_finder,
        "_create_topics",
        Mock(
            return_value=(
                pd.DataFrame({"id": [3]}),
                questions_with_topics,
                topic_model,
            )
        ),
    )
    monkeypatch.setattr(
        gap_finder,
        "_rank_topics",
        Mock(return_value=(ranked_topics, ranked_questions)),
    )
    monkeypatch.setattr(
        gap_finder, "_explain_topics", Mock(return_value=explained_topics)
    )
    monkeypatch.setattr(
        gap_finder, "_score_topic_relevance", Mock(return_value=relevant_topics)
    )
    monkeypatch.setattr(
        gap_finder, "_create_report", Mock(return_value=final_topics)
    )

    assert gap_finder.find() is final_topics
    assert gap_finder.topic_model is topic_model

    run_path = gap_finder.intermediate_results_path
    assert run_path is not None
    expected_files = {
        DataFileNames.QUESTIONS,
        DataFileNames.QUESTIONS_WITH_TOPICS,
        DataFileNames.QUESTIONS_RANKINGS,
        DataFileNames.TOPICS_RANKINGS_JSON,
        DataFileNames.TOPIC_EXPLANATIONS_JSON,
        DataFileNames.REPORT_MD,
    }
    if expect_relevance:
        expected_files.add(DataFileNames.TOPIC_RELEVANCE_SCORES_JSON)
    assert {path.name for path in run_path.iterdir()} == expected_files

    assert pd.read_json(run_path / DataFileNames.QUESTIONS).to_dict("records") == [
        {"id": 0, "question": "What is missing?"}
    ]
    assert pd.read_json(
        run_path / DataFileNames.QUESTIONS_RANKINGS
    ).to_dict("records")[0]["similarity"] == 0.1
    assert (run_path / DataFileNames.REPORT_MD).read_text(
        encoding="utf-8"
    ) == "## Missing evidence\n"
