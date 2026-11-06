import re

import pandas as pd

from gap_finder.io_helper import FileHandler
from gap_finder.singleton.constants import DataFileNames


def test_file_handler_creates_unique_run_directories(tmp_path):
    first = FileHandler(tmp_path)
    second = FileHandler(tmp_path)

    assert first.run_path.parent == tmp_path
    assert second.run_path.parent == tmp_path
    assert first.run_path != second.run_path
    assert first.run_path.is_dir()
    assert second.run_path.is_dir()
    assert re.fullmatch(
        r"\d{8}T\d{12}Z_[0-9a-f]{8}", first.run_path.name
    )


def test_file_handler_saves_pipeline_artefacts(tmp_path):
    file_handler = FileHandler(tmp_path)
    questions = pd.DataFrame(
        {"id": [1], "question": ["What gap remains? ε"]}
    )
    topics = pd.DataFrame({"id": [2], "title": ["Data gap"]})

    file_handler.save_questions(questions)
    file_handler.save_questions_with_topics(questions.assign(topic_id=2))
    file_handler.save_questions_rankings(questions.assign(similarity=0.25))
    file_handler.save_topic_rankings(topics.assign(avg_similarity=0.25))
    file_handler.save_topic_explanations(topics.assign(explanation="Explanation"))
    file_handler.save_topic_relevance_scores(topics.assign(relevance_score=0.9))
    file_handler.save_report("# Report about data gaps\n")

    run_path = file_handler.run_path
    assert pd.read_json(run_path / DataFileNames.QUESTIONS).to_dict("records") == [
        {"id": 1, "question": "What gap remains? ε"}
    ]
    assert pd.read_json(
        run_path / DataFileNames.QUESTIONS_WITH_TOPICS
    ).to_dict("records")[0]["topic_id"] == 2
    assert pd.read_json(
        run_path / DataFileNames.QUESTIONS_RANKINGS
    ).to_dict("records")[0]["similarity"] == 0.25
    assert pd.read_json(
        run_path / DataFileNames.TOPICS_RANKINGS_JSON
    ).to_dict("records")[0]["avg_similarity"] == 0.25
    assert pd.read_json(
        run_path / DataFileNames.TOPIC_EXPLANATIONS_JSON
    ).to_dict("records")[0]["explanation"] == "Explanation"
    assert pd.read_json(
        run_path / DataFileNames.TOPIC_RELEVANCE_SCORES_JSON
    ).to_dict("records")[0]["relevance_score"] == 0.9
    assert (run_path / DataFileNames.REPORT_MD).read_text(
        encoding="utf-8"
    ) == "# Report about data gaps\n"
