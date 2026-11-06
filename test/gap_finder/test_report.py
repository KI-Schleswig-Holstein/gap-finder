import pandas as pd
import pytest

from gap_finder.report import TopicReportCreator


def test_enriches_topics_with_markdown_report_and_relevance():
    topics = pd.DataFrame(
        {
            "id": [1, 2],
            "title": ["Data gaps", pd.NA],
            "explanation": ["The training data is not described clearly.", pd.NA],
            "avg_similarity": [0.25, 0.75],
            "relevance_score": [0.8, pd.NA],
            "relevance_reason": ["This topic is central to the domain.", pd.NA],
        }
    )

    topics_with_report = TopicReportCreator(question_sample_size=1).create(topics=topics)

    assert topics_with_report["id"].tolist() == [1, 2]
    assert pd.isna(topics_with_report.loc[1, "report"])
    report = topics_with_report.loc[0, "report"]
    assert report.startswith("## Rank 1: Data gaps (Score 0.2500)\n")
    assert "## Rank 1: Data gaps (Score 0.2500)" in report
    assert "Topic ID: 1" in report
    assert "The average over all topics is 0.5000." in report
    assert "**Relevance for the domain:** 0.80" in report


def test_report_requires_explanation_columns():
    with pytest.raises(ValueError, match="topics must contain"):
        TopicReportCreator().create(topics=pd.DataFrame({"id": [1], "title": ["Titel"]}))


def test_to_markdown_combines_non_empty_report_sections():
    topics = pd.DataFrame(
        {"report": ["  ## First\n", pd.NA, "", "## Second\n\nDetails.\n"]}
    )

    report = TopicReportCreator.to_markdown(topics)

    assert report == "## First\n\n## Second\n\nDetails.\n"


@pytest.mark.parametrize(
    "topics",
    [pd.DataFrame(), pd.DataFrame({"report": [pd.NA, None, "  "]})],
)
def test_to_markdown_returns_empty_string_without_report_sections(topics):
    assert TopicReportCreator.to_markdown(topics) == ""
