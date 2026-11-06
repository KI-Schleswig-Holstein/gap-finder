import pandas as pd

from gap_finder.language_model.language_model import LanguageModel
from gap_finder.topic_explanation import TopicExplainer


class DummyLanguageModel(LanguageModel):
    def __init__(self):
        self.prompts = []
        self.system_prompt = None

    def invoke(self, prompt: str, system_prompt: str | None = None) -> str:
        return "Title: unused\nExplanation: unused"

    def invoke_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
    ) -> list[str]:
        self.prompts = prompts
        self.system_prompt = system_prompt
        return [
            f"Title: Topic {index}\nExplanation: Explanation {index}"
            for index, _ in enumerate(prompts, start=1)
        ]


def test_explains_lowest_coverage_topics_in_one_batch():
    questions_with_topics = pd.DataFrame(
        {
            "question": ["q1", "q2", "q3", "q4"],
            "topic_id": [0, 0, 1, 2],
        }
    )
    topics = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "topic_words": [
                [("baseline", 0.6)],
                [("attention", 0.5), ("embedding", 0.25)],
                [("retrieval", 0.7)],
            ],
            "avg_similarity": [0.9, 0.1, 0.2],
            "question_count": [2, 1, 1],
        }
    )
    language_model = DummyLanguageModel()

    explanations = TopicExplainer(
        language_model=language_model,
        top_k=2,
        max_sample_size=2,
    ).explain(
        questions_with_topics=questions_with_topics,
        topics=topics,
    )

    assert explanations["id"].tolist() == [0, 1, 2]
    assert pd.isna(explanations.loc[0, "title"])
    assert explanations.loc[1, "title"] == "Topic 1"
    assert explanations.loc[2, "title"] == "Topic 2"
    assert pd.isna(explanations.loc[0, "explanation"])
    assert explanations.loc[1, "explanation"] == "Explanation 1"
    assert explanations.loc[2, "explanation"] == "Explanation 2"
    assert len(language_model.prompts) == 2
    assert "q3" in language_model.prompts[0]
    assert "attention (0.500)" in language_model.prompts[0]
    assert "analyze topic clusters" in language_model.prompts[0]
    assert language_model.system_prompt is None
