import logging
from unittest.mock import Mock, call

import pandas as pd
import pytest

from gap_finder.question_generation import QuestionGenerator
from gap_finder.singleton.constants import PromptTemplates


class DummyPromptLoader:
    def __init__(self):
        self.calls = []

    def render(self, template_name: str, **kwargs) -> str:
        self.calls.append((template_name, kwargs))
        if template_name == PromptTemplates.SUMMARIZE:
            return f"summary prompt: {kwargs['text']}"
        if template_name == PromptTemplates.QUESTION_GENERATION:
            return f"question prompt: {kwargs['chunk']}"
        raise ValueError(f"Unexpected template: {template_name}")


class QuietProgress:
    def __init__(self, iterable, **_kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix_str(self, _text: str) -> None:
        return None


def make_question_generator(
    language_model: Mock,
    summarize_chunks: bool = False,
    batch_size: int = 5,
    domain_specific_questions: bool = False,
    domain_description: str | None = None,
) -> QuestionGenerator:
    generator = QuestionGenerator(
        language_model=language_model,
        summarize_chunks=summarize_chunks,
        batch_size=batch_size,
        domain_specific_questions=domain_specific_questions,
        domain_description=domain_description,
    )
    generator._prompt_loader = DummyPromptLoader()
    return generator


def make_chunks(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"id": range(len(texts)), "text": texts})


@pytest.fixture(autouse=True)
def quiet_tqdm(monkeypatch):
    monkeypatch.setattr(
        "gap_finder.question_generation.tqdm",
        lambda iterable, **kwargs: QuietProgress(iterable, **kwargs),
    )


def test_generate_happy_path_batches_chunks_and_returns_questions():
    language_model = Mock()
    language_model.invoke_batch.side_effect = [
        [
            '{"questions": ["Welche Luecke gibt es in Chunk 1?"]}',
            '{"questions": ["Welche Methode fehlt in Chunk 2?"]}',
        ],
        [
            '{"questions": ["Welche Annahme fehlt in Chunk 3?"]}',
            '{"questions": ["Welche Daten fehlen in Chunk 4?"]}',
        ],
        [
            '{"questions": ["Welche Evaluation fehlt in Chunk 5?"]}',
        ],
    ]
    chunks = make_chunks(
        ["chunk 1", "chunk 2", "chunk 3", "chunk 4", "chunk 5"]
    )
    generator = make_question_generator(
        language_model=language_model,
        batch_size=2,
    )

    questions = generator.generate(chunks)

    assert questions.to_dict("records") == [
        {"id": 0, "question": "Welche Luecke gibt es in Chunk 1?", "chunk_id": 0},
        {"id": 1, "question": "Welche Methode fehlt in Chunk 2?", "chunk_id": 1},
        {"id": 2, "question": "Welche Annahme fehlt in Chunk 3?", "chunk_id": 2},
        {"id": 3, "question": "Welche Daten fehlen in Chunk 4?", "chunk_id": 3},
        {"id": 4, "question": "Welche Evaluation fehlt in Chunk 5?", "chunk_id": 4},
    ]
    language_model.invoke_batch.assert_has_calls(
        [
            call(["question prompt: chunk 1", "question prompt: chunk 2"]),
            call(["question prompt: chunk 3", "question prompt: chunk 4"]),
            call(["question prompt: chunk 5"]),
        ]
    )
    assert language_model.invoke_batch.call_count == 3
    language_model.invoke.assert_not_called()
    language_model.cleanup.assert_not_called()


def test_generate_summarizes_when_summarize_chunks_is_true():
    language_model = Mock()
    language_model.invoke_batch.side_effect = [
        [" summary 1 ", "summary 2\n"],
        [
            '{"questions": ["Welche Luecke folgt aus Summary 1?"]}',
            '{"questions": ["Welche Luecke folgt aus Summary 2?"]}',
        ],
    ]
    generator = make_question_generator(
        language_model=language_model,
        summarize_chunks=True,
    )

    questions = generator.generate(make_chunks(["chunk 1", "chunk 2"]))

    assert questions.to_dict("records") == [
        {
            "id": 0,
            "question": "Welche Luecke folgt aus Summary 1?",
            "chunk_id": 0,
        },
        {
            "id": 1,
            "question": "Welche Luecke folgt aus Summary 2?",
            "chunk_id": 1,
        },
    ]
    language_model.invoke_batch.assert_has_calls(
        [
            call(["summary prompt: chunk 1", "summary prompt: chunk 2"]),
            call(["question prompt: summary 1", "question prompt: summary 2"]),
        ]
    )


def test_generate_does_not_summarize_when_summarize_chunks_is_false():
    language_model = Mock()
    language_model.invoke_batch.return_value = [
        '{"questions": ["Welche Luecke gibt es in Chunk 1?"]}',
        '{"questions": ["Welche Luecke gibt es in Chunk 2?"]}',
    ]
    generator = make_question_generator(
        language_model=language_model,
        summarize_chunks=False,
    )

    generator.generate(make_chunks(["chunk 1", "chunk 2"]))

    language_model.invoke_batch.assert_called_once_with(
        ["question prompt: chunk 1", "question prompt: chunk 2"]
    )


def test_generate_passes_domain_description_to_question_prompt():
    language_model = Mock()
    language_model.invoke_batch.return_value = [
        '{"questions": ["Welche Trainingsdaten fehlen?"]}',
    ]
    generator = make_question_generator(
        language_model=language_model,
        domain_specific_questions=True,
        domain_description="The domain is machine learning research.",
    )

    generator.generate(make_chunks(["chunk 1"]))

    _, kwargs = generator._prompt_loader.calls[0]
    assert kwargs["domain_specific_questions"] is True
    assert kwargs["domain_description"] == "The domain is machine learning research."


def test_generate_skips_invalid_responses_and_logs_failure(caplog):
    caplog.set_level(logging.INFO, logger="gap_finder.question_generation")
    language_model = Mock()
    language_model.invoke_batch.return_value = [
        '{"questions": ["Welche Luecke gibt es?"]}',
        '{"questions": ["Welche Methode fehlt?",]}',
    ]
    generator = make_question_generator(
        language_model=language_model,
    )

    questions = generator.generate(make_chunks(["valid chunk", "invalid chunk"]))

    assert questions.to_dict("records") == [
        {"id": 0, "question": "Welche Luecke gibt es?", "chunk_id": 0},
    ]
    assert "LLM generated questions for 1/2 chunks." in caplog.text
    assert "(1 responses were invalid JSONs.)" in caplog.text


def test_extract_json_raises_error_for_invalid_json():
    generator = make_question_generator(
        language_model=Mock(),
    )

    with pytest.raises(ValueError, match="Invalid JSON:"):
        generator._extract_json('{"questions": ["Welche Luecke fehlt?",]}')


def test_extract_json_raises_error_when_questions_key_is_missing():
    generator = make_question_generator(
        language_model=Mock(),
    )

    with pytest.raises(ValueError, match="Missing required key 'questions'"):
        generator._extract_json('{"answers": ["Welche Luecke fehlt?"]}')
