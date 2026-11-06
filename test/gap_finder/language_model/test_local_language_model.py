from unittest.mock import Mock

import pytest

from gap_finder.language_model.local_language_model import LocalLanguageModel


class DummyPipeline:
    task = "text-generation"

    def __init__(self):
        self.tokenizer = Mock()
        self.tokenizer.apply_chat_template.side_effect = lambda messages, **_: str(messages)
        self.model = Mock()
        self.model.hf_device_map = None
        self.device = "cuda"
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if isinstance(text, list):
            return [[{"generated_text": f"answer {index}"}] for index, _ in enumerate(text)]
        return [{"generated_text": "answer"}]


def test_invokes_pipeline_with_chat_template():
    pipeline = DummyPipeline()
    model = LocalLanguageModel(pipeline)
    pipeline.model.to.reset_mock()

    assert model.invoke("question", "system") == "answer"
    assert pipeline.calls[0][1] == {
        "do_sample": False,
        "return_full_text": False,
    }
    pipeline.model.to.assert_not_called()


def test_invokes_pipeline_in_batch():
    pipeline = DummyPipeline()
    model = LocalLanguageModel(pipeline)

    assert model.invoke_batch(["first", "second"]) == ["answer 0", "answer 1"]


def test_rejects_non_text_chat_template_result():
    pipeline = DummyPipeline()
    pipeline.tokenizer.apply_chat_template.side_effect = None
    pipeline.tokenizer.apply_chat_template.return_value = [1, 2, 3]
    model = LocalLanguageModel(pipeline)

    with pytest.raises(TypeError, match="must return text"):
        model.invoke("question")


def test_rejects_non_generation_pipeline():
    pipeline = DummyPipeline()
    pipeline.task = "feature-extraction"

    with pytest.raises(ValueError, match="text-generation"):
        LocalLanguageModel(pipeline)


def test_accepts_accelerate_managed_pipeline_without_moving_it():
    pipeline = DummyPipeline()
    pipeline.model.hf_device_map = {"model.layers.0": 0, "lm_head": "cpu"}
    model = LocalLanguageModel(pipeline)

    model.activate()
    assert model.invoke("question") == "answer"
    model.deactivate()
    model.cleanup()

    pipeline.model.to.assert_not_called()
