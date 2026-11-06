from typing import Any

from transformers.pipelines.text_generation import TextGenerationPipeline

from gap_finder.language_model.language_model import LanguageModel


class LocalLanguageModel(LanguageModel):
    """Language model backed by a Transformers text-generation pipeline."""

    def __init__(
        self,
        client: TextGenerationPipeline,
    ) -> None:
        if getattr(client, "task", None) != "text-generation":
            raise ValueError(
                "language_model pipeline must have task 'text-generation'."
            )
        self.client = client

    def invoke(self, prompt: str, system_prompt: str | None = None) -> str:
        text = self._format_prompt(prompt, system_prompt)
        return self._extract_text(self._generate(text))

    def invoke_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
    ) -> list[str]:
        if not prompts:
            return []
        texts = [self._format_prompt(prompt, system_prompt) for prompt in prompts]
        results = self._generate(texts)
        return [self._extract_text(result) for result in results]

    def _format_prompt(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        formatted_prompt = self.client.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(formatted_prompt, str):
            raise TypeError(
                "Tokenizer chat template must return text when tokenize=False."
            )
        return formatted_prompt

    def _generate(self, text: str | list[str]) -> Any:
        return self.client(
            text,
            do_sample=False,
            return_full_text=False,
        )

    @staticmethod
    def _extract_text(result: Any) -> str:
        while isinstance(result, list):
            if not result:
                raise ValueError("Language model returned no generations.")
            result = result[0]
        if not isinstance(result, dict) or not isinstance(
            result.get("generated_text"), str
        ):
            raise ValueError("Language model returned an invalid generation.")
        return result["generated_text"]
