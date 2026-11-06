from gap_finder.language_model.client import ClientError, OpenAIClient
from gap_finder.language_model.language_model import LanguageModel


class OpenAILanguageModel(LanguageModel):
    """Language model backed by a configured OpenAI-compatible client."""

    def __init__(self, client: OpenAIClient) -> None:
        self.client = client

    def invoke(self, prompt: str, system_prompt: str | None = None) -> str:
        response = self.client.invoke(prompt=prompt, system_prompt=system_prompt)
        return self._require_response(response, prompt, system_prompt)

    def invoke_batch(
        self,
        prompts: list[str],
        system_prompt: str | None = None,
    ) -> list[str]:
        return [
            self._require_response(
                self.client.invoke(prompt=prompt, system_prompt=system_prompt),
                prompt,
                system_prompt,
                index=index,
            )
            for index, prompt in enumerate(prompts)
        ]

    def _require_response(
        self,
        response: str | None,
        prompt: str,
        system_prompt: str | None,
        **context,
    ) -> str:
        if response is None:
            raise ClientError(
                "OpenAIClient returned None response.",
                prompt=prompt,
                system_prompt=system_prompt,
                client=self.client,
                **context,
            )
        return response
