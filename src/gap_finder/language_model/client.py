from openai import OpenAI

from gap_finder._auth import resolve_api_key


class ClientError(Exception):
    """Error raised when the language-model client returns an invalid result."""

    def __init__(self, *args, **context):
        super().__init__(*args)
        self.context = context


class OpenAIClient:
    """Configured OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=resolve_api_key(api_key),
        )

    def invoke(self, prompt: str, system_prompt: str | None = None) -> str | None:
        """Send a chat request and return its response text."""
        try:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as error:
            raise ClientError(*error.args) from error
