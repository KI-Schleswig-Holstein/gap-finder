from pathlib import Path

from jinja2 import Environment, FileSystemLoader, PackageLoader

STANDARD_LANGUAGE = "english"


class PromptLoader:
    def __init__(
        self,
        templates_dir: Path | None = None,
        language: str | None = None,
        input_text_language: str | None = None,
        question_language: str | None = None,
        output_language: str | None = None,
    ):
        loader = (
            FileSystemLoader(str(templates_dir))
            if templates_dir is not None
            else PackageLoader("gap_finder.prompts", "templates")
        )
        self._env = Environment(
            loader=loader,
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        explicit_language = language is not None
        language = language if explicit_language else STANDARD_LANGUAGE
        self._default_context = {
            "language": language,
            "input_text_language": input_text_language
            or language,
            "question_language": question_language
            or language,
            "output_language": output_language
            or language,
        }

    def render(self, template_name: str, **kwargs) -> str:
        """Load and render a Jinja2 template with the given variables."""
        template = self._env.get_template(template_name)
        context = {**self._default_context, **kwargs}
        return template.render(**context)
