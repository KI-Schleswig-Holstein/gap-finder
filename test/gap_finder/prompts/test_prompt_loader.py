from gap_finder.prompts import PromptLoader
from gap_finder.singleton.constants import PromptTemplates


def test_prompt_loader_renders_template_with_variables(tmp_path):
    template = tmp_path / "template.txt.j2"
    template.write_text(
        "Hello {{ name }}. You have {{ count }} open questions.",
        encoding="utf-8",
    )
    loader = PromptLoader(templates_dir=tmp_path)

    rendered = loader.render("template.txt.j2", name="Ada", count=3)

    assert rendered == "Hello Ada. You have 3 open questions."


def test_prompt_loader_injects_standard_language_defaults(tmp_path):
    template = tmp_path / "template.txt.j2"
    template.write_text(
        "{{ language }}|{{ input_text_language }}|{{ question_language }}|{{ output_language }}",
        encoding="utf-8",
    )

    loader = PromptLoader(templates_dir=tmp_path)

    assert loader.render("template.txt.j2") == "english|english|english|english"


def test_prompt_loader_allows_language_overrides(tmp_path):
    template = tmp_path / "template.txt.j2"
    template.write_text(
        "{{ language }}|{{ input_text_language }}|{{ question_language }}|{{ output_language }}",
        encoding="utf-8",
    )
    loader = PromptLoader(
        templates_dir=tmp_path,
        language="italian",
        input_text_language="english",
        question_language="german",
        output_language="french",
    )

    rendered = loader.render("template.txt.j2", question_language="spanish")

    assert rendered == "italian|english|spanish|french"


def test_prompt_loader_uses_language_as_constructor_fallback(tmp_path):
    template = tmp_path / "template.txt.j2"
    template.write_text(
        "{{ language }}|{{ input_text_language }}|{{ question_language }}|{{ output_language }}",
        encoding="utf-8",
    )
    loader = PromptLoader(templates_dir=tmp_path, language="english")

    assert loader.render("template.txt.j2") == "english|english|english|english"


def test_question_generation_prompt_uses_domain_description_variable():
    loader = PromptLoader(language="english")

    rendered = loader.render(
        PromptTemplates.QUESTION_GENERATION,
        chunk="Example text.",
        k_questions=1,
        questions_placeholder='"Frage 1"',
        domain_specific_questions=True,
        domain_description="The domain is machine learning research.",
    )

    assert "DOMAIN:" in rendered
    assert "The domain is machine learning research." in rendered
    assert "question_generation/domain.txt.j2" not in rendered
