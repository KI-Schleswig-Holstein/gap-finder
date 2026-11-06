import json
import logging
import re
import pandas as pd
from tqdm import tqdm
from gap_finder.language_model.language_model import LanguageModel
from gap_finder.prompts import PromptLoader
from gap_finder.singleton.constants import PromptTemplates

logger = logging.getLogger(__name__)
class QuestionGenerator():
    """Class for generating questions from PDF document chunks."""

    def __init__(self,
                 language_model: LanguageModel,
                 k_questions: int = 2,
                 summarize_chunks: bool = False,
                 domain_specific_questions: bool = False,
                 batch_size: int = 5,
                 language: str = "english",
                 domain_description: str | None = None,
                 ) -> None:
        self.language_model = language_model
        self.k_questions = k_questions
        self.summarize_chunks = summarize_chunks
        self._current_call_type: str = "unknown"
        self._prompt_loader = PromptLoader()
        self.domain_specific_questions = domain_specific_questions
        self.batch_size = batch_size
        self.language = language
        self.domain_description = domain_description

    def generate(
        self,
        chunks: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate unanswerable questions for each chunk.

        Args:
            chunks: DataFrame with one row per chunk and the columns ``id`` and
                ``text``.

        Returns:
            A DataFrame with one generated question per row and the columns
            ``id``, ``question``, and ``chunk_id``.
        """
        total_chunks = len(chunks)
        logger.info(f"Generating questions for {total_chunks} chunks "
                    f"with batch size {self.batch_size}...")
        metadata = []
        total_batches = (
            total_chunks + self.batch_size - 1
        ) // self.batch_size

        progress = tqdm(
            range(0, total_chunks, self.batch_size),
            total=total_batches,
            desc=(
                "Generating questions "
                f"({total_chunks} chunks, {total_batches} batches)"
            ),
            unit="batch",
        )
        for batch_number, start in enumerate(progress, start=1):
            end = min(start + self.batch_size, total_chunks)
            progress.set_postfix_str(
                f"batch {batch_number}/{total_batches}, "
                f"chunks {start + 1}-{end}/{total_chunks}, "
                "waiting for question generation model response..."
            )
            batch = chunks.iloc[start:start + self.batch_size]
            try:
                texts = batch["text"].tolist()
                if self.summarize_chunks:
                    texts = self._summarize_texts(texts)

                questions_batch = self._generate_questions_batch(texts)
                chunk_ids = batch["id"].tolist()
                for i, questions in enumerate(questions_batch):
                    for q in questions.get("questions", []):
                        if isinstance(q, str) and q.strip():
                            metadata.append({"question": q, "chunk_id": chunk_ids[i]})
            except Exception as e:
                logger.error(f"[BATCH ERROR] batch starting at {start}: {e}")
        result_df = pd.DataFrame(metadata, columns=["question", "chunk_id"])
        result_df.insert(0, "id", range(len(result_df)))
        return result_df

    def _summarize_texts(self, texts: list[str]) -> list[str]:
        """Summarize a list of text chunks before question generation."""
        prompts = [self._get_summarize_prompt(text) for text in texts]
        logger.info(f"Generating summary for {len(prompts)} text chunks.")
        responses = self.language_model.invoke_batch(prompts)
        return [response.strip() for response in responses]

    def _get_summarize_prompt(self, text: str) -> str:
        """Render the summarization prompt for a single text chunk."""
        return self._prompt_loader.render(PromptTemplates.SUMMARIZE, text=text, language=self.language)

    def _generate_questions_batch(self, texts: list[str]) -> list[dict]:
        """
        Generate questions for a batch of texts.
        Creates prompts, calls the model, and parses responses.

        args:
            texts [list[str]]: list of text chunks
        returns:
            list[dict]: list of question dictionaries, where each dict has the structure:
                {"questions": [list of questions]}
        """
        prompts = [self._get_questions_prompt(text) for text in texts]
        logger.info(f"Generating questions for {len(prompts)} text chunks.")
        responses = self.language_model.invoke_batch(prompts)

        results = []
        fail = 0
        for response in responses:
            try:
                results.append(self._extract_json(response))
            except Exception:
                fail += 1
                logger.error(
                    f"Error occurred while processing response: {response}")
                results.append({"questions": []})
        logger.info(
            f"LLM generated questions for {len(results)-fail}/{len(results)} chunks.")
        if fail > 0:
            logger.info(f"({fail} responses were invalid JSONs.)")
        return results

    def _extract_json(self, text: str) -> dict:
        """Parse the first JSON object from a raw model response."""
        match = re.search(r"\{[\s\S]*\}", text)

        if match is None:
            raise ValueError("No JSON object found in text")

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        if "questions" not in data:
            raise ValueError("Missing required key 'questions'")

        return data

    def _get_questions_prompt(self, chunk: str) -> str:
        """Render the question-generation prompt for a single text chunk."""
        questions_placeholder = ", ".join(
            [f'"Frage {i+1}"' for i in range(self.k_questions)])
        return self._prompt_loader.render(
            PromptTemplates.QUESTION_GENERATION,
            language=self.language,
            chunk=chunk,
            k_questions=self.k_questions,
            questions_placeholder=questions_placeholder,
            domain_specific_questions=self.domain_specific_questions,
            domain_description=self.domain_description,
        )
