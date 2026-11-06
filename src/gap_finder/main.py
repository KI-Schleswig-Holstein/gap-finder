from __future__ import annotations

import logging
import shutil
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pandas as pd
from tqdm import tqdm

from gap_finder.embedding_model import factory as embedding_model_factory
from gap_finder.embedding_model.embedding_model import EmbeddingModel
from gap_finder.io_helper import FileHandler
from gap_finder.language_model import factory as language_model_factory
from gap_finder.language_model.language_model import LanguageModel
from gap_finder.model_types import (
    EmbeddingModelInput,
    LanguageModelInput,
    TopicEmbeddingModelInput,
)
from gap_finder import persistence
from gap_finder.question_generation import QuestionGenerator
from gap_finder.report import TopicReportCreator
from gap_finder.vector_store.pdf_chunker import PDFChunker
from gap_finder.vector_store.vector_store import VectorStore
from gap_finder.vector_store.numpy_vector_store import NumpyVectorStore

if TYPE_CHECKING:
    from bertopic import BERTopic

logger = logging.getLogger(__name__)

STANDARD_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STANDARD_TOPIC_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
STANDARD_LANGUAGE_MODEL = "Qwen/Qwen3.8-27B"
STANDARD_EMBEDDING_BATCH_SIZE = 32

class GapFinder:
    """Identify potential knowledge gaps in an indexed text corpus.

    A ``GapFinder`` combines an indexed corpus with embedding, language, and
    topic models. :meth:`find` generates or reuses questions, groups them with
    BERTopic, and ranks the resulting topics by corpus coverage.

    Attributes:
        vector_store: Indexed chunk texts and their corpus embeddings.
        embedding_model: Embedding model used for corpus indexing and question
            ranking.
        language_model: Language model used to generate questions and explain
            topics.
        topic_embedding_model: Embedding model configuration passed to
            BERTopic.
        questions: Configured questions before ``find()``, then the questions
            enriched with topic assignments, embeddings, and similarity scores.
        topics: Topic results produced by ``find()``. This attribute is not
            created by normal construction and becomes available after the
            pipeline creates topics.
        topic_model: Fitted ``BERTopic`` instance produced by ``find()``. This
            attribute is not created by normal construction.
        k_questions: Number of questions requested per corpus chunk.
        embedding_batch_size: Batch size used when embedding text.
        question_generation_batch_size: Batch size used for language-model
            question generation.
        summarize_chunks: Whether to summarize chunks before generating
            questions. This can improve question quality, but adds an extra
            language-model call per chunk.
        language: Language used for prompts and BERTopic.
        domain_specific_questions: Whether generated questions should focus on
            the configured domain.
        domain_description: Optional domain context for question generation and
            topic-relevance scoring.
        explain_top_k_topics: Number of lowest-coverage topics to explain.
        topic_explanation_sample_size: Maximum questions sampled per topic
            explanation.
        base_url: Base URL supplied to an OpenAI-compatible chat client.
        api_key: API key supplied to an OpenAI-compatible chat client.
        cleanup_models_after_usage: Whether model resources are released after
            applicable pipeline steps.
        save_intermediate_results: Whether pipeline artifacts are written to
            disk.
        base_path: Parent directory for intermediate results and snapshots.
        intermediate_results_path: Directory created for the current saved run,
            or ``None`` when intermediate saving is disabled or has not run.
    """

    vector_store: VectorStore
    embedding_model: EmbeddingModel
    language_model: LanguageModel
    topic_embedding_model: TopicEmbeddingModelInput
    questions: pd.DataFrame | None
    topics: pd.DataFrame | None
    if TYPE_CHECKING:
        topic_model: BERTopic | None
    k_questions: int
    embedding_batch_size: int
    question_generation_batch_size: int
    summarize_chunks: bool
    language: str
    domain_specific_questions: bool
    domain_description: str | None
    explain_top_k_topics: int
    topic_explanation_sample_size: int
    base_url: str | None
    api_key: str | None
    cleanup_models_after_usage: bool
    save_intermediate_results: bool
    base_path: Path | None
    intermediate_results_path: Path | None

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModelInput = STANDARD_EMBEDDING_MODEL,
        language_model: LanguageModelInput = STANDARD_LANGUAGE_MODEL,
        topic_embedding_model: TopicEmbeddingModelInput = STANDARD_TOPIC_EMBEDDING_MODEL,
        questions: list[str] | pd.DataFrame | None = None,
        *,
        k_questions: int = 3,
        embedding_batch_size: int = STANDARD_EMBEDDING_BATCH_SIZE,
        question_generation_batch_size: int = 1,
        summarize_chunks: bool = True,
        language: str = "english",
        domain_specific_questions: bool = False,
        domain_description: str | None = None,
        explain_top_k_topics: int = 10,
        topic_explanation_sample_size: int = 10,
        base_url: str | None = None,
        api_key: str | None = None,
        cleanup_models_after_usage: bool = False,
        save_intermediate_results: bool = False,
        base_path: str | Path | None = None,
    ):
        """Construct a GapFinder around an existing vector store.

        Use :meth:`from_pdf` or :meth:`from_chunks` when the corpus has not yet
        been embedded. Direct construction expects a populated ``vector_store``.

        Args:
            vector_store: Store containing corpus embeddings and matching text
                chunks.
            embedding_model: Corpus and question embedding model. Accepts a
                model name, ``SentenceTransformer``, ``OpenAIEmbeddingClient``,
                or ``EmbeddingModel``.
            language_model: Model for question generation and topic
                explanations. A string configures an OpenAI-compatible model;
                configured clients, Transformers text-generation pipelines,
                and ``LanguageModel`` instances are also accepted.
            topic_embedding_model: Model name or ``SentenceTransformer`` used
                by BERTopic.
            questions: Existing question strings or a DataFrame containing at
                least a ``question`` column. Supplying questions skips the most
                expensive automatic generation step when :meth:`find` runs.
            k_questions: Questions requested for each corpus chunk.
            embedding_batch_size: Batch size for embedding operations.
            question_generation_batch_size: Prompts processed in one language
                model batch.
            summarize_chunks: Whether to summarize chunks before question
                generation.
            language: Language used in generated text and by BERTopic.
            domain_specific_questions: Whether question prompts should focus on
                ``domain_description``.
            domain_description: Optional domain context. When set, ``find()``
                also scores explained topics for domain relevance.
            explain_top_k_topics: Number of lowest-coverage topics to title and
                explain.
            topic_explanation_sample_size: Maximum questions sampled to explain
                each selected topic.
            base_url: Base URL for a string-configured OpenAI-compatible
                language model.
            api_key: API key for that language model. If omitted, the
                ``GAP_FINDER_API_KEY`` environment variable is used.
            cleanup_models_after_usage: Whether to release owned model
                resources after applicable steps and after ``find()``.
            save_intermediate_results: Whether to save intermediate DataFrames
                and the Markdown report.
            base_path: Parent directory for saved intermediate results and
                snapshots. Required when ``save_intermediate_results`` is true.

        Raises:
            ValueError: If intermediate saving is enabled without ``base_path``.
        """
        self.vector_store: VectorStore = vector_store
        self.embedding_model = embedding_model_factory.create_embedding_model(
            embedding_model=embedding_model,
        )
        self.language_model = language_model_factory.create_language_model(
            language_model=language_model,
            base_url=base_url,
            api_key=api_key,
        )
        self.topic_embedding_model = topic_embedding_model
        self.language: str = language
        self.k_questions: int = k_questions
        self.embedding_batch_size: int = embedding_batch_size
        self.question_generation_batch_size: int = question_generation_batch_size
        self.summarize_chunks: bool = summarize_chunks
        self.domain_specific_questions: bool = domain_specific_questions
        self.domain_description: str | None = domain_description
        self.explain_top_k_topics: int = explain_top_k_topics
        self.topic_explanation_sample_size: int = topic_explanation_sample_size
        self.base_url: str | None = base_url
        self.api_key: str | None = api_key
        self.cleanup_models_after_usage: bool = cleanup_models_after_usage
        if save_intermediate_results and base_path is None:
            raise ValueError(
                "base_path must be provided when save_intermediate_results is enabled."
            )
        self.save_intermediate_results: bool = save_intermediate_results
        self.base_path: Path | None = (
            Path(base_path) if base_path is not None else None
        )
        self.intermediate_results_path: Path | None = None
        if questions is not None:
            self.set_questions(questions)
        else:
            self.questions: pd.DataFrame | None = None

    @classmethod
    def from_pdf(
        cls,
        folder_path: str,
        embedding_model: EmbeddingModelInput = STANDARD_EMBEDDING_MODEL,
        chunk_size: int = 2000,
        chunk_overlap: int = 50,
        vector_store_cls: type[VectorStore] = NumpyVectorStore,
        embedding_batch_size: int = STANDARD_EMBEDDING_BATCH_SIZE,
        **gap_finder_kwargs,
    ):
        """Build a GapFinder by chunking and indexing PDFs recursively.

        Args:
            folder_path: Directory searched recursively for ``.pdf`` files.
            embedding_model: Model used to embed the resulting chunks. See
                :meth:`__init__` for accepted forms.
            chunk_size: Maximum chunk size used by the PDF text splitter.
            chunk_overlap: Character overlap between adjacent chunks.
            vector_store_cls: ``VectorStore`` implementation to construct for
                the embedded chunks.
            embedding_batch_size: Number of chunks embedded per batch.
            **gap_finder_kwargs: Additional arguments forwarded to
                :meth:`__init__`.

        Returns:
            A GapFinder with all discovered PDFs indexed in a new vector store.

        Warns:
            UserWarning: If no PDFs are found below ``folder_path``.
        """
        pdf_chunker = PDFChunker(chunk_size, chunk_overlap)

        chunks = []

        pdf_files = list(Path(folder_path).rglob('*.pdf'))
        if not pdf_files:
            warnings.warn(f"No PDFs found under '{folder_path}'.")

        for file in tqdm(pdf_files, desc="Chunking PDFs", unit="file"):
            chunks.extend(pdf_chunker.chunk_pdf(file))

        return cls.from_chunks(
            chunks=chunks,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            vector_store_cls=vector_store_cls,
            **gap_finder_kwargs,
        )

    @classmethod
    def from_chunks(
        cls,
        chunks: list[str],
        embedding_model: EmbeddingModelInput = STANDARD_EMBEDDING_MODEL,
        vector_store_cls: type[VectorStore] = NumpyVectorStore,
        embedding_batch_size: int = STANDARD_EMBEDDING_BATCH_SIZE,
        **gap_finder_kwargs,
    ):
        """Build a GapFinder by embedding and indexing text chunks.

        Args:
            chunks: Pre-split corpus text.
            embedding_model: Model used to embed ``chunks``. See
                :meth:`__init__` for accepted forms.
            vector_store_cls: ``VectorStore`` implementation to construct. Its
                constructor must accept the embedding dimension as ``dim``.
            embedding_batch_size: Number of chunks embedded per batch.
            **gap_finder_kwargs: Additional arguments forwarded to
                :meth:`__init__`.

        Returns:
            A GapFinder containing a populated vector store.
        """
        embedding_model_instance = embedding_model_factory.create_embedding_model(
            embedding_model=embedding_model,
        )

        try:
            embeddings = embedding_model_instance.encode(
                chunks,
                show_progress_bar=True,
                batch_size=embedding_batch_size,
            )
            vector_store = vector_store_cls(dim=embeddings.shape[1])
            vector_store.add(embeddings, chunks)
            gap_finder = cls(
                vector_store=vector_store,
                embedding_model=embedding_model_instance,
                embedding_batch_size=embedding_batch_size,
                **gap_finder_kwargs,
            )
        except Exception:
            # Construction failed, so no GapFinder exists to own the model.
            embedding_model_instance.cleanup()
            raise

        if gap_finder.cleanup_models_after_usage:
            embedding_model_instance.deactivate()

        return gap_finder

    def create_questions(
        self,
        chunks: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate and store questions for the indexed chunks.

        Calling this method before :meth:`find` lets callers inspect or persist
        the computationally expensive result. ``find()`` reuses the stored
        questions instead of generating them again.

        Args:
            chunks: DataFrame with ``id`` and ``text`` columns. If omitted, the
                chunks and embeddings are loaded from the vector store.

        Returns:
            A DataFrame with one generated question per row and the columns
            ``id``, ``question``, and ``chunk_id``.
        """
        logger.info("Generating questions from indexed chunks...")
        if chunks is None:
            chunks = self._load_chunks_and_embeddings()

        with self._model_step(self.language_model):
            question_generator = QuestionGenerator(
                language_model=self.language_model,
                k_questions=self.k_questions,
                batch_size=self.question_generation_batch_size,
                summarize_chunks=self.summarize_chunks,
                domain_specific_questions=self.domain_specific_questions,
                language=self.language,
                domain_description=self.domain_description,
            )
            questions = question_generator.generate(chunks=chunks)
        logger.info(
            f"Generated {len(questions)} questions from {len(chunks)} chunks.")
        self.questions = questions
        return self.questions

    def set_questions(self, questions: list[str] | pd.DataFrame):
        """Store questions and bypass automatic question generation.

        Args:
            questions: Questions as strings or as a DataFrame containing at
                least a ``question`` column. Existing DataFrame columns are
                preserved as the pipeline adds its results.
        """
        if isinstance(questions, list):
            questions = pd.DataFrame({"question": questions})
        self.questions = questions

    def _get_or_create_questions(self, chunks: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return configured questions or generate them from ``chunks``."""
        if self.questions is None:
            logger.info(
                "No questions provided, generating questions from indexed chunks...")
            return self.create_questions(chunks=chunks)
        return self.questions

    def _create_topics(self, questions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, BERTopic]:
        """Add topic assignments to questions and create the topics DataFrame."""
        from gap_finder.topic_generation import TopicGenerator

        if self.questions is None:
            raise ValueError(
                "Questions must be set before creating topics. Call _get_or_create_questions() first.")

        topic_generator = TopicGenerator(
            language=self.language,
            embedding_model=self.topic_embedding_model,
        )

        topics, questions_with_topics, topic_model = topic_generator.generate(
            questions=questions,
            cleanup_after_generation=self.cleanup_models_after_usage)

        return topics, questions_with_topics, topic_model

    def _rank_topics(self, topics: pd.DataFrame, questions_with_topics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Enrich questions and topics with corpus-similarity statistics."""
        from gap_finder.question_ranking import QuestionRanker
        from gap_finder.topic_ranking import TopicRanker

        question_ranker = QuestionRanker(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            top_k=1,
            batch_size=self.embedding_batch_size,
        )
        with self._model_step(self.embedding_model):
            questions_with_rankings = question_ranker.rank(questions=questions_with_topics)

        topics = TopicRanker().rank(
            questions_with_topics=questions_with_rankings,
            topics=topics,
        )
        return topics, questions_with_rankings

    def _explain_topics(
        self,
        questions_with_topics: pd.DataFrame,
        topics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Enrich the lowest-coverage topics with titles and explanations."""
        from gap_finder.topic_explanation import TopicExplainer

        explainer = TopicExplainer(
            language_model=self.language_model,
            top_k=self.explain_top_k_topics,
            max_sample_size=self.topic_explanation_sample_size,
            language=self.language,
        )
        with self._model_step(self.language_model):
            topics_with_explanations = explainer.explain(
                questions_with_topics=questions_with_topics,
                topics=topics,
            )

        return topics_with_explanations

    def _score_topic_relevance(
        self,
        topics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Enrich explained topics with domain-relevance scores."""
        from gap_finder.topic_relevance import TopicRelevanceScorer

        if self.domain_description is None:
            logger.info("Skipping topic relevance scoring: no domain description set.")
            return topics

        scorer = TopicRelevanceScorer(
            language_model=self.language_model,
            language=self.language,
            domain_description=self.domain_description,
        )
        with self._model_step(self.language_model):
            topics = scorer.score(topics)

        return topics

    def _create_report(
        self,
        topics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Enrich explained topics with Markdown report sections."""
        topics = TopicReportCreator().create(
            topics=topics,
        )
        return topics

    def _load_chunks_and_embeddings(self) -> pd.DataFrame:
        """Return indexed chunks and embeddings as a DataFrame.

        Returns:
            A DataFrame with the columns ``id``, ``embedding``, and ``text``.
        """
        if self.vector_store is None:
            raise ValueError(
                "Vector store is empty. Please provide a valid vector store with chunks and embeddings.")
        embeddings, texts = self.vector_store.get_all()
        return pd.DataFrame({
            "id": range(len(texts)),
            "embedding": list(embeddings),
            "text": texts,
        })

    def find(self) -> pd.DataFrame:
        """Run the complete gap-finding pipeline.

        The pipeline obtains questions, creates BERTopic topics, ranks their
        corpus coverage, explains the lowest-coverage topics, optionally scores
        domain relevance, and creates Markdown report sections. On success the
        returned object is also available as :attr:`topics`; enriched question
        records and the fitted BERTopic model are available as :attr:`questions`
        and :attr:`topic_model`.

        Returns:
            The enriched topics DataFrame, including ranking statistics,
            explanations, optional relevance scores, and Markdown reports.
        """
        try:
            return self._run_pipeline()
        finally:
            if self.cleanup_models_after_usage:
                self.cleanup_models()

    def _run_pipeline(self) -> pd.DataFrame:
        """Run the pipeline body without applying final model cleanup."""
        file_handler = self._create_file_handler()
        step = 1
        chunks = self._load_chunks_and_embeddings()
        logger.info(f"Step {step}: Get or create questions.")
        questions = self._get_or_create_questions(chunks=chunks)
        if file_handler is not None:
            file_handler.save_questions(questions)
        self.questions = questions

        step += 1
        logger.info(f"Step {step}: Create topics.")
        topics, questions, topic_model = self._create_topics(questions=questions)
        if file_handler is not None:
            file_handler.save_questions_with_topics(questions)
        self.topics = topics
        self.questions = questions
        self.topic_model = topic_model

        step += 1
        logger.info(f"Step {step}: Rank topics.")
        topics, questions = self._rank_topics(topics=topics, questions_with_topics=questions)
        if file_handler is not None:
            file_handler.save_questions_rankings(questions)
            file_handler.save_topic_rankings(topics)
        self.questions = questions
        self.topics = topics

        step += 1
        logger.info(f"Step {step}: Explain topics.")
        topics = self._explain_topics(questions_with_topics=questions,
                                      topics=topics,)
        if file_handler is not None:
            file_handler.save_topic_explanations(topics)
        self.topics = topics

        if self.domain_description is not None:
            step += 1
            logger.info(f"Step {step}: Score topic relevance.")
            topics = self._score_topic_relevance(topics=topics)
            if file_handler is not None:
                file_handler.save_topic_relevance_scores(topics)
            self.topics = topics

        step += 1
        logger.info(f"Step {step}: Create report.")
        topics = self._create_report(topics=topics)
        if file_handler is not None:
            file_handler.save_report(TopicReportCreator.to_markdown(topics))
        self.topics = topics

        return topics

    @contextmanager
    def _model_step(
        self,
        model: EmbeddingModel | LanguageModel,
    ) -> Iterator[None]:
        """Keep a model active for one complete pipeline step."""
        model.activate()
        try:
            yield
        finally:
            if self.cleanup_models_after_usage:
                model.deactivate()

    def cleanup_models(self) -> None:
        """Release model resources owned by this GapFinder.

        Caller-supplied models are not moved between devices. The fitted
        BERTopic embedding backend is detached when one is present.
        """
        self.embedding_model.cleanup()
        self.language_model.cleanup()

        topic_model = getattr(self, "topic_model", None)
        if topic_model is not None:
            from gap_finder.topic_generation import release_topic_model_embedding

            release_topic_model_embedding(
                topic_model,
                owns_embedding_model=isinstance(self.topic_embedding_model, str),
            )

    def _create_file_handler(self) -> FileHandler | None:
        """Create a fresh output folder when intermediate saving is enabled."""
        if not self.save_intermediate_results:
            self.intermediate_results_path = None
            return None

        if self.base_path is None:
            raise ValueError(
                "base_path must be provided when save_intermediate_results is enabled."
            )
        file_handler = FileHandler(self.base_path)
        self.intermediate_results_path = file_handler.run_path
        return file_handler

    def save_vector_store(self, folder_path: str | Path):
        """Persist the indexed corpus for reuse with another GapFinder.

        Args:
            folder_path: Directory in which the vector store writes its files.

        The stored index can be loaded through its vector-store implementation
        and passed to :meth:`__init__`, for example to use another language
        model without embedding the corpus again.
        """
        self.vector_store.save_pdf_vector_store(
            folder_path, self._embedding_model_identifier())

    def _embedding_model_identifier(self) -> str:
        """Return the configured corpus embedding model's persistent name."""
        if isinstance(self.embedding_model, str):
            return self.embedding_model
        model_name = getattr(self.embedding_model, "model_name", None)
        if isinstance(model_name, str):
            return model_name
        model_name = getattr(self.embedding_model, "model_name_or_path", None)
        if isinstance(model_name, str):
            return model_name
        raise ValueError("embedding_model does not expose a persistent model name.")

    def save_model(self) -> Path:
        """Save a resumable, versioned GapFinder snapshot.

        The snapshot contains supported model descriptors, the vector store,
        configuration, and any available questions, topics, and fitted topic
        model. Only the built-in NumPy and FAISS vector stores are currently
        registered for complete snapshots.

        Returns:
            Directory containing the completed snapshot.

        Raises:
            ValueError: If a model or vector-store implementation cannot be
                represented by the snapshot format.
        """
        embedding_model_descriptor = persistence.describe_embedding_model(
            self.embedding_model
        )
        language_model_descriptor = persistence.describe_language_model(
            self.language_model
        )
        topic_embedding_model_descriptor = (
            persistence.describe_topic_embedding_model(self.topic_embedding_model)
        )
        persistence.describe_vector_store(self.vector_store)

        questions_state, questions = self._dataframe_attribute_state("questions")
        topics_state, topics = self._dataframe_attribute_state("topics")
        topic_model_state, topic_model = self._attribute_state("topic_model")

        snapshot_base_path = (
            self.base_path
            if self.base_path is not None
            else Path.cwd() / persistence.DEFAULT_SNAPSHOT_BASE_PATH
        )
        temporary_path, final_path = persistence.create_snapshot_paths(
            snapshot_base_path
        )

        try:
            if questions_state == "present":
                persistence.save_dataframe(
                    questions,
                    temporary_path / persistence.QUESTIONS_FILE,
                )
            if topics_state == "present":
                persistence.save_dataframe(
                    topics,
                    temporary_path / persistence.TOPICS_FILE,
                )

            vector_store_descriptor = persistence.save_vector_store(
                self.vector_store,
                temporary_path,
            )
            if topic_model_state == "present":
                persistence.save_topic_model(
                    topic_model,
                    temporary_path,
                    embedding_model_name=topic_embedding_model_descriptor[
                        "model_name"
                    ],
                )

            manifest = {
                "format_version": persistence.SNAPSHOT_FORMAT_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "configuration": self._persistent_configuration(),
                "artifacts": {
                    "questions": questions_state,
                    "topics": topics_state,
                    "topic_model": topic_model_state,
                },
                "models": {
                    "embedding": embedding_model_descriptor,
                    "language": language_model_descriptor,
                    "topic_embedding": topic_embedding_model_descriptor,
                },
                "vector_store": vector_store_descriptor,
                "environment": persistence.environment_metadata(),
            }
            persistence.write_manifest(
                manifest,
                temporary_path / persistence.MANIFEST_FILE,
            )
            temporary_path.rename(final_path)
        except Exception:
            if temporary_path.exists():
                shutil.rmtree(temporary_path, ignore_errors=True)
            raise

        logger.info("Saved GapFinder snapshot to %s", final_path)
        return final_path

    @classmethod
    def load_model(
        cls,
        model_path: str | Path,
        *,
        api_key: str | None = None,
        embedding_api_key: str | None = None,
        local_language_model_kwargs: dict[str, object] | None = None,
    ) -> GapFinder:
        """Restore a resumable GapFinder snapshot from ``model_path``.

        Args:
            model_path: Snapshot directory returned by :meth:`save_model`.
            api_key: API key used when restoring an OpenAI-compatible language
                model.
            embedding_api_key: API key used when restoring an
                OpenAI-compatible embedding model.
            local_language_model_kwargs: Options forwarded to
                ``transformers.pipeline`` for a restored local language model,
                such as ``device``, ``device_map``, ``torch_dtype``, or
                ``model_kwargs``.

        Returns:
            A GapFinder with the saved configuration and available artifacts.
        """
        snapshot_path = Path(model_path)
        manifest = persistence.load_manifest(snapshot_path)
        configuration = cls._validated_persistent_configuration(
            manifest["configuration"]
        )
        artifacts = manifest["artifacts"]
        models = manifest["models"]
        questions_state = cls._artifact_state(artifacts, "questions")
        topics_state = cls._artifact_state(artifacts, "topics")
        topic_model_state = cls._artifact_state(artifacts, "topic_model")

        embedding_model = persistence.restore_embedding_model(
            cls._manifest_mapping(models, "embedding"),
            api_key=embedding_api_key,
        )
        language_model_descriptor = cls._manifest_mapping(models, "language")
        language_model = persistence.restore_language_model(
            language_model_descriptor,
            api_key=api_key,
            pipeline_kwargs=local_language_model_kwargs,
        )
        topic_embedding_model = persistence.restore_topic_embedding_model(
            cls._manifest_mapping(models, "topic_embedding")
        )
        vector_store = persistence.restore_vector_store(
            manifest["vector_store"],
            snapshot_path,
        )

        questions = (
            persistence.load_dataframe(snapshot_path / persistence.QUESTIONS_FILE)
            if questions_state == "present"
            else None
        )
        base_url = (
            language_model_descriptor.get("base_url")
            if language_model_descriptor.get("kind") == "openai"
            else None
        )
        gap_finder = cls(
            vector_store=vector_store,
            embedding_model=embedding_model,
            language_model=language_model,
            topic_embedding_model=topic_embedding_model,
            questions=questions,
            base_url=base_url,
            api_key=api_key,
            base_path=snapshot_path.parent,
            **configuration,
        )

        if topics_state == "none":
            gap_finder.topics = None
        elif topics_state == "present":
            gap_finder.topics = persistence.load_dataframe(
                snapshot_path / persistence.TOPICS_FILE
            )

        if topic_model_state == "none":
            gap_finder.topic_model = None
        elif topic_model_state == "present":
            gap_finder.topic_model = persistence.restore_topic_model(
                snapshot_path,
                embedding_model=topic_embedding_model,
            )

        gap_finder.intermediate_results_path = None
        return gap_finder

    def _persistent_configuration(self) -> dict[str, object]:
        return {
            "k_questions": self.k_questions,
            "embedding_batch_size": self.embedding_batch_size,
            "question_generation_batch_size": self.question_generation_batch_size,
            "summarize_chunks": self.summarize_chunks,
            "language": self.language,
            "domain_specific_questions": self.domain_specific_questions,
            "domain_description": self.domain_description,
            "explain_top_k_topics": self.explain_top_k_topics,
            "topic_explanation_sample_size": self.topic_explanation_sample_size,
            "cleanup_models_after_usage": self.cleanup_models_after_usage,
            "save_intermediate_results": self.save_intermediate_results,
        }

    def _attribute_state(self, name: str) -> tuple[str, object | None]:
        if not hasattr(self, name):
            return "absent", None
        value = getattr(self, name)
        if value is None:
            return "none", None
        return "present", value

    def _dataframe_attribute_state(
        self,
        name: str,
    ) -> tuple[str, pd.DataFrame | None]:
        state, value = self._attribute_state(name)
        if value is None:
            return state, None
        if not isinstance(value, pd.DataFrame):
            raise ValueError(f"GapFinder.{name} must be a pandas DataFrame or None.")
        return state, value

    @classmethod
    def _validated_persistent_configuration(
        cls,
        configuration: dict[str, object],
    ) -> dict[str, object]:
        required = {
            "k_questions",
            "embedding_batch_size",
            "question_generation_batch_size",
            "summarize_chunks",
            "language",
            "domain_specific_questions",
            "domain_description",
            "explain_top_k_topics",
            "topic_explanation_sample_size",
            "cleanup_models_after_usage",
            "save_intermediate_results",
        }
        missing = required - configuration.keys()
        if missing:
            raise ValueError(
                "GapFinder snapshot configuration is missing: "
                f"{sorted(missing)}."
            )
        positive_integers = {
            "k_questions",
            "embedding_batch_size",
            "question_generation_batch_size",
            "explain_top_k_topics",
            "topic_explanation_sample_size",
        }
        for key in positive_integers:
            value = configuration[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    f"GapFinder snapshot configuration '{key}' must be positive."
                )
        for key in (
            "summarize_chunks",
            "domain_specific_questions",
            "cleanup_models_after_usage",
            "save_intermediate_results",
        ):
            if not isinstance(configuration[key], bool):
                raise ValueError(
                    f"GapFinder snapshot configuration '{key}' must be boolean."
                )
        language = configuration["language"]
        if not isinstance(language, str) or not language.strip():
            raise ValueError(
                "GapFinder snapshot configuration 'language' must be a string."
            )
        domain_description = configuration["domain_description"]
        if domain_description is not None and not isinstance(
            domain_description, str
        ):
            raise ValueError(
                "GapFinder snapshot configuration 'domain_description' must be "
                "a string or null."
            )
        return {key: configuration[key] for key in required}

    @staticmethod
    def _manifest_mapping(mapping: dict, key: str) -> dict:
        value = mapping.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"GapFinder snapshot manifest is missing '{key}'.")
        return value

    @staticmethod
    def _artifact_state(artifacts: dict, name: str) -> str:
        state = artifacts.get(name)
        if state not in {"absent", "none", "present"}:
            raise ValueError(
                f"GapFinder snapshot has invalid state for artifact '{name}': "
                f"{state!r}."
            )
        return state
