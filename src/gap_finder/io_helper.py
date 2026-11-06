"""I/O helpers for loading and saving all pipeline artifacts."""
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from gap_finder.singleton import constants

logger = logging.getLogger(__name__)


def _save_dataframe(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient='records', force_ascii=False, indent=2)
    logger.info(f"Saved {len(df)} entries to {path}")


def _save_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    logger.info(f"Saved report to {path}")


def bertopic_model_path(base_path: str, id: str) -> Path:
    return Path(base_path) / Path(id) / constants.DataFileNames.BERTOPIC_MODEL


class FileHandler:
    """Save the artifacts produced by one GapFinder run."""

    def __init__(self, base_path: str | Path):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.run_path = Path(base_path) / f"{timestamp}_{uuid4().hex[:8]}"
        self.run_path.mkdir(parents=True, exist_ok=False)
        logger.info(f"Saving intermediate results to {self.run_path}")

    def _path(self, filename: str) -> Path:
        return self.run_path / filename

    def save_questions(self, questions: pd.DataFrame):
        _save_dataframe(
            questions, self._path(constants.DataFileNames.QUESTIONS))

    def save_questions_with_topics(self, questions: pd.DataFrame):
        _save_dataframe(
            questions,
            self._path(constants.DataFileNames.QUESTIONS_WITH_TOPICS),
        )

    def save_questions_rankings(self, questions: pd.DataFrame):
        _save_dataframe(
            questions, self._path(constants.DataFileNames.QUESTIONS_RANKINGS))

    def save_topic_rankings(self, topics: pd.DataFrame):
        _save_dataframe(
            topics, self._path(constants.DataFileNames.TOPICS_RANKINGS_JSON))

    def save_topic_explanations(self, topics: pd.DataFrame):
        _save_dataframe(
            topics, self._path(constants.DataFileNames.TOPIC_EXPLANATIONS_JSON))

    def save_topic_relevance_scores(self, topics: pd.DataFrame):
        _save_dataframe(
            topics,
            self._path(constants.DataFileNames.TOPIC_RELEVANCE_SCORES_JSON),
        )

    def save_report(self, report: str):
        _save_text(report, self._path(constants.DataFileNames.REPORT_MD))
