import pandas as pd


class TopicReportCreator:
    """Create topic-level Markdown report sections."""

    def __init__(self, question_sample_size: int = 5):
        if question_sample_size < 0:
            raise ValueError("question_sample_size must be non-negative.")
        self.question_sample_size = question_sample_size

    def create(
        self,
        topics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create one Markdown report section per explained topic.

        Args:
            topics: Topic DataFrame with all information needed for reporting.
                It must contain ``id``, ``title``, ``explanation``, and
                ``avg_similarity``. ``relevance_score`` and
                ``relevance_reason`` are used when present.

        Returns:
            A copy of ``topics`` enriched with a ``report`` column. Topics
            without a title or explanation keep an empty report value.
        """
        self._validate_inputs(topics)

        topics_with_report = topics.copy()
        topics_with_report["report"] = pd.Series(dtype="object")
        if topics_with_report.empty:
            return topics_with_report

        overall_avg = topics_with_report["avg_similarity"].mean()
        explained_topics = topics_with_report[
            topics_with_report["title"].apply(self._has_text)
            & topics_with_report["explanation"].apply(self._has_text)
        ].sort_values("avg_similarity", na_position="last")

        for rank, (index, row) in enumerate(explained_topics.iterrows(), start=1):
            topics_with_report.loc[index, "report"] = self._create_topic_report(
                row=row,
                rank=rank,
                overall_avg=overall_avg,
            )

        return topics_with_report

    @staticmethod
    def to_markdown(topics: pd.DataFrame) -> str:
        """Combine non-empty topic report sections into one Markdown document."""
        if "report" not in topics:
            return ""

        sections = [
            section.strip()
            for section in topics["report"]
            if isinstance(section, str) and section.strip()
        ]
        if not sections:
            return ""
        return "\n\n".join(sections) + "\n"

    @staticmethod
    def _validate_inputs(topics: pd.DataFrame):
        required_columns = {"id", "title", "explanation", "avg_similarity"}
        missing_columns = required_columns - set(topics.columns)
        if missing_columns:
            raise ValueError(
                "topics must contain "
                f"{', '.join(sorted(missing_columns))}."
            )

    @classmethod
    def _create_topic_report(
        cls,
        row: pd.Series,
        rank: int,
        overall_avg: float,
    ) -> str:
        avg_similarity = row["avg_similarity"]
        lines = [
            f"## Rank {rank}: {row['title']} (Score {avg_similarity:.4f})",
            "",
            f"Topic ID: {int(row['id'])}",
            "",
            str(row["explanation"]),
            "",
            (
                "The average coverage score for this topic "
                f"is {avg_similarity:.4f}. "
                f"The average over all topics is {overall_avg:.4f}."
            ),
            "",
        ]

        if "relevance_score" in row and not pd.isna(row["relevance_score"]):
            reason = row.get("relevance_reason", "")
            reason_text = (
                f" - Justification: {reason}"
                if cls._has_text(reason)
                else ""
            )
            lines.extend(
                [
                    (
                        "**Relevance for the domain:** "
                        f"{float(row['relevance_score']):.2f}"
                        f"{reason_text}"
                    ),
                    "",
                ]
            )

        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _has_text(value) -> bool:
        return isinstance(value, str) and bool(value.strip())
