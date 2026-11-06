"""Generate original PDF fixtures used by examples and integration tests."""

# /// script
# dependencies = ["reportlab>=4.0"]
# ///

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
TEST_PDF_DIR = ROOT / "test" / "data" / "pdf"
EXAMPLE_PDF_DIR = ROOT / "examples" / "data" / "pdf"


DOCUMENTS = {
    TEST_PDF_DIR / "synthetic_language_systems.pdf": {
        "title": "Reliable Language Systems in Public Administration",
        "subtitle": "A synthetic technical report for software integration testing",
        "sections": [
            (
                "1. Purpose and operating context",
                [
                    "A regional service center is evaluating a language-assisted workflow for sorting incoming requests. The proposed system identifies the subject of each request, suggests a responsible team, and drafts a short internal summary. Staff members remain responsible for every decision and may revise or reject any suggestion. The evaluation therefore focuses on decision support rather than autonomous case handling.",
                    "Requests arrive through web forms, email, scanned letters, and telephone notes. They vary in length, terminology, and writing quality. Some contain several unrelated questions, while others omit information needed for routing. A reliable workflow must preserve the original message, record every transformation, and make uncertainty visible to the employee reviewing the case.",
                    "The initial pilot covers waste collection, building permits, and local transport. These domains were selected because they have distinct vocabularies and established escalation paths. The pilot does not include emergency services, legal determinations, benefit eligibility, or other decisions where an incorrect suggestion could immediately affect a person's rights or safety.",
                ],
            ),
            (
                "2. Data preparation and evaluation",
                [
                    "The evaluation corpus contains synthetic requests written by the project team. Each request has a documented intended category, a short rationale, and one or more ambiguity labels. Personal names, addresses, account numbers, and health information are deliberately excluded. This design permits repeatable experiments without exposing residents or employees to privacy risks.",
                    "Performance is measured at several levels. Routing accuracy indicates whether the suggested team matches the documented category. Coverage measures how often the system makes a usable suggestion. Calibration compares confidence values with observed accuracy. Review time records whether the suggestion helps staff complete the task faster without encouraging superficial approval.",
                    "Aggregate scores can conceal important weaknesses. Results are therefore broken down by channel, topic, text length, ambiguity, and language variety. Every failed case is assigned an error type such as missing context, competing topics, uncommon terminology, or unsupported inference. The error catalogue guides changes to prompts, reference material, and escalation rules.",
                ],
            ),
            (
                "3. Retrieval and response grounding",
                [
                    "Draft summaries are grounded in an approved collection of service descriptions and internal routing notes. Each retrieved passage carries a source identifier, publication date, and owning team. Material that has passed its review date is excluded until the owner confirms that it remains valid. The system displays the supporting passages beside every generated suggestion.",
                    "Retrieval quality is tested separately from generation quality. A relevant passage may be absent because the search query is weak, the index is incomplete, or the document uses different terminology. Conversely, a retrieved passage may be topically related but unsuitable for the specific request. Separate labels make these failure modes visible and prevent the generator from receiving credit for information it could not access.",
                    "The system must decline to draft a confident summary when evidence is missing or contradictory. A structured fallback message tells the reviewer which detail could not be established. This behavior is evaluated with intentionally incomplete requests and with pairs of source passages that describe different procedures for different municipalities.",
                ],
            ),
            (
                "4. Human review and operational safeguards",
                [
                    "Reviewers see the original request, the proposed category, the draft summary, confidence indicators, and cited source passages in one interface. Accepting a suggestion requires an explicit action. Corrections are logged as structured feedback, but they do not enter a training set automatically. A data steward first checks whether the correction contains personal or confidential information.",
                    "Operational monitoring tracks error rates, response latency, unavailable dependencies, and the proportion of suggestions that reviewers change. Sudden shifts trigger an investigation rather than automatic adaptation. Model versions, prompt versions, index snapshots, and configuration values are recorded for each suggestion so that an incident can be reproduced.",
                    "A rollback plan keeps the previous validated configuration available. If monitoring reveals an unexpected pattern, the service can return to manual routing without losing incoming requests. Staff training emphasizes that fluent text is not evidence of correctness and that citations must be checked when a request is unusual or consequential.",
                ],
            ),
            (
                "5. Open questions",
                [
                    "The pilot leaves several questions unresolved. The team has not yet established how performance changes during seasonal peaks, how well the workflow handles mixed-language messages, or how reviewers interpret confidence indicators under time pressure. It is also unclear how often source owners can maintain routing notes when procedures change rapidly.",
                    "Future work will examine accessibility, long-term reviewer behavior, and the cost of maintaining the evidence collection. A successful pilot requires more than a high accuracy score: the workflow must remain understandable, contestable, secure, and useful under ordinary operating constraints.",
                ],
            ),
        ],
    },
    TEST_PDF_DIR / "synthetic_coastal_planning.pdf": {
        "title": "Coastal Adaptation Workshop Notes",
        "subtitle": "Synthetic planning material",
        "sections": [
            (
                "Workshop summary",
                [
                    "A fictional coastal district compared three adaptation measures: restoring wetlands, raising selected roads, and creating shaded emergency gathering points. Participants scored each measure for maintenance effort, accessibility, ecological impact, and usefulness during compound events. The exercise used invented locations and values so that the document can be shared as a software test fixture.",
                    "The group identified missing evidence about winter groundwater levels, the travel needs of residents without cars, and maintenance responsibilities after grant funding ends. No preferred measure was selected. The next workshop will test combinations of measures and document which assumptions drive the ranking.",
                ],
            )
        ],
    },
    TEST_PDF_DIR / "synthetic_library_services.pdf": {
        "title": "Mobile Library Service Review",
        "subtitle": "Synthetic service-design material",
        "sections": [
            (
                "Review summary",
                [
                    "A fictional library network tested a mobile service at six rural stops. The review tracked visit frequency, requested media, wheelchair access, mobile connectivity, and the time required to load reserved items. Survey responses suggested that predictable schedules mattered more than the total number of weekly stops.",
                    "Important gaps remain. The pilot did not include school holidays, severe weather, or residents who need home delivery. Future evaluation should compare operating costs with social outcomes and should record why potential users choose not to visit the service.",
                ],
            )
        ],
    },
    EXAMPLE_PDF_DIR / "synthetic_knowledge_gaps.pdf": {
        "title": "Community Energy Advice Pilot",
        "subtitle": "Synthetic example document for Gap Finder",
        "sections": [
            (
                "Pilot overview",
                [
                    "A fictional municipality offered energy consultations at three neighborhood centers for twelve weeks. Advisors explained heating controls, insulation options, and ways to compare electricity contracts. Attendance was recorded by session and broad housing type, without collecting names or exact addresses.",
                    "Participants most often asked about apartment buildings with shared heating systems. The prepared guidance focused mainly on detached homes, so advisors created temporary notes for recurring questions. These notes were useful during the pilot but were not reviewed as formal guidance.",
                ],
            ),
            (
                "Observed limitations",
                [
                    "The pilot did not measure whether households later changed equipment or reduced energy use. Evening sessions were offered only once, and interpretation was available for two languages. Renters, people with limited mobility, and residents without internet access may therefore be underrepresented in the feedback.",
                    "A follow-up study should define outcome measures, compare different appointment formats, and examine who does not use the service. It should also clarify which organization is responsible for updating technical advice when prices, subsidies, or regulations change.",
                ],
            ),
        ],
    },
}


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="DocumentTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=colors.HexColor("#123B5D"),
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#486879"),
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#176B87"),
            spaceBefore=10,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCopy",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=colors.HexColor("#25333A"),
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Notice",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#5D6D75"),
            alignment=TA_CENTER,
        )
    )
    return styles


def _draw_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#B7CDD7"))
    canvas.setLineWidth(0.5)
    canvas.line(22 * mm, 17 * mm, width - 22 * mm, 17 * mm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#5D6D75"))
    canvas.drawString(22 * mm, 11 * mm, "Gap Finder synthetic fixture")
    canvas.drawRightString(width - 22 * mm, 11 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(path: Path, title: str, subtitle: str, sections):
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=24 * mm,
        leftMargin=24 * mm,
        topMargin=22 * mm,
        bottomMargin=24 * mm,
        title=title,
        author="KI.SH",
        subject="Original synthetic document for Gap Finder testing",
        creator="Gap Finder fixture generator",
        invariant=1,
    )

    notice = (
        "This is an original synthetic document created for software testing. "
        "It contains no real people, cases, measurements, or external source text."
    )
    story = [
        Spacer(1, 8 * mm),
        Paragraph(title, styles["DocumentTitle"]),
        Paragraph(subtitle, styles["Subtitle"]),
        Table(
            [[Paragraph(notice, styles["Notice"])]],
            colWidths=[150 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF6F8")),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9DC5D2")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            ),
        ),
        Spacer(1, 8 * mm),
    ]

    for heading, paragraphs in sections:
        story.append(Paragraph(heading, styles["SectionHeading"]))
        story.extend(Paragraph(paragraph, styles["BodyCopy"]) for paragraph in paragraphs)

    doc.build(story, onFirstPage=_draw_page, onLaterPages=_draw_page)


def main():
    for path, content in DOCUMENTS.items():
        build_pdf(path=path, **content)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
