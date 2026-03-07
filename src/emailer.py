from __future__ import annotations

from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import html
import smtplib

from models import LibraryLoadStats, Recommendation
from settings import SMTPSettings


def _truncate(text: str, limit: int = 420) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def build_email_subject(subject_prefix: str, recommendation_count: int, generated_at: datetime) -> str:
    return f"{subject_prefix} {generated_at:%Y-%m-%d} ({recommendation_count} matches)"


def build_email_html(
    recommendations: list[Recommendation],
    stats: LibraryLoadStats,
    include_pdf_links: bool,
    generated_at: datetime,
) -> str:
    summary = (
        f"<p>Generated at {generated_at:%Y-%m-%d %H:%M UTC}. "
        f"Library papers with abstracts: {stats.entries_with_abstract}. "
        f"Skipped missing abstracts: {stats.skipped_missing_abstract}. "
        f"Duplicates removed: {stats.duplicates_removed}.</p>"
    )

    if not recommendations:
        body = (
            "<div class='paper-card'>"
            "<h2>No new matches today</h2>"
            "<p>The workflow ran successfully but did not find any new arXiv papers close to your bib corpus.</p>"
            "</div>"
        )
        return _wrap_html(summary + body)

    blocks = []
    for recommendation in recommendations:
        paper = recommendation.candidate
        neighbor_titles = ", ".join(
            f"{html.escape(match.title)} ({match.similarity:.3f})" for match in recommendation.neighbors
        )
        links = [f"<a href='{html.escape(paper.arxiv_url)}'>arXiv</a>"]
        if include_pdf_links and paper.pdf_url:
            links.append(f"<a href='{html.escape(paper.pdf_url)}'>PDF</a>")
        authors = ", ".join(html.escape(author) for author in paper.authors) or "Unknown authors"
        published = paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown date"
        blocks.append(
            "<div class='paper-card'>"
            f"<h2>{html.escape(paper.title)}</h2>"
            f"<p><strong>Score:</strong> {recommendation.score:.4f}</p>"
            f"<p><strong>Published:</strong> {published}</p>"
            f"<p><strong>Authors:</strong> {authors}</p>"
            f"<p><strong>Closest bib papers:</strong> {neighbor_titles}</p>"
            f"<p>{html.escape(_truncate(paper.abstract))}</p>"
            f"<p>{' | '.join(links)}</p>"
            "</div>"
        )

    return _wrap_html(summary + "".join(blocks))


def _wrap_html(content: str) -> str:
    return (
        "<!DOCTYPE html>"
        "<html><head><meta charset='utf-8'>"
        "<style>"
        "body { font-family: Arial, sans-serif; background: #f5f7fb; color: #16202a; margin: 0; padding: 24px; }"
        ".paper-card { background: #ffffff; border: 1px solid #d8dee9; border-radius: 12px; padding: 20px; margin-bottom: 18px; }"
        "a { color: #0057b8; text-decoration: none; }"
        "a:hover { text-decoration: underline; }"
        "p { line-height: 1.5; }"
        "</style>"
        "</head><body>"
        "<h1>arXiv Daily Recommendations</h1>"
        f"{content}"
        "</body></html>"
    )


def send_email(subject: str, html_body: str, smtp_settings: SMTPSettings) -> None:
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = smtp_settings.sender
    message["To"] = smtp_settings.recipient
    message.attach(MIMEText("Open this email in an HTML-capable client to view the recommendation list.", "plain", "utf-8"))
    message.attach(MIMEText(html_body, "html", "utf-8"))

    if smtp_settings.use_ssl:
        with smtplib.SMTP_SSL(smtp_settings.host, smtp_settings.port, timeout=60) as server:
            server.login(smtp_settings.username, smtp_settings.password)
            server.sendmail(smtp_settings.sender, [smtp_settings.recipient], message.as_string())
        return

    with smtplib.SMTP(smtp_settings.host, smtp_settings.port, timeout=60) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(smtp_settings.username, smtp_settings.password)
        server.sendmail(smtp_settings.sender, [smtp_settings.recipient], message.as_string())

