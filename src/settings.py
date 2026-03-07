from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import yaml


@dataclass(frozen=True)
class ArxivSettings:
    categories: tuple[str, ...]
    max_candidates: int


@dataclass(frozen=True)
class EmbeddingSettings:
    model: str
    batch_size: int


@dataclass(frozen=True)
class RankingSettings:
    top_k_neighbors: int
    max_results: int


@dataclass(frozen=True)
class EmailSettings:
    subject_prefix: str
    include_pdf_links: bool
    send_empty_email: bool


@dataclass(frozen=True)
class RuntimeSettings:
    data_dir: Path
    output_html: Path
    cache_dir: Path


@dataclass(frozen=True)
class AppSettings:
    arxiv: ArxivSettings
    embedding: EmbeddingSettings
    ranking: RankingSettings
    email: EmailSettings
    runtime: RuntimeSettings


@dataclass(frozen=True)
class SMTPSettings:
    host: str
    port: int
    username: str
    password: str
    recipient: str
    sender: str
    use_ssl: bool


def _require_int(section: dict, key: str, default: int) -> int:
    value = section.get(key, default)
    return int(value)


def _require_bool(section: dict, key: str, default: bool) -> bool:
    value = section.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def load_settings(config_path: Path) -> AppSettings:
    raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    arxiv_section = raw_data.get("arxiv", {})
    embedding_section = raw_data.get("embedding", {})
    ranking_section = raw_data.get("ranking", {})
    email_section = raw_data.get("email", {})
    runtime_section = raw_data.get("runtime", {})

    categories = tuple(str(item).strip() for item in arxiv_section.get("categories", []) if str(item).strip())

    return AppSettings(
        arxiv=ArxivSettings(
            categories=categories,
            max_candidates=_require_int(arxiv_section, "max_candidates", 80),
        ),
        embedding=EmbeddingSettings(
            model=str(embedding_section.get("model", "BAAI/bge-small-en-v1.5")),
            batch_size=_require_int(embedding_section, "batch_size", 32),
        ),
        ranking=RankingSettings(
            top_k_neighbors=max(1, _require_int(ranking_section, "top_k_neighbors", 5)),
            max_results=max(1, _require_int(ranking_section, "max_results", 15)),
        ),
        email=EmailSettings(
            subject_prefix=str(email_section.get("subject_prefix", "[arXiv Daily]")),
            include_pdf_links=_require_bool(email_section, "include_pdf_links", True),
            send_empty_email=_require_bool(email_section, "send_empty_email", False),
        ),
        runtime=RuntimeSettings(
            data_dir=Path(runtime_section.get("data_dir", "data")),
            output_html=Path(runtime_section.get("output_html", "output/latest_report.html")),
            cache_dir=Path(runtime_section.get("cache_dir", ".cache/recommender")),
        ),
    )


def load_smtp_settings() -> SMTPSettings:
    required_keys = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "EMAIL_TO")
    missing = [key for key in required_keys if not os.environ.get(key)]
    if missing:
        missing_joined = ", ".join(missing)
        raise ValueError(f"Missing required SMTP environment variables: {missing_joined}")

    port = int(os.environ["SMTP_PORT"])
    sender = os.environ.get("EMAIL_FROM", os.environ["SMTP_USER"])
    use_ssl_raw = os.environ.get("SMTP_USE_SSL")
    if use_ssl_raw is None:
        use_ssl = port == 465
    else:
        use_ssl = use_ssl_raw.strip().lower() in {"1", "true", "yes", "on"}

    return SMTPSettings(
        host=os.environ["SMTP_HOST"],
        port=port,
        username=os.environ["SMTP_USER"],
        password=os.environ["SMTP_PASSWORD"],
        recipient=os.environ["EMAIL_TO"],
        sender=sender,
        use_ssl=use_ssl,
    )
