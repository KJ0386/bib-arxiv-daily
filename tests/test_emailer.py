from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emailer import build_email_html
from models import CandidatePaper, LibraryLoadStats, NeighborMatch, Recommendation


class EmailerTest(unittest.TestCase):
    def test_build_email_html_includes_pdf_link_and_neighbor_titles(self) -> None:
        recommendation = Recommendation(
            candidate=CandidatePaper(
                title="Graph Signal Learning",
                abstract="Learning on graph-structured data with spectral methods.",
                authors=("Author One", "Author Two"),
                entry_id="http://arxiv.org/abs/2501.00001v1",
                pdf_url="http://arxiv.org/pdf/2501.00001v1",
                published=datetime(2025, 1, 1),
                arxiv_id="2501.00001v1",
            ),
            score=0.9123,
            neighbors=(
                NeighborMatch(title="Graph Neural Networks", similarity=0.95),
                NeighborMatch(title="Spectral Graph Theory", similarity=0.88),
            ),
        )
        stats = LibraryLoadStats(
            files_scanned=2,
            entries_total=10,
            entries_with_abstract=8,
            duplicates_removed=1,
            skipped_missing_title=0,
            skipped_missing_abstract=1,
        )

        html = build_email_html([recommendation], stats, include_pdf_links=True, generated_at=datetime(2025, 1, 1))

        self.assertIn("Graph Signal Learning", html)
        self.assertIn("Graph Neural Networks", html)
        self.assertIn("http://arxiv.org/pdf/2501.00001v1", html)
        self.assertIn("Library papers with abstracts: 8", html)


if __name__ == "__main__":
    unittest.main()

