from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bib_loader import build_library_identity_set, load_library


class BibLoaderTest(unittest.TestCase):
    def test_load_library_merges_duplicates_and_skips_missing_abstract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            (data_dir / "library_a.bib").write_text(
                textwrap.dedent(
                    """
                    @article{paper1,
                      title = {Graph Representation Learning},
                      abstract = {A survey on graph representation learning.},
                      doi = {10.1000/graph}
                    }

                    @article{paper_missing,
                      title = {No Abstract Example}
                    }
                    """
                ).strip(),
                encoding="utf-8",
            )
            (data_dir / "library_b.bib").write_text(
                textwrap.dedent(
                    """
                    @article{paper2,
                      title = {Graph Representation Learning},
                      abstract = {A longer survey on graph representation learning with additional detail.},
                      doi = {10.1000/graph}
                    }

                    @article{paper3,
                      title = {Vision Transformer Advances},
                      abstract = {Transformers are now useful for dense visual tasks.},
                      eprint = {2401.12345}
                    }
                    """
                ).strip(),
                encoding="utf-8",
            )

            papers, stats = load_library(data_dir)

        self.assertEqual(2, len(papers))
        self.assertEqual(4, stats.entries_total)
        self.assertEqual(1, stats.duplicates_removed)
        self.assertEqual(1, stats.skipped_missing_abstract)
        self.assertTrue(any("longer survey" in paper.abstract.lower() for paper in papers))

        identities = build_library_identity_set(papers)
        self.assertIn("doi:10.1000/graph", identities)
        self.assertIn("arxiv:2401.12345", identities)


if __name__ == "__main__":
    unittest.main()

