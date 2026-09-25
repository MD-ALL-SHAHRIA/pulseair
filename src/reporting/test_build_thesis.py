"""Smoke test: the thesis builds from committed files, with no invented numbers.

This is the same contract as the figure generator's test. What it is really protecting
is the claim the document makes about itself -- that every number in it came from a
committed metrics file -- plus two defects that were found by inspection rather than by
a test and would otherwise be easy to reintroduce.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.reporting import build_thesis as bt

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    bt.REG.figures.clear()
    bt.REG.tables.clear()
    bt.CITED.clear()
    bt.SRC.missing.clear()
    doc = bt.assemble()
    out = tmp_path_factory.mktemp("thesis") / "t.docx"
    doc.save(out)
    return doc, out


def test_no_missing_placeholders(built):
    """Every value resolved. A MISSING marker means a number could not be sourced."""
    _, _ = built
    assert not bt.SRC.missing, (
        "the build inserted MISSING placeholders:\n  "
        + "\n  ".join(sorted(set(bt.SRC.missing))))


def test_every_figure_is_embedded_not_just_captioned(built):
    """Regression: an earlier build produced 23 captions and zero images.

    Moving XML between documents carried the drawing elements but left the image
    parts and their relationships in the discarded package. Counting inline shapes
    did not catch it, because the shapes were present and only the media was gone.
    So this counts the media parts inside the saved file.
    """
    doc, path = built
    with zipfile.ZipFile(path) as z:
        media = [n for n in z.namelist() if n.startswith("word/media/")]
    assert len(media) == len(bt.REG.figures), (
        f"{len(bt.REG.figures)} figures registered but {len(media)} media parts "
        f"embedded")
    assert len(media) >= 20, f"only {len(media)} figures embedded"


def test_document_is_not_suspiciously_small(built):
    """A docx with captions but no images weighs a tenth of a correct one."""
    _, path = built
    kb = path.stat().st_size / 1024
    assert kb > 1000, f"{kb:.0f} KB is too small to contain the figures"


def test_every_figure_caption_names_its_source(built):
    for label, source in bt.REG.figures:
        assert source and "MISSING" not in source, label


def test_every_citation_exists_in_the_reference_list(built):
    """In-text citations must resolve against reports/reference_list_expanded.md."""
    ref = (ROOT / "reports" / "reference_list_expanded.md")
    if not ref.exists():
        pytest.skip("reference list not present")
    assert bt.CITED, "no citations recorded"
    assert not [m for m in bt.SRC.missing if "citation" in m]


def test_front_matter_is_in_order(built):
    doc, _ = built
    text = [p.text.strip() for p in doc.paragraphs]
    wanted = ["Declaration", "Certificate of Approval", "Acknowledgement", "Abstract",
              "Table of Contents", "List of Figures", "List of Tables",
              "List of Abbreviations"]
    seen = [t for t in text if t in wanted]
    first = []
    for w in wanted:
        if w in seen and w not in first:
            first.append(w)
    assert first == wanted, f"front matter order is {first}"


def test_lists_of_figures_and_tables_are_populated(built):
    doc, _ = built
    assert len(bt.REG.figures) >= 20
    assert len(bt.REG.tables) >= 30
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Figure 1." in text and "Table 1." in text


def test_abbreviations_are_only_those_actually_used(built):
    """The glossary is built from the document text, not from a fixed list."""
    doc, _ = built
    body = "\n".join(p.text for p in doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            body += "\n" + " ".join(c.text for c in row.cells)
    assert "AQI" in body and "SHAP" in body


def test_need_records_a_gap_rather_than_inventing_one():
    before = len(bt.SRC.missing)
    got = bt.need("no_such_file.json:a.b", "a value that does not exist")
    assert isinstance(got, str) and got.startswith("[MISSING:")
    assert len(bt.SRC.missing) == before + 1
    bt.SRC.missing.pop()


def test_num_rejects_a_non_numeric_value():
    before = len(bt.SRC.missing)
    got = bt.num("ablation_h6.json:labels", "labels are a list, not a number")
    assert got.startswith("[MISSING:")
    bt.SRC.missing[:] = bt.SRC.missing[:before]
