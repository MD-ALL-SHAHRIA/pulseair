"""Smoke test: every figure renders from the committed metrics files.

This is an end-to-end check, not a pixel check. It runs the real generator against
the real ``reports/metrics/*.json`` and asserts that each registered figure produced
a non-empty PNG. What it is actually protecting is the link between the figures and
the data: if a metrics file is renamed, a field is dropped, or a phase is re-run with
a different schema, a figure stops being reproducible and this test says so before
the thesis does.

The strict-accessor tests below are the other half. A figure that silently invents a
number is worse than a figure that fails, so :func:`require` is tested for raising.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.reporting import generate_figures as gf

REPO_ROOT = Path(__file__).resolve().parents[2]
MIN_PNG_BYTES = 5_000          # a 300-dpi plot is tens of KB; anything smaller is a stub


@pytest.fixture(scope="module")
def rendered():
    """Render the whole set once and share it across the assertions."""
    return gf.generate()


def test_registry_is_not_empty():
    assert len(gf.REGISTRY) >= 20, f"only {len(gf.REGISTRY)} figures registered"


def test_slugs_are_unique():
    slugs = [f.slug for f in gf.REGISTRY]
    assert len(slugs) == len(set(slugs)), "duplicate figure slugs would overwrite files"


def test_declared_sources_exist():
    """Every source a figure claims must be a file that is actually committed."""
    missing = [(f.slug, src) for f in gf.REGISTRY for src in f.sources
               if not (gf.METRICS / src).exists()]
    assert not missing, f"figures declare metrics files that do not exist: {missing}"


def test_every_figure_renders(rendered):
    failed = [(r.fig.slug, r.error) for r in rendered if not r.path]
    assert not failed, "figures failed to render:\n" + "\n".join(
        f"  {slug}: {err}" for slug, err in failed)


def test_every_png_is_written_and_non_empty(rendered):
    for r in rendered:
        assert r.path is not None, f"{r.fig.slug} produced no file"
        assert r.path.exists(), f"{r.path} does not exist"
        size = r.path.stat().st_size
        assert size > MIN_PNG_BYTES, f"{r.path.name} is only {size} bytes"


def test_pngs_are_really_pngs(rendered):
    """Guard against a truncated or mis-typed write."""
    for r in rendered:
        assert r.path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", \
            f"{r.path.name} is not a PNG"


def test_output_directory_holds_one_file_per_figure(rendered):
    on_disk = {p.stem for p in gf.FIGURES.glob("*.png")}
    expected = {f.slug for f in gf.REGISTRY}
    assert expected <= on_disk, f"not written: {sorted(expected - on_disk)}"


# ------------------------------------------------------- the never-guess guarantee


def test_require_returns_a_nested_value():
    assert gf.require({"a": {"b": {"c": 7}}}, "a.b.c", "x.json") == 7


def test_require_indexes_into_lists():
    assert gf.require({"rows": [{"v": 1}, {"v": 2}]}, "rows.1.v", "x.json") == 2


def test_require_raises_naming_the_missing_field():
    with pytest.raises(gf.MissingMetric) as exc:
        gf.require({"a": {"b": 1}}, "a.zzz", "some_file.json")
    msg = str(exc.value)
    assert "some_file.json" in msg and "a.zzz" in msg, msg
    assert "b" in msg, "the message should list the keys that are present"


def test_require_raises_on_a_missing_file():
    with pytest.raises(gf.MissingMetric) as exc:
        gf.load("no_such_metrics_file.json")
    assert "no_such_metrics_file.json" in str(exc.value)


def test_req_num_rejects_a_non_number():
    with pytest.raises(gf.MissingMetric, match="not a number"):
        gf.req_num({"a": "0.51"}, "a", "x.json")


def test_req_num_rejects_a_bool():
    """True is an int in Python, and a metric that is a flag is not a metric."""
    with pytest.raises(gf.MissingMetric, match="not a number"):
        gf.req_num({"significant": True}, "significant", "x.json")


def test_a_broken_metrics_field_fails_the_figure_rather_than_faking_it(tmp_path,
                                                                      monkeypatch):
    """Delete a required field and the figure must fail, not substitute."""
    src = "horizon_comparison.json"
    real = json.loads((gf.METRICS / src).read_text())
    del real["rows"]["6"]["observed"]["macro_f1"]

    monkeypatch.setitem(gf._CACHE, src, real)
    spec = next(f for f in gf.REGISTRY if f.slug.startswith("01_"))
    with pytest.raises(gf.MissingMetric, match="macro_f1"):
        spec.fn()
    gf._CACHE.pop(src, None)


def test_class_order_matches_the_committed_labels():
    """The fixed class order must be the labels the pipeline actually wrote.

    Ordinal classes sorted any other way make every per-class figure misread.
    """
    labels = gf.require(gf.load("ablation_h6.json"), "labels", "ablation_h6.json")
    assert list(labels) == gf.CLASS_ORDER
    assert set(gf.CLASS_SHORT) == set(gf.CLASS_ORDER)
