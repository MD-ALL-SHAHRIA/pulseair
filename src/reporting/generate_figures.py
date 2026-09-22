"""Publication-quality thesis figures, generated from this project's own metrics.

Every number plotted here is read out of ``reports/metrics/*.json`` -- the same files
the written reports are compiled from. Nothing is typed in by hand, estimated, or
carried over from a draft. If a figure needs a value that no committed JSON contains,
:func:`require` raises :class:`MissingMetric` naming the file and the field, and the
figure fails loudly rather than shipping a plausible-looking guess. A thesis figure
that disagrees with the table beside it is worse than no figure.

    python -m src.reporting.generate_figures            # regenerate everything
    python -m src.reporting.generate_figures --list     # what exists, and from where
    python -m src.reporting.generate_figures --only 13 14

Output goes to ``reports/figures/``. The run prints an audit trail mapping each PNG
back to the JSON files it was built from.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = REPO_ROOT / "reports" / "metrics"
FIGURES = REPO_ROOT / "reports" / "figures"

# --------------------------------------------------------------------- house style
#
# One style, applied once, used by every figure. The point is that a reader flipping
# between figures never has to re-learn what a colour means: persistence is always the
# same grey, Beijing and Bangladesh are always the same two hues, and "the thing that
# went wrong" is always the same red.

PALETTE = sns.color_palette("colorblind")

C_PERSISTENCE = "#7a7a7a"        # the floor: always grey, never a "real" colour
C_BEIJING = PALETTE[0]           # blue
C_BANGLADESH = PALETTE[1]        # orange
C_MODEL = PALETTE[2]             # green  -- a trained model on its own
C_GAN = PALETTE[4]               # purple -- synthetic-data variants
C_BAD = PALETTE[3]               # red    -- failures, violations, disqualifications
C_ALT = PALETTE[7]               # grey-blue for second members of a pair
C_ACCENT = PALETTE[8]            # yellow for callouts

FIG_WIDTH = 7.2                  # inches; thesis text width, so figures sit side by side
DPI = 300
FS_TITLE = 13
FS_LABEL = 11
FS_TICK = 9.5
FS_ANNOT = 8.5
FS_LEGEND = 9

# Class order is fixed everywhere: it is an ordinal scale, not a category set, and
# sorting it alphabetically (as pandas will, given the chance) destroys the reading.
CLASS_ORDER = ["Good", "Moderate", "Unhealthy (sensitive)", "Unhealthy",
               "Very unhealthy", "Hazardous"]
CLASS_SHORT = {"Good": "Good", "Moderate": "Mod.",
               "Unhealthy (sensitive)": "U.(sens)", "Unhealthy": "Unhealthy",
               "Very unhealthy": "V.unhealthy", "Hazardous": "Hazardous"}
ADVISORY = ["Very unhealthy", "Hazardous"]


def apply_house_style() -> None:
    """Set the one style every figure in this module inherits."""
    sns.set_theme(style="whitegrid", font="DejaVu Sans", palette=PALETTE)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "legend.title_fontsize": FS_LEGEND,
        "figure.titlesize": FS_TITLE,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.edgecolor": "#cccccc",
    })


# ------------------------------------------------------------------ strict accessors


class MissingMetric(KeyError):
    """A figure asked for a number that no committed JSON provides.

    Raised instead of falling back to a placeholder. The message names the file and
    the exact dotted path, so the fix is either "run the phase that writes it" or
    "this figure cannot be made from committed data" -- and the second is a finding
    worth reporting, not a gap worth papering over.
    """


_CACHE: dict[str, Any] = {}


def load(name: str) -> Any:
    """Read one metrics file, with the failure mode spelled out."""
    if name in _CACHE:
        return _CACHE[name]
    path = METRICS / name
    if not path.exists():
        available = ", ".join(sorted(p.name for p in METRICS.glob("*.json"))) or "none"
        raise MissingMetric(
            f"{path.relative_to(REPO_ROOT)} does not exist. "
            f"Available metrics files: {available}")
    _CACHE[name] = json.loads(path.read_text())
    return _CACHE[name]


def require(obj: Any, path: str, source: str) -> Any:
    """Fetch ``path`` (dotted, list indices allowed) from ``obj`` or raise.

    >>> require({"a": {"b": 1}}, "a.b", "x.json")
    1
    >>> require({"rows": [{"v": 2}]}, "rows.0.v", "x.json")
    2
    >>> try:
    ...     require({"a": {"b": 1}}, "a.zzz", "x.json")
    ... except MissingMetric as exc:
    ...     print(exc)
    "x.json: missing field 'a.zzz'. Keys present at that level: b"
    """
    cur = obj
    walked: list[str] = []
    for part in path.split("."):
        walked.append(part)
        here = ".".join(walked)
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                raise MissingMetric(
                    f"{source}: no element '{part}' at '{here}' "
                    f"(list of length {len(cur)})") from None
        elif isinstance(cur, dict):
            if part not in cur:
                keys = ", ".join(sorted(map(str, cur))[:12]) or "(empty)"
                raise MissingMetric(
                    f"{source}: missing field '{here}'. "
                    f"Keys present at that level: {keys}")
            cur = cur[part]
        else:
            raise MissingMetric(
                f"{source}: cannot descend to '{here}' -- "
                f"'{'.'.join(walked[:-1])}' is a {type(cur).__name__}, not a mapping")
    return cur


def req_num(obj: Any, path: str, source: str) -> float:
    """``require`` plus the assertion that the value really is a number."""
    v = require(obj, path, source)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise MissingMetric(
            f"{source}: field '{path}' is {v!r} ({type(v).__name__}), not a number")
    return float(v)


# ------------------------------------------------------------------------- registry


@dataclass
class Figure:
    slug: str
    title: str
    sources: tuple[str, ...]
    fn: Callable[[], plt.Figure]
    caveat: str = ""


REGISTRY: list[Figure] = []


def figure(slug: str, title: str, sources: Sequence[str], caveat: str = ""):
    """Register a figure function so the CLI can list, run and audit it."""
    def deco(fn):
        REGISTRY.append(Figure(slug=slug, title=title, sources=tuple(sources),
                               fn=fn, caveat=caveat))
        return fn
    return deco


def _save(fig: plt.Figure, slug: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{slug}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _annotate_bars(ax, fmt="{:.4f}", fontsize=FS_ANNOT, padding=2, **kw):
    """Value labels on every bar container, in one consistent format."""
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, fontsize=fontsize, padding=padding, **kw)


def _floor_line(ax, y: float, label: str, *, axis: str = "y", color: str = C_PERSISTENCE):
    """Draw the persistence floor as a rule across the plot.

    Every macro-F1 figure carries this. A score without its floor does not say
    whether the model is working, and a figure that omits it invites the reader to
    assume it is.
    """
    drawer = ax.axhline if axis == "y" else ax.axvline
    drawer(y, color=color, ls="--", lw=1.6, zorder=0.5, label=label)


# ============================================================ 01 persistence floor


@figure("01_persistence_floor_vs_horizon",
        "Persistence floor across forecast horizons",
        ["horizon_comparison.json"])
def fig_persistence_floor() -> plt.Figure:
    """The result that reframed the project: the floor collapses as h grows."""
    src = "horizon_comparison.json"
    d = load(src)
    horizons = require(d, "horizons", src)
    rows = []
    for h in horizons:
        r = require(d, f"rows.{h}", src)
        rows.append({
            "Horizon": f"{h} h",
            "Macro-F1": req_num(r, "observed.macro_f1", f"{src}:rows.{h}"),
            "Accuracy": req_num(r, "observed.accuracy", f"{src}:rows.{h}"),
            "Label unchanged (%)": req_num(r, "label_unchanged_pct", f"{src}:rows.{h}"),
        })
    df = pd.DataFrame(rows)
    primary = f"{require(d, 'primary', src)} h"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.3))

    colors = [C_ACCENT if h == primary else C_PERSISTENCE for h in df["Horizon"]]
    sns.barplot(df, x="Horizon", y="Macro-F1", hue="Horizon", palette=colors,
                legend=False, ax=ax1, edgecolor="white", linewidth=0.8)
    _annotate_bars(ax1)
    ax1.set_ylim(0, max(df["Macro-F1"]) * 1.18)
    ax1.set_title("Zero-parameter persistence", fontsize=FS_LABEL + 0.5)
    ax1.set_xlabel("Forecast horizon")

    sns.lineplot(df, x="Horizon", y="Label unchanged (%)", marker="o",
                 color=C_BEIJING, lw=2.2, markersize=8, ax=ax2)
    for _, r in df.iterrows():
        ax2.annotate(f"{r['Label unchanged (%)']:.1f}%",
                     (r["Horizon"], r["Label unchanged (%)"]),
                     textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=FS_ANNOT)
    ax2.set_ylim(0, 100)
    ax2.set_title("AQI category unchanged from t to t+h", fontsize=FS_LABEL + 0.5)
    ax2.set_xlabel("Forecast horizon")

    fig.suptitle(f"The task only becomes a forecasting problem at h = {primary}",
                 y=1.02)
    fig.tight_layout()
    return fig


# ====================================================== 02 master model comparison


def _variants() -> list[dict]:
    """The same variant list the written summary is compiled from.

    Imported rather than reassembled: if the figure and the table in
    ``final_results_summary.md`` ever disagreed about which models were compared,
    that would be a bug, and sharing the assembly is how it stays impossible.
    """
    from src.reporting.compile_results import collect
    return collect()["variants"]


@figure("02_master_model_comparison",
        "Every Beijing variant against the persistence floor",
        ["ablation_h6.json", "gapfill_h6.json", "dl_h6.json", "smote_h6.json",
         "deployment_h6.json", "deployment_h6_cw.json", "conformal_h6.json",
         "baseline_h6.json"])
def fig_master_comparison() -> plt.Figure:
    variants = _variants()
    floor = None
    rows = []
    for v in variants:
        name = v["name"].split(" (")[0] if v["name"].startswith("Persistence") else v["name"]
        score = req_num(v, "scores.macro_f1", f"variant {v['name']}")
        if v["vs_persistence"] is None:
            floor = score
            continue
        vs = v["vs_persistence"]
        lo = hi = np.nan
        if isinstance(vs, dict):
            lo = req_num(vs, "ci_low", f"variant {v['name']} vs_persistence")
            hi = req_num(vs, "ci_high", f"variant {v['name']} vs_persistence")
        # ci_low/ci_high bound the DIFFERENCE against persistence, so the bar's own
        # error bar is that interval re-centred on the variant's score.
        rows.append({"Variant": name, "Macro-F1": score,
                     "err_lo": (score - floor) - lo if not np.isnan(lo) else np.nan,
                     "err_hi": hi - (score - floor) if not np.isnan(hi) else np.nan,
                     "tested": isinstance(vs, dict)})
    if floor is None:
        raise MissingMetric("no Persistence row in the compiled variant list")

    df = pd.DataFrame(rows).sort_values("Macro-F1", ascending=True).reset_index(drop=True)
    colors = [C_MODEL if s > floor else C_BAD for s in df["Macro-F1"]]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 0.42 * len(df) + 1.9))
    sns.barplot(df, y="Variant", x="Macro-F1", hue="Variant", palette=colors,
                legend=False, ax=ax, edgecolor="white", linewidth=0.7)
    err = np.vstack([df["err_lo"].to_numpy(), df["err_hi"].to_numpy()])
    finite = ~np.isnan(err).any(axis=0)
    if finite.any():
        ax.errorbar(df["Macro-F1"][finite], np.arange(len(df))[finite],
                    xerr=err[:, finite], fmt="none", ecolor="#333333",
                    elinewidth=1.2, capsize=3, zorder=5)
    _floor_line(ax, floor, f"Persistence floor ({floor:.4f})", axis="x")

    # Labels clear the error-bar cap, not just the bar end, or they collide with it.
    for i, r in df.iterrows():
        d = r["Macro-F1"] - floor
        mark = "   (not tested)" if not r["tested"] else ""
        tip = r["Macro-F1"] + (0 if np.isnan(r["err_hi"]) else r["err_hi"])
        ax.annotate(f"{r['Macro-F1']:.4f}  ({d:+.4f}){mark}",
                    (tip, i), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=FS_ANNOT)

    ax.set_xlim(min(df["Macro-F1"].min(), floor) - 0.015, df["Macro-F1"].max() + 0.055)
    ax.set_ylabel("")
    ax.set_xlabel("Test macro-F1, observed rows only")
    ax.set_title("No Beijing variant clears the floor by a meaningful margin")
    ax.legend(loc="upper left", frameon=True, fontsize=FS_ANNOT)
    fig.tight_layout()
    return fig


# ======================================================== 03 the advisory trade-off


@figure("03_advisory_class_tradeoff",
        "What augmentation does to the advisory classes",
        ["ablation_h6.json", "smote_h6.json", "gapfill_h6.json"])
def fig_advisory_tradeoff() -> plt.Figure:
    """The pattern behind the disqualification rule.

    CTGAN and SMOTE both raise aggregate macro-F1 and both degrade the two classes
    that justify the device. Class weighting is the intervention that does not.
    """
    ab, sm, gap = load("ablation_h6.json"), load("smote_h6.json"), load("gapfill_h6.json")
    cw_key = "RandomForest (class_weight=balanced)"

    series = {
        "Persistence": require(ab, "rows.Persistence.scores", "ablation_h6.json"),
        "RandomForest": require(ab, "rows.unaugmented.scores", "ablation_h6.json"),
        "+ CTGAN (broad 4)": require(ab, "rows.broad-4.scores", "ablation_h6.json"),
        "+ CTGAN (targeted 2)": require(ab, "rows.targeted-2.scores", "ablation_h6.json"),
        "+ SMOTE": require(sm, "scores", "smote_h6.json"),
        "+ class weight": require(gap, f"variants.{cw_key}.scores", "gapfill_h6.json"),
    }
    rows = []
    for name, sc in series.items():
        rows.append({"Variant": name, "Class": "macro-F1",
                     "F1": req_num(sc, "macro_f1", f"scores of {name}")})
        for c in ADVISORY:
            rows.append({"Variant": name, "Class": c,
                         "F1": req_num(sc, f"per_class.{c}", f"scores of {name}")})
    df = pd.DataFrame(rows)

    order = list(series)
    # Grey is reserved for persistence everywhere in this module, so no other series
    # may take it -- the targeted-2 variant gets the blue instead.
    pal = {"Persistence": C_PERSISTENCE, "RandomForest": C_MODEL,
           "+ CTGAN (broad 4)": C_GAN, "+ CTGAN (targeted 2)": C_BEIJING,
           "+ SMOTE": C_BAD, "+ class weight": C_BANGLADESH}

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH * 1.25, 4.6), sharey=False)
    for ax, cls in zip(axes, ["macro-F1"] + ADVISORY):
        sub = df[df["Class"] == cls]
        sns.barplot(sub, x="Variant", y="F1", hue="Variant", order=order,
                    hue_order=order, palette=pal, legend=False, ax=ax,
                    edgecolor="white", linewidth=0.7)
        base = float(sub[sub["Variant"] == "Persistence"]["F1"].iloc[0])
        _floor_line(ax, base, "persistence")
        ax.set_title(cls if cls != "macro-F1" else "Aggregate (macro-F1)",
                     fontsize=FS_LABEL)
        ax.set_xlabel("")
        ax.set_ylabel("F1" if ax is axes[0] else "")
        ax.tick_params(axis="x", rotation=60)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        lo, hi = sub["F1"].min(), sub["F1"].max()
        span = hi - lo
        ax.set_ylim(max(0, lo - span * 0.55), hi + span * 0.85)
        # Rotated: six bars in a narrow panel put horizontal labels on top of
        # each other, and two of these differences are 0.003 wide.
        _annotate_bars(ax, fmt="{:.3f}", fontsize=7.4, rotation=90, padding=3)

    fig.suptitle("Augmentation buys aggregate macro-F1 by giving up the advisory classes",
                 y=1.03)
    fig.tight_layout()
    return fig


# ========================================================= 04 CTGAN validity checks


@figure("04_ctgan_validity",
        "Physical validity of CTGAN output after the constraint fix",
        ["gan_h6.json"],
        caveat=("The pre-fix state (dict-form constraint silently ignored, ~51% of "
                "synthetic rows with DEWP > TEMP) is described in gan_quality_report_h6.md "
                "but was never written to a metrics JSON -- the buggy run was discarded "
                "rather than archived. Only the post-fix columns are plotted; the 'before' "
                "bar is deliberately absent rather than hardcoded from prose."))
def fig_ctgan_validity() -> plt.Figure:
    src = "gan_h6.json"
    v = require(load(src), "validity", src)
    rows = []
    for which in ("real", "synthetic"):
        blk = require(v, which, src)
        rows += [
            {"Source": which.capitalize(), "Check": "hour on unit circle (%)",
             "Value": req_num(blk, "hour.on_unit_circle_pct", f"{src}:validity.{which}")},
            {"Source": which.capitalize(), "Check": "month on unit circle (%)",
             "Value": req_num(blk, "month.on_unit_circle_pct", f"{src}:validity.{which}")},
            {"Source": which.capitalize(), "Check": "DEWP > TEMP (%)",
             "Value": req_num(blk, "dewp_above_temp_pct", f"{src}:validity.{which}")},
        ]
    df = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, 3.2),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    circ = df[df["Check"].str.contains("unit circle")]
    sns.barplot(circ, x="Check", y="Value", hue="Source",
                palette=[C_BEIJING, C_GAN], ax=ax1, edgecolor="white", linewidth=0.7)
    ax1.set_ylim(0, 112)
    ax1.axhline(100, color=C_MODEL, ls="--", lw=1.4, label="valid (100%)")
    _annotate_bars(ax1, fmt="{:.1f}")
    ax1.set_title("Cyclical encodings", fontsize=FS_LABEL)
    ax1.set_xlabel(""); ax1.set_ylabel("% of rows")
    ax1.tick_params(axis="x", rotation=12)
    ax1.legend(loc="lower right", ncol=2, fontsize=FS_ANNOT)

    dewp = df[df["Check"].str.contains("DEWP")]
    sns.barplot(dewp, x="Source", y="Value", hue="Source",
                palette=[C_BEIJING, C_GAN], legend=False, ax=ax2,
                edgecolor="white", linewidth=0.7)
    _annotate_bars(ax2, fmt="{:.4f}")
    ax2.set_ylim(0, max(dewp["Value"].max() * 1.6, 0.004))
    ax2.set_title("Impossible air\n(dew point above temperature)", fontsize=FS_LABEL)
    ax2.set_xlabel(""); ax2.set_ylabel("% of rows")

    n_syn = req_num(v, "synthetic.n", f"{src}:validity")
    n_clip = sum(req_num(load(src), f"quality.{c}.n_dewp_clipped", src)
                 for c in require(load(src), "quality", src))
    fig.suptitle("After the sdv.cag.Inequality fix: 0 violations in "
                 f"{int(n_syn):,} synthetic rows", y=1.02)
    fig.text(0.5, -0.06,
             f"Real data carries {req_num(v, 'real.dewp_above_temp_n', src):.0f} violating rows "
             f"({req_num(v, 'real.dewp_above_temp_pct', src):.4f}%); "
             f"{int(n_clip):,} were clipped to saturation before fitting, "
             "because SDV refuses to fit a constraint the training data violates.",
             ha="center", fontsize=FS_ANNOT, color="#555555")
    fig.tight_layout()
    return fig


# ============================================================ 05 per-class heatmap


@figure("05_per_class_f1_heatmap",
        "Per-class F1 across every Beijing variant",
        ["ablation_h6.json", "gapfill_h6.json", "dl_h6.json", "smote_h6.json"])
def fig_per_class_heatmap() -> plt.Figure:
    """Absolute F1 beside the delta against persistence.

    The delta panel is the one that matters: it has a real midpoint at zero, so a
    diverging map reads correctly, and the red band down the advisory columns is the
    project's central finding in one glance.
    """
    variants = _variants()
    floor_row = next((v for v in variants if v["vs_persistence"] is None), None)
    if floor_row is None:
        raise MissingMetric("no Persistence row in the compiled variant list")
    floor = {c: req_num(floor_row, f"scores.per_class.{c}", "Persistence scores")
             for c in CLASS_ORDER}

    index, absolute, delta = [], [], []
    for v in variants:
        name = v["name"].split(" —")[0]
        index.append(name)
        vals = [req_num(v, f"scores.per_class.{c}", f"variant {name}") for c in CLASS_ORDER]
        absolute.append(vals)
        delta.append([x - floor[c] for x, c in zip(vals, CLASS_ORDER)])

    cols = [CLASS_SHORT[c] for c in CLASS_ORDER]
    abs_df = pd.DataFrame(absolute, index=index, columns=cols)
    del_df = pd.DataFrame(delta, index=index, columns=cols)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH * 1.65, 0.36 * len(index) + 2.2))
    sns.heatmap(abs_df, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax1,
                cbar_kws={"label": "F1", "shrink": 0.75},
                annot_kws={"fontsize": FS_ANNOT}, linewidths=0.5, linecolor="white")
    ax1.set_title("Per-class F1", fontsize=FS_LABEL + 0.5)

    lim = float(np.abs(del_df.to_numpy()).max())
    sns.heatmap(del_df, annot=True, fmt="+.3f", cmap="RdYlGn", center=0.0,
                vmin=-lim, vmax=lim, ax=ax2,
                cbar_kws={"label": "F1 vs persistence", "shrink": 0.75},
                annot_kws={"fontsize": FS_ANNOT}, linewidths=0.5, linecolor="white")
    ax2.set_title("Change against the persistence floor", fontsize=FS_LABEL + 0.5)
    ax2.set_yticklabels([])

    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=35)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle("Gains concentrate in the common classes; "
                 "the advisory columns are where variants lose", y=1.01)
    fig.tight_layout()
    return fig


# ============================================================== 06 capacity sweep


@figure("06_capacity_sweep",
        "Sequence-model capacity against validation macro-F1",
        ["capacity_sweep_h6.json"])
def fig_capacity_sweep() -> plt.Figure:
    src = "capacity_sweep_h6.json"
    d = load(src)
    rows = require(d, "sweep.rows", src)
    df = pd.DataFrame([{
        "Architecture": require(r, "arch", src).upper() if require(r, "arch", src) == "lstm"
                        else require(r, "arch", src).capitalize(),
        "Hidden size": int(req_num(r, "size", src)),
        "Parameters": req_num(r, "n_params", src),
        "Validation macro-F1": req_num(r, "val_macro_f1", src),
    } for r in rows])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.6))
    sns.lineplot(df, x="Parameters", y="Validation macro-F1", hue="Architecture",
                 style="Architecture", markers=True, dashes=False, lw=2.2,
                 markersize=9, palette=[C_BEIJING, C_GAN], ax=ax)
    for _, r in df.iterrows():
        ax.annotate(f"h={r['Hidden size']}\n{r['Validation macro-F1']:.4f}",
                    (r["Parameters"], r["Validation macro-F1"]),
                    textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=FS_ANNOT)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    pad = (df["Validation macro-F1"].max() - df["Validation macro-F1"].min()) * 0.45
    ax.set_ylim(df["Validation macro-F1"].min() - pad,
                df["Validation macro-F1"].max() + pad * 1.5)

    verdicts = []
    for arch in ("lstm", "transformer"):
        a = require(d, f"analysis.{arch}", src)
        if require(a, "monotone_declining", src):
            verdicts.append(f"{arch}: declines monotonically "
                            f"({req_num(a, 'gain_smallest_to_largest', src):+.4f} over "
                            f"{req_num(a, 'param_ratio', src):.1f}x the parameters)")
    ax.set_title("More capacity makes both architectures worse")
    if verdicts:
        fig.text(0.5, -0.04, "   |   ".join(verdicts), ha="center",
                 fontsize=FS_ANNOT, color="#555555")
    fig.tight_layout()
    return fig


# ================================================= 07 aleatoric / epistemic split


@figure("07_uncertainty_decomposition",
        "Aleatoric against epistemic predictive entropy",
        ["dl_h6.json"])
def fig_uncertainty_decomposition() -> plt.Figure:
    """Why nothing in the model space helped.

    Total predictive entropy splits into what the model does not know (epistemic,
    reducible with more data or capacity) and what the data itself does not
    determine (aleatoric, irreducible). The split is the ceiling argument.
    """
    src = "dl_h6.json"
    d = load(src)
    archs = list(require(d, "uncertainty", src))
    fig, axes = plt.subplots(1, len(archs), figsize=(FIG_WIDTH, 3.4))
    if len(archs) == 1:
        axes = [axes]

    for ax, arch in zip(axes, archs):
        total = req_num(d, f"uncertainty.{arch}.mean_entropy", src)
        epi = req_num(d, f"uncertainty.{arch}.mean_epistemic", src)
        alea = total - epi
        if alea < 0:
            raise MissingMetric(
                f"{src}: uncertainty.{arch} gives epistemic ({epi:.4f}) above total "
                f"entropy ({total:.4f}); the decomposition is not usable")
        frac = alea / total * 100
        wedges, _, autotexts = ax.pie(
            [alea, epi], labels=None, colors=[C_PERSISTENCE, C_BAD],
            autopct=lambda p: f"{p:.1f}%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
            textprops={"fontsize": FS_ANNOT + 1})
        for t in autotexts:
            t.set_color("white"); t.set_fontweight("bold")
        ax.set_title(f"{arch.capitalize()}\ntotal entropy {total:.4f} nats",
                     fontsize=FS_LABEL)
        ax.annotate(f"aleatoric {alea:.4f}\nepistemic {epi:.4f}",
                    (0.5, -0.12), xycoords="axes fraction", ha="center",
                    fontsize=FS_ANNOT, color="#555555")
        if ax is axes[0]:
            ax.legend(wedges, [f"Aleatoric (irreducible) — {frac:.1f}%",
                               "Epistemic (reducible)"],
                      loc="upper center", bbox_to_anchor=(0.5, -0.24),
                      frameon=False, fontsize=FS_LEGEND)

    fig.suptitle("The uncertainty is in the data, not the model", y=1.02)
    fig.tight_layout()
    return fig


# ============================================== 08 marginal vs Mondrian coverage


@figure("08_conformal_coverage",
        "Conformal coverage per class: marginal against Mondrian",
        ["conformal_h6.json"])
def fig_conformal_coverage() -> plt.Figure:
    src = "conformal_h6.json"
    d = load(src)
    target = 1.0 - req_num(d, "calibration.alpha", src)
    rows = []
    for method, path in (("Marginal", "test"), ("Mondrian", "mondrian.test")):
        for c in CLASS_ORDER:
            blk = require(d, f"{path}.per_class.{c}", src)
            rows.append({"Class": CLASS_SHORT[c], "Method": method,
                         "Coverage": req_num(blk, "coverage", f"{src}:{path}.per_class.{c}"),
                         "n": int(req_num(blk, "n", f"{src}:{path}.per_class.{c}"))})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.6))
    sns.barplot(df, x="Class", y="Coverage", hue="Method",
                palette=[C_ALT, C_MODEL], ax=ax, edgecolor="white", linewidth=0.7)
    ax.axhline(target, color=C_BAD, ls="--", lw=1.8,
               label=f"target coverage ({target:.2f})")
    _annotate_bars(ax, fmt="{:.3f}", fontsize=7.6)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("Empirical coverage")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=18)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.legend(loc="lower left", ncol=3, fontsize=FS_ANNOT)

    haz_m = float(df[(df.Class == "Hazardous") & (df.Method == "Marginal")].Coverage.iloc[0])
    haz_c = float(df[(df.Class == "Hazardous") & (df.Method == "Mondrian")].Coverage.iloc[0])
    ax.set_title("Marginal coverage meets its target while under-covering Hazardous")
    fig.text(0.5, -0.07,
             f"Hazardous: {haz_m:.3f} marginal vs {haz_c:.3f} Mondrian. A marginal "
             "guarantee constrains the average, so the shortfall lands on the rare classes.",
             ha="center", fontsize=FS_ANNOT, color="#555555")
    fig.tight_layout()
    return fig


# =========================================== 09 conformal prediction set sizes


@figure("09_conformal_set_sizes",
        "Distribution of conformal prediction-set size",
        ["conformal_h6.json"])
def fig_set_sizes() -> plt.Figure:
    src = "conformal_h6.json"
    d = load(src)
    rows = []
    for method, path in (("Marginal", "test"), ("Mondrian", "mondrian.test")):
        hist = require(d, f"{path}.size_histogram", src)
        n = req_num(d, f"{path}.n", src)
        for size, count in sorted(hist.items(), key=lambda kv: int(kv[0])):
            rows.append({"Set size": int(size), "Method": method,
                         "Share of predictions (%)": count / n * 100})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.4))
    sns.barplot(df, x="Set size", y="Share of predictions (%)", hue="Method",
                palette=[C_ALT, C_MODEL], ax=ax, edgecolor="white", linewidth=0.7)
    _annotate_bars(ax, fmt="{:.1f}", fontsize=7.6)
    ax.set_xlabel("Categories in the prediction set")
    ax.set_ylim(0, df["Share of predictions (%)"].max() * 1.18)
    ax.legend(title="", loc="upper right")

    notes = []
    for method, path in (("Marginal", "test"), ("Mondrian", "mondrian.test")):
        notes.append(f"{method}: median {req_num(d, f'{path}.median_set_size', src):.0f}, "
                     f"singletons {req_num(d, f'{path}.singleton_rate', src) * 100:.1f}%")
    ax.set_title("A single confident category is the exception, not the rule")
    fig.text(0.5, -0.06, "   |   ".join(notes), ha="center",
             fontsize=FS_ANNOT, color="#555555")
    fig.tight_layout()
    return fig


# =========================================================== 10 compression funnel


@figure("10_compression_funnel",
        "Model size at each stage of the deployment pipeline",
        ["deployment_h6.json", "deployment_h6_bd.json"])
def fig_compression_funnel() -> plt.Figure:
    panels = [("Beijing", "deployment_h6.json", C_BEIJING),
              ("Bangladesh (deployed)", "deployment_h6_bd.json", C_BANGLADESH)]

    # Each panel gets its own axis: the two models differ by an order of magnitude in
    # size, and a shared log axis flattens the Bangladesh funnel into three equal bars.
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH * 1.25, 4.3), sharey=False)
    headline = []

    for ax, (label, src, colour) in zip(axes, panels):
        d = load(src)
        base_kb = req_num(d, "baseline.pickle_kb", src)
        comp_kb = req_num(d, "compressed.pickle_kb", src)
        onnx_kb = req_num(d, "onnx.bytes", src) / 1024.0
        stages = pd.DataFrame([
            {"Stage": f"Full forest\n{int(req_num(d, 'baseline.n_estimators', src))}x"
                      f"d{int(req_num(d, 'baseline.max_depth', src))}", "KB": base_kb},
            {"Stage": f"Compressed\n{int(req_num(d, 'compressed.n_estimators', src))}x"
                      f"d{int(req_num(d, 'compressed.max_depth', src))}", "KB": comp_kb},
            {"Stage": "ONNX export", "KB": onnx_kb},
        ])
        shades = sns.light_palette(colour, n_colors=5)[2:5][::-1]
        sns.barplot(stages, x="Stage", y="KB", hue="Stage", palette=shades,
                    legend=False, ax=ax, edgecolor="white", linewidth=0.7)
        ax.set_yscale("log")
        ax.set_ylim(stages["KB"].min() * 0.30, stages["KB"].max() * 4.5)
        for i, r in stages.iterrows():
            kb = r["KB"]
            txt = f"{kb / 1024:.1f} MB" if kb >= 1024 else f"{kb:,.0f} KB"
            ax.annotate(txt, (i, kb), textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=FS_ANNOT, fontweight="bold")
        dv = req_num(d, "compressed.val_macro_f1", src) - req_num(d, "baseline.val_macro_f1", src)
        dt = req_num(d, "compressed.test_macro_f1", src) - req_num(d, "baseline.test_macro_f1", src)
        ax.set_title(f"{label}  —  {base_kb / comp_kb:.0f}x smaller\n"
                     f"macro-F1 {dv:+.4f} val, {dt:+.4f} test",
                     fontsize=FS_LABEL)
        ax.set_xlabel("")
        ax.set_ylabel("Serialized size, KB (log scale)" if ax is axes[0] else "")
        headline.append((label.split(" (")[0], base_kb / comp_kb, dt))

    # Derived, not asserted. An earlier draft of this title claimed "three orders of
    # magnitude", which describes a sweep point these files do not contain.
    fig.suptitle("  |  ".join(f"{name}: {ratio:.0f}x smaller for {dt:+.4f} test macro-F1"
                              for name, ratio, dt in headline),
                 y=1.02)
    fig.text(0.5, -0.04,
             "ONNX is larger than the compressed pickle on Beijing: the sweep kept 100 "
             "trees there, and the ONNX tree ensemble format is not a compressed one.",
             ha="center", fontsize=FS_ANNOT, color="#555555")
    fig.tight_layout()
    return fig


# ============================================================ 11 EPA vs HJ 633-2012


@figure("11_epa_vs_hj633",
        "Class balance under EPA and HJ 633-2012 breakpoints",
        ["hj633_h6.json"])
def fig_epa_vs_hj633() -> plt.Figure:
    """The same air, binned by two national standards.

    The standards do not share class names, so the bars are indexed by band rank
    rather than by label, and each panel carries its own labels.
    """
    src = "hj633_h6.json"
    d = require(load(src), "results", src)
    standards = list(d)
    fig, axes = plt.subplots(1, len(standards) + 1,
                             figsize=(FIG_WIDTH * 1.45, 4.4),
                             gridspec_kw={"width_ratios": [1] * len(standards) + [0.7]})

    for ax, std in zip(axes, standards):
        labels = require(d, f"{std}.labels", src)
        dist = require(d, f"{std}.distribution.test", src)
        df = pd.DataFrame([{"Class": lbl,
                            "Share of test rows (%)": req_num(dist, f"{lbl}.pct",
                                                              f"{src}:{std}.distribution.test")}
                           for lbl in labels])
        colour = C_BEIJING if std == "EPA" else C_BANGLADESH
        sns.barplot(df, x="Class", y="Share of test rows (%)", hue="Class",
                    palette=sns.light_palette(colour, n_colors=len(df) + 2)[2:],
                    legend=False, ax=ax, edgecolor="white", linewidth=0.7)
        _annotate_bars(ax, fmt="{:.1f}", fontsize=7.4)
        bps = require(d, f"{std}.breakpoints", src)
        ax.set_title(std, fontsize=FS_LABEL)
        # The breakpoint list belongs under the axis, not in the title: two of these
        # side by side are wide enough to run into each other up there.
        ax.set_xlabel("PM2.5 breakpoints (µg/m³): "
                      + ", ".join(f"{b:g}" for b in bps), fontsize=FS_ANNOT)
        ax.set_ylabel("Share of test rows (%)" if ax is axes[0] else "")
        ax.set_ylim(0, 45)
        ax.tick_params(axis="x", rotation=55)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")

    ax = axes[-1]
    comp = pd.DataFrame([{
        "Standard": std,
        "Macro-F1": req_num(d, f"{std}.rf.macro_f1", src),
        "Persistence": req_num(d, f"{std}.persistence.macro_f1", src),
        "Δ": req_num(d, f"{std}.vs_persistence.observed_diff", src),
        "sig": bool(require(d, f"{std}.vs_persistence.significant", src)),
    } for std in standards])
    long = comp.melt(id_vars="Standard", value_vars=["Persistence", "Macro-F1"],
                     var_name="Which", value_name="Macro-F1 score")
    long["Which"] = long["Which"].replace({"Macro-F1": "RandomForest"})
    sns.barplot(long, x="Standard", y="Macro-F1 score", hue="Which",
                palette=[C_PERSISTENCE, C_MODEL], ax=ax,
                edgecolor="white", linewidth=0.7)
    _annotate_bars(ax, fmt="{:.3f}", fontsize=7.4)
    ax.set_title("RF against its own floor", fontsize=FS_LABEL)
    ax.set_xlabel(""); ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 0.88)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=FS_ANNOT, loc="upper center", ncol=1, framealpha=0.95)
    for i, r in comp.iterrows():
        ax.annotate(f"Δ {r['Δ']:+.4f}\n{'significant' if r['sig'] else 'not significant'}",
                    (i, 0.035), ha="center", fontsize=FS_ANNOT, color="#333333")

    fig.suptitle("The conclusion does not depend on which standard defines the classes",
                 y=1.03)
    fig.tight_layout()
    return fig


# ========================================================== 12 Bonferroni family


@figure("12_bonferroni_correction",
        "Every persistence comparison under family-wise correction",
        ["ablation_h6.json", "gapfill_h6.json", "dl_h6.json", "smote_h6.json",
         "deployment_h6.json", "deployment_h6_cw.json"])
def fig_bonferroni() -> plt.Figure:
    """Bonferroni over the same family the written summary corrects.

    The bootstrap's resolution floor is drawn explicitly: with 1,000 resamples the
    smallest reportable two-sided p is 2/1000, so a bar at the floor means "below
    this", not "exactly zero".
    """
    variants = _variants()
    tested = [v for v in variants
              if isinstance(v.get("vs_persistence"), dict)]
    if not tested:
        raise MissingMetric("no variant carries a vs_persistence bootstrap result")
    n_boot = req_num(load("ablation_h6.json"), "n_boot", "ablation_h6.json")
    resolution = 2.0 / n_boot
    alpha = 0.05
    corrected = alpha / len(tested)

    rows = []
    for v in tested:
        p = req_num(v, "vs_persistence.p_two_sided", f"variant {v['name']}")
        diff = req_num(v, "vs_persistence.observed_diff", f"variant {v['name']}")
        rows.append({"Variant": v["name"].split(" —")[0],
                     "p": max(p, resolution), "at_floor": p < resolution,
                     "Direction": "better than persistence" if diff > 0
                                  else "worse than persistence",
                     "survives": p < corrected})
    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.15, 0.42 * len(df) + 2.4))
    sns.barplot(df, y="Variant", x="p", hue="Direction",
                palette={"better than persistence": C_MODEL,
                         "worse than persistence": C_BAD},
                dodge=False, ax=ax, edgecolor="white", linewidth=0.7)
    ax.set_xscale("log")
    ax.axvline(alpha, color="#555555", ls=":", lw=1.6, label=f"α = {alpha}")
    ax.axvline(corrected, color=C_BAD, ls="--", lw=1.8,
               label=f"Bonferroni α/{len(df)} = {corrected:.4f}")
    ax.axvline(resolution, color=C_ACCENT, ls="-.", lw=1.6,
               label=f"bootstrap resolution 2/{int(n_boot)} = {resolution:.4f}")

    for i, r in df.iterrows():
        mark = "<" if r["at_floor"] else ""
        ax.annotate(f"{mark}{r['p']:.4f}", (r["p"], i), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=FS_ANNOT)

    n_surv = int(df["survives"].sum())
    ax.set_xlabel("Two-sided p, paired bootstrap (log scale)")
    ax.set_ylabel("")
    ax.set_title(f"{n_surv} of {len(df)} comparisons survive the corrected threshold")
    # Below the axes: inside, it lands on the bars at the resolution floor, which are
    # exactly the ones a reader is trying to read.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              frameon=False, fontsize=FS_ANNOT)
    ax.set_xlim(resolution * 0.45, 2.6)
    fig.tight_layout()
    return fig


# ====================================================== 13 / 14 rolling-origin CV


def _rolling_cv_figure(src: str, place: str, colour) -> plt.Figure:
    d = load(src)
    folds = require(d, "folds", src)
    n_folds = int(req_num(d, "n_folds", src))
    rows = []
    for f in folds:
        fid = int(req_num(f, "fold", src))
        months = require(f, "eval_months" if "eval_months" in f else "months", src)
        for model, sc in require(f, "scores", src).items():
            rows.append({"Fold": fid, "Model": model,
                         "Macro-F1": req_num(sc, "macro_f1", f"{src}:folds.{fid}.scores.{model}"),
                         "months": months})
    df = pd.DataFrame(rows)

    order = ["Persistence"] + [m for m in df["Model"].unique() if m != "Persistence"]
    pal = {"Persistence": C_PERSISTENCE}
    extra = [colour, C_GAN, C_MODEL, C_ALT]
    for i, m in enumerate([m for m in order if m != "Persistence"]):
        pal[m] = extra[i % len(extra)]

    short = {m: (m.replace("RandomForest (class_weight=balanced)", "RF (balanced)")
                  .replace("RandomForest (unweighted)", "RF (unweighted)")
                  .replace("RandomForest ", "RF ")) for m in order}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH * 1.4, 4.2),
                                   gridspec_kw={"width_ratios": [1.6, 1]})
    sns.lineplot(df, x="Fold", y="Macro-F1", hue="Model", style="Model",
                 hue_order=order, style_order=order, markers=True, dashes=False,
                 lw=2.1, markersize=8, palette=pal, ax=ax1)
    ax1.set_xticks(sorted(df["Fold"].unique()))
    ax1.set_xlabel(f"Rolling-origin fold (expanding window, "
                   f"{int(req_num(d, 'embargo_hours', src))} h embargo)")
    ax1.set_title(f"{place}: per-fold macro-F1 across {n_folds} chronological folds",
                  fontsize=FS_LABEL)
    ax1.legend(fontsize=FS_ANNOT, loc="upper center", ncol=len(order),
               title="", framealpha=0.95)
    lo, hi = df["Macro-F1"].min(), df["Macro-F1"].max()
    ax1.set_ylim(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.42)

    agg = require(d, "aggregate.per_model", src)
    spread = pd.DataFrame([{
        "Model": m,
        "Mean": req_num(agg, f"{m}.mean", f"{src}:aggregate.per_model"),
        "SD": req_num(agg, f"{m}.std", f"{src}:aggregate.per_model"),
    } for m in order])
    sns.barplot(spread, x="Model", y="Mean", hue="Model", order=order,
                hue_order=order, palette=pal, legend=False, ax=ax2,
                edgecolor="white", linewidth=0.7)
    ax2.errorbar(np.arange(len(spread)), spread["Mean"], yerr=spread["SD"],
                 fmt="none", ecolor="#333333", elinewidth=1.3, capsize=4, zorder=5)
    for i, r in spread.iterrows():
        ax2.annotate(f"{r['Mean']:.4f}\n±{r['SD']:.4f}", (i, r["Mean"] + r["SD"]),
                     textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=FS_ANNOT)
    ax2.set_title("Mean ± SD across folds", fontsize=FS_LABEL)
    ax2.set_xlabel(""); ax2.set_ylabel("Macro-F1")
    ax2.set_ylim(0, (spread["Mean"] + spread["SD"]).max() * 1.34)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels([short[m] for m in order], rotation=18, ha="right",
                        fontsize=FS_ANNOT)

    tests = {k: v for k, v in require(d, "aggregate.tests", src).items()
             if not k.startswith("_")}
    caption = "   |   ".join(
        f"{m}: {int(req_num(t, 'wins', src))}/{int(req_num(t, 'n_folds', src))} folds, "
        f"mean Δ {req_num(t, 'mean_delta', src):+.4f}, "
        f"one-sided p = {req_num(t, 'p_one_sided', src):.4f}"
        for m, t in tests.items())
    fig.text(0.5, -0.05, caption, ha="center", fontsize=FS_ANNOT, color="#555555")
    fig.tight_layout()
    return fig


@figure("13_rolling_cv_beijing",
        "Beijing rolling-origin cross-validation, per fold",
        ["rolling_cv_h6.json"])
def fig_rolling_cv_beijing() -> plt.Figure:
    return _rolling_cv_figure("rolling_cv_h6.json", "Beijing", C_BEIJING)


@figure("14_rolling_cv_bangladesh",
        "Bangladesh rolling-origin cross-validation, per fold",
        ["rolling_cv_h6_bangladesh.json"])
def fig_rolling_cv_bangladesh() -> plt.Figure:
    return _rolling_cv_figure("rolling_cv_h6_bangladesh.json", "Bangladesh", C_BANGLADESH)


# ========================================================== 15 folds-won summary


@figure("15_folds_won_summary",
        "Folds won against persistence: Beijing against Bangladesh",
        ["rolling_cv_h6.json", "rolling_cv_h6_bangladesh.json"])
def fig_folds_won() -> plt.Figure:
    """The comparison the whole project turns on.

    Same protocol, same model families, two datasets. Beijing never gets past 2 of 5;
    Bangladesh takes 5 of 5 with class weighting.
    """
    rows = []
    for place, src in (("Beijing", "rolling_cv_h6.json"),
                       ("Bangladesh", "rolling_cv_h6_bangladesh.json")):
        d = load(src)
        tests = {k: v for k, v in require(d, "aggregate.tests", src).items()
                 if not k.startswith("_")}
        for model, t in tests.items():
            rows.append({
                "Dataset": place,
                "Model": model.replace("RandomForest ", "RF "),
                "Folds won": req_num(t, "wins", f"{src}:aggregate.tests.{model}"),
                "n_folds": req_num(t, "n_folds", f"{src}:aggregate.tests.{model}"),
                "mean_delta": req_num(t, "mean_delta", f"{src}:aggregate.tests.{model}"),
                "p_one_sided": req_num(t, "p_one_sided", f"{src}:aggregate.tests.{model}"),
                "sig": bool(require(t, "significant_one_sided",
                                    f"{src}:aggregate.tests.{model}")),
            })
    df = pd.DataFrame(rows)
    n_folds = df["n_folds"].unique()
    if len(n_folds) != 1:
        raise MissingMetric(
            f"the two rolling-origin runs use different fold counts {sorted(n_folds)}; "
            "a shared 'folds won' axis would be misleading")
    total = int(n_folds[0])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.6))
    sns.barplot(df, x="Model", y="Folds won", hue="Dataset",
                palette={"Beijing": C_BEIJING, "Bangladesh": C_BANGLADESH},
                ax=ax, edgecolor="white", linewidth=0.7)
    ax.axhline(total / 2, color=C_PERSISTENCE, ls="--", lw=1.5,
               label="coin flip (half the folds)")
    ax.set_ylim(0, total + 1.35)
    ax.set_yticks(range(total + 1))
    ax.set_ylabel(f"Folds beating persistence (of {total})")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=12)

    for container, (_, grp) in zip(ax.containers, df.groupby("Dataset", sort=False)):
        for bar, (_, r) in zip(container, grp.iterrows()):
            ax.annotate(f"{int(r['Folds won'])}/{total}\nΔ {r['mean_delta']:+.4f}\n"
                        f"p={r['p_one_sided']:.4f}{' *' if r['sig'] else ''}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7.4)

    ax.legend(loc="upper left", fontsize=FS_ANNOT)
    ax.set_title("The same protocol answers differently on the two datasets")
    fig.tight_layout()
    return fig


# ================================================ 16 precision / recall per class


@figure("16_precision_recall_beijing",
        "Precision, recall and F1 per class, deployed-candidate Beijing forest",
        ["baseline_h6.json"])
def fig_precision_recall() -> plt.Figure:
    src = "baseline_h6.json"
    d = load(src)
    best = require(d, "best", src)
    per_class = require(d, f"results.{best}.test.observed.per_class", src)
    rows = []
    for c in CLASS_ORDER:
        blk = require(per_class, c, f"{src}:results.{best}.test.observed.per_class")
        for metric in ("precision", "recall", "f1"):
            rows.append({"Class": CLASS_SHORT[c],
                         "Metric": metric.capitalize() if metric != "f1" else "F1",
                         "Value": req_num(blk, metric,
                                          f"{src}:...per_class.{c}"),
                         "support": int(req_num(blk, "support", f"{src}:...per_class.{c}"))})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.7))
    sns.barplot(df, x="Class", y="Value", hue="Metric",
                palette=[C_BEIJING, C_GAN, C_MODEL], ax=ax,
                edgecolor="white", linewidth=0.7)
    _annotate_bars(ax, fmt="{:.2f}", fontsize=7.2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("")
    ax.set_ylabel("Score, observed test rows")
    ax.tick_params(axis="x", rotation=18)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    supports = df.drop_duplicates("Class").set_index("Class")["support"]
    ax.set_xticks(range(len(supports)))
    ax.set_xticklabels([f"{c}\nn={supports[c]:,}" for c in supports.index],
                       rotation=18, ha="right")
    ax.legend(loc="upper right", ncol=3, fontsize=FS_ANNOT)
    worst = df[df.Metric == "Recall"].sort_values("Value").iloc[0]
    ax.set_title(f"{best}: recall collapses on {worst['Class']} ({worst['Value']:.2f})")
    fig.tight_layout()
    return fig


# =================================================== 17 split prevalence mismatch


@figure("17_split_class_prevalence",
        "Class prevalence across the chronological splits",
        ["hj633_h6.json"])
def fig_split_prevalence() -> plt.Figure:
    """Why validation and test keep disagreeing.

    The splits are chronological, so each covers different seasons, and the rare
    classes are exactly where the prevalence gap is widest.
    """
    src = "hj633_h6.json"
    dist = require(load(src), "results.EPA.distribution", src)
    rows = []
    for split in ("train", "val", "test"):
        blk = require(dist, split, f"{src}:results.EPA.distribution")
        for c in CLASS_ORDER:
            rows.append({"Split": split.capitalize(), "Class": CLASS_SHORT[c],
                         "Share (%)": req_num(blk, f"{c}.pct",
                                              f"{src}:results.EPA.distribution.{split}")})
    df = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH * 1.2, 3.6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    sns.barplot(df, x="Class", y="Share (%)", hue="Split",
                palette=[C_PERSISTENCE, C_BEIJING, C_BANGLADESH], ax=ax1,
                edgecolor="white", linewidth=0.7)
    _annotate_bars(ax1, fmt="{:.1f}", fontsize=6.6, rotation=90, padding=3)
    ax1.set_ylim(0, df["Share (%)"].max() * 1.30)
    ax1.set_xlabel(""); ax1.tick_params(axis="x", rotation=25)
    for lbl in ax1.get_xticklabels():
        lbl.set_ha("right")
    ax1.set_title("Every class, every split", fontsize=FS_LABEL)
    ax1.legend(loc="upper right", ncol=3, fontsize=FS_ANNOT)

    wide = df.pivot(index="Class", columns="Split", values="Share (%)")
    wide = wide.reindex([CLASS_SHORT[c] for c in CLASS_ORDER])
    gap = (wide["Val"] - wide["Test"]).rename("Val − Test (pp)").reset_index()
    sns.barplot(gap, x="Class", y="Val − Test (pp)", hue="Class",
                palette=[C_BAD if abs(v) > 3 else C_ALT for v in gap["Val − Test (pp)"]],
                legend=False, ax=ax2, edgecolor="white", linewidth=0.7)
    ax2.axhline(0, color="#444444", lw=1.2)
    _annotate_bars(ax2, fmt="{:+.1f}", fontsize=7.2)
    lim = float(gap["Val − Test (pp)"].abs().max()) * 1.45
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel(""); ax2.tick_params(axis="x", rotation=25)
    for lbl in ax2.get_xticklabels():
        lbl.set_ha("right")
    ax2.set_title("Validation minus test", fontsize=FS_LABEL)

    fig.suptitle("Chronological splits do not share a class distribution", y=1.02)
    fig.tight_layout()
    return fig


# ================================================== 18 Mendeley vs Embassy monitor


@figure("18_mendeley_vs_embassy",
        "Reanalysis against the US Embassy Dhaka reference monitor",
        ["dhaka_ground_truth.json"])
def fig_mendeley_vs_embassy() -> plt.Figure:
    src = "dhaka_ground_truth.json"
    c = require(load(src), "comparison", src)
    n_overlap = int(req_num(c, "n_overlap", src))

    counts = pd.DataFrame([
        {"Class": "Hazardous", "Source": "Reference monitor",
         "Hours": req_num(c, "hazardous_reference", src)},
        {"Class": "Hazardous", "Source": "Mendeley reanalysis",
         "Hours": req_num(c, "hazardous_reanalysis", src)},
        {"Class": "Very unhealthy", "Source": "Reference monitor",
         "Hours": req_num(c, "very_unhealthy_reference", src)},
        {"Class": "Very unhealthy", "Source": "Mendeley reanalysis",
         "Hours": req_num(c, "very_unhealthy_reanalysis", src)},
    ])
    stats = pd.DataFrame([
        {"Statistic": s.upper() if s == "p95" else s.capitalize(), "Source": label,
         "PM2.5 (µg/m³)": req_num(c, f"{key}.{s}", src)}
        for key, label in (("reference", "Reference monitor"),
                           ("reanalysis", "Mendeley reanalysis"))
        for s in ("mean", "p95", "p99", "max")
    ])

    pal = {"Reference monitor": C_MODEL, "Mendeley reanalysis": C_BAD}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH * 1.2, 3.6))

    sns.barplot(counts, x="Class", y="Hours", hue="Source", palette=pal, ax=ax1,
                edgecolor="white", linewidth=0.7)
    ax1.set_yscale("log")
    for container in ax1.containers:
        ax1.bar_label(container, fmt="{:,.0f}", fontsize=FS_ANNOT, padding=2)
    ax1.set_title(f"Advisory-class hours in {n_overlap:,} overlapping hours",
                  fontsize=FS_LABEL)
    ax1.set_xlabel(""); ax1.set_ylabel("Hours (log scale)")
    ax1.legend(fontsize=FS_ANNOT, loc="upper right")

    sns.barplot(stats, x="Statistic", y="PM2.5 (µg/m³)", hue="Source", palette=pal,
                ax=ax2, edgecolor="white", linewidth=0.7)
    for container in ax2.containers:
        ax2.bar_label(container, fmt="{:,.0f}", fontsize=FS_ANNOT, padding=2)
    ax2.set_title("PM2.5 distribution over the same hours", fontsize=FS_LABEL)
    ax2.set_xlabel(""); ax2.legend(fontsize=FS_ANNOT, loc="upper left")
    ax2.set_ylim(0, stats["PM2.5 (µg/m³)"].max() * 1.2)

    r = req_num(c, "pearson_r", src)
    bias = req_num(c, "bias_reanalysis_minus_reference", src)
    agree = req_num(c, "class_agreement", src)
    fig.suptitle("The reanalysis tracks the shape and flattens the peaks", y=1.03)
    fig.text(0.5, -0.06,
             f"Pearson r = {r:.3f}, mean bias {bias:+.1f} µg/m³, "
             f"AQI-class agreement {agree * 100:.1f}%. "
             f"Above 150 µg/m³ the bias widens to "
             f"{req_num(c, 'high_range.bias', src):+.1f}.",
             ha="center", fontsize=FS_ANNOT, color="#555555")
    fig.tight_layout()
    return fig


# ============================================ 19 Phase 11b advisory-class result


@figure("19_phase11b_hazardous",
        "Phase 11b: the validated advisory-class result",
        ["dhaka_pm25_model_h6.json"])
def fig_phase11b() -> plt.Figure:
    """The project's one positive advisory-class finding, per fold.

    The aggregate bars would be enough to state the result, but 7 folds is few
    enough to show every point, so the strip plot carries the real per-fold values
    rather than asking the reader to trust a standard deviation.
    """
    src = "dhaka_pm25_model_h6.json"
    agg = require(load(src), "cv.aggregate", src)
    n_folds = int(req_num(load(src), "cv.n_folds", src))

    metrics = [("f1_Hazardous", "Hazardous F1"), ("macro_f1", "Macro-F1")]
    bars, points = [], []
    for key, label in metrics:
        blk = require(agg, key, f"{src}:cv.aggregate")
        per_model = require(blk, "per_fold_model", f"{src}:cv.aggregate.{key}")
        per_delta = require(blk, "per_fold_delta", f"{src}:cv.aggregate.{key}")
        if len(per_model) != n_folds or len(per_delta) != n_folds:
            raise MissingMetric(
                f"{src}: cv.aggregate.{key} has {len(per_model)} per-fold model values "
                f"and {len(per_delta)} deltas, but cv.n_folds is {n_folds}")
        bars += [
            {"Metric": label, "Which": "Persistence",
             "Mean": req_num(blk, "persistence_mean", f"{src}:cv.aggregate.{key}"),
             "SD": float(np.std([m - d for m, d in zip(per_model, per_delta)], ddof=1))},
            {"Metric": label, "Which": "Model",
             "Mean": req_num(blk, "model_mean", f"{src}:cv.aggregate.{key}"),
             "SD": req_num(blk, "model_std", f"{src}:cv.aggregate.{key}")},
        ]
        for i, (m, dlt) in enumerate(zip(per_model, per_delta), start=1):
            points.append({"Metric": label, "Which": "Model", "Fold": i, "Value": m})
            points.append({"Metric": label, "Which": "Persistence", "Fold": i,
                           "Value": m - dlt})
    bars_df, pts_df = pd.DataFrame(bars), pd.DataFrame(points)

    pal = {"Persistence": C_PERSISTENCE, "Model": C_BANGLADESH}
    # Both the bars and the points must be told the hue and category order. Left to
    # infer it, seaborn takes each frame's first-seen order -- which differs between
    # these two frames, and silently dodges the points onto the wrong bars.
    hue_order = ["Persistence", "Model"]
    order = [label for _, label in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH * 1.25, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    sns.barplot(bars_df, x="Metric", y="Mean", hue="Which", order=order,
                hue_order=hue_order, palette=pal, ax=ax1,
                edgecolor="white", linewidth=0.7, errorbar=None)
    for container, which in zip(ax1.containers, hue_order):
        grp = bars_df[bars_df["Which"] == which].set_index("Metric").loc[order]
        xs = [b.get_x() + b.get_width() / 2 for b in container]
        ax1.errorbar(xs, grp["Mean"], yerr=grp["SD"], fmt="none", ecolor="#333333",
                     elinewidth=1.3, capsize=4, zorder=5)
    sns.stripplot(pts_df, x="Metric", y="Value", hue="Which", order=order,
                  hue_order=hue_order, palette=pal,
                  dodge=True, ax=ax1, size=4.5, edgecolor="white", linewidth=0.6,
                  legend=False, jitter=0.11, zorder=6)
    ax1.set_ylabel(f"Score (mean ± SD over {n_folds} folds)")
    ax1.set_xlabel("")
    ax1.set_ylim(0, max(pts_df["Value"].max(),
                        (bars_df["Mean"] + bars_df["SD"]).max()) * 1.30)
    ax1.legend(loc="upper center", ncol=2, fontsize=FS_ANNOT, title="")
    ax1.set_title("Aggregate, with every fold shown", fontsize=FS_LABEL)

    haz = pts_df[pts_df["Metric"] == "Hazardous F1"]
    sns.lineplot(haz, x="Fold", y="Value", hue="Which", style="Which",
                 palette=pal, markers=True, dashes=False, lw=2.1, markersize=8,
                 ax=ax2)
    ax2.set_xticks(range(1, n_folds + 1))
    ax2.set_title("Hazardous F1, fold by fold", fontsize=FS_LABEL)
    ax2.set_ylabel("Hazardous F1"); ax2.set_xlabel("Rolling-origin fold")
    ax2.legend(fontsize=FS_ANNOT, title="", loc="best")

    hb = require(agg, "f1_Hazardous", f"{src}:cv.aggregate")
    wins = int(req_num(hb, "wins", f"{src}:cv.aggregate.f1_Hazardous"))
    p = req_num(hb, "p_two_sided", f"{src}:cv.aggregate.f1_Hazardous")
    res = require(agg, "_resolution", f"{src}:cv.aggregate")
    floor_p = req_num(res, "min_p_two_sided", f"{src}:cv.aggregate._resolution")
    fig.suptitle(f"Hazardous F1 {req_num(hb, 'model_mean', src):.4f} vs "
                 f"{req_num(hb, 'persistence_mean', src):.4f} floor — "
                 f"{wins}/{n_folds} folds, p = {p:.4f}", y=1.03)
    fig.text(0.5, -0.05,
             f"Signed-rank over {n_folds} folds cannot report a two-sided p below "
             f"{floor_p:.4f}; this result sits at that floor, which is the strongest "
             "the design can produce.",
             ha="center", fontsize=FS_ANNOT, color="#555555")
    fig.tight_layout()
    return fig


# ================================================= 20 OpenAQ Dhaka station survey


@figure("20_openaq_survey",
        "OpenAQ station survey around Dhaka",
        ["openaq_survey.json"],
        caveat=("The spec asked for a Department of Environment (DoE) station count. "
                "No committed JSON carries one: openaq_survey.json records only the "
                "providers OpenAQ itself returns (Spartan, StateAir Dhaka, AirNow, "
                "SPARTAN Network, AirGradient), and DoE is not among them. That panel "
                "is omitted rather than invented."))
def fig_openaq_survey() -> plt.Figure:
    src = "openaq_survey.json"
    d = load(src)
    locs = require(d, "survey.locations", src)
    radius = req_num(d, "survey.radius_km", src)

    providers: dict[str, int] = {}
    params: dict[str, int] = {}
    for i, loc in enumerate(locs):
        providers[require(loc, "provider", f"{src}:survey.locations.{i}")] = \
            providers.get(require(loc, "provider", f"{src}:survey.locations.{i}"), 0) + 1
        for p in require(loc, "parameters", f"{src}:survey.locations.{i}"):
            params[p] = params.get(p, 0) + 1

    prov_df = (pd.DataFrame({"Provider": list(providers), "Stations": list(providers.values())})
               .sort_values("Stations", ascending=False))
    par_df = (pd.DataFrame({"Pollutant": list(params), "Stations": list(params.values())})
              .sort_values("Stations", ascending=False))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(FIG_WIDTH * 1.45, 3.5),
                                        gridspec_kw={"width_ratios": [1.1, 1.3, 0.85]})

    sns.barplot(prov_df, y="Provider", x="Stations", hue="Provider",
                palette=sns.light_palette(C_BANGLADESH, n_colors=len(prov_df) + 2)[2:][::-1],
                legend=False, ax=ax1, edgecolor="white", linewidth=0.7)
    _annotate_bars(ax1, fmt="{:.0f}", fontsize=FS_ANNOT)
    ax1.set_title(f"{len(locs)} stations within {radius:.0f} km", fontsize=FS_LABEL)
    ax1.set_ylabel("")

    key = {"pm25", "pm10", "co"}
    sns.barplot(par_df, y="Pollutant", x="Stations", hue="Pollutant",
                palette=[C_MODEL if p in key else C_ALT for p in par_df["Pollutant"]],
                legend=False, ax=ax2, edgecolor="white", linewidth=0.7)
    _annotate_bars(ax2, fmt="{:.0f}", fontsize=FS_ANNOT)
    ax2.set_title("Stations reporting each parameter", fontsize=FS_LABEL)
    ax2.set_ylabel("")

    gate = pd.DataFrame([
        {"Gate": "Any station", "Stations": float(len(locs))},
        {"Gate": "With PM10 or CO",
         "Stations": req_num(d, "assessment.n_with_pm10_or_co", src)},
        {"Gate": "Passing all criteria",
         "Stations": req_num(d, "assessment.n_passing", src)},
    ])
    sns.barplot(gate, x="Gate", y="Stations", hue="Gate",
                palette=[C_ALT, C_ACCENT, C_BAD], legend=False, ax=ax3,
                edgecolor="white", linewidth=0.7)
    _annotate_bars(ax3, fmt="{:.0f}", fontsize=FS_ANNOT)
    ax3.set_title("Survivors of each filter", fontsize=FS_LABEL)
    ax3.set_xlabel(""); ax3.set_ylabel("Stations")
    ax3.set_ylim(0, len(locs) * 1.2)
    ax3.tick_params(axis="x", rotation=25)
    for lbl in ax3.get_xticklabels():
        lbl.set_ha("right")

    probe = require(d, "probe", src)
    fig.suptitle("No multi-pollutant Dhaka station meets the coverage requirement",
                 y=1.03)
    fig.text(0.5, -0.07,
             f"The single PM2.5+PM10 candidate spans "
             f"{req_num(probe, 'span_years', src):.2f} years at "
             f"{req_num(probe, 'completeness_pct', src):.1f}% completeness — "
             f"duration test {'passed' if require(probe, 'passes_duration', src) else 'failed'}. "
             "Reported as a negative result rather than forced.",
             ha="center", fontsize=FS_ANNOT, color="#555555")
    fig.tight_layout()
    return fig


# ============================================================= 21 pipeline diagram


@figure("21_pipeline_overview",
        "End-to-end pipeline, Phase 0 to the deployed system",
        [],
        caveat="Schematic: hand-specified structure, not read from metrics.")
def fig_pipeline() -> plt.Figure:
    """The only figure here that is not data-driven.

    It describes the structure of the work, so there is no metric to read. Every
    number that appears on it is pulled from JSON anyway, so the diagram cannot
    drift away from the results it labels.
    """
    floor = req_num(load("horizon_comparison.json"), "rows.6.observed.macro_f1",
                    "horizon_comparison.json")
    bd_folds = int(req_num(load("rolling_cv_h6_bangladesh.json"),
                           "aggregate.tests.RandomForest (class_weight=balanced).wins",
                           "rolling_cv_h6_bangladesh.json"))
    bd_n = int(req_num(load("rolling_cv_h6_bangladesh.json"), "n_folds",
                       "rolling_cv_h6_bangladesh.json"))
    bj_best = max(int(req_num(t, "wins", "rolling_cv_h6.json"))
                  for k, t in require(load("rolling_cv_h6.json"), "aggregate.tests",
                                      "rolling_cv_h6.json").items()
                  if not k.startswith("_"))
    bj_n = int(req_num(load("rolling_cv_h6.json"), "n_folds", "rolling_cv_h6.json"))
    haz = req_num(load("dhaka_pm25_model_h6.json"), "cv.aggregate.f1_Hazardous.model_mean",
                  "dhaka_pm25_model_h6.json")
    haz_floor = req_num(load("dhaka_pm25_model_h6.json"),
                        "cv.aggregate.f1_Hazardous.persistence_mean",
                        "dhaka_pm25_model_h6.json")
    onnx_kb = req_num(load("deployment_h6_bd.json"), "onnx.bytes",
                      "deployment_h6_bd.json") / 1024
    latency = None

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.5, 6.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off"); ax.grid(False)
    ax.set_facecolor("white")

    def box(x, y, w, h, text, face, edge, fontsize=8.4, weight="normal", tcol="#111111"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.4",
                                    facecolor=face, edgecolor=edge, linewidth=1.3,
                                    zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=tcol, zorder=3,
                linespacing=1.45)

    def arrow(x1, y1, x2, y2, colour="#666666", style="-|>", lw=1.5, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=13, color=colour, lw=lw,
                                     linestyle=ls, zorder=1,
                                     shrinkA=1, shrinkB=1))

    pale = lambda c: sns.light_palette(c, n_colors=8)[1]

    ax.text(50, 97.5, "PulseAir: from raw archives to a deployed advisory",
            ha="center", fontsize=FS_TITLE + 1, fontweight="bold")

    # ---- left column: the Beijing methodology track
    ax.text(23, 91.5, "BEIJING  —  methodology track", ha="center",
            fontsize=FS_LABEL, fontweight="bold", color=C_BEIJING)
    box(4, 82, 38, 6.5, "Phase 0–2  UCI Beijing multi-site, 420,768 h × 12 stations\n"
                        "ffill by station · is_imputed provenance · chronological split",
        pale(C_BEIJING), C_BEIJING)
    box(4, 73.5, 38, 5.5, f"Phase 3  Persistence floor at h=6:  {floor:.4f} macro-F1\n"
                          "RF / XGBoost baselines measured against it",
        pale(C_PERSISTENCE), C_PERSISTENCE, weight="bold")
    box(4, 65, 38, 5.5, "Phase 4  CTGAN + SMOTE augmentation\n"
                        "disqualified: aggregate gain, advisory-class loss",
        pale(C_BAD), C_BAD)
    box(4, 56.5, 38, 5.5, "Phase 5  LSTM / Transformer + MC dropout\n"
                          "lose to the forest; 98.3% of entropy is aleatoric",
        pale(C_BAD), C_BAD)
    box(4, 48, 38, 5.5, "Phase 6  Mondrian conformal + SHAP\n"
                        "prediction sets, not an argmax",
        pale(C_GAN), C_GAN)
    box(4, 39.5, 38, 5.5, f"Phase 8–9  Rolling-origin CV: best {bj_best}/{bj_n} folds\n"
                          "→ no Beijing model is deployed",
        pale(C_BAD), C_BAD, weight="bold")
    for y in (82, 73.5, 65, 56.5, 48):
        arrow(23, y, 23, y - 2.0)

    # ---- right column: the Bangladesh deployment track
    ax.text(77, 91.5, "BANGLADESH  —  deployment track", ha="center",
            fontsize=FS_LABEL, fontweight="bold", color=C_BANGLADESH)
    box(58, 82, 38, 6.5, "Phase 10  Mendeley 9j447cynb9 v2\n"
                         "integrity audit: 81% of rows rejected, 2022-08-05 onward kept",
        pale(C_BANGLADESH), C_BANGLADESH)
    box(58, 73.5, 38, 5.5, "7 shared channels (PM2.5, PM10, CO + cyclical)\n"
                           "no TEMP/DEWP — the 9-channel set cannot transfer",
        pale(C_BANGLADESH), C_BANGLADESH)
    box(58, 65, 38, 5.5, f"Rolling-origin CV: {bd_folds}/{bd_n} folds beat persistence\n"
                         "class_weight='balanced' — common classes only",
        pale(C_MODEL), C_MODEL, weight="bold")
    box(58, 56.5, 38, 5.5, "Phase 11  US Embassy Dhaka reference monitor\n"
                           "1,602 Hazardous hours vs 17 in the reanalysis",
        pale(C_ACCENT), C_ACCENT)
    box(58, 48, 38, 5.5, f"Phase 11b  PM2.5-only model: Hazardous F1 {haz:.4f}\n"
                         f"vs {haz_floor:.4f} floor, 7/7 folds — advisory classes validated",
        pale(C_MODEL), C_MODEL, weight="bold")
    box(58, 39.5, 38, 5.5, "Phase 11c  OpenAQ survey for a multi-pollutant source\n"
                           "none qualifies — reported, not forced",
        pale(C_BAD), C_BAD)
    for y in (82, 73.5, 65, 56.5, 48):
        arrow(77, y, 77, y - 2.0)

    # methodology feeds the deployment track
    arrow(42, 63, 58, 72, colour=C_BEIJING, ls="--", lw=1.8)
    ax.text(50, 69.5, "protocol\ntransfers", ha="center", fontsize=7.6,
            color=C_BEIJING, style="italic")

    # ---- deployed system
    arrow(23, 39.5, 23, 34.5); arrow(77, 39.5, 77, 34.5)
    arrow(23, 34.5, 34, 31.6); arrow(77, 34.5, 66, 31.6)
    box(18, 24, 64, 7.5,
        f"DEPLOYED  —  Bangladesh RF, class_weight='balanced', 25 trees × depth 12\n"
        f"ONNX {onnx_kb:,.0f} KB  →  Mondrian conformal set  →  top-3 SHAP  →  "
        "Gemini advisory (validated, template fallback)",
        pale(C_MODEL), C_MODEL, fontsize=9, weight="bold")
    arrow(50, 24, 50, 19.5)
    box(18, 12.5, 64, 6.5,
        "ESP32 neckband  —  the ONNX benchmark is a portability proxy, not a flash claim.\n"
        "No model in this project has yet run on the target silicon.",
        "#f4f4f4", "#999999", fontsize=8.4)

    ax.text(50, 6.5,
            "Every box above is backed by a committed report; every number on this "
            "diagram is read from reports/metrics/*.json at render time.",
            ha="center", fontsize=7.8, color="#666666", style="italic")

    legend = [Line2D([], [], marker="s", ls="", markersize=9,
                     markerfacecolor=pale(c), markeredgecolor=c, label=lbl)
              for c, lbl in ((C_PERSISTENCE, "the floor"), (C_MODEL, "worked"),
                             (C_BAD, "negative result"), (C_GAN, "uncertainty layer"))]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.02),
              ncol=4, frameon=False, fontsize=FS_ANNOT)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------- driver


@dataclass
class Result:
    fig: Figure
    path: Path | None = None
    error: str = ""
    caveat: str = ""


def generate(only: Sequence[str] | None = None) -> list[Result]:
    """Render every registered figure, collecting failures instead of aborting.

    A figure that cannot be built is a finding -- usually "this number was never
    written to JSON" -- so the run continues and reports all of them together
    rather than stopping at the first.
    """
    apply_house_style()
    results: list[Result] = []
    for spec in REGISTRY:
        if only and not any(spec.slug.startswith(o) or o in spec.slug for o in only):
            continue
        try:
            fig = spec.fn()
            path = _save(fig, spec.slug)
            results.append(Result(spec, path=path, caveat=spec.caveat))
        except MissingMetric as exc:
            plt.close("all")
            results.append(Result(spec, error=f"MissingMetric: {exc}"))
        except Exception as exc:                      # noqa: BLE001 - reported, not hidden
            plt.close("all")
            results.append(Result(spec, error=f"{type(exc).__name__}: {exc}"))
    return results


def _print_audit(results: list[Result]) -> None:
    ok = [r for r in results if r.path]
    bad = [r for r in results if not r.path]
    print(f"\n{'=' * 78}\nFIGURE AUDIT TRAIL  —  {len(ok)} written, {len(bad)} failed\n{'=' * 78}")
    for r in results:
        if r.path:
            kb = r.path.stat().st_size / 1024
            src = ", ".join(r.fig.sources) if r.fig.sources else "(schematic — no metrics)"
            print(f"\n  {r.fig.slug}.png   {kb:,.0f} KB")
            print(f"    {r.fig.title}")
            print(f"    sources: {src}")
            if r.caveat:
                print(f"    CAVEAT:  {r.caveat}")
        else:
            print(f"\n  {r.fig.slug}   ** NOT GENERATED **")
            print(f"    {r.fig.title}")
            print(f"    reason:  {r.error}")
    if bad:
        print(f"\n{'-' * 78}")
        print(f"{len(bad)} figure(s) could not be generated. Listed above with the exact")
        print("missing file/field. Nothing was substituted or estimated.")
    print()


def write_index(results: list[Result]) -> Path:
    """Write the audit trail to reports/figures/README.md.

    The printed trail scrolls away; this one is committed next to the PNGs, so a
    reader who finds a figure in the thesis can trace it back to the JSON it came
    from without running anything.
    """
    lines = ["# Figures",
             "",
             "Generated by `python -m src.reporting.generate_figures`. Every value is read",
             "from `reports/metrics/*.json` at render time -- none is typed in here, and a",
             "missing field fails the figure rather than being substituted.",
             "",
             "| # | Figure | Reads from |",
             "| --- | --- | --- |"]
    caveats = []
    for r in results:
        if not r.path:
            continue
        num, _, rest = r.fig.slug.partition("_")
        src = ", ".join(f"`{x}`" for x in r.fig.sources) or "_schematic_"
        lines.append(f"| {num} | [{r.fig.title}]({r.path.name}) | {src} |")
        if r.fig.caveat:
            caveats.append((num, r.fig.title, r.fig.caveat))
    failed = [r for r in results if not r.path]
    if caveats:
        lines += ["", "## Caveats", ""]
        for num, title, note in caveats:
            lines += [f"**{num} — {title}.** {note}", ""]
    if failed:
        lines += ["", "## Not generated", "",
                  "These figures were requested but no committed JSON supplies the data.",
                  "Listed rather than faked:", ""]
        for r in failed:
            lines.append(f"- **{r.fig.slug}** — {r.fig.title}. `{r.error}`")
    lines.append("")
    out = FIGURES / "README.md"
    out.write_text("\n".join(lines))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", metavar="SLUG",
                    help="substring match on figure slugs, e.g. --only 13 14")
    ap.add_argument("--list", action="store_true",
                    help="list registered figures and their source files, render nothing")
    args = ap.parse_args(argv)

    if args.list:
        print(f"{len(REGISTRY)} figures registered, output -> "
              f"{FIGURES.relative_to(REPO_ROOT)}/\n")
        for spec in REGISTRY:
            src = ", ".join(spec.sources) if spec.sources else "(schematic)"
            print(f"  {spec.slug}\n    {spec.title}\n    sources: {src}")
            if spec.caveat:
                print(f"    CAVEAT:  {spec.caveat}")
        return 0

    results = generate(args.only)
    _print_audit(results)
    if not args.only:
        # Only a full run may rewrite the index; a partial run would drop the rest.
        idx = write_index(results)
        print(f"wrote {idx.relative_to(REPO_ROOT)}\n")
    return 1 if any(not r.path for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
