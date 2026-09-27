"""One command to reproduce the whole pipeline, in dependency order.

This is an orchestration layer only. It runs the project's already-existing steps
(``python -m src.…``) in the correct sequence with progress output; it changes nothing
about what any step does. Every command here is one a reader could otherwise copy out of
the README by hand — the value added is the order, the timing, and stop-on-failure.

    python -m src.reproduce_all                 # full pipeline, environment already set up
    python -m src.reproduce_all --list          # print the plan and exit
    python -m src.reproduce_all --skip-slow      # skip the multi-hour GAN + deep-learning steps
    python -m src.reproduce_all --skip-network   # skip steps that fetch data / call an API
    python -m src.reproduce_all --only-reporting # just recompile results, figures and the thesis
    python -m src.reproduce_all --from rolling_cv # resume from a named stage
    python -m src.reproduce_all --with-install    # pip install first, then run everything

Flags compose. `--dry-run` prints what would run without running it. On a failure the
run stops and reports which stage failed (use `--keep-going` to continue past failures).

Notes on the heaviest and least self-contained steps, all tagged below:
  * SLOW   — GAN training (~2.5 h) and the deep-learning sweeps; skip with --skip-slow.
  * NET    — data downloads, the OpenAQ survey and the DoE crosscheck fetch over the
             network; skip with --skip-network.
  * OPT    — the LLM advisory layer needs GEMINI_API_KEY in .env; skip with --skip-optional.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field


@dataclass
class Stage:
    name: str
    argv: list[str]
    desc: str
    tags: set[str] = field(default_factory=set)   # {"slow","net","opt","report"}


PY = [sys.executable, "-m"]


def _m(mod: str, *args: str) -> list[str]:
    return PY + [mod, *args]


# Ordered by dependency. Each entry is a step already documented in the README or added
# by an extension phase; nothing here is new behaviour.
STAGES: list[Stage] = [
    # --- environment (only with --with-install) ---
    Stage("install", ["__pip__"], "install the package and test extras", {"install"}),

    # --- data ---
    Stage("download", _m("src.preprocessing.download"),
          "fetch the UCI Beijing dataset", {"net"}),
    Stage("preprocess", _m("src.preprocessing.pipeline", "--all-horizons"),
          "build processed frames for all horizons"),

    # --- Beijing baselines ---
    Stage("baseline", _m("src.models.baseline"), "RandomForest / XGBoost baseline (h=6)"),
    Stage("horizon_sweep", _m("src.models.baseline", "--horizon-sweep"),
          "persistence and model across h=1/6/12/24"),

    # --- GAN augmentation (slow) ---
    Stage("gan_augment", _m("src.gan.augment"), "train the CTGAN synthesizers", {"slow"}),
    Stage("baseline_augmented", _m("src.models.baseline", "--augmented"),
          "retrain on the augmented table", {"slow"}),
    Stage("gan_ablation", _m("src.gan.ablation"), "broad/targeted/unaugmented ablation"),
    Stage("smote_control", _m("src.gan.smote_control"), "SMOTE and class-weight controls"),
    Stage("gan_validity", _m("src.gan.validity_before_fix"),
          "re-measure the before-fix CTGAN validity"),

    # --- deep learning (slow) ---
    Stage("dl_forecast", _m("src.models.dl_forecast"),
          "LSTM/Transformer, MC dropout, calibration", {"slow"}),
    Stage("capacity_sweep", _m("src.models.capacity_sweep"),
          "hidden-size sweep for both architectures", {"slow"}),

    # --- conformal + explainability ---
    Stage("conformal", _m("src.models.conformal"), "split vs Mondrian conformal coverage"),
    Stage("shap", _m("src.explainability.shap_analysis"), "SHAP attributions and case plots"),
    Stage("llm_advisory", _m("src.explainability.llm_advisory"),
          "LLM advisory examples (needs GEMINI_API_KEY)", {"net", "opt"}),

    # --- deployment ---
    Stage("compress", _m("src.deployment.compress_export"),
          "compression sweep + ONNX export (h=6)"),
    Stage("compress_cw", _m("src.deployment.compress_export", "--class-weight"),
          "class-weighted deployment variant"),

    # --- rolling-origin CV + sensitivity ---
    Stage("rolling_cv", _m("src.models.rolling_cv"), "5-fold rolling-origin CV (Beijing)"),
    Stage("rolling_cv_8", _m("src.models.rolling_cv", "--folds", "8"),
          "8-fold rolling-origin CV, tabular families"),
    Stage("rolling_cv_8_seq",
          _m("src.models.rolling_cv", "--folds", "8", "--only-sequence-models"),
          "8-fold rolling-origin CV, sequence models", {"slow"}),
    Stage("hj633", _m("src.models.hj633_sensitivity"), "EPA vs HJ 633-2012 breakpoints"),

    # --- extended validation ---
    Stage("station_holdout", _m("src.models.station_holdout_cv"),
          "leave-one-station-out generalisation"),
    Stage("sensor_noise", _m("src.models.sensor_noise_robustness"),
          "low-cost-sensor noise robustness"),
    Stage("selective", _m("src.models.selective_prediction"),
          "selective prediction on the conformal sets"),
    Stage("fold_count", _m("src.reporting.fold_count_comparison"),
          "5-fold vs 8-fold comparison report"),

    # --- Bangladesh external validation ---
    Stage("bd_preprocess", _m("src.preprocessing.bangladesh"),
          "Bangladesh adapter + data-integrity audit"),
    Stage("bd_validation", _m("src.models.bangladesh_validation"),
          "transfer vs native on Bangladesh"),
    Stage("bd_rolling", _m("src.models.rolling_cv", "--dataset", "bangladesh"),
          "rolling-origin CV on Bangladesh"),
    Stage("bd_compress", _m("src.deployment.compress_export", "--dataset", "bangladesh"),
          "deployed compression point for Bangladesh"),
    Stage("bd_selective", _m("src.models.selective_prediction", "--dataset", "bangladesh"),
          "selective prediction for the deployed model"),
    Stage("dhaka_ground_truth", _m("src.models.dhaka_ground_truth"),
          "US Embassy reference-monitor comparison", {"net"}),
    Stage("dhaka_model", _m("src.models.dhaka_pm25_model"),
          "PM2.5-only Hazardous detector (Phase 11b)"),
    Stage("openaq", _m("src.preprocessing.openaq_survey"),
          "OpenAQ station survey around Dhaka", {"net"}),
    Stage("integrity_audit", _m("src.models.integrity_audit"),
          "pulsebench.dataset_audit on the raw Mendeley file"),
    Stage("dhaka_crosscheck", _m("src.preprocessing.dhaka_monthly_crosscheck", "--download"),
          "fifth fabrication signature vs DoE monthly reports", {"net"}),

    # --- reporting (final) ---
    Stage("compile", _m("src.reporting.compile_results"),
          "master results summary + report index", {"report"}),
    Stage("figures", _m("src.reporting.generate_figures"),
          "render all figures from committed metrics", {"report"}),
    Stage("thesis", _m("src.reporting.build_thesis"),
          "assemble the thesis .docx", {"report"}),
]

INSTALL_ARGV = [sys.executable, "-m", "pip", "install", "-e", ".[test]"]


def select(args) -> list[Stage]:
    stages = list(STAGES)
    if not args.with_install:
        stages = [s for s in stages if "install" not in s.tags]
    if args.only_reporting:
        return [s for s in stages if "report" in s.tags]
    if args.only:
        chosen = [s for s in stages if s.name in set(args.only)]
        missing = set(args.only) - {s.name for s in chosen}
        if missing:
            sys.exit(f"unknown stage(s): {', '.join(sorted(missing))}")
        return chosen
    if args.from_stage:
        names = [s.name for s in stages]
        if args.from_stage not in names:
            sys.exit(f"unknown --from stage: {args.from_stage}")
        stages = stages[names.index(args.from_stage):]
    if args.skip_slow:
        stages = [s for s in stages if "slow" not in s.tags]
    if args.skip_network:
        stages = [s for s in stages if "net" not in s.tags]
    if args.skip_optional:
        stages = [s for s in stages if "opt" not in s.tags]
    return stages


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--dry-run", action="store_true", help="show commands without running")
    ap.add_argument("--skip-slow", action="store_true", help="skip multi-hour steps")
    ap.add_argument("--skip-network", action="store_true", help="skip steps needing network")
    ap.add_argument("--skip-optional", action="store_true", help="skip steps needing an API key")
    ap.add_argument("--only-reporting", action="store_true",
                    help="only recompile results, figures and thesis from committed metrics")
    ap.add_argument("--only", nargs="+", metavar="STAGE", help="run only these named stages")
    ap.add_argument("--from", dest="from_stage", metavar="STAGE", help="resume from this stage")
    ap.add_argument("--with-install", action="store_true", help="pip install before running")
    ap.add_argument("--keep-going", action="store_true", help="continue past a failed stage")
    args = ap.parse_args(argv)

    stages = select(args)
    n = len(stages)

    def tagstr(s: Stage) -> str:
        order = [("slow", "SLOW"), ("net", "NET"), ("opt", "OPT")]
        badges = [b for t, b in order if t in s.tags]
        return f"  [{','.join(badges)}]" if badges else ""

    if args.list or args.dry_run:
        print(f"Plan: {n} stage(s)\n")
        for i, s in enumerate(stages, 1):
            cmd = "pip install -e '.[test]'" if "install" in s.tags else " ".join(s.argv)
            print(f"  {i:2}/{n}  {s.name:20} {s.desc}{tagstr(s)}")
            print(f"          $ {cmd}")
        return 0

    print(f"Reproducing the pipeline: {n} stage(s). "
          f"Stop on first failure unless --keep-going.\n")
    results: list[tuple[str, str, float]] = []
    t0 = time.time()
    for i, s in enumerate(stages, 1):
        argv_i = INSTALL_ARGV if "install" in s.tags else s.argv
        print(f"[{i}/{n}] {s.name} — {s.desc}{tagstr(s)}")
        print(f"      $ {' '.join(argv_i)}", flush=True)
        started = time.time()
        rc = subprocess.run(argv_i).returncode
        dt = time.time() - started
        status = "ok" if rc == 0 else f"FAILED (exit {rc})"
        results.append((s.name, status, dt))
        print(f"      -> {status} in {dt:.1f}s\n", flush=True)
        if rc != 0 and not args.keep_going:
            print(f"Stopped at stage '{s.name}'. Fix it and resume with "
                  f"`--from {s.name}`.")
            _summary(results, time.time() - t0)
            return rc
    _summary(results, time.time() - t0)
    return 0 if all(st == "ok" for _, st, _ in results) else 1


def _summary(results, total: float) -> None:
    ok = sum(1 for _, st, _ in results if st == "ok")
    print("=" * 60)
    print(f"Summary: {ok}/{len(results)} stages ok, {total:.1f}s total")
    for name, st, dt in results:
        if st != "ok":
            print(f"  {st:18} {name}")


if __name__ == "__main__":
    raise SystemExit(main())
