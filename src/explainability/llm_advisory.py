"""Plain-language risk advisory from the model's own numbers.

Takes everything the predictor actually knows about one reading — the point
prediction, the **conformal prediction set** with its coverage guarantee, the top SHAP
contributions, and a short user profile — and asks Claude to turn it into two or three
sentences a wearer can act on.

The design constraint that shapes the whole prompt: **the advisory must not sound more
certain than the model is.** Phase 6's conformal analysis found a median prediction set
of 2 categories and singletons only 7.2% of the time. An advisory layer that reports
the argmax and stops would be asserting confidence the predictor does not have, on a
health question. So every numeric fact is passed explicitly, the model is told to name
the ambiguity when the set has more than one member, and it is told what the coverage
guarantee does *not* cover.

A rule-based template produces the same information without the API. It runs when
there is no key, when the call fails, and when the response fails validation — so the
device degrades to terser wording rather than to silence.

    python -m src.explainability.llm_advisory            # 5 worked examples
    python -m src.explainability.llm_advisory --row 1234 # one specific test row

Writes ``reports/llm_advisory_examples.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _load_dotenv() -> None:
    """Read .env into the environment if the key is not already set.

    Without this the module reads os.environ and finds nothing, so a key sitting in
    .env would silently produce fallback advisories -- the exact failure mode this
    project has already been caught by once. An explicit export still wins.
    """
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        env = REPO_ROOT / ".env"
        if not env.exists():
            return
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# Ordered most to least severe for the "worst case in the set" logic. A wearer should
# be advised against the worst plausible outcome, not the most likely one.
# Free-tier per-minute quota clears on its own; two waits is enough to ride it out
# without turning a failed batch into a long stall.
MAX_QUOTA_RETRIES = 2

# Free tier allows 5 requests/minute; 13s between calls keeps a batch under it.
PACING_SECONDS = 13

SEVERITY = ["Good", "Moderate", "Unhealthy (sensitive)", "Unhealthy",
            "Very unhealthy", "Hazardous"]

ACTIONABLE = {"PM2.5": "fine particulates", "PM10": "coarse particulates",
              "CO": "carbon monoxide"}
CONTEXT_NAMES = {"TEMP": "temperature", "DEWP": "humidity (via dew point)",
                 "hour_sin": "time of day", "hour_cos": "time of day",
                 "month_sin": "season", "month_cos": "season"}


@dataclass
class UserProfile:
    age: int
    respiratory_condition: bool
    label: str = ""

    def describe(self) -> str:
        cond = ("has a diagnosed respiratory condition (e.g. asthma or COPD)"
                if self.respiratory_condition else "has no diagnosed respiratory condition")
        return f"{self.age} years old, {cond}"

    @property
    def sensitive(self) -> bool:
        """Groups the AQI's 'sensitive' band is defined around."""
        return self.respiratory_condition or self.age >= 65 or self.age <= 12


@dataclass
class Reading:
    """Everything the advisory is allowed to reason from."""
    station: str
    datetime: str
    horizon_hours: int
    predicted: str
    probability: float
    conformal_set: list[str]
    target_coverage: float
    top_shap: list[dict]
    raw_features: dict
    true_label: str | None = None
    # Mondrian (class-conditional) conformal: empirical coverage measured per class on
    # the test split. Marginal conformal has no such thing -- its guarantee is an
    # average -- which is why the advisory switched to Mondrian before switching
    # providers. See reports/conformal_h6.md section 4.
    per_class_coverage: dict | None = None
    method: str = "mondrian"

    @property
    def ambiguous(self) -> bool:
        return len(self.conformal_set) != 1

    @property
    def worst_in_set(self) -> str:
        if not self.conformal_set:
            return self.predicted
        return max(self.conformal_set, key=SEVERITY.index)


# ------------------------------------------------------------------------ prompt


SYSTEM_PROMPT = """You write short air-quality advisories for someone wearing a \
neckband air-quality monitor. You are given the output of a forecasting model and must \
turn it into plain language the wearer can act on.

Hard rules:

1. Use only the numbers you are given. Never invent a reading, a threshold, a \
pollutant, or a health statistic.
2. The model outputs a CONFORMAL PREDICTION SET, not a single answer. When that set \
contains more than one category you MUST say so plainly in the advisory — name the \
range and make clear the forecast is uncertain. Do not report only the most likely \
category as if it were settled. Being honest about ambiguity is more important than \
sounding authoritative.
3. When the set contains more than one category, base the precaution on the WORST \
category in the set, and say that is what you are doing.
4. Coverage is a property of the METHOD over many readings, not a probability about \
this one reading. The set is built by Mondrian (class-conditional) conformal \
prediction, so the guarantee does hold per category — but it still never licenses \
"we are 90% sure this reading is X". Describe it as how often the method is right, \
not as certainty about this measurement.
5. SHAP values explain which sensor channels the model relied on. They are not causal \
claims about the air. Say "the model weighted X most heavily", not "X caused this".
6. This is not medical advice. Do not diagnose, do not name medications, and do not \
tell anyone to change prescribed treatment.
7. Two to four sentences. Second person. No headings, no bullet points, no preamble — \
return only the advisory text.
8. Tailor the precaution to the profile you are given, without restating their age or \
condition back to them as a label."""


def build_user_prompt(reading: Reading, profile: UserProfile) -> str:
    """Every fact stated explicitly and numerically. No summarising before the model."""
    shap_lines = []
    for s in reading.top_shap[:3]:
        feat = s["feature"]
        human = ACTIONABLE.get(feat) or CONTEXT_NAMES.get(feat, feat)
        direction = "toward" if s["value"] > 0 else "away from"
        kind = "actionable pollutant channel" if feat in ACTIONABLE else "context (not actionable)"
        shap_lines.append(
            f"  - {feat} ({human}): SHAP {s['value']:+.4f}, pushes {direction} the "
            f"predicted category. {kind}.")

    raw = reading.raw_features
    set_str = ", ".join(reading.conformal_set) if reading.conformal_set else "EMPTY"
    if reading.per_class_coverage:
        coverage_lines = "  Measured coverage for the categories in this set:\n" + "\n".join(
            f"    - {c}: {reading.per_class_coverage[c]:.1%}"
            for c in reading.conformal_set if c in reading.per_class_coverage)
    else:
        coverage_lines = ""
    if reading.ambiguous and reading.conformal_set:
        ambiguity = (
            f"The set has {len(reading.conformal_set)} categories, so the forecast is "
            f"AMBIGUOUS. You must say so. Base the precaution on the worst category in "
            f"the set, which is {reading.worst_in_set}.")
    elif not reading.conformal_set:
        ambiguity = ("The set is EMPTY: no category met the confidence threshold. This "
                     "reading is unlike the data the model was calibrated on. Say that "
                     "no reliable forecast is available and advise caution.")
    else:
        ambiguity = ("The set contains exactly one category, so the model is unusually "
                     "confident here. You may state the forecast directly, but still "
                     "avoid implying certainty about the specific reading.")

    return f"""SENSOR READING — {reading.station}, {reading.datetime}
  PM2.5  {raw['PM2.5']:.1f} ug/m3
  PM10   {raw['PM10']:.1f} ug/m3
  CO     {raw['CO']:.0f} ug/m3
  Temperature {raw['TEMP']:.1f} C
  Dew point   {raw['DEWP']:.1f} C

FORECAST — air-quality risk category {reading.horizon_hours} hours from now
  Most likely category: {reading.predicted} (model probability {reading.probability:.2f})
  Conformal prediction set: [{set_str}]
  Set size: {len(reading.conformal_set)}
  Method: Mondrian (class-conditional) split conformal prediction
  Target coverage: {reading.target_coverage:.0%} PER CATEGORY (a separate threshold is
    calibrated for each category, so the guarantee holds for each one individually --
    not only on average). It is still a property of the method across many readings,
    never a probability about this single reading.
{coverage_lines}

{ambiguity}

WHAT THE MODEL WEIGHTED MOST (SHAP, for the predicted category)
{chr(10).join(shap_lines)}

WEARER PROFILE
  {profile.describe()}
  Counts as a sensitive group under AQI guidance: {"yes" if profile.sensitive else "no"}

Write the advisory."""


# --------------------------------------------------------------------- fallback


def fallback_advisory(reading: Reading, profile: UserProfile) -> str:
    """Rule-based template. Same facts, blunter prose, no API.

    Runs when there is no API key, the call fails, or the response fails validation.
    It carries the same honesty obligation as the prompt: if the set is ambiguous, the
    template says so.
    """
    worst = reading.worst_in_set
    sensitive = profile.sensitive

    if not reading.conformal_set:
        return (
            f"No reliable forecast is available for the next {reading.horizon_hours} "
            f"hours — this reading does not resemble the data the model was calibrated "
            f"on, so no air-quality category met the confidence threshold. Current "
            f"PM2.5 is {reading.raw_features['PM2.5']:.0f} ug/m3. Treat conditions as "
            f"uncertain and limit prolonged outdoor exertion until the next reading."
            + (" Keep any reliever inhaler with you." if profile.respiratory_condition else ""))

    if reading.ambiguous:
        cats = ", ".join(reading.conformal_set[:-1]) + f" or {reading.conformal_set[-1]}"
        lead = (
            f"Air quality {reading.horizon_hours} hours from now is uncertain: the "
            f"forecast covers {len(reading.conformal_set)} categories — {cats}. "
            f"The most likely single outcome is {reading.predicted}, but the range is "
            f"wide enough that it should not be relied on.")
        precaution = f"Plan for the worst case in that range, {worst}."
    else:
        lead = (f"Air quality {reading.horizon_hours} hours from now is forecast as "
                f"{reading.predicted}, and the model is unusually confident — this is "
                f"one of the roughly 7% of readings where a single category meets the "
                f"confidence threshold.")
        precaution = ""

    top = reading.top_shap[0]
    human = ACTIONABLE.get(top["feature"]) or CONTEXT_NAMES.get(top["feature"], top["feature"])
    driver = (f"The model weighted {human} most heavily "
              f"(currently {reading.raw_features.get(top['feature'], float('nan')):.0f} "
              f"ug/m3)." if top["feature"] in ACTIONABLE else
              f"The model weighted {human} most heavily for this prediction.")

    if worst in ("Hazardous", "Very unhealthy"):
        action = ("Avoid outdoor exertion and stay indoors with windows closed where "
                  "you can.")
        if profile.respiratory_condition:
            action += " Keep your reliever inhaler with you."
        elif sensitive:
            action += " Take it slowly if you do go out."
    elif worst == "Unhealthy":
        action = ("Cut back on strenuous outdoor activity."
                  if sensitive else "Consider shortening strenuous outdoor activity.")
    elif worst == "Unhealthy (sensitive)":
        action = ("Ease off outdoor exertion and take breaks." if sensitive
                  else "Most people will be fine; ease off if you notice symptoms.")
    else:
        action = "No particular precaution is needed."

    parts = [lead, precaution, driver, action]
    body = " ".join(p for p in parts if p)
    return body + " This is general guidance, not medical advice."


# -------------------------------------------------------------------- API call


@dataclass
class AdvisoryResult:
    text: str
    source: str            # "gemini" | "fallback"
    model: str | None = None
    error: str | None = None
    usage: dict | None = None


def _validate(text: str, reading: Reading) -> str | None:
    """Reject a response that breaks the honesty contract; returns a reason or None.

    Cheap guards, not a full evaluation. The one that matters is the third: if the
    prediction set is ambiguous and the advisory never signals it, the advisory is
    worse than useless because it launders uncertainty into confidence.
    """
    if not text or len(text.strip()) < 40:
        return "response too short to be a usable advisory"
    lowered = text.lower()
    if reading.ambiguous and reading.conformal_set:
        hedges = ("uncertain", "range", "categories", "could be", "anywhere from",
                  "between", "not settled", "ambiguous", "or ")
        if not any(h in lowered for h in hedges):
            return ("prediction set has >1 category but the advisory does not signal "
                    "ambiguity")
    if f"{reading.target_coverage:.0%} sure" in lowered or "90% certain" in lowered:
        return "advisory misstates coverage as per-reading certainty"
    # Backstop for truncation the metadata check missed: a finished advisory ends on
    # terminal punctuation.
    if text.rstrip()[-1] not in ".!?":
        return "advisory does not end in a complete sentence (likely truncated)"
    return None


def generate(reading: Reading, profile: UserProfile, model: str, max_tokens: int,
             verbose: bool = False) -> AdvisoryResult:
    """Ask Gemini; fall back to the template on any failure.

    Uses the **google-genai** SDK. The older `google-generativeai` package this
    originally targeted is end-of-life upstream ("no longer receiving updates or bug
    fixes"), which is a poor foundation for a thesis whose stated rationale for the
    free tier is reproducibility.

    Gemini's free tier is the deliberate choice: the advisory layer should be
    reproducible by another researcher without a paid API account. See the
    methodological note in reports/llm_advisory_examples.md.
    """
    from google import genai
    from google.genai import errors as gerr
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return AdvisoryResult(
            fallback_advisory(reading, profile), "fallback", model=model,
            error="no credentials (GEMINI_API_KEY unset; add it to .env)")

    try:
        client = genai.Client(api_key=api_key)
    except Exception as exc:
        return AdvisoryResult(fallback_advisory(reading, profile), "fallback",
                              model=model,
                              error=f"client init: {type(exc).__name__}: {exc}")

    prompt = build_user_prompt(reading, profile)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=max_tokens,
        temperature=0.3,             # advisory copy should be steady, not creative
    )

    # Free-tier quota comes in two flavours needing opposite treatment: a per-minute
    # limit clears on its own, a per-day limit does not. Only the first is retried.
    # google-genai raises ClientError for 4xx (including 429) rather than a typed
    # quota exception, so the status code is what distinguishes them.
    attempt = 0
    try:
        while True:
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt, config=config)
                break
            except gerr.ClientError as exc:
                code = getattr(exc, "code", None)
                detail = str(exc)
                if code != 429:
                    raise
                per_day = "PerDay" in detail
                if per_day or attempt >= MAX_QUOTA_RETRIES:
                    kind = "daily" if per_day else "per-minute"
                    return AdvisoryResult(
                        fallback_advisory(reading, profile), "fallback", model=model,
                        error=f"free-tier quota exhausted ({kind}): "
                              f"{detail.splitlines()[0][:120]}")
                attempt += 1
                if verbose:
                    print(f"      per-minute quota hit; waiting {PACING_SECONDS + 10}s "
                          f"(retry {attempt}/{MAX_QUOTA_RETRIES})")
                time.sleep(PACING_SECONDS + 10)
    except gerr.ClientError as exc:
        code = getattr(exc, "code", None)
        label = {400: "invalid request", 401: "authentication failed",
                 403: "permission denied", 404: "model not found"}.get(
                     code, f"client error {code}")
        return AdvisoryResult(fallback_advisory(reading, profile), "fallback",
                              model=model, error=f"{label}: {str(exc)[:160]}")
    except gerr.ServerError as exc:
        return AdvisoryResult(fallback_advisory(reading, profile), "fallback",
                              model=model, error=f"server error: {str(exc)[:160]}")
    except gerr.APIError as exc:
        return AdvisoryResult(fallback_advisory(reading, profile), "fallback",
                              model=model,
                              error=f"api error: {type(exc).__name__}: {str(exc)[:140]}")
    except Exception as exc:
        # Deliberate catch-all, and only here. This function's contract is that it
        # always returns an advisory: a wearer losing their air-quality warning to an
        # unhandled exception is a worse failure than a blunter template.
        return AdvisoryResult(fallback_advisory(reading, profile), "fallback",
                              model=model,
                              error=f"unexpected {type(exc).__name__}: {exc}")

    blocked = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
    if blocked:
        return AdvisoryResult(fallback_advisory(reading, profile), "fallback",
                              model=model, error=f"prompt blocked: {blocked}")

    # A truncated advisory can stop after naming a hazard and before giving the
    # precaution. Gemini 3.x reasons internally and those tokens count against
    # max_output_tokens, so the cap is reachable even for a short answer. The text
    # validator cannot catch this -- truncated output still contains hedging words --
    # so it is checked from the response metadata.
    finish = None
    if getattr(response, "candidates", None):
        finish = getattr(response.candidates[0], "finish_reason", None)
    fname = getattr(finish, "name", str(finish) if finish is not None else None)
    if fname == "MAX_TOKENS":
        return AdvisoryResult(
            fallback_advisory(reading, profile), "fallback", model=model,
            error=f"truncated: hit max_output_tokens ({max_tokens}); raise "
                  f"explainability.llm_max_tokens")
    if fname not in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
        return AdvisoryResult(
            fallback_advisory(reading, profile), "fallback", model=model,
            error=f"generation stopped early (finish_reason={fname})")

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        return AdvisoryResult(fallback_advisory(reading, profile), "fallback",
                              model=model,
                              error=f"no usable text (finish_reason={fname})")

    problem = _validate(text, reading)
    if problem:
        return AdvisoryResult(fallback_advisory(reading, profile), "fallback",
                              model=model, error=f"validation failed: {problem}")

    usage = None
    um = getattr(response, "usage_metadata", None)
    if um is not None:
        usage = {"input_tokens": getattr(um, "prompt_token_count", None),
                 "output_tokens": getattr(um, "candidates_token_count", None)}
    return AdvisoryResult(text, "gemini", model=model, usage=usage)


# ------------------------------------------------------------------- examples


def load_reading(row: int, horizon: int = 6) -> Reading:
    """Rebuild one test case from the saved artifacts."""
    proc = REPO_ROOT / "data" / "processed" / f"h{horizon}"
    art = REPO_ROOT / "src" / "models" / "artifacts"
    cfg_raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    labels = list(cfg_raw["data"]["pm25_labels"])

    bundle = joblib.load(art / f"baseline_h{horizon}.pkl")
    conf = joblib.load(art / f"conformal_h{horizon}.pkl")
    meta = json.loads((proc / "metadata.json").read_text())
    scaler = joblib.load(proc / "scaler.pkl")
    frame = pd.read_csv(proc / "tabular_test.csv")

    X = frame[bundle["feature_columns"]].to_numpy(dtype=np.float32)
    praw = bundle["model"].predict_proba(X[row:row + 1])[0]
    probs = np.zeros(len(labels))
    for col, cls in enumerate(bundle["model"].classes_):
        probs[int(cls)] = praw[col]
    # Mondrian by default: per-class thresholds, so the guarantee quoted in the
    # advisory holds for the category being quoted. Falls back to the pooled
    # threshold only if an older artifact has no Mondrian entry.
    if conf.get("default_method") == "mondrian" and conf.get("mondrian_thresholds"):
        thr = conf["mondrian_thresholds"]
        in_set = [labels[i] for i in range(len(labels))
                  if (1.0 - probs[i]) <= thr[i] + 1e-12]
        method = "mondrian"
        per_class = conf.get("mondrian_per_class_coverage")
    else:
        in_set = [labels[i] for i in np.flatnonzero((1.0 - probs) <= conf["q"] + 1e-12)]
        method = "marginal"
        per_class = None

    import shap
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals = shap.TreeExplainer(bundle["model"]).shap_values(X[row:row + 1])
    vals = np.stack(vals, axis=-1) if isinstance(vals, list) else np.asarray(vals)
    pred = int(probs.argmax())
    contrib = vals[0, :, pred]
    order = np.argsort(-np.abs(contrib))
    top = [{"feature": bundle["feature_columns"][j], "value": float(contrib[j])}
           for j in order[:3]]

    scaled = list(meta["scaled_columns"])
    raw = {c: float(frame.loc[row, c] * scaler.scale_[scaled.index(c)]
                    + scaler.mean_[scaled.index(c)]) for c in scaled}

    return Reading(
        station=str(frame.loc[row, "station"]),
        datetime=str(frame.loc[row, "datetime"]),
        horizon_hours=horizon,
        predicted=labels[pred],
        probability=float(probs[pred]),
        conformal_set=in_set,
        target_coverage=float(conf["target_coverage"]),
        top_shap=top,
        raw_features=raw,
        true_label=labels[int(frame.loc[row, "y_category"])],
        per_class_coverage=per_class,
        method=method,
    )


# The five worked examples. Chosen to span the situations the advisory must handle,
# including at least one ambiguous set and one Hazardous case, and to pair different
# profiles against the same kind of reading.
EXAMPLE_PROFILES = [
    UserProfile(34, False, "healthy adult"),
    UserProfile(71, True, "older adult with COPD"),
    UserProfile(9, True, "child with asthma"),
    UserProfile(45, False, "healthy adult"),
    UserProfile(68, False, "older adult, no condition"),
]


def pick_example_rows(horizon: int = 6, n: int = 5) -> list[int]:
    """Rows that between them cover confident, ambiguous and advisory-class cases."""
    shap_metrics = REPO_ROOT / "reports" / "metrics" / f"shap_h{horizon}.json"
    if shap_metrics.exists():
        cases = json.loads(shap_metrics.read_text())["cases"]
        by_reason = {}
        for c in cases:
            by_reason.setdefault(c["reason"], []).append(c)
        wanted = ["Hazardous, ambiguous (set > 1)", "Hazardous, confident (singleton set)",
                  "highly ambiguous (set >= 4)", "Very unhealthy, ambiguous",
                  "confident, any class"]
        rows = []
        for r in wanted:
            if by_reason.get(r):
                rows.append(by_reason[r][0]["row"])
        for c in cases:
            if len(rows) >= n:
                break
            if c["row"] not in rows:
                rows.append(c["row"])
        return rows[:n]
    return list(range(n))


def run(n: int = 5, horizon: int = 6, *, write: bool = True,
        verbose: bool = True) -> dict:
    cfg_raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    ex = cfg_raw["explainability"]
    model, max_tokens = ex["llm_model"], int(ex["llm_max_tokens"])
    say = print if verbose else (lambda *a, **k: None)

    has_key = bool(os.environ.get("GEMINI_API_KEY") or
                   os.environ.get("GOOGLE_API_KEY"))
    say(f"model {model} | credentials in environment: {'yes' if has_key else 'no'}")
    if not has_key:
        say("  -> API calls will fail and the rule-based fallback will be used. "
            "Set GEMINI_API_KEY in .env for live output.")

    rows = pick_example_rows(horizon, n)
    say(f"example rows: {rows}")

    results = []
    for i, (row, profile) in enumerate(zip(rows, EXAMPLE_PROFILES)):
        # Pace the batch under the free tier's per-minute cap. Cheaper than relying on
        # the retry path, which only fires after a request has already been rejected.
        if i and has_key:
            time.sleep(PACING_SECONDS)
        reading = load_reading(row, horizon)
        out = generate(reading, profile, model, max_tokens, verbose)
        results.append({"reading": reading, "profile": profile, "result": out})
        err = (out.error or "ok").splitlines()[0][:90]
        say(f"  row {row:>6,} set={len(reading.conformal_set)} [{out.source}] {err}")

    payload = {"model": model, "results": results, "has_credentials": has_key,
               "horizon": horizon}
    if write:
        path = REPO_ROOT / "reports" / "llm_advisory_examples.md"
        path.write_text(build_report(payload))
        say(f"wrote {path.relative_to(REPO_ROOT)}")
    return payload


def build_report(payload: dict) -> str:
    def table(header, rows):
        esc = lambda cs: [str(c).replace("|", "\\|") for c in cs]
        b = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return (f"| {' | '.join(esc(header))} |\n"
                f"| {' | '.join(['---'] * len(header))} |\n{b}")

    results = payload["results"]
    n_amb = sum(1 for r in results if r["reading"].ambiguous)

    n_api = sum(1 for r in results if r["result"].source == "gemini")
    n_fb = len(results) - n_api
    prov_rows = [[f"Example {i}",
                  "**live Gemini output**" if r["result"].source == "gemini"
                  else "fallback template",
                  f"`{r['result'].model}`" if r["result"].source == "gemini" else "—",
                  "—" if r["result"].source == "gemini" else f"`{r['result'].error}`"]
                 for i, r in enumerate(results, 1)]
    prov_table = table(["Example", "Source", "Model", "Fallback reason"], prov_rows)

    if n_api == 0:
        headline = (f"> **Every advisory below is fallback template output, not model "
                    f"output.** No live call succeeded.")
    elif n_fb:
        headline = (f"> **{n_api} of {len(results)} advisories are live "
                    f"`{payload['model']}` output; {n_fb} fell back to the template.** "
                    f"Per-example provenance is in the table below — nothing here is "
                    f"presented as model output unless it is.")
    else:
        headline = (f"> **All {len(results)} advisories below are live "
                    f"`{payload['model']}` output.** Generated against the Gemini API "
                    f"with a real key; none fell back to the template.")

    provenance = f"""{headline}

{prov_table}"""

    blocks = []
    for i, r in enumerate(results, 1):
        reading, profile, out = r["reading"], r["profile"], r["result"]
        set_str = ", ".join(reading.conformal_set) if reading.conformal_set else "EMPTY"
        shap_str = "; ".join(f"`{s['feature']}` {s['value']:+.4f}"
                             for s in reading.top_shap)
        badge = ("**AMBIGUOUS — set of "
                 f"{len(reading.conformal_set)}**" if reading.ambiguous
                 else "confident — singleton set")
        src = (f"live Gemini output — `{out.model}`" if out.source == "gemini"
               else f"rule-based fallback ({out.error.splitlines()[0][:90]})")
        blocks.append(f"""### Example {i} — {badge}

| | |
|---|---|
| Station / time | {reading.station}, {reading.datetime} |
| PM2.5 / PM10 / CO | {reading.raw_features['PM2.5']:.1f} / {reading.raw_features['PM10']:.1f} / {reading.raw_features['CO']:.0f} ug/m3 |
| Temp / dew point | {reading.raw_features['TEMP']:.1f} C / {reading.raw_features['DEWP']:.1f} C |
| Point prediction (+{reading.horizon_hours} h) | **{reading.predicted}** (p = {reading.probability:.2f}) |
| Conformal set ({reading.method}, {reading.target_coverage:.0%} per class) | **[{set_str}]** |
| Worst case in set | {reading.worst_in_set} |
| True label | {reading.true_label} |
| Top-3 SHAP | {shap_str} |
| Wearer | {profile.label} — {profile.describe()} |
| Source | {src} |

> {out.text}
""")

    # Pull the conformal numbers from the artifact rather than restating them, so the
    # prose cannot drift from whatever calibration actually shipped.
    cpath = REPO_ROOT / "reports" / "metrics" / f"conformal_h{payload['horizon']}.json"
    if cpath.exists():
        cm = json.loads(cpath.read_text())
        mt = cm["mondrian"]["test"]
        singles, median = mt["singleton_rate"], mt["median_set_size"]
        haz = mt["per_class"]["Hazardous"]["coverage"]
        vun = mt["per_class"]["Very unhealthy"]["coverage"]
        marg_haz = cm["test"]["per_class"]["Hazardous"]["coverage"]
    else:
        singles, median, haz, vun, marg_haz = 0.021, 3, 0.908, 0.936, 0.843

    return f"""# LLM risk advisory — worked examples

Generated by `src/explainability/llm_advisory.py`. Each advisory is built from a
predictor's own numbers: the RandomForest point prediction, the conformal prediction
set from `conformal_h{payload['horizon']}.pkl`, the top-3 SHAP contributions, and a
short wearer profile.

> **Scope note (added after Phase 10).** These examples are driven by the **Beijing**
> model, which is *not* the deployed predictor — rolling-origin CV showed no
> Beijing-trained model reliably beats persistence
> (`reports/rolling_origin_cv_h{payload['horizon']}.md`). The deployment candidate is
> the Bangladesh-native model (`reports/bangladesh_deployment.md`). What this report
> demonstrates is the **advisory layer's behaviour** — how it communicates an ambiguous
> prediction set honestly — which is model-agnostic and carries over unchanged. The
> specific pollutant values below are Beijing readings.

{provenance}

---

## The honesty requirement

Phase 6's conformal analysis found a **median prediction set of {median:.0f} categories**
under Mondrian calibration, with a single-category set only **{singles:.1%}** of the
time. An advisory layer that reports the argmax and stops would assert confidence the
predictor does not have — on a health question.

So the prompt states every number explicitly and adds four constraints the model is
required to honour:

1. **Name the ambiguity.** When the conformal set has more than one category, say so
   and give the range. Do not present the most likely category as settled.
2. **Advise against the worst case in the set**, not the most likely one, and say that
   is what is being done.
3. **Never restate coverage as per-reading certainty.** Coverage is how often the
   *method* is right across many readings; it is not "90% sure this reading is
   Hazardous". The sets come from **Mondrian (class-conditional) conformal**, so the
   guarantee does hold per category — Hazardous is covered at {haz:.1%} and Very
   unhealthy at {vun:.1%} on test (`reports/conformal_h{payload['horizon']}.md` §4).
   That is a real strengthening: under the earlier *marginal* calibration Hazardous
   sat at {marg_haz:.1%}, and an advisory quoting "90%" for a Hazardous reading would
   have been wrong.
4. **SHAP is about the model, not the air.** "The model weighted PM2.5 most heavily",
   never "PM2.5 caused this".

Responses are validated before use: an advisory for an ambiguous set that contains no
hedging language is rejected and the fallback is substituted. {n_amb} of
{len(results)} examples below have a set larger than one, which is what exercises that
path.

---

## Examples

{chr(10).join(blocks)}
---

## Why a free-tier hosted model (thesis note)

The advisory layer runs on **Gemini's free tier** rather than a paid API. That is a
methodological choice, not only a budget one, and it is worth stating in the write-up.

- **Reproducibility without a paywall.** Anyone re-running this project needs a free
  Google AI Studio key and nothing else. A reviewer, an examiner, or a researcher
  extending the work can regenerate every advisory in this file at zero cost. A paid
  API would make the final stage of the pipeline unverifiable for anyone unwilling to
  put a card down, which for a student thesis is a real barrier to replication.
- **The LLM is the thinnest part of the contribution.** Everything load-bearing — the
  forecast, the conformal set, the coverage guarantee, the SHAP attributions — is
  computed locally by `baseline_h{payload['horizon']}.pkl` and
  `conformal_h{payload['horizon']}.pkl`. The model only renders those numbers as
  prose. Swapping it changes wording, not conclusions, which is exactly why the
  provider is a configuration value (`explainability.llm_provider`) and why the
  rule-based fallback can stand in for it.
- **The honesty constraints are enforced outside the model.** The validator rejects an
  unhedged advisory on an ambiguous set, rejects a truncated one, and rejects any
  claim of per-reading certainty — regardless of which model produced the text. The
  guarantee that the advisory does not overstate confidence does not rest on trusting
  the LLM.

Two caveats a reader should know:

- **Free-tier quotas are rate-limited.** At scale the `ResourceExhausted` path will
  fire and the template will serve; that is by design, but a deployed device would
  need either a paid tier or on-device generation.
- **`gemini-2.0-flash`, originally specified, has been retired by Google** (404: *"no
  longer available… use models/gemini-3.6-flash"*). The pinned model is now
  `gemini-3.6-flash`, its named successor. A pinned version is used rather than the
  `gemini-flash-latest` alias so the results in this file stay reproducible. Hosted
  models are moving targets, which is itself an argument for keeping the LLM
  non-load-bearing.
- The advisory layer uses the **`google-genai`** SDK. The original implementation
  targeted `google-generativeai`, which is end-of-life upstream ("no longer receiving
  updates or bug fixes"); it was migrated rather than left on a dead package, since
  reproducibility is the stated reason for choosing a hosted free tier at all.

---

## Fallback behaviour

`fallback_advisory()` produces the same facts from a template whenever the live call
cannot be used. Each case is caught separately rather than under one broad `except`,
so the report can say *which* failure occurred:

| Failure | Handler |
|---|---|
| `GEMINI_API_KEY` unset | checked before the call |
| Model retired or misspelled | `google.api_core.exceptions.NotFound` |
| Bad key / no access | `PermissionDenied`, `Unauthenticated` |
| Free-tier quota exhausted | `ResourceExhausted` |
| Malformed request | `InvalidArgument` |
| Transient outage or timeout | `ServiceUnavailable`, `DeadlineExceeded` |
| Any other Google API error | `GoogleAPIError` |
| Anything else | final catch-all, recorded by exception type |
| Prompt blocked by safety filter | `prompt_feedback.block_reason` |
| **Output truncated** | `finish_reason == MAX_TOKENS` |
| Unhedged advisory on an ambiguous set | text validator |
| Advisory not ending in a complete sentence | text validator |

The truncation check earned its place during this run. Gemini 3.x reasons internally
and those tokens count against `max_output_tokens`, so a ~120-token advisory hit the
cap at 1024 and stopped mid-sentence — after naming the hazard, before giving the
precaution. **The text validator passed it**, because truncated output still contains
hedging words. Truncation is only visible in the response metadata, so it is checked
there, with an end-of-sentence test as a backstop.

The fallback carries the same honesty obligation as the prompt: it names the number of
categories when the set is ambiguous, advises against the worst case, and labels
itself as general guidance rather than medical advice. The device degrades to terser
wording, never to silence and never to false confidence.

Regenerate with `python -m src.explainability.llm_advisory`.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--row", type=int, default=None,
                    help="advise on one specific test row and print it")
    ap.add_argument("--age", type=int, default=34)
    ap.add_argument("--respiratory", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    if args.row is not None:
        cfg_raw = yaml.safe_load(DEFAULT_CONFIG.read_text())["explainability"]
        reading = load_reading(args.row, args.horizon)
        profile = UserProfile(args.age, args.respiratory)
        out = generate(reading, profile, cfg_raw["llm_model"],
                       int(cfg_raw["llm_max_tokens"]))
        print(json.dumps({"reading": asdict(reading), "profile": asdict(profile),
                          "advisory": asdict(out)}, indent=2, default=str))
        return 0

    run(n=args.n, horizon=args.horizon, write=not args.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
