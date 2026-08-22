"""What the benchmark measured, read from the committed data where possible.

The serving layer makes claims - "cascade is the right policy on this ladder",
"escalating buys about 15 points here". Those claims come from the experiment
next door, and this module is the seam between them.

Two rules govern everything below.

**Real numbers are computed, not typed.** The probe cross-tab is recomputed from
`runs/results.probe.jsonl` on demand rather than transcribed. A hard-coded figure
rots the first time the task set is rebuilt or a grader is fixed, and both have
moved this exact number.

**A number with no run behind it is not served.** Every ladder's economics come
from its own committed frontier run. There is no table of fallback constants -
there was, and it had two of three verdicts backwards. A ladder with no run
reports `verdict: None` and names the command that would produce one.

**Simulated numbers are labelled at the point of use.** Anything derived from
mock output carries `simulated: True` in its payload, so a consumer that
surfaces it has to walk past the label to do so.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from llm_routing import paths
from llm_routing import response_cache

PROBE_PATH = paths.RUNS / "results.probe.jsonl"


def redraw_path(ladder: str) -> Path:
    """Where this ladder's decisive-cell redraw lives, if it has one.

    Only `wide` has been redrawn - it is a paid run and the noise correction it
    measures is a property of the task set more than of the ladder. Consumers
    must handle absence.
    """
    return paths.RUNS / f"redraw.{ladder}.json"


def frontier_path(ladder: str) -> Path:
    """Where this ladder's frontier run lives.

    Per-ladder rather than one shared `frontier.jsonl`. The shared name was a
    copy of whichever ladder ran last, so a fresh clone either found the wrong
    ladder's economics or - because the copy was gitignored - found nothing and
    silently fell back to a table of hard-coded constants. The per-ladder files are
    committed, so a clone now reports measured economics for every ladder that
    has one.
    """
    return paths.RUNS / f"frontier.{ladder}.jsonl"


# ---------------------------------------------------------------------------
# Price ratios - exact, and the only part of the finding that needs no run
# ---------------------------------------------------------------------------

def price_ratios(ladder: str) -> dict | None:
    """The top rung's price over the bottom rung's, from the price table.

    Computed rather than recorded, because it is pure arithmetic over
    `models.MODEL_SPECS` and therefore cannot drift from the ladder it
    describes. Nothing here depends on a run, a mode, or a task set.

    `effective` folds in the tokenizer asymmetry the benchmark models: Claude
    4.7 and later emit roughly 30% more tokens for identical text, so on the
    `claude` and `wide` ladders the rungs disagree about how long the same
    prompt is. That works against escalation, so ignoring it would under-price
    exactly the rung a cascade escalates to.
    """
    from llm_routing import models

    ids = models.LADDERS.get(ladder)
    if not ids:
        return None
    bottom, top = models.MODEL_SPECS[ids[0]], models.MODEL_SPECS[ids[-1]]
    list_ratio = top["price_in"] / bottom["price_in"]
    effective = (
        (top["price_in"] * top["tokenizer_factor"])
        / (bottom["price_in"] * bottom["tokenizer_factor"])
    )
    out = {
        "rungs": " -> ".join(ids),
        "list_ratio": round(list_ratio, 2),
        "effective_ratio": round(effective, 2),
    }
    measured = realized_ratio(ladder)
    if measured:
        out["realized_ratio"] = measured["ratio"]
        out["realized_n_tasks"] = measured["n_tasks"]
    return out


def realized_ratio(ladder: str, path: Path | None = None) -> dict | None:
    """The ratio the provider actually billed, from cached greedy answers.

    `price_ratios` is arithmetic over the price table and cannot be wrong about
    the table - but it prices a call using INPUT rates, and on a reasoning
    workload roughly 93% of the bill is OUTPUT tokens, where these rungs are
    much further apart. On `wide` the input rates differ by 36x and the output
    rates by 89x, so which one is used is not a detail.

    On `wide` the realized figure is **117x against a quoted 46x**, over 427
    benchmark tasks with both rungs cached. The direction matters: understating the price
    gap understates the case for cascading on exactly the hard tasks where
    routing is worth doing.

    Restricted to tasks where BOTH rungs answered, so the two means describe the
    same population rather than two different task mixes. Greedy answers only -
    self-consistency samples are a different, shorter action and would drag the
    cheap rung's mean toward it.

    Returns None when no ladder cache exists, which is the normal state for a
    ladder that has never been run for real.
    """
    path = path or paths.CACHE / f"raw_calls.{ladder}.jsonl"
    if not path.exists():
        return None

    per_task: dict[str, dict[str, list[float]]] = {}
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("kind") != "answer" or d.get("mode") != "real":
            continue
        if d.get("temperature") not in (0, 0.0):
            continue
        if str(d.get("task_id") or "").startswith(response_cache.LIVE_ID_PREFIX):
            # A response bought by SERVING a live query, not by the benchmark.
            # `response_cache` writes those to serving.<ladder>.jsonl now, so
            # this only fires on a cache written before that split - but it has
            # to fire, because one served query is enough to make this function
            # report a price ratio over n=1.
            continue
        per_task.setdefault(d["task_id"], {}).setdefault(d["tier"], []).append(
            d["cost_usd"])

    both = [v for v in per_task.values() if v.get("cheap") and v.get("expensive")]
    if not both:
        return None

    def mean_of(tier):
        calls = [c for v in both for c in v[tier]]
        return sum(calls) / len(calls)

    lo, hi = mean_of("cheap"), mean_of("expensive")
    if lo <= 0:
        return None
    return {
        "ratio": round(hi / lo, 1),
        "n_tasks": len(both),
        "mean_cheap_usd": lo,
        "mean_expensive_usd": hi,
    }


# Below roughly this effective price ratio, the cascade's fixed costs - the
# wasted cheap call, plus verification - exceed what skipping an expensive call
# saves.
#
# Deliberately NOT used to compute a verdict: on real models the cascade is
# CHEAPER at deepseek's 3.11x and DEARER at claude's 6.5x, so the outcome is not
# monotonic in the price ratio and no threshold can express it. What decides it
# is verification cost on that ladder and how much better the top rung is. Kept
# as the literature's rule of thumb, reported alongside the measured verdict.
CROSSOVER_RATIO = 3.0


# ---------------------------------------------------------------------------
# The economics - derived from this ladder's own frontier run, or not at all
# ---------------------------------------------------------------------------

# Every ladder in `models.LADDERS` has a frontier run committed under runs/, so
# the economics below are derived and never quoted.
#
# There is deliberately NO fallback table of constants. One existed, from early
# mock runs, and it had TWO OF THREE VERDICTS BACKWARDS once the ladders were
# measured. A fallback that fires silently and is wrong is worse than none: a
# ladder with no frontier run says so, and names the command that produces one.

@lru_cache(maxsize=4)
def frontier_economics(path: str | None = None, ladder: str | None = None) -> dict | None:
    """Derive the cascade's economics from a frontier run, if one is on disk.

    Returns None when this ladder has no frontier run committed, which is the
    honest answer for a ladder nobody has swept.

    Two quantities, both computed by `frontier.economics` rather than
    reimplemented here, so the router, the report and the figures can never
    disagree about what they mean:

      cascade_vs_always_best_pct
          The cheapest cascade setting that is at least as accurate as always
          paying for the top rung, against that rung's cost. Negative means the
          cascade is cheaper at matched accuracy. This is the headline.
      auc_gain_over_random_pct
          Mean accuracy across the whole shared budget range, minus the same
          for a cost-matched coin flip. Answers "how good is this at every
          budget" rather than "at the budget somebody tuned it to".

    The ladder each file was generated for is returned alongside, and the
    caller still checks it matches - the rows carry the ladder, so a mislabelled
    or hand-copied file cannot pass itself off as another ladder's.
    """
    if path:
        p = Path(path)
    else:
        from llm_routing import models
        p = frontier_path(ladder or models.LADDER)
    if not p.exists():
        return None

    try:
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    if not rows:
        return None

    # Imported here rather than at module scope: `frontier` pulls in policies,
    # run_eval and splits, and `findings` is imported by the MCP server on
    # every start. The probe path and the price ratios must stay cheap.
    from llm_routing import frontier as frontier_mod

    econ = frontier_mod.economics(rows)
    if econ is None:
        return None

    return {
        "ladder": rows[0].get("ladder"),
        "cascade_vs_always_best_pct": econ["cascade_vs_always_best_pct"],
        "auc_gain_over_random_pct": econ["auc_gain_over_random_pct"],
        "verdict": econ["verdict"],
        # Every row of a mock frontier carries simulated: true. If ANY row is
        # simulated the derived figure is, so this is `any`, not `all`.
        "simulated": any(r.get("simulated", True) for r in rows),
        "n": rows[0].get("n"),
        "split": rows[0].get("split"),
        "source": "frontier.jsonl",
    }


def ratio_verdict(ladder: str) -> dict:
    """Should this ladder cascade or route? The product's core decision.

    Two sources, both labelled in the payload:

      price ratios   exact arithmetic over the price table. Always present, and
                     reported for context - it does NOT decide the verdict,
                     because the measurement is not monotonic in it.
      frontier       derived from `runs/frontier.<ladder>.jsonl`, which is
                     committed for every ladder in `models.LADDERS`.

    A ladder with no frontier run gets `verdict: None` and a note naming the
    command that would produce one. It does not get a guess.
    """
    ratios = price_ratios(ladder)
    if ratios is None:
        return {
            "ladder": ladder,
            "known": False,
            "note": (
                f"unknown ladder {ladder!r} - not in models.LADDERS, so not "
                f"even its price ratio can be computed"
            ),
        }

    out: dict = {"ladder": ladder, "known": True, **ratios}

    live = frontier_economics(ladder=ladder)
    if live and live.get("ladder") == ladder and live.get("verdict"):
        out.update({
            "cascade_vs_always_best_pct": live["cascade_vs_always_best_pct"],
            "auc_gain_over_random_pct": live["auc_gain_over_random_pct"],
            "verdict": live["verdict"],
            "economics_source": f"frontier.{ladder}.jsonl",
            "economics_simulated": live["simulated"],
            "economics_n": live.get("n"),
        })
    else:
        out.update({
            "verdict": None,
            "economics_source": None,
            "note": (
                f"no frontier run on disk for {ladder!r}. Derive one with "
                f"`ROUTER_LADDER={ladder} ROUTER_MODE=replay "
                f"python -m llm_routing.frontier`. This router will not guess "
                f"a verdict from a price ratio - the measurement is not "
                f"monotonic in it."
            ),
        })

    # Every committed ladder is measured on real responses, so this is now a
    # property of the frontier file rather than a per-ladder exception.
    out["accuracy_data_is_real"] = bool(
        live and live.get("ladder") == ladder and not live.get("simulated"))
    return out


# ---------------------------------------------------------------------------
# The two-arm probe - recomputed from the real data
# ---------------------------------------------------------------------------

@dataclass
class Probe:
    """The cheap-vs-expensive cross-tab, over tasks where both arms ran."""

    n: int
    both_ok: int
    routable: int
    both_fail: int
    inverted: int
    cheap_acc: float
    expensive_acc: float
    by_domain: dict

    @property
    def routable_pct(self) -> float:
        return 100.0 * self.routable / self.n if self.n else 0.0

    @property
    def ceiling_pct(self) -> float:
        """The most any router can add over always_cheap: routable + inverted."""
        return 100.0 * (self.routable + self.inverted) / self.n if self.n else 0.0

    @property
    def rescue_rate(self) -> float:
        """Of the cheap rung's failures, the fraction escalating actually fixes.

        The number a cascade lives on. Escalating on a `both_fail` task pays
        the expensive rung and still returns a wrong answer.
        """
        failures = self.routable + self.both_fail
        return self.routable / failures if failures else 0.0

    def ci95(self) -> tuple[float, float]:
        """Wilson interval on the routable fraction.

        Wilson rather than normal-approximation: at n=100 and p near 0.15 the
        normal interval is visibly wrong at the tails, and this estimate sits
        on the boundary of the band it is being compared against, so the tail
        behaviour is exactly what matters.
        """
        if not self.n:
            return (0.0, 0.0)
        z, n, p = 1.96, self.n, self.routable / self.n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))

    def to_dict(self) -> dict:
        lo, hi = self.ci95()
        return {
            "n": self.n,
            "cells": {
                "both_ok": self.both_ok, "routable": self.routable,
                "both_fail": self.both_fail, "inverted": self.inverted,
            },
            "always_cheap_pct": round(100 * self.cheap_acc, 1),
            "always_expensive_pct": round(100 * self.expensive_acc, 1),
            "routable_pct": round(self.routable_pct, 1),
            "routable_ci95_pct": [round(lo, 1), round(hi, 1)],
            "ceiling_over_cheap_pct": round(self.ceiling_pct, 1),
            "rescue_rate_pct": round(100 * self.rescue_rate, 1),
            "by_domain": self.by_domain,
            "simulated": False,
            "source": "results.probe.jsonl",
        }


@lru_cache(maxsize=4)
def load_probe(path: str | None = None) -> Probe | None:
    """Recompute the cross-tab from the committed probe rows.

    Returns None when the file is absent, so a consumer can degrade rather than
    crash - the agent must still route a query in a checkout where the paid
    artefacts were not fetched.
    """
    p = Path(path) if path else PROBE_PATH
    if not p.exists():
        return None

    cheap: dict[str, bool] = {}
    expensive: dict[str, bool] = {}
    domains: dict[str, str] = {}

    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Simulated rows must never reach a "measured" figure.
            if row.get("simulated", True):
                continue
            tid, policy = row.get("task_id"), row.get("policy")
            if tid is None or policy not in ("always_cheap", "always_expensive"):
                continue
            domains[tid] = row.get("domain", "unknown")
            (cheap if policy == "always_cheap" else expensive)[tid] = bool(
                row.get("correct")
            )

    shared = sorted(set(cheap) & set(expensive))
    if not shared:
        return None

    cells = {"both_ok": 0, "routable": 0, "both_fail": 0, "inverted": 0}
    per_domain: dict[str, dict] = {}

    for tid in shared:
        c, e = cheap[tid], expensive[tid]
        cell = (
            "both_ok" if c and e
            else "routable" if not c and e
            else "inverted" if c and not e
            else "both_fail"
        )
        cells[cell] += 1
        d = per_domain.setdefault(
            domains.get(tid, "unknown"),
            {"n": 0, "both_ok": 0, "routable": 0, "both_fail": 0,
             "inverted": 0, "cheap_ok": 0, "expensive_ok": 0},
        )
        d["n"] += 1
        d[cell] += 1
        d["cheap_ok"] += int(c)
        d["expensive_ok"] += int(e)

    for d in per_domain.values():
        d["routable_pct"] = round(100 * d["routable"] / d["n"], 1)
        d["cheap_pct"] = round(100 * d["cheap_ok"] / d["n"], 1)
        d["expensive_pct"] = round(100 * d["expensive_ok"] / d["n"], 1)

    n = len(shared)
    return Probe(
        n=n,
        both_ok=cells["both_ok"], routable=cells["routable"],
        both_fail=cells["both_fail"], inverted=cells["inverted"],
        cheap_acc=sum(cheap[t] for t in shared) / n,
        expensive_acc=sum(expensive[t] for t in shared) / n,
        by_domain=per_domain,
    )


@lru_cache(maxsize=4)
def load_redraw(ladder: str) -> dict | None:
    """The decisive-cell redraw for a ladder: observed vs reproducible routable.

    Returns None when the ladder has not been redrawn, so the caveat that
    quotes it can drop the magnitude rather than invent one.
    """
    p = redraw_path(ladder)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not all(k in d for k in ("observed", "reproducible", "noise_share")):
        return None
    return d


# ---------------------------------------------------------------------------
# The verifier-quality finding - why the product cares which verifier it got
# ---------------------------------------------------------------------------

VERIFIER_TRANSFER = {
    "self_consistency": {
        "benchmark_name": "verify_math",
        "quality": "proxy",
        "transfers_to_production": True,
        "why": (
            "agreement is computed over the model's own draws, so it never "
            "consults an answer key and needs nothing serving cannot supply"
        ),
        "cost": "k model calls per verification, linear in k",
    },
    "tests": {
        "benchmark_name": "verify_code",
        "quality": "perfect",
        "transfers_to_production": False,
        "why": (
            "free and exact in the benchmark only because MBPP+ ships the "
            "asserts; a served query carries none unless the caller supplies "
            "them, so most production traffic cannot have this verifier"
        ),
        "cost": "no model calls; one subprocess",
    },
    "none": {
        "benchmark_name": "(predictive routing)",
        "quality": "absent",
        "transfers_to_production": True,
        "why": (
            "verifying nothing is always available; it converts the cascade "
            "into a one-shot predictive router that never learns it was wrong"
        ),
        "cost": "free",
    },
}

DEGRADATION_NOTE = (
    "sweep_degraded.py is the experiment that prices this: it corrupts "
    "verify_code by a controlled amount inside the code domain - holding the "
    "models, prompts, domain and grader fixed - and measures how cascade "
    "quality falls as verifier quality does. Losing the real tests in "
    "production is a move along that curve, not a step off it. The sweep has "
    "been run in replay against real responses on all three ladders "
    "(runs/sweep_degraded.<ladder>.jsonl), so its shape is measured."
)


def caveats(ladder: str) -> list[str]:
    """What the agent must say about itself, derived rather than pinned.

    THIS LIST WENT STALE ONCE, which is why it is a function now. It carried
    three literals describing a superseded run: `McNemar p=0.002`, a claim that
    only `wide` had real accuracy data and nine of eleven policies were mock,
    and a self-disagreement rate of 8.7% with no measurement anywhere in the
    repo behind it. All three survived the runs that falsified them, because a
    literal in a caveat is quoted, never recomputed.

    So anything with a magnitude now comes out of runs/, and anything that
    cannot be derived here names the doc section that owns it instead of
    restating the number. A caveat that has drifted from the data is worse than
    no caveat: it is read as the current state of the evidence.

    Same reasoning as scripts/provenance/freeze_probe.py, one layer up - that keeps the
    tests off pinned magnitudes, this keeps the agent's own prose off them.
    """
    out = [
        # Ladder-dependence is the caveat that replaced "only wide is real".
        # All three ladders are measured now; what is NOT general is the
        # headline, and quoting it without the ladder is the live error.
        "`cheaper AND better` is a `wide` result, not a general one. On "
        "`claude` the cascade buys its accuracy at a cost premium, and on "
        "`deepseek` the two rungs are not measurably different, so there is "
        "nothing to route at all. Name the ladder or the claim is empty - "
        "docs/RESULTS.md.",
    ]

    probe = load_probe()
    if probe:
        # Thin, but real: the ceiling is computed from the same cross-tab the
        # rest of the payload reports, so it cannot disagree with it.
        out.append(
            f"The routing signal is real and thin. The cheap and top rungs "
            f"differ at exact-McNemar p<0.001 on `wide` and `claude`, but "
            f"every policy competes over a ceiling of only "
            f"{probe.ceiling_pct:.1f} points over always_cheap "
            f"({probe.routable} routable and {probe.inverted} inverted of "
            f"{probe.n}). On `deepseek` that test does not reject at all "
            f"(p=1.000)."
        )

    redraw = load_redraw(ladder)
    if redraw:
        out.append(
            f"Part of even that ceiling is luck. Redrawing the decisive cells "
            f"moved the routable fraction from {100 * redraw['observed']:.1f}% "
            f"observed to {100 * redraw['reproducible']:.1f}% reproducible, "
            f"because one rung had a bad draw. "
            f"{100 * redraw['noise_share']:.1f}% of what fresh draws say is "
            f"there does not reproduce. Only the decisive cells were "
            f"redrawn, so that correction is a floor on itself, not an "
            f"estimate - see the limitation on routing opportunity in "
            f"docs/LIMITATIONS.md."
        )
    else:
        out.append(
            f"Part of even that ceiling is luck, and on `{ladder}` it is "
            f"unquantified: only `wide` has a decisive-cell redraw. Greedy "
            f"decoding is not deterministic for either provider, so a "
            f"single-draw probe is partly measuring which draw it got - "
            f"see the noise and determinism findings in docs/RESULTS.md."
        )

    return out


def summary(ladder: str) -> dict:
    """Everything the product wants to say about itself, in one payload."""
    probe = load_probe()
    return {
        "ladder": ladder,
        "ratio": ratio_verdict(ladder),
        "crossover_ratio": CROSSOVER_RATIO,
        # Labelled in the payload, not only in the comment above the constant.
        # A consumer reading this JSON sees a threshold sitting next to a
        # verdict, and the whole finding is that the threshold does not produce
        # the verdict.
        "crossover_ratio_note": (
            "The literature's rule of thumb, reported for comparison and used "
            "for nothing. These three ladders are not monotonic in the price "
            "ratio, so a threshold gets two of them backwards. `ratio.verdict` "
            "is the measured answer."
        ),
        "probe": probe.to_dict() if probe else None,
        "probe_note": (
            None if probe else
            "results.probe.jsonl not found; real cross-tab unavailable"
        ),
        "verifiers": VERIFIER_TRANSFER,
        "degradation_note": DEGRADATION_NOTE,
        "caveats": caveats(ladder),
    }
