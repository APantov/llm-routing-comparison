"""What the benchmark measured, read from the committed data where possible.

The serving layer makes claims - "cascade is the right policy on this ladder",
"escalating buys about 15 points here". Those claims come from the experiment
next door, and this module is the seam between them.

Two rules govern everything below.

**Real numbers are computed, not typed.** The probe cross-tab is recomputed
from `results.probe.jsonl` on demand rather than transcribed into a constant.
A hard-coded 15.0% would silently rot the first time the task set is rebuilt or
the grader is fixed - and the grader fix of 6 August 2026 moved this exact
figure from 13.0% to 15.0%, so that is a demonstrated failure mode, not a
hypothetical one.

**Simulated numbers are labelled at the point of use.** Nine of the eleven
policies have never been run against a real model; their accuracy figures come
from mock mode and mean nothing about any model. Anything sourced from them
carries `simulated: True` in its payload, so a consumer that surfaces it has to
walk past the label to do so.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_PATH = REPO_ROOT / "results.probe.jsonl"
FRONTIER_PATH = REPO_ROOT / "frontier.jsonl"


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
    import models

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

    Measured on the 7 August redraw the realized figure was **69x against a
    quoted 46x**, and the direction matters: this repository's whole thesis is
    that cascading pays in proportion to the price gap, so understating the gap
    understates the case for cascading on exactly the hard tasks where routing
    is worth doing.

    Restricted to tasks where BOTH rungs answered, so the two means describe the
    same population rather than two different task mixes. Greedy answers only -
    self-consistency samples are a different, shorter action and would drag the
    cheap rung's mean toward it.

    Returns None when no ladder cache exists, which is the normal state for a
    ladder that has never been run for real.
    """
    path = path or REPO_ROOT / "cache" / f"raw_calls.{ladder}.jsonl"
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
# Deliberately NOT used to compute a verdict. The `deepseek` ladder sits at
# 3.1x and the cascade still loses there, so the true crossover is somewhere
# above 3.1 and below `claude`'s 6.5 - a single threshold would get deepseek
# wrong. Treat this as "around 3x, and the sign is what to check" rather than
# as a decision boundary.
CROSSOVER_RATIO = 3.0


# ---------------------------------------------------------------------------
# The economics - derived from a frontier run, or quoted from a historical one
# ---------------------------------------------------------------------------

# Recorded 30 July 2026 from README "The finding this made possible" and
# STATUS.md §1.
#
# THESE ARE SUPERSEDED and are kept only as a fallback and a comparison point.
# They were produced BEFORE the 6 August task-set rebuild (sanitized MBPP ->
# MBPP+, MATH500 level 3 -> level 5), and a mock frontier run over the current
# task set gives materially different magnitudes - `claude` moves from -12% to
# about -46%, `deepseek` from +33% to about +10%.
#
# The SIGN survives on all three ladders, which is the actual finding: cascading
# pays in proportion to the price gap, and below roughly 3x it loses. The
# magnitudes do not survive, which is exactly why they are no longer presented
# as current.
HISTORICAL_ECONOMICS = {
    "deepseek": {"cascade_vs_always_best_pct": +33.0,
                 "auc_gain_over_random_pct": 7.2, "verdict": "route"},
    "claude": {"cascade_vs_always_best_pct": -12.0,
               "auc_gain_over_random_pct": 4.8, "verdict": "cascade"},
    "wide": {"cascade_vs_always_best_pct": -74.0,
             "auc_gain_over_random_pct": 8.8, "verdict": "cascade"},
}
HISTORICAL_AS_OF = "2026-07-30"
HISTORICAL_NOTE = (
    "quoted from the 30 July 2026 frontier run, which predates the 6 August "
    "task-set rebuild. The sign is reliable; the magnitude is not. Run "
    "`python frontier.py` on this ladder to derive current figures."
)


@lru_cache(maxsize=4)
def frontier_economics(path: str | None = None) -> dict | None:
    """Derive the cascade's economics from a frontier run, if one is on disk.

    Returns None when `frontier.jsonl` is absent - it is gitignored, because
    while the inputs are simulated so is every number in it.

    Two quantities, both computed with `frontier.py`'s own hull and AUC code
    rather than a reimplementation, so the router and the report can never
    disagree about what they mean:

      cascade_vs_always_best_pct
          The cheapest cascade setting that is at least as accurate as always
          paying for the top rung, against that rung's cost. Negative means the
          cascade is cheaper at matched accuracy. This is the headline.
      auc_gain_over_random_pct
          Mean accuracy across the whole shared budget range, minus the same
          for a cost-matched coin flip. Answers "how good is this at every
          budget" rather than "at the budget somebody tuned it to".

    The file holds ONE ladder at a time - `frontier.py` overwrites it per run -
    so the ladder it was generated for is returned alongside, and the caller
    must check it matches the one they are asking about.
    """
    p = Path(path) if path else FRONTIER_PATH
    if not p.exists():
        return None

    try:
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    if not rows:
        return None

    families: dict[str, list] = {}
    for r in rows:
        families.setdefault(r.get("family", ""), []).append(r)

    needed = ("always_cheap", "always_expensive", "cascade", "random")
    if any(f not in families for f in needed):
        return None

    # Imported here rather than at module scope: `frontier` pulls in policies,
    # run_eval and splits, and `findings` is imported by the MCP server on
    # every start. The probe path and the price ratios must stay cheap.
    import frontier as frontier_mod

    def points(name):
        return [(p_["cost_per_task"], p_["accuracy"]) for p_ in families[name]]

    # The same budget interval frontier.py integrates over: cheapest rung to
    # most expensive rung.
    lo = min(c for c, _ in points("always_cheap"))
    hi = max(c for c, _ in points("always_expensive"))

    random_auc = frontier_mod.auc(frontier_mod.upper_hull(points("random")), lo, hi)
    cascade_auc = frontier_mod.auc(frontier_mod.upper_hull(points("cascade")), lo, hi)

    expensive = max(
        families["always_expensive"],
        key=lambda p_: (p_["accuracy"], -p_["cost_per_task"]),
    )
    matched = [
        p_ for p_ in families["cascade"]
        if p_["accuracy"] >= expensive["accuracy"] - 1e-9
    ]
    cost_pct = None
    if matched and expensive["cost_per_task"] > 0:
        cheapest = min(matched, key=lambda p_: p_["cost_per_task"])
        cost_pct = round(
            100.0 * (cheapest["cost_per_task"] - expensive["cost_per_task"])
            / expensive["cost_per_task"], 1
        )

    auc_gain = (
        None if random_auc is None or cascade_auc is None
        else round(100.0 * (cascade_auc - random_auc), 1)
    )

    return {
        "ladder": rows[0].get("ladder"),
        "cascade_vs_always_best_pct": cost_pct,
        "auc_gain_over_random_pct": auc_gain,
        "verdict": (
            None if cost_pct is None else ("cascade" if cost_pct < 0 else "route")
        ),
        # Every row of a mock frontier carries simulated: true. If ANY row is
        # simulated the derived figure is, so this is `any`, not `all`.
        "simulated": any(r.get("simulated", True) for r in rows),
        "n": rows[0].get("n"),
        "split": rows[0].get("split"),
        "source": "frontier.jsonl",
    }


def ratio_verdict(ladder: str) -> dict:
    """Should this ladder cascade or route? The product's core decision.

    Assembled from three sources of decreasing reliability, each labelled:

      price ratios   exact arithmetic over the price table. Always present.
      frontier       derived from `frontier.jsonl` if a run for THIS ladder is
                     on disk. Current, but simulated while the inputs are.
      historical     the 30 July constants. Superseded; sign only.
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

    live = frontier_economics()
    if live and live.get("ladder") == ladder and live.get("verdict"):
        out.update({
            "cascade_vs_always_best_pct": live["cascade_vs_always_best_pct"],
            "auc_gain_over_random_pct": live["auc_gain_over_random_pct"],
            "verdict": live["verdict"],
            "economics_source": "frontier.jsonl",
            "economics_simulated": live["simulated"],
            "economics_n": live.get("n"),
        })
    else:
        hist = HISTORICAL_ECONOMICS.get(ladder)
        if hist is None:
            out.update({
                "verdict": None,
                "economics_source": None,
                "note": (
                    "no frontier run on disk for this ladder and no historical "
                    "figure; run `python frontier.py` to derive one"
                ),
            })
        else:
            out.update({
                **hist,
                "economics_source": "historical",
                "economics_simulated": True,
                "economics_as_of": HISTORICAL_AS_OF,
                "economics_note": HISTORICAL_NOTE,
            })

    # Only `wide` has real accuracy data behind any of this, and even there the
    # frontier itself has never been run against real models.
    out["accuracy_data_is_real"] = ladder == "wide"
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
    "production is a move along that curve, not a step off it. NOTE: the "
    "sweep has only ever been run in mock mode, so its shape is a modelled "
    "prediction rather than a measurement."
)


def summary(ladder: str) -> dict:
    """Everything the product wants to say about itself, in one payload."""
    probe = load_probe()
    return {
        "ladder": ladder,
        "ratio": ratio_verdict(ladder),
        "crossover_ratio": CROSSOVER_RATIO,
        "probe": probe.to_dict() if probe else None,
        "probe_note": (
            None if probe else
            "results.probe.jsonl not found; real cross-tab unavailable"
        ),
        "verifiers": VERIFIER_TRANSFER,
        "degradation_note": DEGRADATION_NOTE,
        "caveats": [
            "Only the `wide` ladder has real accuracy data. Two of eleven "
            "policies (always_cheap, always_expensive) have been run against "
            "real models; the other nine remain mock-mode figures.",
            "The routing signal is real, significant (McNemar p=0.002) and "
            "thin: every policy competes over a ~17-point ceiling.",
            "Models disagree with themselves on 8.7% of maths tasks between "
            "independent draws of the same prompt, which is noise sitting "
            "underneath a 15% signal.",
        ],
    }
