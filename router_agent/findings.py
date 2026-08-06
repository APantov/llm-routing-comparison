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
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_PATH = REPO_ROOT / "results.probe.jsonl"


# ---------------------------------------------------------------------------
# The ratio finding - the headline, and the one the router acts on
# ---------------------------------------------------------------------------

# Source: STATUS.md §1 and README "The finding this made possible". These come
# from the verified price tables and the escalation logic rather than from
# MOCK_SKILL, which is why STATUS.md expects them to survive a real run - but
# only `wide` has real accuracy data behind it, so `measured` is set per row.
RATIO_FINDING = {
    "deepseek": {
        "rungs": "v4-flash -> v4-pro",
        "list_ratio": 3.1,
        "effective_ratio": 3.1,
        "cascade_vs_always_best_pct": +33.0,
        "verdict": "route",
        "auc_gain_over_random_pct": 7.2,
        "measured": False,
    },
    "claude": {
        "rungs": "Haiku 4.5 -> Sonnet 5 -> Opus 5",
        "list_ratio": 5.0,
        "effective_ratio": 6.5,
        "cascade_vs_always_best_pct": -12.0,
        "verdict": "cascade",
        "auc_gain_over_random_pct": 4.8,
        "measured": False,
    },
    "wide": {
        "rungs": "DeepSeek v4-flash -> Opus 5",
        "list_ratio": 36.0,
        "effective_ratio": 46.0,
        "cascade_vs_always_best_pct": -74.0,
        "verdict": "cascade",
        "auc_gain_over_random_pct": 8.8,
        "measured": True,
    },
}

# Below roughly this effective price ratio, the cascade's fixed costs - the
# wasted cheap call, plus verification - exceed what skipping an expensive call
# saves. The number is where the sign flips between the deepseek and claude
# ladders, so treat it as "between 3.1 and 6.5, nearer the bottom" rather than
# as a measured constant.
CROSSOVER_RATIO = 3.0


def ratio_verdict(ladder: str) -> dict:
    """Should this ladder cascade or route? The product's core decision."""
    row = RATIO_FINDING.get(ladder)
    if row is None:
        return {
            "ladder": ladder,
            "known": False,
            "note": (
                f"no measurement for ladder {ladder!r}; the crossover sits near "
                f"an effective price ratio of {CROSSOVER_RATIO}x - above it "
                f"cascade, below it route"
            ),
        }
    return {"ladder": ladder, "known": True, **row}


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
