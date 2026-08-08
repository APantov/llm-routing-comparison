#!/usr/bin/env python3
r"""Re-draw the tasks that decided the routable fraction, to price its noise.

WHY THIS EXISTS
---------------
`results.probe.jsonl` holds one draw per (task, tier). From that single draw the
repository reports `routable = 15%` - the fraction of tasks the cheap rung gets
wrong and the expensive rung gets right - and treats it as the ceiling on any
router.

A single draw cannot tell "the cheap model cannot do this" apart from "the cheap
model can usually do this and missed once". *How Much of the Routing Gap Is
Real?* (arXiv:2607.03436) decomposes exactly this measurement and puts the
single-draw noise share at 36% on MATH-500 - and the maths half of this task set
IS MATH-500 level 5. See NOTES.md issue 3.

WHAT IT DOES
------------
Redraws only the cells that can move the answer. A task in `both_ok` contributes
nothing to `routable` on any draw where both rungs behave as observed, and a
task in `inverted` is already counted against the cascade. The cells that decide
the number are:

    routable    cheap wrong, expensive right  -> is this reproducible?
    both_fail   neither rung right            -> or was the expensive rung unlucky?

For each, it takes `--draws` further samples per rung and reports the empirical
per-rung success probability, then re-estimates the routable fraction from those
probabilities instead of from one coin flip.

WHAT IT DOES NOT DO
-------------------
It does not redraw `both_ok` or `inverted`, so a task where the CHEAP rung is
secretly flaky but happened to succeed keeps its single-draw verdict. That
biases the re-estimate DOWNWARD (some hidden routable mass is never found), so
the corrected figure this prints is a lower bound on the correction, not a
two-sided one. `--cells all` redraws everything and removes the caveat, at
roughly five times the cost.

Temperature defaults to 0.0 - the probe's own setting - so what is measured is
the reproducibility of the protocol that produced the headline, not the variance
of some other protocol.

SPENDING
--------
Prints a costed plan and exits. Nothing is called until `--go` is passed.
Every response is written to the ladder's cache as it arrives, so a run that
dies halfway keeps what it paid for, and once complete the whole re-estimate
replays for free.

    python3 scripts/redraw_decisive.py                    # plan and price only
    ROUTER_MODE=real ROUTER_LADDER=wide \
        python3 scripts/redraw_decisive.py --draws 10 --go
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import models          # noqa: E402
import routable        # noqa: E402
from graders import grade  # noqa: E402

TASKSET = REPO / "taskset.jsonl"


def observed_cells(tasks, ladder):
    """The four cells of the probe's cross-tab, as {cell: [task, ...]}."""
    verdicts = routable.real_verdicts(tasks, ladder)
    by_id = {t["id"]: t for t in tasks}
    cells = {"both_ok": [], "routable": [], "both_fail": [], "inverted": []}
    for task_id, v in verdicts.items():
        if "cheap" not in v or "expensive" not in v:
            continue
        name = {
            (True, True): "both_ok", (False, True): "routable",
            (False, False): "both_fail", (True, False): "inverted",
        }[(v["cheap"], v["expensive"])]
        cells[name].append(by_id[task_id])
    return cells


def mean_call_cost(ladder, task_ids=None):
    """Mean observed cost of one greedy answer, per tier, from the paid run.

    Restricted to `task_ids` when given, and that restriction is the whole
    point rather than a refinement. The tasks this script redraws are the ones
    that DEFEATED a rung, so they are the long ones: they draw more reasoning
    tokens than the task set's average, and output tokens are what cost money.

    Estimating from the whole cache under-quoted the first real run of this
    script by 51% - $1.96 predicted against $2.96 spent. Averaging over the
    population you are about to sample, at the temperature you will sample it
    at, is the fix.
    """
    path = REPO / "cache" / f"raw_calls.{ladder}.jsonl"
    totals, counts = {}, {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("kind") != "answer":
                continue
            # Greedy draws only. Self-consistency samples sit at temperature
            # 0.8 and are a different, shorter action.
            if d.get("temperature") not in (0, 0.0):
                continue
            if task_ids is not None and d.get("task_id") not in task_ids:
                continue
            totals[d["tier"]] = totals.get(d["tier"], 0.0) + d["cost_usd"]
            counts[d["tier"]] = counts.get(d["tier"], 0) + 1
    return {t: totals[t] / counts[t] for t in totals if counts[t]}


def redraw(tasks, draws, temperature):
    """p_hat[task_id][tier] = share of draws that graded correct."""
    out = {}
    for i, task in enumerate(tasks, 1):
        row = {}
        for tier in ("cheap", "expensive"):
            ok = 0
            for s in range(1, draws + 1):
                # sample_idx starts at 1: index 0 is the probe's original draw,
                # already on disk. Including it would double-count it.
                r = models.call(tier, task, temperature=temperature, sample_idx=s)
                ok += bool(grade(task, r.text))
            row[tier] = ok / draws
        out[task["id"]] = row
        print(f"  [{i}/{len(tasks)}] {task['id']:<16} "
              f"cheap={row['cheap']:.2f}  expensive={row['expensive']:.2f}",
              file=sys.stderr)
    return out


def reestimate(cells, p_hat, n_total, tau):
    """Re-estimate routable from per-rung success probabilities.

    Three numbers, and the gap between them is the point:

      observed      what one draw said           (the published 15%)
      expected      E[routable] under a fresh draw, = sum (1-p_c) * p_e
      reproducible  tasks the cheap rung reliably fails and the expensive rung
                    reliably gets, at tolerance tau

    `expected` says what re-running the probe would give on average.
    `reproducible` says how much of that a router could actually capture -
    the rest is a coin the router has no way to predict.
    """
    observed = len(cells["routable"]) / n_total

    expected = 0.0
    reproducible = 0.0
    for task in cells["routable"] + cells["both_fail"]:
        p = p_hat[task["id"]]
        expected += (1.0 - p["cheap"]) * p["expensive"]
        if p["cheap"] <= tau and p["expensive"] >= 1.0 - tau:
            reproducible += 1.0

    # Tasks outside the redrawn cells keep their single-draw verdict, and none
    # of them was counted routable, so they add nothing to either estimate.
    return {
        "observed": observed,
        "expected": expected / n_total,
        "reproducible": reproducible / n_total,
        "noise_share": (
            1.0 - (reproducible / n_total) / (expected / n_total)
            if expected > 0 else float("nan")
        ),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Re-draw the tasks that decided the routable fraction.")
    ap.add_argument("--draws", type=int, default=10,
                    help="extra samples per task per rung (default 10)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 matches the probe, and measures ITS reproducibility")
    ap.add_argument("--cells", choices=["decisive", "all"], default="decisive",
                    help="'decisive' redraws routable+both_fail only (default)")
    ap.add_argument("--tau", type=float, default=0.2,
                    help="tolerance for calling a rung's outcome reliable")
    ap.add_argument("--go", action="store_true",
                    help="actually make the calls. Without this, plan only.")
    args = ap.parse_args()

    if not TASKSET.exists():
        sys.exit("taskset.jsonl not built. Run: python3 build_taskset.py")
    tasks = routable.load_tasks(TASKSET)
    ladder = models.LADDER
    cells = observed_cells(tasks, ladder)

    n_graded = sum(len(v) for v in cells.values())
    if n_graded == 0:
        sys.exit(
            f"No two-arm data for ladder {ladder!r}.\n"
            f"  cache/raw_calls.{ladder}.jsonl has no gradeable cheap+expensive pairs.\n"
            f"  Run the probe first:\n"
            f"    ROUTER_MODE=real python3 run_eval.py "
            f"--policy always_cheap --policy always_expensive --split all"
        )

    targets = (cells["routable"] + cells["both_fail"] if args.cells == "decisive"
               else [t for v in cells.values() for t in v])
    n_calls = len(targets) * 2 * args.draws
    unit = mean_call_cost(ladder, {t["id"] for t in targets})
    est = sum(unit.get(t, 0.0) for t in ("cheap", "expensive")) * len(targets) * args.draws

    print(f"ladder    {ladder}   mode {models.MODE}")
    print(f"probe     n={n_graded}  " + "  ".join(
        f"{k}={len(v)}" for k, v in cells.items()))
    print(f"redraw    {len(targets)} tasks x 2 rungs x {args.draws} draws "
          f"= {n_calls} calls at temperature {args.temperature}")
    print(f"estimate  ${est:.4f}   "
          f"(mean greedy call cost on THESE tasks, not the task set average)")
    print(f"          a reasoning model's output length varies per draw, so "
          f"treat this as +/- 25%.")

    if not args.go:
        print("\nPlan only - nothing was called. Add --go to spend.")
        print("Set ROUTER_MODE=real first, or this replays and measures nothing.")
        return

    if models.MODE != "real":
        sys.exit(f"\nROUTER_MODE is {models.MODE!r}. Refusing: a redraw in "
                 f"{models.MODE!r} mode would re-read one cached answer "
                 f"{args.draws} times and report perfect reproducibility.")

    print(f"\nDrawing...", file=sys.stderr)
    p_hat = redraw(targets, args.draws, args.temperature)
    out = reestimate(cells, p_hat, n_graded, args.tau)

    print("\n" + "=" * 62)
    print("  routable fraction, re-estimated")
    print("=" * 62)
    print(f"  observed      {out['observed']:6.1%}   one draw per cell (published)")
    print(f"  expected      {out['expected']:6.1%}   mean over fresh draws")
    print(f"  reproducible  {out['reproducible']:6.1%}   cheap reliably fails, "
          f"expensive reliably succeeds")
    print(f"  noise share   {out['noise_share']:6.1%}   of `expected` that no "
          f"single-commit router can capture")
    if args.cells == "decisive":
        print("\n  both_ok and inverted were not redrawn, so hidden routable mass\n"
              "  in those cells is not counted. Treat `reproducible` as a lower\n"
              "  bound; --cells all removes the caveat.")

    path = REPO / f"redraw.{ladder}.json"
    path.write_text(json.dumps(
        {"ladder": ladder, "draws": args.draws, "temperature": args.temperature,
         "tau": args.tau, "cells_redrawn": args.cells,
         "n_graded": n_graded, "p_hat": p_hat, **out}, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
