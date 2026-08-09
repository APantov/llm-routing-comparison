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

There is also a hard cap, `--max-spend`, checked twice: against the ESTIMATE
before the first call, and against MEASURED backend spend after every call. The
estimate alone is not a guard - the first real run of this script came in at
$2.96 against a $1.96 prediction, a 51% under-quote, and only the second check
would have caught that.

    python3 scripts/redraw_decisive.py                    # plan and price only
    ROUTER_MODE=real ROUTER_LADDER=wide \
        python3 scripts/redraw_decisive.py --draws 10 --go
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import models          # noqa: E402
import routable        # noqa: E402
from graders import grade  # noqa: E402

TASKSET = REPO / "taskset.jsonl"

# The cap itself lives in models.call, next to the one line that can charge a
# card, so this script does not implement one - it only chooses the number.
# `--max-spend` overwrites models.MAX_SPEND_USD for the process, so exactly one
# figure is ever in force and the flag is authoritative for this run.
DEFAULT_MAX_SPEND_USD = models.MAX_SPEND_USD


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


def mean_call_cost(ladder, task_ids=None, domain=None):
    """Mean observed cost of one greedy answer, per tier, from the paid run.

    Restricted to `task_ids` when given, and that restriction is the whole
    point rather than a refinement. The tasks this script redraws are the ones
    that DEFEATED a rung, so they are the long ones: they draw more reasoning
    tokens than the task set's average, and output tokens are what cost money.

    Estimating from the whole cache under-quoted the first real run of this
    script by 51% - $1.96 predicted against $2.96 spent. Averaging over the
    population you are about to sample, at the temperature you will sample it
    at, is the fix.

    `domain` is the fallback for screening a RAW candidate pool, where by
    definition no task has been called yet so `task_ids` would match nothing and
    the estimate would come out at $0.00 - a spend guard reading zero is worse
    than no guard. A domain-wide mean is a weaker estimate than a per-task one,
    which is why the second, measured cap in redraw() is the real protection.
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
            if domain is not None and d.get("domain") != domain:
                continue
            totals[d["tier"]] = totals.get(d["tier"], 0.0) + d["cost_usd"]
            counts[d["tier"]] = counts.get(d["tier"], 0) + 1
    return {t: totals[t] / counts[t] for t in totals if counts[t]}


# Raw candidate pools, for screening rather than redrawing. Each returns
# (tasks, domain); `build_taskset.drop_quarantined` is applied to every one of
# them at the call site, because SHIP_PLAN.md section 0.5 makes that rule apply
# to "every rerun, every ladder, every figure" - a screener that re-bought the
# five unpassable tasks would be the first thing to break it.
def _pool_mbppplus(min_math_level):
    import build_taskset
    return build_taskset.load_mbppplus(), "code"


def _pool_math500(min_math_level):
    import build_taskset
    return build_taskset.load_math500(min_math_level), "math"


POOLS = {"mbppplus": _pool_mbppplus, "math500": _pool_math500}


def redraw(tasks, draws, temperature, tiers=("cheap", "expensive")):
    """p_hat[task_id][tier] = share of draws that graded correct.

    The running total shown per task is `models.backend_spend_usd` - what has
    actually reached a backend - not the sum of `cost_usd`. A draw already on
    disk from an interrupted earlier run is served from the cache and charges
    the card nothing, so a resumed run would otherwise appear to be re-spending
    money it is not spending.

    The cap that stops this loop is enforced inside models.call. It raises
    SpendCapExceeded from wherever it binds, which is why nothing here catches
    it: a partial p_hat must not be written.
    """
    out = {}
    for i, task in enumerate(tasks, 1):
        row = {}
        for tier in tiers:
            ok = 0
            for s in range(1, draws + 1):
                # sample_idx starts at 1: index 0 is the probe's original draw,
                # already on disk. Including it would double-count it.
                r = models.call(tier, task, temperature=temperature, sample_idx=s)
                ok += bool(grade(task, r.text))
            row[tier] = ok / draws
        out[task["id"]] = row
        print(f"  [{i}/{len(tasks)}] {task['id']:<16} "
              + "  ".join(f"{t}={row[t]:.2f}" for t in tiers)
              + f"   spent ${models.backend_spend_usd:.4f}", file=sys.stderr)
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


def screen_summary(p_hat, tiers, tau):
    """Cross-tab a screen by reproducible per-rung outcome.

    Deliberately NOT the same table as `reestimate`. That one corrects a
    published number against a probe; this one has no probe behind it - it is
    describing a pool nobody has measured before, so all it can honestly say is
    how each rung behaved and, when both were drawn, how they combined.

    `reliably fails` / `reliably passes` use the same tau as the re-estimate, so
    the two outputs are talking about reproducibility in the same units.
    """
    n = len(p_hat)
    out = {"n": n}
    for tier in tiers:
        ps = [p[tier] for p in p_hat.values()]
        out[tier] = {
            "mean_p": sum(ps) / n if n else float("nan"),
            "reliably_fails": sum(1 for p in ps if p <= tau),
            "reliably_passes": sum(1 for p in ps if p >= 1.0 - tau),
            "flaky": sum(1 for p in ps if tau < p < 1.0 - tau),
        }
    if set(tiers) >= {"cheap", "expensive"}:
        out["routable_reproducible"] = sum(
            1 for p in p_hat.values()
            if p["cheap"] <= tau and p["expensive"] >= 1.0 - tau
        )
        out["both_fail_reproducible"] = sum(
            1 for p in p_hat.values()
            if p["cheap"] <= tau and p["expensive"] <= tau
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Re-draw the tasks that decided the routable fraction, or "
                    "screen a raw candidate pool at one rung.")
    ap.add_argument("--draws", type=int, default=10,
                    help="extra samples per task per rung (default 10)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 matches the probe, and measures ITS reproducibility")
    ap.add_argument("--cells", choices=["decisive", "all"], default="decisive",
                    help="'decisive' redraws routable+both_fail only (default). "
                         "Ignored unless --pool probe.")
    ap.add_argument("--tau", type=float, default=0.2,
                    help="tolerance for calling a rung's outcome reliable")
    ap.add_argument("--tier", choices=["cheap", "expensive", "both"], default="both",
                    help="which rung(s) to draw. 'cheap' is the screening case: "
                         "cheap draws cost $0.000029-0.000245 against the "
                         "expensive rung's $0.002562-$0.019648, so a "
                         "cheap-only pass over a whole pool is affordable and a "
                         "two-rung one is not.")
    ap.add_argument("--pool", choices=["probe", *sorted(POOLS)], default="probe",
                    help="'probe' (default) redraws tasks already measured in "
                         "this ladder's cache. The others screen a RAW "
                         "candidate pool that has never been called.")
    ap.add_argument("--min-math-level", type=int, default=3,
                    help="MATH500 level floor for --pool math500 (default 3)")
    ap.add_argument("--limit", type=int, default=None,
                    help="screen only the first N tasks of the pool")
    ap.add_argument("--max-spend", type=float, default=DEFAULT_MAX_SPEND_USD,
                    help=f"hard cap on MEASURED backend spend "
                         f"(default ${DEFAULT_MAX_SPEND_USD:.2f}, or "
                         f"$ROUTER_MAX_SPEND_USD)")
    ap.add_argument("--go", action="store_true",
                    help="actually make the calls. Without this, plan only.")
    args = ap.parse_args()

    tiers = ("cheap", "expensive") if args.tier == "both" else (args.tier,)
    # One number in force for the process, so the flag and the env var cannot
    # disagree about which cap applies.
    models.MAX_SPEND_USD = args.max_spend
    ladder = models.LADDER
    cells, n_graded, domain = None, None, None

    if args.pool == "probe":
        if not TASKSET.exists():
            sys.exit("taskset.jsonl not built. Run: python3 build_taskset.py")
        tasks = routable.load_tasks(TASKSET)
        cells = observed_cells(tasks, ladder)
        n_graded = sum(len(v) for v in cells.values())
        if n_graded == 0:
            sys.exit(
                f"No two-arm data for ladder {ladder!r}.\n"
                f"  cache/raw_calls.{ladder}.jsonl has no gradeable cheap+expensive pairs.\n"
                f"  Run the probe first:\n"
                f"    ROUTER_MODE=real python3 run_eval.py "
                f"--policy always_cheap --policy always_expensive --split all\n"
                f"  Or screen a raw pool instead:  --pool mbppplus --tier cheap"
            )
        targets = (cells["routable"] + cells["both_fail"] if args.cells == "decisive"
                   else [t for v in cells.values() for t in v])
        # Per-task means: these tasks defeated a rung, so they are the long ones.
        unit = mean_call_cost(ladder, {t["id"] for t in targets})
        basis = "mean greedy call cost on THESE tasks, not the task set average"
    else:
        import build_taskset
        pool, domain = POOLS[args.pool](args.min_math_level)
        targets = build_taskset.drop_quarantined(pool)
        if args.limit is not None:
            targets = targets[:args.limit]
        # Domain means: nothing in a raw pool has been called, so there is no
        # per-task history to average over. See mean_call_cost.
        unit = mean_call_cost(ladder, domain=domain)
        basis = f"mean greedy call cost over all cached {domain} tasks"
        if not unit:
            sys.exit(
                f"Cannot price a {domain!r} screen on ladder {ladder!r}: "
                f"cache/raw_calls.{ladder}.jsonl holds no greedy {domain} "
                f"answers to average over.\n"
                f"  Running blind is refused - an unpriced pass over a whole "
                f"pool is exactly the run that should not start by accident."
            )

    if not targets:
        sys.exit("nothing to draw: 0 target tasks after filtering.")

    n_calls = len(targets) * len(tiers) * args.draws
    est = sum(unit.get(t, 0.0) for t in tiers) * len(targets) * args.draws

    print(f"ladder    {ladder}   mode {models.MODE}")
    if cells is not None:
        print(f"probe     n={n_graded}  " + "  ".join(
            f"{k}={len(v)}" for k, v in cells.items()))
    else:
        print(f"pool      {args.pool} ({domain})  {len(targets)} tasks after "
              f"quarantine" + (f", limited to {args.limit}" if args.limit else ""))
    print(f"{'redraw' if cells is not None else 'screen'}    {len(targets)} tasks "
          f"x {len(tiers)} rung(s) [{', '.join(tiers)}] x {args.draws} draws "
          f"= {n_calls} calls at temperature {args.temperature}")
    print(f"estimate  ${est:.4f}   ({basis})")
    print(f"          a reasoning model's output length varies per draw, so "
          f"treat this as +/- 25%.")
    print(f"cap       ${args.max_spend:.2f} measured backend spend, "
          f"checked after every call")

    # First of the two checks. The estimate is weak evidence - this script's own
    # first real run came in 51% over - so the cap it guards is the plan being
    # obviously wrong, not the plan being slightly off. The measured check inside
    # redraw() is what catches the rest.
    if est > args.max_spend:
        sys.exit(
            f"\nRefusing: the ESTIMATE (${est:.4f}) already exceeds the cap "
            f"(${args.max_spend:.2f}), before the +/- 25%.\n"
            f"  Draw fewer tasks (--limit), fewer draws (--draws), or one rung "
            f"(--tier cheap).\n"
            f"  Raise the cap only once you have decided the number is right: "
            f"--max-spend {est * 1.5:.2f}"
        )

    if not args.go:
        print("\nPlan only - nothing was called. Add --go to spend.")
        print("Set ROUTER_MODE=real first, or this replays and measures nothing.")
        return

    if models.MODE != "real":
        sys.exit(f"\nROUTER_MODE is {models.MODE!r}. Refusing: a redraw in "
                 f"{models.MODE!r} mode would re-read one cached answer "
                 f"{args.draws} times and report perfect reproducibility.")

    print(f"\nDrawing...", file=sys.stderr)
    p_hat = redraw(targets, args.draws, args.temperature, tiers)

    common = {"ladder": ladder, "draws": args.draws,
              "temperature": args.temperature, "tau": args.tau,
              "tiers": list(tiers), "p_hat": p_hat}

    # A screen and a re-estimate answer different questions and are written to
    # different files on purpose. `redraw.<ladder>.json` is a CORRECTION to a
    # published number and downstream code reads it as one; a screen has no
    # published number behind it and must not be mistaken for one.
    if cells is None:
        summary = screen_summary(p_hat, tiers, args.tau)
        print("\n" + "=" * 62)
        print(f"  screen: {args.pool} ({domain}), {summary['n']} tasks")
        print("=" * 62)
        for tier in tiers:
            s = summary[tier]
            print(f"  {tier:<10} mean p={s['mean_p']:.3f}   "
                  f"reliably fails {s['reliably_fails']:>4}   "
                  f"flaky {s['flaky']:>4}   "
                  f"reliably passes {s['reliably_passes']:>4}")
        if "routable_reproducible" in summary:
            print(f"\n  reproducibly routable  {summary['routable_reproducible']}"
                  f"/{summary['n']}   cheap reliably fails, expensive reliably passes")
            print(f"  reproducibly both_fail {summary['both_fail_reproducible']}"
                  f"/{summary['n']}   candidates for manual adjudication")
        else:
            print(f"\n  One rung only, so this says which tasks the {tiers[0]} "
                  f"rung reliably\n  fails - not which are routable. That needs "
                  f"the other rung.")
        path = REPO / f"screen.{ladder}.{args.pool}.json"
        path.write_text(json.dumps(
            {**common, "pool": args.pool, "domain": domain,
             "min_math_level": args.min_math_level, "limit": args.limit,
             "summary": summary}, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")
        return

    if set(tiers) < {"cheap", "expensive"}:
        # Refusing to print a re-estimate is the point. reestimate() needs
        # (1 - p_cheap) * p_expensive per task, so with one rung drawn it would
        # have to substitute the single-draw verdict for the other rung - which
        # is the exact quantity this script exists to stop trusting.
        path = REPO / f"redraw.{ladder}.{tiers[0]}.json"
        path.write_text(json.dumps(
            {**common, "cells_redrawn": args.cells, "n_graded": n_graded},
            indent=2), encoding="utf-8")
        print(f"\nwrote {path}")
        print(f"No re-estimate: only the {tiers[0]} rung was drawn, and the "
              f"routable fraction\nis a property of both. Re-run with --tier "
              f"both to get one.")
        return

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
        {**common, "cells_redrawn": args.cells,
         "n_graded": n_graded, **out}, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    try:
        main()
    except models.SpendCapExceeded as exc:
        print(
            "\n" + "=" * 62
            + "\n  SPEND CAP - REDRAW ABORTED, NOTHING WRITTEN\n"
            + "=" * 62 + f"\n{exc}",
            file=sys.stderr,
        )
        raise SystemExit(2)
