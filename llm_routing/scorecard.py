"""Per-policy error attribution: what each router got right, and what it got wrong.

The rest of the analysis reports policies in aggregate and answers "which policy
is better". None of it answers "what did this one do wrong", and those differ:
two routers can reach the same accuracy at the same price by making opposite
mistakes in equal numbers.

`routable.py` establishes that only some tasks carry routing information at all,
by crossing the cheap rung's verdict with the expensive rung's:

    both_ok     cheap right, expensive right   any router scores 1
    routable    cheap wrong, expensive right   the ONLY cell routing can win
    both_fail   cheap wrong, expensive wrong   any router scores 0
    inverted    cheap right, expensive wrong   routing UP loses

This module joins that cross-tab against each policy's actual decision, which
turns a percentage into a named account:

                     | stayed cheap            | escalated
    -----------------+-------------------------+---------------------------
    both_ok          | correct thrift          | WASTED escalation
    routable         | MISSED rescue           | correct rescue
    both_fail        | unavoidable loss, cheap | WASTED on a lost cause
    inverted         | lucky thrift            | HARMFUL escalation

"cascade scored 97.9%" becomes "cascade rescued 12 of 14 routable tasks, burned
$0.31 escalating 21 tasks the cheap rung already had, and escalated 2 tasks it
then got wrong".

Two caveats. The cross-tab is a property of the LADDER, not the policy, so a
scorecard is only comparable within one ladder's results file. And a single-draw
cross-tab treats a flaky task as decided: `redraw.<ladder>.json` measures how much
of the routable cell reproduces - about two thirds on `wide` - so read the rescue
counts as being about this measurement, not an infinitely repeated one.

    python -m llm_routing.scorecard                    # results.<ladder>.jsonl
    python -m llm_routing.scorecard --results runs/results.claude.jsonl
    python -m llm_routing.scorecard --by-domain --json card.json

Reads only files already on disk. No API calls, no spend.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from llm_routing import paths

RESULTS = paths.RUNS / f"results.{paths.default_ladder()}.jsonl"

# The cell a task falls in, from (cheap_ok, expensive_ok). Same four names
# routable.crosstab uses, deliberately - this is that table, per policy.
CELLS = {
    (True, True): "both_ok",
    (False, True): "routable",
    (False, False): "both_fail",
    (True, False): "inverted",
}

ORDER = [
    "correct_rescue", "rescued_without_top_rung", "missed_rescue",
    "wasted_escalation", "harmful_escalation", "wasted_on_lost_cause",
    "correct_thrift", "lucky_thrift", "unavoidable_loss", "unexpected_loss",
]

# Outcomes that cost money without being able to improve the answer.
WASTEFUL = {"wasted_escalation", "harmful_escalation", "wasted_on_lost_cause"}

# Outcomes on which the policy got the task RIGHT. Note `wasted_escalation` is
# here: on a both_ok task both rungs are correct, so escalating buys the right
# answer at the wrong price. "Wasted" is a statement about the money, not the
# answer, and conflating the two is what the reconciliation check below caught
# the first time it ran.
CORRECT_OUTCOMES = {
    "correct_rescue", "rescued_without_top_rung", "correct_thrift",
    "lucky_thrift", "wasted_escalation",
}


def outcome_of(cell, used_expensive, correct):
    """What this policy did on this task, and whether it was a mistake.

    THE THIRD ACTION. Reading the decision as binary - escalated or not - gets
    the oracle wrong. It rescues `math-422` and `math-432`, both in the `routable`
    cell, with `calls = ['cheap'] * 5`: that is cheap-rung SELF-CONSISTENCY, a
    third action costing five cheap calls and no expensive one. A binary reading
    files both as missed rescues while the same rows say `correct: true`.

    So the cell is a prediction and `correct` is the observation, and where they
    disagree the observation wins. Those disagreements are not noise to be
    smoothed over - `rescued_without_top_rung` is the measured size of the effect
    `scripts/provenance/resample_vs_reroute.py` exists to ask about, which is whether
    majority-of-k substitutes for escalating.
    """
    if used_expensive:
        return {
            "routable": ("correct_rescue", False),
            "both_ok": ("wasted_escalation", True),
            "both_fail": ("wasted_on_lost_cause", True),
            "inverted": ("harmful_escalation", True),
        }[cell]
    if cell in ("routable", "both_fail"):
        # The cross-tab says one greedy cheap call fails here and only the TOP
        # rung fixes it, so getting it right without reaching the top means
        # something else did the work. Two mechanisms do, on this data:
        # cheap-rung self-consistency (the oracle recovers math-422 and math-432
        # with calls = ['cheap'] * 5), and a MIDDLE rung (on the three-rung
        # claude ladder always_mid solves 12 routable tasks, escalating and
        # resampling neither). Hence a bucket named for what it is NOT.
        if correct:
            return "rescued_without_top_rung", False
        return ("missed_rescue", True) if cell == "routable" else \
               ("unavoidable_loss", False)
    # cell says the cheap rung handles it.
    if not correct:
        return "unexpected_loss", True
    return ("correct_thrift", False) if cell == "both_ok" else \
           ("lucky_thrift", False)


def load_results(path):
    if not path.exists():
        sys.exit(f"{path.name} not found. Generate it first:\n  python -m llm_routing.run_eval")
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if not rows:
        sys.exit(f"{path.name} is empty")
    return rows


def cells_from_cache(tasks, ladder):
    """task_id -> cell, by grading what the paid run put on disk.

    Reuses `routable.real_verdicts`, which already carries the two exclusions
    that matter: responses stranded under a superseded cache parameter, and
    truncated ones, which are unmeasured rather than wrong.
    """
    from llm_routing import routable

    verdicts = routable.real_verdicts(tasks, ladder)
    out = {}
    for task_id, row in verdicts.items():
        if "cheap" not in row or "expensive" not in row:
            # Needs both rungs to be classified at all. routable drops these too.
            continue
        out[task_id] = CELLS[(bool(row["cheap"]), bool(row["expensive"]))]
    return out


def score(rows, cells, by_domain=False):
    """policy -> group -> counts, dollars and the task ids behind each bucket."""
    acc = defaultdict(lambda: defaultdict(lambda: {
        "n": 0, "correct": 0, "cost": 0.0, "truncated": 0,
        "outcomes": defaultdict(int), "ids": defaultdict(list),
        "wasted_cost": 0.0, "rescue_cost": 0.0,
    }))
    for r in rows:
        cell = cells.get(r["task_id"])
        if cell is None:
            continue
        escalated = "expensive" in (r.get("calls") or [])
        outcome, _is_mistake = outcome_of(cell, escalated, bool(r["correct"]))
        groups = ["all"] + ([r["domain"]] if by_domain else [])
        for g in groups:
            d = acc[r["policy"]][g]
            d["n"] += 1
            d["correct"] += bool(r["correct"])
            d["cost"] += r.get("cost_usd") or 0.0
            d["truncated"] += bool(r.get("truncated"))
            d["outcomes"][outcome] += 1
            d["ids"][outcome].append(r["task_id"])
            if outcome in WASTEFUL:
                # What the mistake cost. Only escalation mistakes have a price;
                # a missed rescue costs accuracy, not money.
                d["wasted_cost"] += r.get("cost_usd") or 0.0
            if outcome == "correct_rescue":
                d["rescue_cost"] += r.get("cost_usd") or 0.0
    return acc


def fmt_policy(name, d):
    o = d["outcomes"]
    n = d["n"]
    # Every task the cross-tab says only the expensive rung gets right, however
    # the policy went on to handle it.
    rescuable = (o["correct_rescue"] + o["missed_rescue"]
                 + o["rescued_without_top_rung"])
    escalations = (o["correct_rescue"] + o["wasted_escalation"]
                   + o["harmful_escalation"] + o["wasted_on_lost_cause"])
    # Numerator matches the denominator: every winnable task the policy got
    # right, by escalating OR by resampling at the cheap rung. Counting only
    # escalations here reported the oracle at 71% on a set where it got 7 of 7.
    recovered = o["correct_rescue"] + o["rescued_without_top_rung"]
    recall = recovered / rescuable if rescuable else float("nan")
    precision = o["correct_rescue"] / escalations if escalations else float("nan")
    return {
        "n": n,
        "accuracy": d["correct"] / n if n else float("nan"),
        "cost": d["cost"],
        "cost_per_task": d["cost"] / n if n else float("nan"),
        "rescue_recall": recall,
        "escalation_precision": precision,
        "wasted_cost": d["wasted_cost"],
        "truncated": d["truncated"],
        **{k: o[k] for k in ORDER},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", metavar="PATH", default=None,
                    help="results file to score. Default: runs/results.<ladder>.jsonl for the ladder in ROUTER_LADDER - there is no unsuffixed file.")
    ap.add_argument("--ladder", default=None,
                    help="cache to build the cross-tab from. Default: whichever "
                         "ladder the results file says it was measured on.")
    ap.add_argument("--by-domain", action="store_true",
                    help="also break every policy down by domain. Worth doing "
                         "whenever the halves differ in size - an aggregate over "
                         "357 code and 60 maths tasks is a code number.")
    ap.add_argument("--show-ids", metavar="OUTCOME", default=None,
                    help="list the task ids behind one outcome, e.g. missed_rescue")
    ap.add_argument("--json", metavar="PATH", default=None)
    args = ap.parse_args()

    path = Path(args.results) if args.results else RESULTS
    rows = load_results(path)

    ladders = {r.get("ladder") for r in rows}
    if len(ladders) > 1:
        sys.exit(f"{path.name} mixes ladders {sorted(ladders)}. A cross-tab is a "
                 f"property of one ladder; score them separately.")
    ladder = args.ladder or (ladders.pop() if ladders else "wide")

    # SET THE LADDER BEFORE IMPORTING models, VIA run_eval.
    #
    # `models` reads ROUTER_LADDER at module scope and builds MODELS/TIERS once.
    # Scoring results.claude.jsonl with the environment still on `wide` fails
    # models.is_reachable for every response, and the cross-tab comes back empty
    # and divides by zero. The ladder a results file was measured on is a
    # property of the file, so it is read from there rather than the environment.
    if os.environ.get("ROUTER_LADDER") != ladder:
        os.environ["ROUTER_LADDER"] = ladder
    from llm_routing import models
    from llm_routing import run_eval

    # Two guards for two ways in. `cells_from_cache` goes through
    # `routable.real_verdicts`, which names the REAL cache files explicitly and
    # so cannot be fed fabricated responses whatever the mode says - but the
    # results file it is joined against can be anything on disk, hence the first
    # check. The second is the same rule this pipeline applies everywhere: a
    # module that writes a published artefact does not run in a mode that
    # cannot measure.
    simulated = any(r.get("simulated", r.get("mode") == "mock") for r in rows)
    models.refuse_simulated_artefact("scorecard", simulated, path.name)
    models.require_measured_mode("scorecard")

    tasks = run_eval.load_tasks()
    cells = cells_from_cache(tasks, ladder)
    scored = {tid for r in rows for tid in [r["task_id"]] if tid in cells}
    unclassified = {r["task_id"] for r in rows} - cells.keys()

    print()
    print(f"scorecard: {path.name}  |  ladder {ladder}  |  "
          f"{len(scored)} tasks classified")
    if unclassified:
        print(f"  {len(unclassified)} task(s) unclassified - the cross-tab needs "
              f"BOTH rungs' greedy answers,\n  and truncated or stranded "
              f"responses are excluded as unmeasured rather than wrong.")

    cellcount = defaultdict(int)
    for tid in scored:
        cellcount[cells[tid]] += 1
    total = sum(cellcount.values())
    print(f"\n  task cells: " + "  ".join(
        f"{k}={cellcount[k]}" for k in
        ("both_ok", "routable", "both_fail", "inverted")))
    if not total:
        sys.exit(
            f"\nNo task could be classified, so there is nothing to score.\n"
            f"  The cross-tab needs both rungs' greedy answers for the "
            f"{ladder!r} ladder,\n  graded through models.is_reachable. Zero "
            f"classified usually means the cache\n  for this ladder is empty or "
            f"was recorded under superseded parameters.\n"
            f"  Check:  ROUTER_LADDER={ladder} python -m llm_routing.routable --real"
        )
    win = cellcount["routable"] + cellcount["inverted"]
    print(f"  dynamic range: {win} of {total} tasks ({win/total:.1%}) can "
          f"distinguish two routers at all.")
    print("  On the rest every policy scores identically by construction, so "
          "accuracy\n  differences can only ever come from those.")

    acc = score(rows, cells, by_domain=args.by_domain)

    groups = ["all"] + (sorted({r["domain"] for r in rows})
                        if args.by_domain else [])
    for g in groups:
        present = [(p, d[g]) for p, d in sorted(acc.items()) if d[g]["n"]]
        if not present:
            continue
        print()
        print(f"--- {g} " + "-" * (72 - len(g)))
        print(f"{'policy':<18}{'acc':>7}{'$/task':>10}"
              f"{'rescued':>9}{'no-top':>8}{'missed':>8}{'wasted':>8}"
              f"{'harmful':>9}{'recall':>8}{'prec':>7}{'$ wasted':>10}")
        print("-" * 104)
        for name, d in present:
            f = fmt_policy(name, d)
            rec = "  n/a" if f["rescue_recall"] != f["rescue_recall"] else \
                  f"{f['rescue_recall']:>6.0%}"
            pre = "  n/a" if f["escalation_precision"] != f["escalation_precision"] \
                  else f"{f['escalation_precision']:>5.0%}"
            print(f"{name:<18}{f['accuracy']:>6.1%}{f['cost_per_task']:>10.6f}"
                  f"{f['correct_rescue']:>9}{f['rescued_without_top_rung']:>8}"
                  f"{f['missed_rescue']:>8}"
                  f"{f['wasted_escalation']:>8}{f['harmful_escalation']:>9}"
                  f"{rec:>8}{pre:>7}{f['wasted_cost']:>10.4f}")

    print()
    print("  rescued  = routable task, escalated        (the only way to win)")
    print("  no-top   = routable/both_fail task, got it right WITHOUT the top")
    print("             rung - cheap self-consistency, or a middle rung")
    print("  missed   = routable task, stayed cheap and got it wrong")
    print("  wasted   = both_ok task, escalated         (paid, changed nothing)")
    print("  harmful  = inverted task, escalated        (paid to get it wrong)")
    print("  recall   = (rescued + no-top) / routable   (of what was winnable)")
    print("  prec     = rescued / all escalations       (of what it paid for)")
    print("  $ wasted = spend on escalations that could not improve the answer")

    # The buckets partition the tasks, so they must reproduce the accuracy
    # run_eval reported. If they do not, the join is wrong and every number
    # above is wrong with it - which is exactly the kind of silent
    # disagreement this repo has been bitten by before.
    bad = []
    for name, d in sorted(acc.items()):
        o = d["all"]["outcomes"]
        right = sum(o[k] for k in CORRECT_OUTCOMES)
        if right != d["all"]["correct"]:
            bad.append(f"{name}: buckets say {right} correct, rows say "
                       f"{d['all']['correct']}")
    print()
    if bad:
        print("!! BUCKETS DO NOT RECONCILE WITH THE RESULT ROWS:")
        for line in bad:
            print(f"   {line}")
        print("   Do not quote anything above until this is resolved.")
    else:
        print("  reconciled: every policy's correct-outcome buckets sum to the "
              "accuracy\n  its result rows report.")

    if args.show_ids:
        print()
        for name, d in sorted(acc.items()):
            ids = d["all"]["ids"].get(args.show_ids) or []
            if ids:
                print(f"{name} {args.show_ids} ({len(ids)}): {', '.join(sorted(ids))}")

    if args.json:
        out = {
            "results_file": path.name, "ladder": ladder,
            "simulated": simulated, "cells": dict(cellcount),
            "policies": {
                p: {g: fmt_policy(p, d[g]) for g in d if d[g]["n"]}
                for p, d in acc.items()
            },
            "ids": {p: {g: {k: sorted(v) for k, v in d[g]["ids"].items()}
                        for g in d if d[g]["n"]}
                    for p, d in acc.items()},
        }
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
