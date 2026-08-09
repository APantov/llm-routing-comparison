"""Routable-fraction analysis: is there anything for a router to decide?

Every routing result in this repo is a comparison between policies that choose
which rung to call. That comparison can only carry information on tasks where the
choice CHANGES THE OUTCOME. Cross-tabulate the cheap rung's verdict against the
expensive rung's verdict and the task set falls into four cells:

    both_ok      cheap right, expensive right   -> any router scores 1. Free.
    routable     cheap wrong, expensive right   -> the ONLY cell routing can win
    both_fail    cheap wrong, expensive wrong   -> any router scores 0. Hopeless.
    inverted     cheap right, expensive wrong   -> routing UP loses. Noise, or not.

`both_ok` and `both_fail` are ties by construction: the best conceivable router
and the worst conceivable router score identically on them. So the entire
accuracy dynamic range of the experiment is

    ceiling = |routable| + |inverted|

and if that is two points, no sample size and no router cleverness produces a
publishable accuracy result on this task set. This script measures it.

    python3 routable.py                       # mock, current ROUTER_LADDER
    python3 routable.py --ladders all         # mock, all three ladders
    python3 routable.py --real                # grade the real cached responses
    python3 routable.py --taskset pool.jsonl  # any candidate task set

Mock mode costs nothing. --real reads `cache/raw_calls.<ladder>.jsonl` and grades
what is already on disk; it never calls a model, so it also costs nothing.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Target band for the routable fraction.
#
# NOT the same quantity as the pilot gate's failure rate, and the difference is
# the point of this file. The gate measures P(cheap fails); this measures
# P(cheap fails AND expensive succeeds). The first is an upper bound on the
# second, and the gap between them is everything the expensive rung cannot fix.
#
# 0.15 floor: below this the whole experiment has under 15 points of dynamic
# range, so a 5-point difference between routers needs n in the thousands.
# 0.45 ceiling: above this the cheap rung is failing so often that always_cheap
# is not a serious baseline and the interesting comparison stops being routing.
ROUTABLE_FLOOR = 0.15
ROUTABLE_CEILING = 0.45


def wilson(k, n, z=1.96):
    """Wilson score interval. Behaves at k=0 and k=n, where Wald does not."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar on the discordant pair (b, c).

    b = routable (cheap wrong, expensive right), c = inverted. Under the null
    that the two rungs are equally accurate, b ~ Binomial(b + c, 0.5). This is
    the significance test for `always_expensive` beating `always_cheap`, which
    is the strongest accuracy claim the task set could ever support.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_tasks(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def crosstab(verdicts):
    """verdicts: list of (cheap_ok, exp_ok). Returns the four counts."""
    c = Counter(verdicts)
    return {
        "both_ok": c[(True, True)],
        "routable": c[(False, True)],
        "both_fail": c[(False, False)],
        "inverted": c[(True, False)],
    }


def summarise(cells, n_label=""):
    n = sum(cells.values())
    if n == 0:
        return None
    routable = cells["routable"]
    inverted = cells["inverted"]
    lo, hi = wilson(routable, n)
    clo, chi = wilson(routable + inverted, n)
    return {
        "n": n,
        "cells": cells,
        "cheap_acc": (cells["both_ok"] + inverted) / n,
        "exp_acc": (cells["both_ok"] + routable) / n,
        "cheap_fail_rate": (cells["routable"] + cells["both_fail"]) / n,
        "routable_frac": routable / n,
        "routable_ci": (lo, hi),
        "inverted_frac": inverted / n,
        "ceiling": (routable + inverted) / n,
        "ceiling_ci": (clo, chi),
        # Of the tasks the cheap rung gets wrong, the share the expensive rung
        # actually rescues. This is the number that says whether "make it harder"
        # helps: if it is low, harder tasks add both_fail, not routable.
        "rescue_rate": routable / (routable + cells["both_fail"]) if (routable + cells["both_fail"]) else float("nan"),
        "mcnemar_p": mcnemar_exact(routable, inverted),
        "label": n_label,
    }


def fmt(s):
    c = s["cells"]
    lo, hi = s["routable_ci"]
    return (
        f"n={s['n']:<4} "
        f"both_ok={c['both_ok']:<4} routable={c['routable']:<4} "
        f"both_fail={c['both_fail']:<4} inverted={c['inverted']:<3} | "
        f"cheap={s['cheap_acc']:6.1%} exp={s['exp_acc']:6.1%} | "
        f"routable={s['routable_frac']:6.1%} [{lo:.1%},{hi:.1%}] | "
        f"ceiling={s['ceiling']:6.1%} | rescue={s['rescue_rate']:6.1%} | "
        f"McNemar p={s['mcnemar_p']:.3f}"
    )


# ---------------------------------------------------------------------------
# Mock path: the mock's verdict is a pure function of (task, tier), so this
# needs no run_eval and no cache priming.
# ---------------------------------------------------------------------------
def mock_verdicts(tasks, ladder):
    os.environ["ROUTER_LADDER"] = ladder
    os.environ["ROUTER_MODE"] = "mock"
    for mod in ("models", "policies", "response_cache"):
        sys.modules.pop(mod, None)
    import models
    from graders import grade

    out = {}
    for t in tasks:
        row = {}
        for tier in models.TIERS:
            r = models.call(tier, t)
            row[tier] = grade(t, r.text)
        out[t["id"]] = row
    return out


# ---------------------------------------------------------------------------
# Real path: grade what the paid run already put on disk. No calls.
# ---------------------------------------------------------------------------
def real_verdicts(tasks, ladder):
    import models
    from graders import grade

    path = HERE / "cache" / f"raw_calls.{ladder}.jsonl"
    if not path.exists():
        return {}
    by_id = {t["id"]: t for t in tasks}
    out = defaultdict(dict)
    truncated = []
    orphans = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        # Greedy answers only: temperature 0, sample 0. The temperature-0.8
        # self-consistency samples are a different action and belong to the
        # cascade, not to the tier-choice question this file asks.
        if d.get("kind") != "answer" or d.get("temperature") not in (0, 0.0):
            continue
        if d.get("sample_idx") not in (0, None):
            continue
        task = by_id.get(d["task_id"])
        if task is None:
            continue
        if not models.is_reachable(d, task):
            # An orphan from a superseded parameter - see models.is_reachable.
            # This function reads the raw file rather than going through the
            # cache, so without this check it grades responses the experiment
            # itself can never serve, and the LAST such row on disk silently
            # wins.
            orphans += 1
            continue
        if models.is_truncated(d):
            # LEFT UNMEASURED rather than graded False. A response cut off at
            # max_tokens never reached its \boxed{}, so the grader would score
            # it wrong for a reason that is not about the model - and a wrong
            # CHEAP verdict here puts the task in `routable`, inflating the
            # headline this file computes.
            #
            # Concretely, before this: math-96's cheap draw was truncated, so
            # the cross-tab called it routable. Ten fresh cheap draws get it
            # right 10 times out of 10. It was never a routing opportunity.
            #
            # Dropping the tier drops the TASK, because crosstab needs both
            # rungs. That is the correct arithmetic: an unmeasured pair cannot
            # be classified into any of the four cells.
            truncated.append((d["task_id"], d["tier"]))
            continue
        out[d["task_id"]][d["tier"]] = grade(task, d["text"])
    if orphans:
        print(f"  {orphans} stranded response(s) skipped - recorded under a "
              f"parameter the cache key has since moved past, so the "
              f"experiment cannot serve them.")
    if truncated:
        print(f"  !! {len(truncated)} greedy response(s) hit max_tokens and are "
              f"UNMEASURED, not failures.")
        for task_id, tier in sorted(truncated):
            print(f"     {task_id:<16} {tier:<10} dropped from the cross-tab")
        print(f"     n falls by that many pairs. Raising models.MAX_TOKENS "
              f"re-charges every\n     cached response (SHIP_PLAN.md section "
              f"1), so the task is excluded instead.")
    return out


def report(verdicts, tasks, header, lo_tier="cheap", hi_tier="expensive"):
    by_id = {t["id"]: t for t in tasks}
    print(f"\n=== {header} ===")
    groups = defaultdict(list)
    for tid, row in verdicts.items():
        if lo_tier not in row or hi_tier not in row:
            continue
        pair = (row[lo_tier], row[hi_tier])
        groups["all"].append(pair)
        groups[by_id[tid]["domain"]].append(pair)
        lvl = (by_id[tid].get("predict_features") or {}).get("level")
        if lvl is not None:
            groups[f"math level {lvl}"].append(pair)

    out = {}
    order = ["all"] + sorted(k for k in groups if k != "all")
    for key in order:
        s = summarise(crosstab(groups[key]), key)
        if s:
            out[key] = s
            print(f"  {key:<16} {fmt(s)}")

    review_queue(verdicts, by_id, lo_tier, hi_tier)
    return out


def review_queue(verdicts, by_id, lo_tier="cheap", hi_tier="expensive"):
    r"""Name the both_fail tasks so a human looks at them.

    Every both_fail task is one of two things, and the cross-tab cannot tell
    them apart: a genuinely hard task, or a task whose expected answer is not
    derivable from its prompt. On 8 August 2026 all four both_fail code tasks
    turned out to be the second kind, and because they were also ALL of
    always_expensive's failures they were setting the ceiling for every policy
    in the project. Nobody looked, because "both models failed" reads as
    "hard".

    So this is not a diagnostic - it is a queue. A task that lands here is
    unmeasured until somebody runs a textbook-correct solution against its
    tests and decides which kind it is. If unpassable, add it to
    build_taskset.QUARANTINED with the evidence.
    """
    failures = sorted(
        tid for tid, row in verdicts.items()
        if lo_tier in row and hi_tier in row
        and not row[lo_tier] and not row[hi_tier]
    )
    if not failures:
        print("\n  both_fail review queue: empty - no task defeated both rungs.")
        return []
    print(f"\n  !! both_fail review queue: {len(failures)} task(s) defeated BOTH rungs.")
    print("     Each is either genuinely hard or unpassable-by-spec. The cross-tab")
    print("     cannot tell, and an unpassable task silently caps every policy.")
    for tid in failures:
        print(f"       {tid:<16} {by_id[tid]['domain']}")
    print("     Check each against a textbook-correct solution; if unpassable, add")
    print("     it to build_taskset.QUARANTINED with the evidence.")
    return failures


def verdict_line(s):
    lo, hi = s["routable_ci"]
    if hi < ROUTABLE_FLOOR:
        return f"TOO EASY - routable fraction is below {ROUTABLE_FLOOR:.0%} and the CI does not reach it"
    if lo > ROUTABLE_CEILING:
        return f"TOO HARD - routable fraction is above {ROUTABLE_CEILING:.0%} and the CI does not reach it"
    if lo >= ROUTABLE_FLOOR and hi <= ROUTABLE_CEILING:
        return f"IN BAND - routable fraction sits inside [{ROUTABLE_FLOOR:.0%}, {ROUTABLE_CEILING:.0%}]"
    return f"UNRESOLVED - CI straddles a boundary of [{ROUTABLE_FLOOR:.0%}, {ROUTABLE_CEILING:.0%}]; need more tasks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskset", default="taskset.jsonl")
    ap.add_argument("--ladders", default=os.environ.get("ROUTER_LADDER", "claude"))
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    tasks = load_tasks(HERE / args.taskset)
    ladders = ["claude", "deepseek", "wide"] if args.ladders == "all" else args.ladders.split(",")

    dump = {}
    for lad in ladders:
        if args.real:
            os.environ["ROUTER_LADDER"] = lad
            v = real_verdicts(tasks, lad)
            if not v:
                print(f"\n=== REAL, ladder={lad} === no cached responses")
                continue
            dump[f"real:{lad}"] = report(v, tasks, f"REAL (cached responses), ladder={lad}")
        else:
            v = mock_verdicts(tasks, lad)
            dump[f"mock:{lad}"] = report(v, tasks, f"MOCK - SIMULATED, NOT MEASURED, ladder={lad}")

    for k, v in dump.items():
        if "all" in v:
            print(f"\n{k:<16} {verdict_line(v['all'])}")

    if args.json:
        Path(args.json).write_text(json.dumps(dump, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
