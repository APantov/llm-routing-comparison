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

    python -m llm_routing.routable                       # current ROUTER_LADDER
    python -m llm_routing.routable --ladders all         # all three ladders
    python -m llm_routing.routable --taskset pool.jsonl  # any candidate task set

There is one path and it is measured: `cache/raw_calls.<ladder>.jsonl` is read
and what is already on disk is graded. No model is called, so it costs nothing -
which is why the simulated cross-tab this module used to print by default was
never worth the risk of being read as a measurement. `--real` is still accepted,
and now does nothing, because it appears in published commands.
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from llm_routing import paths

# Target band for the routable fraction. NOT the pilot gate's failure rate, and
# the difference is the point of this file: the gate measures P(cheap fails),
# this measures P(cheap fails AND expensive succeeds). The gap between them is
# everything the expensive rung cannot fix.
#
# Below the 0.15 floor the experiment has under 15 points of dynamic range, so a
# 5-point difference between routers needs n in the thousands. Above the 0.45
# ceiling always_cheap is not a serious baseline and the interesting comparison
# stops being routing.
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


# `mock_verdicts` STOOD HERE, and it was the sharpest version of the problem
# this repository had with mock mode. It built the same four-cell cross-tab out
# of `models.MOCK_SKILL` and `models.MOCK_FAILURE_CORRELATION`, printed it in the
# same layout as the measured one, and ran BY DEFAULT - so `python -m
# llm_routing.routable` produced a plausible routable fraction for a ladder
# nobody had ever called. docs/RESULTS.md had to carry a bold warning that the
# `--real` flag was not optional.
#
# The measured path costs nothing either: it grades responses already on disk.
# There was never a reason to simulate this, only an order of implementation
# that outlived its usefulness.


# ---------------------------------------------------------------------------
# Real path: grade what the paid run already put on disk. No calls.
# ---------------------------------------------------------------------------
def real_verdicts(tasks, ladder):
    from llm_routing import models
    from llm_routing import response_cache
    from llm_routing.graders import grade

    # EVERY file the cache would serve this ladder from, not just its own. The
    # ladder is absent from the cache key, so a response bought for one ladder
    # serves any ladder whose rung uses the same model - which is what makes
    # three ladders affordable.
    #
    # Reading only raw_calls.<ladder>.jsonl therefore sees a smaller set than the
    # experiment does: raw_calls.claude.jsonl holds haiku and sonnet but no Opus,
    # so no task has both rungs and the cross-tab comes back empty for a fully
    # measured ladder.
    #
    # The rule, which models.is_reachable below also enforces: if the experiment
    # would not serve it, do not grade it - and if it WOULD, do not miss it.
    cache_files = [p for p in (response_cache._sibling_real_paths(ladder)
                               + [paths.CACHE / f"raw_calls.{ladder}.jsonl"])
                   if p.exists()]
    if not cache_files:
        return {}
    by_id = {t["id"]: t for t in tasks}
    out = defaultdict(dict)
    truncated = []
    orphans = 0
    # Map responses to THE REQUESTED ladder's rungs by MODEL, never by the
    # recorded `tier`: a row's tier label belongs to the ladder it was recorded
    # under, so `wide`'s "expensive" is Opus and `deepseek`'s is v4-pro. Reading
    # the label across files silently compares two different models.
    #
    # Model identity is the right key because it is what the CACHE keys on -
    # `response_cache.make_key` has never heard of a tier.
    #
    # Built from `models.LADDERS[ladder]`, the ARGUMENT, not `models.TIERS`,
    # which is whichever ladder the process was imported under. The two disagree
    # whenever a caller asks for a ladder other than the configured one.
    rung_names = models._TIER_NAMES.get(len(models.LADDERS[ladder]))
    if rung_names is None:
        raise ValueError(f"no rung names for a {len(models.LADDERS[ladder])}-rung ladder")
    tier_of_model = dict(zip(models.LADDERS[ladder], rung_names))

    lines = (line for path in cache_files for line in open(path, encoding="utf-8"))
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        tier = tier_of_model.get(d.get("model"))
        if tier is None:
            # A model that is not a rung of this ladder at all.
            continue
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
            # max_tokens never reached its \boxed{}, so grading it wrong is a
            # statement about the cap, not the model - and a wrong CHEAP verdict
            # puts the task in `routable`, inflating this file's headline.
            # math-96 was called routable that way; ten fresh cheap draws get it
            # right 10 out of 10.
            #
            # Dropping the tier drops the TASK, because crosstab needs both
            # rungs. That is the correct arithmetic: an unmeasured pair cannot be
            # classified into any of the four cells.
            truncated.append((d["task_id"], tier))
            continue
        out[d["task_id"]][tier] = grade(task, d["text"])
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
              f"re-charges every cached\n     response - see the standing "
              f"invariants in docs/ARCHITECTURE.md - so\n     the task is "
              f"excluded instead.")
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
    derivable from its prompt. At one point all four both_fail code tasks
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
    ap.add_argument("--taskset", default=str(paths.TASKSET))
    ap.add_argument("--ladders", default=os.environ.get("ROUTER_LADDER", "claude"))
    ap.add_argument("--real", action="store_true",
                    help="ACCEPTED AND IGNORED. Grading the real cached "
                         "responses is the only thing this module does now; the "
                         "flag survives because published commands pass it.")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    # `real_verdicts` names the real cache files explicitly, so this module
    # cannot be fed fabricated responses whatever the mode says. The guard is
    # here anyway, and the rule it keeps uniform is worth more than the one
    # module it is redundant in: something that writes a published artefact does
    # not run in a mode that fabricates. A reader should not have to work out
    # which analysis scripts are exceptions.
    from llm_routing import models
    models.require_measured_mode("routable")

    tasks = load_tasks(Path(args.taskset))
    ladders = ["claude", "deepseek", "wide"] if args.ladders == "all" else args.ladders.split(",")

    dump = {}
    for lad in ladders:
        os.environ["ROUTER_LADDER"] = lad
        v = real_verdicts(tasks, lad)
        if not v:
            print(f"\n=== ladder={lad} === no cached responses. This cross-tab "
                  f"is grading, not\n    calling: with nothing on disk for this "
                  f"ladder there is nothing to grade.")
            continue
        dump[f"real:{lad}"] = report(v, tasks, f"REAL (cached responses), ladder={lad}")

    for k, v in dump.items():
        if "all" in v:
            print(f"\n{k:<16} {verdict_line(v['all'])}")

    if args.json:
        Path(args.json).write_text(json.dumps(dump, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
