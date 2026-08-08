#!/usr/bin/env python3
"""Given a budget, is another cheap draw better than one expensive call?

A cascade has exactly one move when verification fails: climb the ladder. That
is a choice, not a law, and it is not obviously the right one when the top rung
costs 68x the bottom - for the price of one Opus call you could take
SIXTY-EIGHT more DeepSeek draws. *Resample or Reroute?* (arXiv:2607.08665)
frames the two as competing uses of the same budget and reports cascades losing
by 22-31% on saturated tasks.

This answers it on real data, for free. The 7 August redraw left ten greedy
draws of both rungs for each of the 21 decisive tasks on disk, so every number
below is read from `cache/raw_calls.wide.jsonl` and nothing is called.

THE ASYMMETRY THAT DECIDES IT
-----------------------------
What resampling buys depends entirely on what can pick the winner:

  code   the verifier RUNS THE TESTS. It is exact and free, so best-of-k is
         genuinely deployable: draw k times, keep any draw that passes.

  math   there is no exact check, only self-consistency. The deployable move
         is majority-vote-of-k, which is a strictly weaker thing - it recovers
         the truth only when the wrong answers scatter. When a model is
         confidently wrong the modal answer IS the wrong one, and drawing more
         makes it more confident, not more correct.

Both are reported, plus the oracle best-of-k on math as an upper bound that no
deployable router can reach. The gap between "best-of-k" and "majority-of-k" on
maths is the price of not having a real verifier - which is this repository's
central variable, priced here on real data instead of by corruption sweep.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import routable                                  # noqa: E402
from graders import extract_answer, grade        # noqa: E402

LADDER = "wide"
KS = (1, 3, 5, 7, 9)


def load_draws(ladder):
    """{task_id: {tier: {sample_idx: text}}} for greedy real answers."""
    path = REPO / "cache" / f"raw_calls.{ladder}.jsonl"
    out, costs = collections.defaultdict(lambda: collections.defaultdict(dict)), collections.defaultdict(list)
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("kind") != "answer" or d.get("mode") != "real":
            continue
        if d.get("temperature") not in (0, 0.0):
            continue
        out[d["task_id"]][d["tier"]][d.get("sample_idx") or 0] = d["text"]
        costs[d["tier"]].append(d["cost_usd"])
    unit = {t: sum(v) / len(v) for t, v in costs.items()}
    return out, unit


def majority_of_k(task, texts):
    """What self-consistency would return: the modal answer among k draws.

    Deployable on maths. Ties break toward the first draw seen, which is the
    conservative reading - it does not let the tie-break peek at the truth.
    """
    answers = [extract_answer(t) for t in texts]
    counts = collections.Counter(a for a in answers if a is not None)
    if not counts:
        return False, 0.0
    modal, n = counts.most_common(1)[0]
    text = next(t for t in texts if extract_answer(t) == modal)
    return grade(task, text), n / len(texts)


def best_of_k(task, texts):
    """Keep any draw that passes. Deployable ONLY where the check is exact."""
    return any(grade(task, t) for t in texts)


def main():
    tasks = {t["id"]: t for t in routable.load_tasks(REPO / "taskset.jsonl")}
    draws, unit = load_draws(LADDER)
    verdicts = routable.real_verdicts(list(tasks.values()), LADDER)

    decisive = [
        tid for tid, v in verdicts.items()
        if "cheap" in v and "expensive" in v and not v["cheap"]
        and len(draws.get(tid, {}).get("cheap", {})) >= max(KS)
    ]
    if not decisive:
        sys.exit("No redrawn tasks found. Run scripts/redraw_decisive.py first.")

    c, e = unit["cheap"], unit["expensive"]
    print(f"ladder {LADDER}   mean greedy call: cheap ${c:.6f}  expensive ${e:.6f}")
    print(f"one expensive call buys {e / c:.0f} cheap draws\n")

    for domain, label in (("code", "CODE - exact verifier, best-of-k is deployable"),
                          ("math", "MATH - no exact verifier, only majority-of-k is deployable")):
        ids = [t for t in decisive if tasks[t]["domain"] == domain]
        if not ids:
            continue
        print("=" * 68)
        print(f"  {label}   n={len(ids)}")
        print("=" * 68)
        header = "  strategy".ljust(28) + "solved".rjust(8) + "cost/task".rjust(12) + "  vs escalating"
        print(header)
        print("  " + "-" * (len(header) - 2))

        # Escalating: the cheap call already made, k-1 verifier samples, then
        # the expensive rung. Success is whatever the top rung actually does.
        esc_solved = sum(1 for t in ids if verdicts[t]["expensive"])
        esc_cost = 5 * c + e   # k=5 self-consistency, then escalate
        print(f"  escalate to expensive".ljust(28)
              + f"{esc_solved}/{len(ids)}".rjust(8) + f"${esc_cost:.5f}".rjust(12) + "  --")

        for k in KS:
            texts_for = lambda t: [draws[t]["cheap"][i] for i in sorted(draws[t]["cheap"])[:k]]
            cost = k * c
            if domain == "code":
                solved = sum(1 for t in ids if best_of_k(tasks[t], texts_for(t)))
                name = f"  best-of-{k} cheap (tests)"
            else:
                solved = sum(1 for t in ids if majority_of_k(tasks[t], texts_for(t))[0])
                name = f"  majority-of-{k} cheap"
            delta = f"{solved - esc_solved:+d} solved, {cost / esc_cost:.0%} of the cost"
            print(name.ljust(28) + f"{solved}/{len(ids)}".rjust(8)
                  + f"${cost:.5f}".rjust(12) + f"  {delta}")

        if domain == "math":
            for k in KS[-1:]:
                texts_for = [drs for drs in ()]
                solved = sum(1 for t in ids
                             if best_of_k(tasks[t], [draws[t]["cheap"][i]
                                                     for i in sorted(draws[t]["cheap"])[:k]]))
                print(f"  best-of-{k} cheap (ORACLE)".ljust(28)
                      + f"{solved}/{len(ids)}".rjust(8) + f"${k * c:.5f}".rjust(12)
                      + "  not deployable - no exact check exists")
        print()

    # Does agreement tell you WHICH failure resampling can fix?
    print("=" * 68)
    print("  does low agreement predict that resampling will help?")
    print("=" * 68)
    rows = []
    for t in decisive:
        if tasks[t]["domain"] != "math":
            continue
        texts = [draws[t]["cheap"][i] for i in sorted(draws[t]["cheap"])[:9]]
        ok, agree = majority_of_k(tasks[t], texts)
        n_right = sum(1 for x in texts if grade(tasks[t], x))
        rows.append((t, agree, n_right, ok))
    for t, agree, n_right, ok in sorted(rows, key=lambda r: r[1]):
        print(f"  {t:<12} modal agreement {agree:>5.0%}   {n_right}/9 draws right   "
              f"majority-of-9 {'CORRECT' if ok else 'WRONG'}")


if __name__ == "__main__":
    main()
