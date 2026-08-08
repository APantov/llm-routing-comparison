"""Paired significance testing over results.jsonl.

WHY THIS EXISTS
---------------
At this n, one task is roughly two percentage points, and the gaps between
adjacent policies in the report are of that order. Without a test, "policy A beat
policy B by two points" is not a claim about routing - it is a claim about which
tasks happened to be sampled.

PAIRED, not unpaired, and that is the whole point of the response cache. Every
policy answered the SAME tasks using the SAME cached model responses, so the
comparison can condition on the task and throw away the between-task variance,
which is much larger than the between-policy variance. An unpaired t-test on these
numbers would be both wrong and far less powerful.

Two tests, because they answer different questions:

  McNemar (exact)    Did A and B disagree in an asymmetric way? Looks only at
                     tasks where exactly one was correct, which is exactly the
                     information a paired accuracy comparison contains. Exact
                     binomial rather than the chi-square approximation, because
                     the discordant counts here are small and the approximation
                     is unreliable below about 25.

  Paired bootstrap   How large is the difference, with an interval? Resamples
                     tasks with replacement and recomputes the per-task
                     difference, which also handles COST, where McNemar does not
                     apply because cost is continuous.

Reporting an interval matters more than reporting a p-value: "between -1 and +9
points" and "+4 points, p=0.09" are the same finding, but only the first makes the
uncertainty impossible to overlook.

    python3 stats.py                          # every pair worth comparing
    python3 stats.py --vs cascade             # everything against one policy
    python3 stats.py --bootstrap 20000
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results.jsonl"

# Resamples for the bootstrap. 10k is enough for a 95% interval to be stable to
# about a tenth of a point, and it costs no model calls at all - this module reads
# a file and never touches models.py.
BOOTSTRAP_N = 10000

# Fixed so a reported interval is reproducible. A bootstrap interval that moves
# when you rerun it is not a result.
BOOTSTRAP_SEED = 0

# The comparisons worth printing by default. Every pair is a multiple-comparisons
# problem, so the default is a short list of pre-registered questions rather than
# all 78 pairs mined for whichever came out significant.
#
# Re-based on 8 August 2026: `predictive` was deleted, and five of the eight
# pairs named it. `llm_router` takes its place as the predictive family's
# representative, because it is the member that runs in every mode.
#
# ONLY THE FIRST PAIR IS COST-MATCHED. random_matched is calibrated to
# llm_router's escalation rate, so that one comparison holds spend roughly fixed
# and isolates skill. Every other pair compares policies at DIFFERENT spending
# levels - routellm now runs at a fixed threshold, and the cascades spend what
# they spend. McNemar is blind to cost, so the accuracy verdict alone is not a
# recommendation: read it next to the d_cost column, which this module prints for
# every pair. run_eval's routing-skill table is the cost-adjusted view.
DEFAULT_PAIRS = [
    # Does the LLM-as-router beat a coin flip at its own spend? The null
    # hypothesis, and the one genuinely cost-matched comparison here.
    ("llm_router", "random_matched"),
    # Does a learned router beat a coin flip at all?
    ("routellm", "random_matched"),
    # Learned router against LLM-as-router: the predictive family, internally.
    ("routellm", "llm_router"),
    # The project's headline architecture comparison, cascading vs predictive.
    ("cascade", "llm_router"),
    # Does verifying beat simply always paying for the best model?
    ("cascade", "always_expensive"),
    # Does the unified policy beat each of the two it unifies?
    ("cascade_routing", "cascade"),
    ("cascade_routing", "llm_router"),
    # Is routing at all better than not routing? The floor.
    ("llm_router", "always_cheap"),
]


def load(path=RESULTS):
    if not path.exists():
        sys.exit(
            f"{path.name} not found. Generate it first:\n"
            f"  python3 run_eval.py"
        )
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if not rows:
        sys.exit(f"{path.name} is empty")
    return rows


def index(rows):
    """policy -> {task_id: row}, keeping only policies with full coverage.

    Policies defined on a subset of tasks, like cascade_degraded, cannot be paired
    against a full-coverage policy: the pairing would silently compare different
    task sets. They are dropped with a note rather than compared unsoundly.
    """
    by = defaultdict(dict)
    for r in rows:
        by[r["policy"]][r["task_id"]] = r
    n_max = max(len(v) for v in by.values())
    full = {k: v for k, v in by.items() if len(v) == n_max}
    partial = {k: len(v) for k, v in by.items() if len(v) != n_max}
    return full, partial, n_max


def mcnemar_exact(a_correct, b_correct):
    """Exact two-sided McNemar. Returns (b, c, p).

    b = A right where B was wrong; c = A wrong where B was right. Tasks where both
    agreed carry no information about which policy is better and are correctly
    ignored - that is the point of the test, not a limitation of it.

    Under the null that discordant pairs fall either way with equal probability,
    min(b, c) ~ Binomial(b + c, 0.5).
    """
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * tail)


def paired_bootstrap(pairs, stat, n_resamples=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Percentile interval for the mean paired difference of `stat`.

    Resamples TASKS, not observations, which is what keeps the pairing intact:
    each draw takes both policies' results for the same task or neither.
    """
    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return None, None, None
    diffs = [stat(a) - stat(b) for a, b in pairs]
    point = sum(diffs) / n
    means = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int(0.025 * (n_resamples - 1))]
    hi = means[int(0.975 * (n_resamples - 1))]
    return point, lo, hi


def compare(a_name, b_name, full, n_resamples):
    a, b = full[a_name], full[b_name]
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    pairs = [(a[i], b[i]) for i in ids]
    a_ok = [bool(a[i]["correct"]) for i in ids]
    b_ok = [bool(b[i]["correct"]) for i in ids]

    nb, nc, p = mcnemar_exact(a_ok, b_ok)
    d_acc, acc_lo, acc_hi = paired_bootstrap(
        pairs, lambda r: float(bool(r["correct"])), n_resamples)
    d_cost, cost_lo, cost_hi = paired_bootstrap(
        pairs, lambda r: r["cost_usd"], n_resamples)

    return {
        "a": a_name, "b": b_name, "n": len(ids),
        "acc_a": sum(a_ok) / len(ids), "acc_b": sum(b_ok) / len(ids),
        "discordant_a": nb, "discordant_b": nc, "p": p,
        "d_acc": d_acc, "acc_lo": acc_lo, "acc_hi": acc_hi,
        "d_cost": d_cost, "cost_lo": cost_lo, "cost_hi": cost_hi,
    }


def verdict(r):
    """A one-line reading, written to be hard to over-claim from.

    Always mentions cost. Since 8 August 2026 only one pre-registered pair is
    cost-matched, so "A is more accurate" on its own is an invitation to quote a
    win that was bought rather than earned. If A wins on accuracy while spending
    more, the price is part of the finding and is stated in the same sentence.
    """
    sig = r["p"] < 0.05
    crosses_zero = r["acc_lo"] <= 0 <= r["acc_hi"]
    d_cost = r["d_cost"]
    # "Free" means the cost interval spans zero: no detectable price difference.
    cost_free = r["cost_lo"] <= 0 <= r["cost_hi"]
    if cost_free:
        price = "at no detectable cost difference"
    elif d_cost > 0:
        price = f"while A spends +${d_cost:.6f}/task"
    else:
        price = f"while A spends -${abs(d_cost):.6f}/task"

    if not sig or crosses_zero:
        return f"no detectable accuracy difference at this n, {price}"
    better = "A better" if r["d_acc"] > 0 else "B better"
    return f"accuracy difference is detectable ({better}), {price}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vs", help="compare every policy against this one")
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N)
    ap.add_argument("--all-pairs", action="store_true",
                    help="every pair. Read the multiple-comparisons warning first.")
    args = ap.parse_args()

    rows = load()
    full, partial, n = index(rows)

    simulated = any(r.get("simulated", r.get("mode") == "mock") for r in rows)
    print()
    if simulated:
        print("### SIMULATED INPUT - these tests are arithmetic on FABRICATED ###")
        print("### outcomes. The p-values are real; what they are computed    ###")
        print("### from is not. Nothing here measures a model.                ###")
    else:
        print("### paired tests on measured results ###")
    print()
    print(f"n={n} tasks, {len(full)} policies with full coverage, "
          f"{args.bootstrap} bootstrap resamples (seed {BOOTSTRAP_SEED})")
    if partial:
        print("excluded, defined on a subset of tasks so they cannot be paired:")
        for k, c in sorted(partial.items()):
            print(f"  {k} (n={c})")

    if args.all_pairs:
        names = sorted(full)
        todo = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]]
        print()
        print(f"!! ALL {len(todo)} PAIRS. At p<0.05 roughly one in twenty of these")
        print("   is expected to look significant with nothing going on. Do not")
        print("   pick the winners out of this table; use the pre-registered list.")
    elif args.vs:
        if args.vs not in full:
            sys.exit(f"{args.vs!r} not in results. Available: {', '.join(sorted(full))}")
        todo = [(x, args.vs) for x in sorted(full) if x != args.vs]
    else:
        todo = [(x, y) for x, y in DEFAULT_PAIRS if x in full and y in full]
        skipped = [(x, y) for x, y in DEFAULT_PAIRS if x not in full or y not in full]
        if skipped:
            print()
            print("pre-registered comparisons not available in this results file:")
            for x, y in skipped:
                print(f"  {x} vs {y}")

    print()
    print("A vs B. d_acc is A minus B in accuracy points, with a 95% paired")
    print("bootstrap interval. b/c are the discordant counts McNemar reads.")
    print()
    print(f"{'A':<17}{'B':<17}{'acc A':>7}{'acc B':>7}{'d_acc':>8}"
          f"{'  95% CI':>17}{'b/c':>8}{'p':>8}{'d_$/task':>12}")
    print("-" * 108)
    out = []
    for x, y in todo:
        r = compare(x, y, full, args.bootstrap)
        if r is None:
            continue
        out.append(r)
        ci = f"[{r['acc_lo']:+.1%},{r['acc_hi']:+.1%}]"
        star = "*" if r["p"] < 0.05 else " "
        print(f"{x:<17}{y:<17}{r['acc_a']:>6.1%} {r['acc_b']:>6.1%} "
              f"{r['d_acc']:>+7.1%} {ci:>16} "
              f"{r['discordant_a']}/{r['discordant_b']:<5} {r['p']:>7.3f}{star}"
              f"{r['d_cost']:>+12.6f}")

    print()
    print("* p < 0.05 on the exact McNemar test.")
    print("d_$/task is carried in this table on purpose: only the first pair is")
    print("cost-matched, so an accuracy win at a higher price is not a free win.")
    print()
    print("cost differences, same pairing (dollars per task, A minus B):")
    print(f"{'A':<17}{'B':<17}{'d_cost':>11}{'  95% CI':>26}")
    print("-" * 72)
    for r in out:
        ci = f"[{r['cost_lo']:+.6f}, {r['cost_hi']:+.6f}]"
        print(f"{r['a']:<17}{r['b']:<17}{r['d_cost']:>+11.6f} {ci:>25}")

    print()
    print("readings:")
    for r in out:
        print(f"  {r['a']} vs {r['b']}: {verdict(r)}")

    detectable = [r for r in out if r["p"] < 0.05 and not (r["acc_lo"] <= 0 <= r["acc_hi"])]
    print()
    print(f"{len(detectable)} of {len(out)} comparisons show a detectable accuracy")
    print(f"difference at n={n}. The rest are not ties - they are UNRESOLVED, which")
    print("is a statement about the sample size rather than about the policies.")


if __name__ == "__main__":
    main()
