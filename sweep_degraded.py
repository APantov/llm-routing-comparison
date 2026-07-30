"""The degradation sweep: cascade cost and accuracy against verifier quality.

THIS IS THE EXPERIMENT. Everything else in the repo is scaffolding for it.

The project's stated contribution is that verifier quality is the manipulated
variable. With only the two natural verifiers there are two levels of that
variable, and they are perfectly confounded with task domain:

    perfect verifier | code | MBPP    | run asserts | $0 to verify
    proxy   verifier | math | MATH500 | exact match | k-1 extra cheap calls

Five things differ between those rows, so "the code cascade beats
always_expensive and the math cascade does not" cannot be attributed to verifier
quality. It can only be attributed to "code is different from math", which is a
between-subjects comparison with no controls and the most attackable claim the
project could make.

This sweep fixes that by moving verifier quality WITHIN the code domain. Same
tasks, same two models, same grader, same prompts, same cost structure - only
verify_code's fidelity moves, from perfect to a coin flip. The output is a curve
rather than two points, and two points are not a trend.

    p = 0.00   verifier is verify_code, unchanged. Identical to `cascade`.
    p = 1.00   verifier ignores the tests entirely and flips a coin. Zero
               information, AUC 0.5.

p is the probability the verifier ignores the test result, so the effective error
rate is p/2 and the effective AUC is roughly 1 - p/2.

FREE, because of the response cache: every point in the sweep reuses the same
cheap and expensive responses. The sweep makes zero additional model calls after
the first point, in mock mode and in real mode alike. That property is the entire
reason the cache had to exist before any money was spent.

    python3 sweep_degraded.py                      # mock
    ROUTER_MODE=replay python3 sweep_degraded.py   # from a paid run, free
"""

import argparse
import sys
from pathlib import Path

import models
import policies
import run_eval

HERE = Path(__file__).parent
OUT = HERE / "sweep_degraded.jsonl"

# The corruption levels. Spaced to resolve the low end, because that is where the
# break-even lives: a verifier does not have to be very good before cascading
# stops paying, and a linear grid would put most of its points in the region where
# the answer is already obvious.
SWEEP = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def _mean(xs):
    return sum(xs) / len(xs)


def _sd(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def run_replicate(tasks, p, seed):
    """One corruption draw at one corruption level. Returns stats and rows."""
    policies.VERIFIER_CORRUPTION = p
    policies.VERIFIER_CORRUPTION_SEED = seed
    rows = []
    for t in tasks:
        res = policies.policy_cascade_degraded(t)
        rows.append({
            "task_id": res.task_id, "domain": t["domain"],
            "difficulty": t.get("difficulty_proxy"),
            "policy": "cascade_degraded", "correct": res.correct,
            "cost_usd": res.cost_usd, "latency_s": res.latency_s,
            "escalated": res.escalated, "calls": res.calls,
            **run_eval.provenance(),
        })
    n = len(rows)
    return {
        "verifier_corruption": p,
        "corruption_seed": seed,
        "n": n,
        "accuracy": sum(r["correct"] for r in rows) / n,
        "cost_per_task": sum(r["cost_usd"] for r in rows) / n,
        "latency_per_task": sum(r["latency_s"] for r in rows) / n,
        "escalation_rate": sum(r["escalated"] for r in rows) / n,
    }, rows


def run_point(tasks, p, repeats):
    """One sweep point, averaged over `repeats` independent corruption draws.

    Averaging is not cosmetic smoothing. At p > 0 the verifier's verdict is a
    random variable, so a single draw at this n estimates the mean with a standard
    error of several accuracy points - enough to invert two adjacent points of a
    curve whose true shape is monotonic. Repeating the draw and reporting the
    spread separates the part of the curve that is mechanism from the part that
    is luck.

    Only seed 0's per-task rows are written out, because those are the ones the
    paired statistics need to line up with results.jsonl. The other draws
    contribute to the mean and the spread, then are discarded.
    """
    reps = [run_replicate(tasks, p, s) for s in range(repeats)]
    stats = [s for s, _ in reps]
    agg = {
        "verifier_corruption": p,
        "n": stats[0]["n"],
        "repeats": repeats,
    }
    for field in ("accuracy", "cost_per_task", "latency_per_task", "escalation_rate"):
        vals = [s[field] for s in stats]
        agg[field] = _mean(vals)
        agg[field + "_sd"] = _sd(vals)
        agg[field + "_seed0"] = stats[0][field]
    return agg, reps[0][1]


def reference_points(tasks):
    """The two policies the cascade has to be judged against, on the same tasks.

    always_expensive is the accuracy ceiling the cascade is trying to reach at
    lower cost; always_cheap is the cost floor it is trying to beat on accuracy.
    Without both drawn on the same axes, "cascade cost saving" has no denominator.
    """
    out = {}
    for name in ("always_cheap", "always_expensive"):
        rs = [policies.POLICIES[name](t) for t in tasks]
        out[name] = {
            "accuracy": sum(r.correct for r in rs) / len(rs),
            "cost_per_task": sum(r.cost_usd for r in rs) / len(rs),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="code", choices=["code", "math"],
                    help="code by default - the point is to hold the domain fixed")
    ap.add_argument("--repeats", type=int, default=200,
                    help="independent corruption draws per level. Free: every "
                         "model response is a cache hit.")
    args = ap.parse_args()

    tasks = run_eval.load_tasks(domain=args.domain)
    print(run_eval.banner(), file=sys.stderr)
    for line in models.ladder_summary():
        print(line, file=sys.stderr)
    print(f"sweeping verifier corruption over {SWEEP} on {len(tasks)} {args.domain} tasks, "
          f"{args.repeats} draws each", file=sys.stderr)

    models.reset_call_stats()
    ref = reference_points(tasks)
    after_ref = models.call_stats["backend"]

    points, all_rows = [], []
    for p in SWEEP:
        stats, rows = run_point(tasks, p, args.repeats)
        points.append(stats)
        all_rows.extend(rows)
    # Leave the module as we found it: these are globals, and a later import in
    # the same process must not inherit a swept value.
    policies.VERIFIER_CORRUPTION = 0.0
    policies.VERIFIER_CORRUPTION_SEED = 0

    run_eval.write_jsonl(OUT, all_rows)

    tag = run_eval.tag()
    print()
    print(run_eval.banner())
    print()
    print(tag)
    print(f"cascade_degraded on {args.domain}, n={len(tasks)}, "
          f"mean of {points[0]['repeats']} corruption draws "
          f"(verifier quality is the ONLY thing that varies)")
    print(f"{'corrupt p':>10} {'eff.AUC':>8} {'acc':>15} {'cost/task':>19} "
          f"{'escal':>8} {'$/correct':>11}")
    print("-" * 76)
    for s in points:
        auc = 1 - s["verifier_corruption"] / 2
        per_correct = s["cost_per_task"] / s["accuracy"] if s["accuracy"] else float("nan")
        print(f"{s['verifier_corruption']:>10.2f} {auc:>8.3f} "
              f"{s['accuracy']:>8.1%} +-{s['accuracy_sd']:<4.1%} "
              f"{s['cost_per_task']:>12.6f} +-{s['cost_per_task_sd']:<5.6f} "
              f"{s['escalation_rate']:>7.1%} {per_correct:>11.6f}")

    print()
    print("reference policies on the same tasks:")
    for name, r in ref.items():
        print(f"  {name:<18} acc {r['accuracy']:>6.1%}   cost/task {r['cost_per_task']:>10.6f}"
              f"   $/correct {r['cost_per_task'] / r['accuracy']:>10.6f}")

    # Break-even: the corruption level past which the cascade no longer beats
    # always_expensive. This is the engineering answer the whole sweep exists to
    # produce - the minimum verifier quality at which cascading is worth building.
    ae = ref["always_expensive"]
    ae_per_correct = ae["cost_per_task"] / ae["accuracy"]
    print()
    print("break-even, two ways, because they give different answers:")

    # (a) cost per correct answer. The number a production reader wants, and the
    #     one that can mislead: a policy that is simply less accurate can still
    #     win on $/correct by being much cheaper.
    beaten = [s for s in points
              if s["accuracy"] and s["cost_per_task"] / s["accuracy"] <= ae_per_correct]
    if not beaten:
        print("  (a) $/correct : never beats always_expensive at any p tested.")
    elif len(beaten) == len(points):
        print("  (a) $/correct : beats always_expensive at EVERY p tested, down to a "
              "pure coin flip.")
        print("                  That is a fact about the price ratio, not about the")
        print("                  verifier - a coin flip still sends half the traffic cheap.")
    else:
        worst = max(s["verifier_corruption"] for s in beaten)
        print(f"  (a) $/correct : beats always_expensive up to p={worst:.2f} "
              f"(effective AUC {1 - worst / 2:.3f}).")

    # (b) matched accuracy. The comparison the project actually claims to make. A
    #     cost saving is only a saving if quality held; once accuracy drops below
    #     the expensive tier, the cascade is buying its savings with correctness
    #     and the two policies are no longer comparable on cost.
    tol = 1.0 / points[0]["n"]  # one task
    matched = [s for s in points if s["accuracy"] >= ae["accuracy"] - tol]
    if not matched:
        print(f"  (b) matched acc: never reaches always_expensive's {ae['accuracy']:.1%} "
              f"(+-1 task) at any p tested.")
    else:
        worst = max(s["verifier_corruption"] for s in matched)
        s = next(x for x in points if x["verifier_corruption"] == worst)
        saving = 1 - s["cost_per_task"] / ae["cost_per_task"]
        print(f"  (b) matched acc: holds always_expensive's accuracy (within one task) "
              f"up to p={worst:.2f},")
        print(f"                  where it is {saving:.0%} cheaper. Past that the cascade is")
        print(f"                  paying for its savings in correctness, so (a) flatters it.")

    # Monotonicity is a claim about the mechanism, so check it rather than eyeball
    # the table. Accuracy should fall and escalation should rise as the verifier
    # degrades; at this n a one-task wobble is expected and is not evidence
    # against the mechanism.
    print()
    print(f"monotonicity check over {len(points)} points (1 task = {1 / len(tasks):.1%}):")
    for field, direction, want in (
        ("accuracy", "falls", "down"),
        ("cost_per_task", "rises", "up"),
        ("escalation_rate", "rises", "up"),
    ):
        vals = [s[field] for s in points]
        one = [s[field + "_seed0"] for s in points]
        bad = sum(1 for a, b in zip(vals, vals[1:])
                  if (b > a + 1e-12 if want == "down" else b < a - 1e-12))
        bad1 = sum(1 for a, b in zip(one, one[1:])
                   if (b > a + 1e-12 if want == "down" else b < a - 1e-12))
        verdict = "yes" if bad == 0 else f"NO, {bad} inversion(s)"
        fmt = (lambda v: f"{v:.1%}") if field != "cost_per_task" else (lambda v: f"${v:.6f}")
        print(f"  {field:<16} {direction} monotonically: {verdict:<20} "
              f"{fmt(vals[0])} -> {fmt(vals[-1])}   (single draw: "
              f"{'monotonic' if bad1 == 0 else f'{bad1} inversion(s)'})")
    print("  'single draw' is seed 0 alone - the shape one realisation would have")
    print("  shown, which is what this n buys without repeats.")

    st = models.call_stats
    print()
    print(f"model calls: {st['requested']} requested, {st['from_cache']} from cache, "
          f"{st['backend']} reached a backend "
          f"({after_ref} of those were the reference policies)")
    print(f"  the {len(SWEEP)} sweep points cost {st['backend'] - after_ref} extra model calls")
    print(f"\nwrote {len(all_rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
