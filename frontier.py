"""Cost-quality frontiers, and a single number to compare them by.

WHY A CURVE AND NOT A TABLE ROW
-------------------------------
Every policy in this repo has a knob. `cascade` has an agreement threshold,
`routellm` has a score threshold, `cascade_routing` has lambda, and the random
baseline has its escalation rate. Turning any of them buys accuracy with money.

So comparing two policies at one setting each compares two arbitrary points, and
the winner can be changed by turning either knob. That is the single most common
way routing results mislead: "our router beat the cascade" usually means "our
router was tuned to spend more". run_eval's table has exactly this problem, which
is why the routing-skill column exists to partly correct for it.

The fix is standard in the routing literature. Sweep each knob across its whole
range, plot accuracy against cost, and compare the resulting CURVES. RouterBench
(arXiv:2403.12031) formalises this as a cost-quality convex hull with an
area-under-curve summary; Dekoninck et al. (arXiv:2410.10347) report AUC over
swept lambda for the same reason. This module does that here.

    python3 frontier.py
    python3 frontier.py --split all --points 15

WHAT IS COMPUTED
----------------
1. Each policy's own operating points, from sweeping its knob.
2. Its ACHIEVABLE frontier: the upper convex hull of those points. The hull, not
   just the staircase, because any point on the segment between two achievable
   operating points is itself achievable by randomising between them. That is a
   real construction, not a smoothing convenience.
3. AUC: the area under that hull across a cost interval shared by every policy,
   divided by the width of the interval. It has the units of accuracy, and it
   reads as "average accuracy this policy delivers across the whole budget
   range". Higher is better.
4. The same number for the random baseline, and the difference. That difference is
   the honest headline: how much accuracy the policy buys, at matched spend,
   across every budget rather than at one convenient budget.

Free, because of the response cache: every point in every sweep reuses the same
model responses.
"""

import argparse
import sys
from pathlib import Path

import models
import policies
import run_eval
import splits

HERE = Path(__file__).parent
OUT = HERE / "frontier.jsonl"


# ---------------------------------------------------------------------------
# Running one operating point
# ---------------------------------------------------------------------------

def _measure(tasks, fn):
    """Accuracy and mean cost of one policy at one setting."""
    n_ok = 0
    cost = 0.0
    for t in tasks:
        r = fn(t)
        n_ok += bool(r.correct)
        cost += r.cost_usd
    return n_ok / len(tasks), cost / len(tasks)


def sweep_random(tasks, n_points):
    """The baseline curve: escalate a fixed fraction of traffic at random.

    This is the line every router has to beat. It is a genuine curve rather than
    a single null, and comparing against the whole line is what stops a router
    from claiming credit for simply spending more.
    """
    out = []
    saved = policies.RANDOM_MATCHED_RATES
    try:
        for i in range(n_points + 1):
            rate = i / n_points
            policies.RANDOM_MATCHED_RATES = {"math": rate, "code": rate}
            acc, cost = _measure(tasks, policies.policy_random_matched)
            out.append({"knob": "rate", "value": round(rate, 4), "accuracy": acc,
                        "cost_per_task": cost})
    finally:
        policies.RANDOM_MATCHED_RATES = saved
    return out


def sweep_cascade(tasks, _n_points):
    """Sweep the agreement threshold.

    The knob is discrete, not continuous: with k samples, agreement can only take
    the values 0/k .. k/k, so there are exactly k+1 distinct thresholds and
    sweeping more finely would produce duplicate points dressed up as a smooth
    curve. Thresholds above 1.0 are included so the "never accept" end of the
    range is reachable, which is the point where the cascade degenerates into
    always_expensive-with-a-wasted-cheap-call.
    """
    k = policies.SELF_CONSISTENCY_K
    out = []
    saved = policies.AGREEMENT_THRESHOLD
    try:
        for i in range(k + 2):
            th = i / k
            policies.AGREEMENT_THRESHOLD = th
            acc, cost = _measure(tasks, policies.policy_cascade)
            out.append({"knob": "agreement_threshold", "value": round(th, 4),
                        "accuracy": acc, "cost_per_task": cost})
    finally:
        policies.AGREEMENT_THRESHOLD = saved
    return out


# `sweep_predictive` was removed on 8 August 2026 with the policy it swept.
#
# Worth recording WHY it was not merely unused but wrong. It picked per-domain
# thresholds to hit a target escalation rate, drawn from the values actually
# present in the task set. On the math half those values were {5} - a single
# level, because MIN_MATH_LEVEL = 5 - so the sweep had exactly two attainable
# points there, rate 0 and rate 1, and could not trace a curve at all.
#
# The report that came out of it, "predictive contributes no point to the
# frontier", was therefore a description of a degenerate sweep rather than a
# finding about predictive routing. The predictive family is now represented by
# sweep_routellm below, whose knob is a continuous score threshold.


def sweep_routellm(tasks, n_points):
    """Sweep RouteLLM's score threshold, expressed as a target escalation rate."""
    import routellm_router
    if not routellm_router.available(tasks):
        return []
    out = []
    saved = dict(routellm_router.THRESHOLDS)
    try:
        for i in range(n_points + 1):
            rate = i / n_points
            routellm_router.calibrate(tasks, {"math": rate, "code": rate})
            acc, cost = _measure(tasks, policies.policy_routellm)
            out.append({"knob": "target_rate", "value": round(rate, 4),
                        "accuracy": acc, "cost_per_task": cost})
    finally:
        routellm_router.THRESHOLDS = saved
    return out


def sweep_cascade_routing(tasks, n_points):
    """Sweep lambda, the price of quality in dollars.

    Geometric rather than linear, because lambda multiplies a cost measured in
    thousandths of a dollar against a quality in [0, 1]. The behaviour changes
    over orders of magnitude, so a linear grid would put almost every point in the
    region where lambda is large enough that nothing is ever worth buying.

    lambda = 0 means money is free, so the policy climbs to the top tier. Very
    large lambda means quality is worthless, so it stops at the cheapest.
    """
    if not policies.ESTIMATORS_FITTED:
        return []
    out = []
    saved = policies.CASCADE_ROUTING_LAMBDA
    try:
        for i in range(n_points + 1):
            # 0, then 10^0.5 .. 10^4.5 across the remaining points.
            lam = 0.0 if i == 0 else 10 ** (0.5 + 4.0 * (i - 1) / max(1, n_points - 1))
            policies.CASCADE_ROUTING_LAMBDA = lam
            acc, cost = _measure(tasks, policies.policy_cascade_routing)
            out.append({"knob": "lambda", "value": round(lam, 4),
                        "accuracy": acc, "cost_per_task": cost})
    finally:
        policies.CASCADE_ROUTING_LAMBDA = saved
    return out


SWEEPS = {
    "random": sweep_random,
    "routellm": sweep_routellm,
    "cascade": sweep_cascade,
    "cascade_routing": sweep_cascade_routing,
}


# ---------------------------------------------------------------------------
# Frontier geometry
# ---------------------------------------------------------------------------

def upper_hull(points):
    """Upper-left convex hull of (cost, accuracy) points: the achievable frontier.

    Every returned point is either measured or reachable by randomising between
    two measured points, which is why the hull rather than the raw staircase is
    the right object. Points strictly inside the hull are dominated: some mix of
    two other settings is at least as accurate for the same money.
    """
    pts = sorted({(round(c, 12), a) for c, a in points})
    # Keep only points that are not dominated by a cheaper, at-least-as-accurate one.
    kept = []
    best = float("-inf")
    for c, a in pts:
        if a > best:
            kept.append((c, a))
            best = a
    # Discard any point below the chord between its neighbours: concavity.
    hull = []
    for p in kept:
        while len(hull) >= 2:
            (c0, a0), (c1, a1) = hull[-2], hull[-1]
            # cross product; drop hull[-1] if it lies below the line c0->p
            if (c1 - c0) * (p[1] - a0) >= (p[0] - c0) * (a1 - a0):
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def _interp(hull, cost):
    """Accuracy of the hull at `cost`, extended flat beyond its ends.

    Flat extension is legitimate in both directions. Below the hull's cheapest
    point there is no way to spend less, so the accuracy there is not defined by
    this policy and holding it flat neither rewards nor punishes it; above the
    dearest point, extra budget simply goes unused.
    """
    if not hull:
        return None
    if cost <= hull[0][0]:
        return hull[0][1]
    if cost >= hull[-1][0]:
        return hull[-1][1]
    for (c0, a0), (c1, a1) in zip(hull, hull[1:]):
        if c0 <= cost <= c1:
            if c1 == c0:
                return max(a0, a1)
            f = (cost - c0) / (c1 - c0)
            return a0 + f * (a1 - a0)
    return hull[-1][1]


def auc(hull, lo, hi, steps=400):
    """Mean accuracy of the hull over the cost interval [lo, hi].

    Normalised by the interval width, so the result has the units of accuracy and
    is directly comparable to a number in the run_eval table. Trapezoidal, over a
    fixed grid, so two policies are always integrated identically.
    """
    if not hull or hi <= lo:
        return None
    total = 0.0
    for i in range(steps):
        c0 = lo + (hi - lo) * i / steps
        c1 = lo + (hi - lo) * (i + 1) / steps
        total += 0.5 * (_interp(hull, c0) + _interp(hull, c1)) * (c1 - c0)
    return total / (hi - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["eval", "all"], default="eval")
    ap.add_argument("--domain", choices=["math", "code"])
    ap.add_argument("--points", type=int, default=10,
                    help="sweep resolution. Free: every response is a cache hit.")
    ap.add_argument("--out", metavar="PATH", default=None,
                    help="write the curve here instead of frontier.jsonl. One "
                         "file per ladder, so three ladders' curves can be read "
                         "side by side rather than overwriting each other.")
    args = ap.parse_args()

    global OUT
    if args.out:
        OUT = Path(args.out)

    all_tasks = run_eval.load_tasks(domain=args.domain)
    calibration, evaluation = splits.split(all_tasks)
    tasks = evaluation if args.split == "eval" else all_tasks

    print(run_eval.banner(), file=sys.stderr)
    for line in models.ladder_summary():
        print(line, file=sys.stderr)
    for line in splits.describe(calibration, evaluation):
        print(line, file=sys.stderr)
    print(f"sweeping {len(SWEEPS)} policy families over {len(tasks)} tasks "
          f"({args.split})", file=sys.stderr)

    if calibration:
        try:
            policies.fit_estimators(calibration)
        except models.ReplayMiss as exc:
            # Same rule as run_eval: sit the estimator-dependent families out
            # rather than take the whole sweep down with them. SWEEPS is
            # filtered below on ESTIMATORS_FITTED, which fit_estimators leaves
            # False when it raises.
            print(f"cascade_routing: SKIPPED - estimators cannot be fitted from "
                  f"this cache.\n  {str(exc).splitlines()[0]}", file=sys.stderr)
    policies.calibrate_random_rates(tasks)

    models.reset_call_stats()

    # The fixed reference points, which also set the cost interval everything is
    # integrated over.
    #
    # A family that cannot be replayed from this cache sits out, the same rule
    # run_eval applies. Sitting out is safe for every family EXCEPT the two rungs
    # that define the cost axis, so those are checked explicitly below rather
    # than left to fail as a KeyError several screens later.
    fixed, unreplayable = {}, []
    for name in [f"always_{t}" for t in models.TIERS] + ["oracle"]:
        try:
            acc, cost = _measure(tasks, policies.POLICIES[name])
        except models.ReplayMiss:
            unreplayable.append(name)
            continue
        fixed[name] = {"accuracy": acc, "cost_per_task": cost}

    for axis in ("always_cheap", "always_expensive"):
        if axis not in fixed:
            raise SystemExit(
                f"frontier: {axis} cannot be replayed from this cache, and it "
                f"defines the cost axis every curve is integrated over. There is "
                f"no frontier to draw.\n"
                f"  Record it with: ROUTER_MODE=real python3 run_eval.py "
                f"--policy {axis} --split all"
            )

    lo = fixed["always_cheap"]["cost_per_task"]
    hi = fixed["always_expensive"]["cost_per_task"]

    curves = {}
    for name, fn in SWEEPS.items():
        try:
            pts = fn(tasks, args.points)
        except models.ReplayMiss:
            unreplayable.append(name)
            continue
        if pts:
            curves[name] = pts

    if unreplayable:
        print(f"\n!! not replayable from this cache, so absent from the frontier "
              f"below: {', '.join(sorted(set(unreplayable)))}\n"
              f"   Their calls were never recorded. This is a gap in the cache, "
              f"not a result about the policies.", file=sys.stderr)

    tag = run_eval.tag()
    print()
    print(run_eval.banner())
    print()
    print(tag)
    print(f"cost-quality frontiers, n={len(tasks)} ({args.split} split)")
    print(f"integrating over cost/task ${lo:.6f} (always_cheap) to "
          f"${hi:.6f} (always_expensive)")
    print()
    print("AUC = mean accuracy across that whole budget range, so it has the units")
    print("of accuracy. It answers 'how good is this policy at every budget', not")
    print("'how good is it at the one budget somebody tuned it to'.")
    print()

    base_hull = upper_hull([(p["cost_per_task"], p["accuracy"])
                            for p in curves.get("random", [])])
    base_auc = auc(base_hull, lo, hi)

    print(f"{'policy family':<18} {'pts':>4} {'hull':>5} {'AUC':>8} "
          f"{'vs random':>10} {'best acc':>9} {'at cost':>10}")
    print("-" * 70)
    rows = []
    for name, pts in curves.items():
        hull = upper_hull([(p["cost_per_task"], p["accuracy"]) for p in pts])
        a = auc(hull, lo, hi)
        best = max(pts, key=lambda p: (p["accuracy"], -p["cost_per_task"]))
        delta = "" if base_auc is None or a is None or name == "random" \
            else f"{a - base_auc:>+9.1%}"
        if name == "random":
            delta = "  baseline"
        rows.append((name, len(pts), len(hull), a, delta, best))
        print(f"{name:<18} {len(pts):>4} {len(hull):>5} {a:>8.1%} {delta:>10} "
              f"{best['accuracy']:>8.1%} {best['cost_per_task']:>10.6f}")

    print()
    print("fixed reference points (single settings, so no curve):")
    for name, r in fixed.items():
        print(f"  {name:<18} acc {r['accuracy']:>6.1%}   cost/task "
              f"{r['cost_per_task']:>10.6f}")

    # The combined frontier over DEPLOYABLE policies only.
    #
    # The oracle is deliberately excluded. It needs the answer in order to choose,
    # so it dominates every budget by construction; including it would collapse the
    # frontier to a single point and say nothing about the policies that could
    # actually be shipped. It is printed afterwards as the ceiling it is.
    deployable = {n: p for n, p in curves.items()}
    every = [(p["cost_per_task"], p["accuracy"]) for pts in deployable.values() for p in pts]
    for name, r in fixed.items():
        if name != "oracle":
            every.append((r["cost_per_task"], r["accuracy"]))
    combined = upper_hull(every)

    def owners_of(c, a):
        """Every family that reaches this exact point. Ties are common and real:
        at rate 0 all the routers collapse onto always_cheap."""
        found = []
        for name, pts in deployable.items():
            if any(abs(p["cost_per_task"] - c) < 1e-12 and abs(p["accuracy"] - a) < 1e-12
                   for p in pts):
                found.append(name)
        for name, r in fixed.items():
            if name == "oracle":
                continue
            if abs(r["cost_per_task"] - c) < 1e-12 and abs(r["accuracy"] - a) < 1e-12:
                found.append(name)
        return found

    print()
    print(tag)
    print("the combined frontier over DEPLOYABLE policies - who owns each budget:")
    print("(the oracle is excluded: it needs the answer to choose, so it would own")
    print(" every budget and the comparison would say nothing)")
    print(f"{'cost/task':>11} {'accuracy':>9}   reached by")
    print("-" * 60)
    on_hull = set()
    for c, a in combined:
        who = owners_of(c, a)
        on_hull.update(who)
        print(f"{c:>11.6f} {a:>9.1%}   {', '.join(who) or '(mixture)'}")

    orc = fixed.get("oracle")
    if orc:
        print(f"{orc['cost_per_task']:>11.6f} {orc['accuracy']:>9.1%}   "
              f"<- oracle, for reference. NOT deployable.")
    else:
        print("            (no oracle row - it could not be replayed from this "
              "cache, so the ceiling below is unknown)")

    dominated = [n for n in deployable if n not in on_hull]
    if dominated:
        print()
        print("families contributing NO point to the frontier, i.e. beaten at every")
        print("budget by some other policy here:")
        for n in dominated:
            print(f"  {n}")

    # Whole-run rather than per-row: a frontier point is an aggregate over every
    # task, so one fabricated response anywhere taints every point that could
    # have used it. The conservative reading is the only defensible one here.
    prov = run_eval.provenance(simulated=models.call_stats["served_mock"] > 0)

    rows_out = []
    for name, pts in curves.items():
        for p in pts:
            rows_out.append({"family": name, **p, "n": len(tasks),
                             "split": args.split, **prov})
    for name, r in fixed.items():
        rows_out.append({"family": name, "knob": None, "value": None, **r,
                         "n": len(tasks), "split": args.split, **prov})
    run_eval.write_jsonl(OUT, rows_out)

    st = models.call_stats
    print()
    print(f"model calls: {st['requested']} requested, {st['from_cache']} from cache, "
          f"{st['backend']} reached a backend")
    print(f"\nwrote {len(rows_out)} points -> {OUT}")


if __name__ == "__main__":
    main()
