"""The random baselines, over many seeds. The missing null hypothesis.

One coin flip is not a baseline. results.jsonl carries a single draw
(policies.RANDOM_SEED = 0) because the paired statistics need per-task rows that
line up with the other policies; this script reports the mean and spread over
many draws, which is the number to quote.

WHY IT MATTERS, concretely. `predictive` routes 38 of 100 tasks to the expensive
tier and scores 89.0% against always_cheap's 74.0%. That 15-point gap looks like
routing skill and is not, on its own, evidence of any: a router that picks 38
tasks AT RANDOM also gains accuracy, because a third of the traffic is now going
to a better model. The question is how much of the gap survives once the
spending is held fixed, and that is what a cost-matched random baseline answers.

This is not a general methodological point. cascade_routing_project_plan.md,
Decision #1, defends the heuristic router with the line "you can say where yours
sits between random and optimal". Random was never implemented, so the plan's
answer to its most likely interview question rested on a baseline that did not
exist.

FREE, because of the response cache: every seed re-uses the same cheap and
expensive greedy responses. 200 seeds cost zero additional model calls.

    python3 random_baseline.py
    python3 random_baseline.py --seeds 1000
"""

import argparse
import sys

import models
import policies
import run_eval


def _mean(xs):
    return sum(xs) / len(xs)


def _sd(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _pct(xs, q):
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))]


def sweep_seeds(tasks, policy_fn, seeds):
    """Run one random policy over `seeds` independent draws."""
    accs, costs = [], []
    per_domain = {d: [] for d in ("math", "code")}
    for s in range(seeds):
        policies.RANDOM_SEED = s
        rs = [(t, policy_fn(t)) for t in tasks]
        accs.append(sum(r.correct for _, r in rs) / len(rs))
        costs.append(sum(r.cost_usd for _, r in rs) / len(rs))
        for d in per_domain:
            sub = [r for t, r in rs if t["domain"] == d]
            if sub:
                per_domain[d].append(sum(r.correct for r in sub) / len(sub))
    policies.RANDOM_SEED = 0
    return accs, costs, per_domain


def fixed_policy(tasks, name):
    rs = [(t, policies.POLICIES[name](t)) for t in tasks]
    out = {"all": sum(r.correct for _, r in rs) / len(rs),
           "cost": sum(r.cost_usd for _, r in rs) / len(rs)}
    for d in ("math", "code"):
        sub = [r for t, r in rs if t["domain"] == d]
        if sub:
            out[d] = sum(r.correct for r in sub) / len(sub)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()

    tasks = run_eval.load_tasks()
    rates = policies.calibrate_random_rates(tasks)
    print(run_eval.banner(), file=sys.stderr)
    print(f"{args.seeds} seeds over {len(tasks)} tasks; random_matched calibrated to "
          + ", ".join(f"{d}={r:.0%}" for d, r in sorted(rates.items())), file=sys.stderr)

    import routellm_router
    if routellm_router.available(tasks):
        routellm_router.calibrate(tasks, rates)

    models.reset_call_stats()
    names = ["always_cheap", "always_expensive", "predictive", "llm_router", "oracle"]
    if routellm_router.available(tasks):
        names.insert(3, "routellm")
    fixed = {n: fixed_policy(tasks, n) for n in names}

    results = {}
    for name, fn in (("random_matched", policies.policy_random_matched),
                     ("random_50", policies.policy_random_50)):
        results[name] = sweep_seeds(tasks, fn, args.seeds)

    tag = ("### MOCK MODE - SIMULATED, NOT MEASURED ###" if models.MODE == "mock"
           else f"### {models.MODE.upper()} MODE ###")
    print()
    print(run_eval.banner())
    print()
    print(tag)
    print(f"random baselines over {args.seeds} seeds, n={len(tasks)}")
    print(f"{'policy':<18} {'acc mean':>9} {'sd':>7} {'[p5':>8} {'p95]':>8} {'cost/task':>12}")
    print("-" * 68)
    for name, (accs, costs, _) in results.items():
        print(f"{name:<18} {_mean(accs):>9.1%} {_sd(accs):>7.1%} "
              f"{_pct(accs, 0.05):>8.1%} {_pct(accs, 0.95):>8.1%} {_mean(costs):>12.6f}")
    for name in [n for n in ("always_cheap", "predictive", "routellm", "llm_router",
                             "oracle", "always_expensive") if n in fixed]:
        f = fixed[name]
        print(f"{name:<18} {f['all']:>9.1%} {'-':>7} {'-':>8} {'-':>8} {f['cost']:>12.6f}")

    # The number that makes the predictive arm interpretable.
    print()
    print(tag)
    print("routing skill = (router - random_matched) / (oracle - random_matched)")
    print("  random_matched is the mean over all seeds, so this is skill against the")
    print("  EXPECTED coin flip rather than against one lucky or unlucky draw.")
    print(f"{'router':<16} {'domain':<8} {'random':>8} {'router':>8} {'oracle':>8} {'skill':>9}")
    print("-" * 62)
    rm_accs, _, rm_domain = results["random_matched"]
    for router in [n for n in ("predictive", "routellm", "llm_router") if n in fixed]:
        for domain in ("all", "math", "code"):
            lo = _mean(rm_accs) if domain == "all" else _mean(rm_domain[domain])
            mid, hi = fixed[router][domain], fixed["oracle"][domain]
            if abs(hi - lo) < 1e-9:
                skill = "     n/a"
            else:
                skill = f"{(mid - lo) / (hi - lo):>8.1%}"
            print(f"{router:<16} {domain:<8} {lo:>8.1%} {mid:>8.1%} {hi:>8.1%} {skill:>9}")

    # How often a random draw beats the heuristic outright. If this is not small,
    # the heuristic's advantage is inside the noise and should not be claimed.
    print()
    p_acc = fixed["predictive"]["all"]
    beats = sum(1 for a in rm_accs if a >= p_acc) / len(rm_accs)
    print(f"P(random_matched >= predictive) over {args.seeds} seeds: {beats:.1%}")
    if beats > 0.05:
        print("  NOT a significant edge for the heuristic at this n. Say so.")
    else:
        print("  The heuristic beats a cost-matched coin flip more often than chance.")

    st = models.call_stats
    print()
    print(f"model calls: {st['requested']} requested, {st['from_cache']} from cache, "
          f"{st['backend']} reached a backend")
    print(f"  {args.seeds} seeds x 2 policies cost 0 extra model calls beyond the first draw.")


if __name__ == "__main__":
    main()
