"""
Batch runner. Runs every policy over every task, writes results, prints a report.

Usage:
    python run_eval.py              # mock mode, no API key, no spend
    ROUTER_MODE=real python run_eval.py --limit 10   # 10-task pilot, real calls

Start in mock mode. Get the whole pipeline working, see the report print, THEN
switch to real. Debugging a broken pipeline while paying per call is miserable.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import models
import policies
import response_cache
from policies import POLICIES, POLICY_DOMAINS

HERE = Path(__file__).parent
RESULTS = HERE / "results.jsonl"


def write_jsonl(path, rows):
    r"""Write JSONL with LF endings on every platform.

    Not cosmetic. taskset.jsonl in this repo was written on Windows and carries
    CRLF; results.jsonl was written on Linux and carries LF. The two artefacts
    of one experiment disagree about line endings, which means the same code on
    two machines produces byte-different files and no hash-based regression gate
    is possible. newline="" stops Python's platform translation.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

# Hard spend cap. Enforced in code, not by willpower.
MAX_SPEND_USD = 20.0


MOCK_BANNER = """\
================================================================
  MOCK MODE - these numbers are SIMULATED, not measured.

  Model replies are FABRICATED from answers already stored in the
  taskset. Accuracy below restates models.MOCK_SKILL; it does not
  measure any model. Costs are modelled from synthetic token
  counts - nothing was spent and no network call was made.

  For a real result:  ROUTER_MODE=real python run_eval.py
================================================================"""

REAL_BANNER = """\
================================================================
  REAL MODE - live API calls, real money, real numbers.
  Every response is written to cache/raw_calls.jsonl as it
  arrives, so this run can be replayed for free forever after.
================================================================"""

REPLAY_BANNER = """\
================================================================
  REPLAY MODE - served entirely from cache/raw_calls.jsonl.
  No network call. No spend. These are the SAME responses the
  paid run received, so the numbers are the paid run's numbers.
================================================================"""


def banner():
    return {"mock": MOCK_BANNER, "real": REAL_BANNER, "replay": REPLAY_BANNER}[models.MODE]


def provenance():
    """Everything needed to interpret a row, carried by the row itself.

    Stamped per row rather than in a sidecar header so that a slice of
    results.jsonl - one policy, one domain, one k - is still self-describing.
    The parameters are here specifically so a sweep over k or the agreement
    threshold produces a file you can group by instead of a pile of runs you
    have to remember the settings for.
    """
    return {
        "mode": models.MODE,
        "mock_seed": models.MOCK_SEED if models.MODE == "mock" else None,
        "k": policies.SELF_CONSISTENCY_K,
        "agreement_threshold": policies.AGREEMENT_THRESHOLD,
        # The manipulated variable. Stamped on every row so the degradation
        # sweep produces one file you can group by, rather than a pile of runs
        # whose settings you have to remember.
        "verifier_corruption": policies.VERIFIER_CORRUPTION,
        # Which draw the random baselines took. Meaningless for every other
        # policy, and recorded anyway, because a row that needs an external
        # note to interpret is a row that will eventually be misread.
        "random_seed": policies.RANDOM_SEED,
    }


def load_tasks(limit=None, domain=None):
    with open(HERE / "taskset.jsonl", encoding="utf-8") as f:
        tasks = [json.loads(l) for l in f]
    if domain:
        tasks = [t for t in tasks if t["domain"] == domain]
    return tasks[:limit] if limit else tasks


def applicable(name, task):
    """Does this policy run on this task?

    cascade_degraded is defined on the code domain only, and that is the point
    of it rather than a limitation: it varies verifier fidelity WITHIN a domain,
    holding tasks, models, grader and prompts fixed, so that verifier quality
    stops being confounded with math-vs-code. Running it on math would put the
    confound straight back.
    """
    allowed = POLICY_DOMAINS.get(name)
    return allowed is None or task["domain"] in allowed


def run(tasks):
    rows = []
    spend = 0.0
    total = sum(1 for t in tasks for n in POLICIES if applicable(n, t))
    done = 0

    for task in tasks:
        for name, fn in POLICIES.items():
            if not applicable(name, task):
                continue
            if spend > MAX_SPEND_USD:
                print(f"\n!! spend cap ${MAX_SPEND_USD} hit, stopping early", file=sys.stderr)
                return rows
            res = fn(task)
            spend += res.cost_usd
            rows.append(
                {
                    "task_id": res.task_id,
                    "domain": task["domain"],
                    "difficulty": task.get("difficulty_proxy"),
                    "policy": res.policy,
                    "correct": res.correct,
                    "cost_usd": res.cost_usd,
                    "latency_s": res.latency_s,
                    "escalated": res.escalated,
                    "calls": res.calls,
                    **provenance(),
                }
            )
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{total}  spend=${spend:.4f}", file=sys.stderr)
    return rows


def routing_skill(by_policy, tag):
    """How much of the achievable routing gain each router actually captured.

        skill = (acc_router - acc_random_matched) / (acc_oracle - acc_random_matched)

    Random is the floor and the oracle is the ceiling, so this reads as "what
    fraction of the headroom did the router find". It is the number that makes
    the whole predictive arm interpretable: raw accuracy conflates routing skill
    with willingness to spend, and a cost-matched random baseline holds the
    spending fixed so only skill is left.

    Reported PER DOMAIN as well as overall, because the aggregate here is the
    mean of a router with a real signal (math, where MATH500 ships a difficulty
    level) and one with almost none (code, where nothing in an MBPP prompt
    predicts difficulty). Averaging those two into one number hides the finding.

    A negative value means the router did worse than a coin flip at the same
    spend, which is a result and should be printed as one, not clipped to zero.
    """
    def acc(name, domain=None):
        rs = by_policy.get(name, [])
        if domain:
            rs = [r for r in rs if r["domain"] == domain]
        return (sum(r["correct"] for r in rs) / len(rs)) if rs else None

    routers = [n for n in ("predictive", "llm_router") if by_policy.get(n)]
    if not routers or not by_policy.get("random_matched") or not by_policy.get("oracle"):
        return

    print()
    print(tag)
    print("routing skill = (router - random_matched) / (oracle - random_matched)")
    print(f"{'router':<16} {'domain':<8} {'random':>8} {'router':>8} {'oracle':>8} {'skill':>9}")
    print("-" * 62)
    for name in routers:
        for domain in (None, "math", "code"):
            lo, mid, hi = acc("random_matched", domain), acc(name, domain), acc("oracle", domain)
            if None in (lo, mid, hi):
                continue
            label = domain or "all"
            if abs(hi - lo) < 1e-9:
                # No headroom: random already matches the oracle, so the ratio
                # is 0/0 and any number printed here would be invented.
                skill = "     n/a"
            else:
                skill = f"{(mid - lo) / (hi - lo):>8.1%}"
            print(f"{name:<16} {label:<8} {lo:>8.1%} {mid:>8.1%} {hi:>8.1%} {skill:>9}")


def report(rows):
    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r["policy"]].append(r)

    # Ordered roughly by spend, so the table reads as a cost ladder. The two
    # random policies sit next to predictive on purpose: they are what
    # predictive has to beat before its accuracy means anything.
    order = [
        "always_cheap", "random_matched", "random_50", "predictive",
        "llm_router", "cascade_degraded", "cascade", "oracle", "always_expensive",
    ]

    # Tag EVERY table, not just the top of the report. A screenshot is usually
    # a crop of one table, and a crop that loses the banner is exactly how a
    # simulated number ends up in a README as if it were measured.
    tag = {
        "mock": "### MOCK MODE - SIMULATED, NOT MEASURED ###",
        "real": "### REAL MODE - live API calls ###",
        "replay": "### REPLAY MODE - cached responses from a real run ###",
    }[models.MODE]

    print()
    print(banner())
    print()
    print(tag)
    print(f"{'policy':<20} {'acc':>7} {'cost/task':>12} {'lat/task':>10} {'escal':>7}")
    print("-" * 60)
    for name in order:
        rs = by_policy.get(name, [])
        if not rs:
            continue
        acc = sum(r["correct"] for r in rs) / len(rs)
        cost = sum(r["cost_usd"] for r in rs) / len(rs)
        lat = sum(r["latency_s"] for r in rs) / len(rs)
        esc = sum(r["escalated"] for r in rs) / len(rs)
        # cascade_degraded runs on code only, so its row is over 40 tasks while
        # the others are over 100. Print n rather than let the reader assume.
        note = f"  (n={len(rs)}, code only)" if name == "cascade_degraded" else ""
        print(f"{name:<20} {acc:>6.1%} {cost:>12.6f} {lat:>9.2f}s {esc:>6.1%}{note}")

    routing_skill(by_policy, tag)

    # The verifier contrast: cascade should do noticeably worse where the
    # verifier is a guess (math) than where it is perfect (code).
    print()
    print(tag)
    print("cascade by domain (the verifier contrast):")
    print(f"{'domain':<10} {'acc':>7} {'cost/task':>12} {'escal':>8}")
    print("-" * 40)
    for domain in ("code", "math"):
        rs = [r for r in by_policy.get("cascade", []) if r["domain"] == domain]
        if not rs:
            continue
        acc = sum(r["correct"] for r in rs) / len(rs)
        cost = sum(r["cost_usd"] for r in rs) / len(rs)
        esc = sum(r["escalated"] for r in rs) / len(rs)
        verifier = "perfect" if domain == "code" else "proxy"
        print(f"{domain:<10} {acc:>6.1%} {cost:>12.6f} {esc:>7.1%}   <- {verifier} verifier")

    # The claim under test in DECISION #7: that an LLM routing call "would add a
    # full round trip and defeat the purpose". Half of that is a cost claim, so
    # print the cost rather than argue about it.
    if by_policy.get("llm_router") and by_policy.get("always_cheap"):
        n = len(by_policy["llm_router"])
        router_cost = sum(policies.ROUTER_CALL_COST) / max(1, len(policies.ROUTER_CALL_COST))
        router_lat = sum(policies.ROUTER_CALL_LATENCY) / max(1, len(policies.ROUTER_CALL_LATENCY))
        cheap = sum(r["cost_usd"] for r in by_policy["always_cheap"]) / n
        exp = sum(r["cost_usd"] for r in by_policy.get("always_expensive", [])) / n \
            if by_policy.get("always_expensive") else None
        print()
        print(tag)
        print("LLM-as-router overhead (the claim in policies.py DECISION #4):")
        print(f"  routing call        ${router_cost:.6f}/task, +{router_lat:.2f}s/task")
        print(f"    as % of a cheap answer call     {100 * router_cost / cheap:>5.1f}%")
        if exp:
            print(f"    as % of an expensive answer call{100 * router_cost / exp:>6.1f}%")
        print("  Cost and latency above are arithmetic on the price table and are")
        print("  the only part of this policy that mock mode can measure. Its")
        print("  ACCURACY restates models.MOCK_ROUTER_SKILL and measures nothing.")
        if models.MODE == "mock":
            print("  The PERCENTAGES are softer than the dollar figure: mock answer calls")
            print("  emit a fixed 80/120 output tokens, well under a real reply, so the")
            print("  router's share of a real answer call will be smaller than shown.")
            print("  Quote the absolute cost from a real run, not these ratios.")

    # The pilot gate. This is the number that decides whether the task set works.
    cheap = by_policy.get("always_cheap", [])
    if cheap:
        fail = 1 - sum(r["correct"] for r in cheap) / len(cheap)
        print()
        print(f"cheap-model failure rate: {fail:.1%}")
        if fail < 0.20:
            print("  TOO EASY - the cascade has almost nothing to route. Use harder tasks.")
        elif fail > 0.55:
            print("  TOO HARD - the cascade escalates nearly everything. Use easier tasks.")
        else:
            print("  GOOD - in the 30-40% target band where routing decisions matter.")

    # Two different numbers, and conflating them is the mistake the cache makes
    # easy. `attributed` is what the policies cost: every call charged to every
    # policy that made it, which is what a production deployment of one policy
    # would pay. `backend` is what THIS RUN actually spent, after deduplicating
    # identical calls across policies. Deduplication changes the second and must
    # never change the first.
    attributed = sum(r["cost_usd"] for r in rows)
    st = models.call_stats
    print()
    label = "total MODELLED cost" if models.MODE == "mock" else "total attributed cost"
    print(f"{label}: ${attributed:.4f}   (sum over policies - what they would each pay)")
    print(
        f"model calls: {st['requested']} requested, {st['from_cache']} served from cache, "
        f"{st['backend']} reached a backend"
    )
    if st["requested"]:
        saved = 100 * st["from_cache"] / st["requested"]
        print(f"  cache deduplicated {saved:.1f}% of calls  (cache now holds {response_cache.size()})")
    if models.MODE == "mock":
        print("  (simulated - nothing was spent and no network call was made)")

    # Truncation is invisible in the accuracy table - a cut-off answer just
    # scores as wrong - so it gets its own line.
    if models.truncated_calls:
        print()
        print(
            f"!! {models.truncated_calls} call(s) hit max_tokens and were graded as "
            f"WRONG. Raise models.MAX_TOKENS and re-run; these numbers are not valid."
        )


def guard_clobber(force: bool):
    """Refuse to overwrite REAL results with a MOCK run.

    A real run costs money and cannot be reproduced (temperature > 0). A mock
    run is free and takes seconds. Losing the former to the latter by typing
    `python run_eval.py` out of habit is a mistake worth making impossible
    rather than merely unlikely.
    """
    if models.MODE != "mock" or force or not RESULTS.exists():
        return
    try:
        with RESULTS.open(encoding="utf-8") as f:
            existing = {json.loads(l).get("mode") for l in f if l.strip()}
    except (json.JSONDecodeError, OSError):
        return
    # `replay` counts as real: those rows describe responses that were paid for.
    if existing & {"real", "replay"}:
        sys.exit(
            f"\nREFUSING TO RUN.\n"
            f"  {RESULTS.name} holds REAL results, which cost money and cannot be\n"
            f"  reproduced. This is a MOCK run and would overwrite them.\n\n"
            f"  Back them up:   cp {RESULTS.name} results.real.jsonl\n"
            f"  Or override:    python run_eval.py --force\n"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="run only the first N tasks (use 10 for the pilot)")
    ap.add_argument("--domain", choices=["math", "code"])
    ap.add_argument(
        "--force", action="store_true",
        help="allow a mock run to overwrite existing real results",
    )
    args = ap.parse_args()

    guard_clobber(args.force)

    tasks = load_tasks(args.limit, args.domain)
    # Match the random baseline's spend to predictive's on THESE tasks, before
    # anything runs. Doing it after would compare against a rate measured on a
    # different task set.
    rates = policies.calibrate_random_rates(tasks)
    print(banner(), file=sys.stderr)
    print(f"mode={models.MODE}  tasks={len(tasks)}  policies={len(POLICIES)}", file=sys.stderr)
    print(
        "random_matched calibrated to predictive: "
        + ", ".join(f"{d}={r:.0%}" for d, r in sorted(rates.items())),
        file=sys.stderr,
    )

    rows = run(tasks)
    write_jsonl(RESULTS, rows)
    report(rows)
    print(f"\nwrote {len(rows)} rows -> {RESULTS}")
    # Last line of output as well as the first. Terminal scrollback usually
    # shows the end of a run, not the beginning.
    print()
    print(banner())


if __name__ == "__main__":
    main()
