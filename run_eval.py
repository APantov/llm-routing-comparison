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
from policies import POLICIES

HERE = Path(__file__).parent
RESULTS = HERE / "results.jsonl"

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
  Not reproducible: sampling at temperature > 0 cannot be pinned.
================================================================"""


def banner():
    return MOCK_BANNER if models.MODE == "mock" else REAL_BANNER


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
    }


def load_tasks(limit=None, domain=None):
    tasks = [json.loads(l) for l in open(HERE / "taskset.jsonl")]
    if domain:
        tasks = [t for t in tasks if t["domain"] == domain]
    return tasks[:limit] if limit else tasks


def run(tasks):
    rows = []
    spend = 0.0
    total = len(tasks) * len(POLICIES)
    done = 0

    for task in tasks:
        for name, fn in POLICIES.items():
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


def report(rows):
    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r["policy"]].append(r)

    order = ["always_cheap", "predictive", "cascade", "oracle", "always_expensive"]

    # Tag EVERY table, not just the top of the report. A screenshot is usually
    # a crop of one table, and a crop that loses the banner is exactly how a
    # simulated number ends up in a README as if it were measured.
    tag = (
        "### MOCK MODE - SIMULATED, NOT MEASURED ###"
        if models.MODE == "mock"
        else "### REAL MODE - live API calls ###"
    )

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
        print(f"{name:<20} {acc:>6.1%} {cost:>12.6f} {lat:>9.2f}s {esc:>6.1%}")

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

    spent = sum(r["cost_usd"] for r in rows)
    print()
    if models.MODE == "mock":
        print(f"total MODELLED cost: ${spent:.4f}  (simulated - nothing was spent)")
    else:
        print(f"total spend this run: ${spent:.4f}")

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
        existing = {json.loads(l).get("mode") for l in RESULTS.open() if l.strip()}
    except (json.JSONDecodeError, OSError):
        return
    if "real" in existing:
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
    print(banner(), file=sys.stderr)
    print(f"mode={models.MODE}  tasks={len(tasks)}  policies={len(POLICIES)}", file=sys.stderr)

    rows = run(tasks)
    with open(RESULTS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    report(rows)
    print(f"\nwrote {len(rows)} rows -> {RESULTS}")
    # Last line of output as well as the first. Terminal scrollback usually
    # shows the end of a run, not the beginning.
    print()
    print(banner())


if __name__ == "__main__":
    main()
