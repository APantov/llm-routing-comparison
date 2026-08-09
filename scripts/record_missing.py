#!/usr/bin/env python3
r"""Buy the calls a full replay needs and the paid runs never recorded.

WHY THIS EXISTS
---------------
The cache was populated by two runs with narrow purposes: the two-arm probe
(`always_cheap` and `always_expensive`, one greedy draw per task) and the
decisive-task redraw (21 tasks, ten draws each). Every response in it is real.

But a cache is only as complete as the policies that filled it, and most
policies here make calls neither of those runs had any reason to make:

    verify_math       draws SELF_CONSISTENCY_K - 1 extra samples per maths
                      task at temperature 0.8. The probe drew ONE answer at
                      temperature 0. So `cascade`, `oracle`, `cascade_routing`
                      and anything fitted on them cannot replay at all.
    llm_router        asks the cheap model to classify difficulty first, an
                      8-token call with its own prompt. Never recorded.

Since ROUTER_REPLAY_FALLBACK defaults off (NOTES.md issue 19), those policies
are now correctly DROPPED from a replay rather than silently served fabricated
responses. Dropped is honest; it is not the same as measured. This script buys
the gap once so they can be measured, after which everything replays free
forever.

WHAT IT BUYS, AND WHY THAT LIST IS COMPLETE
-------------------------------------------
The missing calls are enumerable from the code's own constants rather than
discovered by running a policy and seeing what it asks for. That distinction
matters: a policy's control flow depends on the responses it gets, so a
discovery run driven by stubs would request calls a real run never makes, and
this script would buy them.

Two facts make the enumeration exact:

  - `_self_consistency` returns immediately, making zero calls, on any tier
    whose model does not accept a temperature. So only tiers with
    `accepts_temperature` can generate sample calls, whatever the policy.
  - It is reached only through `verify_math` and the maths branch of
    `policy_oracle`, both guarded on `domain == "math"`.

So the set is: maths tasks x temperature-accepting tiers x samples 1..K-1, plus
one routing call per task. Calls already in the cache are skipped, so re-running
this after a partial run costs only the remainder.

SPENDING
--------
Prints a costed plan and exits. Nothing is called until `--go` is passed, and
`--go` refuses unless ROUTER_MODE=real. Every response is written to the
ladder's cache as it arrives, so a run that dies halfway keeps what it paid for.

    python3 scripts/record_missing.py                       # plan and price
    ROUTER_MODE=real ROUTER_LADDER=wide \
        python3 scripts/record_missing.py --go
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import models            # noqa: E402
import policies          # noqa: E402
import response_cache    # noqa: E402

TASKSET = REPO / "taskset.jsonl"


def load_tasks():
    if not TASKSET.exists():
        sys.exit("taskset.jsonl not built. Run: python3 build_taskset.py")
    return [json.loads(l) for l in TASKSET.open(encoding="utf-8") if l.strip()]


def already_cached(tier, task, temperature, sample_idx, kind):
    """Is this exact call already on disk as a REAL response?

    Asks for the real key specifically. A mock entry for the same call must not
    count as recorded - that confusion is the bug this whole exercise came out
    of.
    """
    prompt = models.build_prompt(task, kind)
    key = response_cache.make_key(
        mode="real", model=models.MODELS[tier]["id"], prompt=prompt,
        temperature=temperature, sample_idx=sample_idx,
        max_tokens=models._max_tokens_for(kind), mock_seed=None,
    )
    return response_cache.get(key) is not None


def missing_calls(tasks, want_route=True):
    """Every call a full replay needs that the cache does not have."""
    out = []

    # 1. Self-consistency samples. See the module docstring for why this
    #    enumeration is exact rather than approximate.
    samplable = [t for t in models.TIERS
                 if models.MODELS[t]["accepts_temperature"]]
    for task in tasks:
        if task["domain"] != "math":
            continue
        for tier in samplable:
            for idx in range(1, policies.SELF_CONSISTENCY_K):
                call = (tier, task, 0.8, idx, "answer")
                if not already_cached(*call):
                    out.append(call)

    # 2. llm_router's classification call, on every task in both domains. Eight
    #    output tokens, so it is a rounding error next to the samples above -
    #    but it is the entire reason that policy cannot replay.
    if want_route:
        for task in tasks:
            call = ("cheap", task, 0.0, 0, "route")
            if not already_cached(*call):
                out.append(call)

    return out


def unit_costs(ladder):
    """Mean real cost of one cheap call, split by domain and kind.

    Per domain because the two differ by an order of magnitude here - level-5
    maths draws long derivations and MBPP+ solutions are short - and a single
    average over the cache would misprice whichever domain dominates it.
    Falls back to the price table when a cell has never been called.
    """
    path = REPO / "cache" / f"raw_calls.{ladder}.jsonl"
    totals, counts = {}, {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("mode") != "real":
                continue
            k = (d.get("tier"), d.get("domain"), d.get("kind"))
            totals[k] = totals.get(k, 0.0) + d.get("cost_usd", 0.0)
            counts[k] = counts.get(k, 0) + 1
    return {k: totals[k] / counts[k] for k in totals if counts[k]}


def estimate(calls, unit):
    """Priced from measured calls of the same tier, domain and kind."""
    total = 0.0
    unpriced = 0
    for tier, task, _temp, _idx, kind in calls:
        cell = unit.get((tier, task["domain"], kind))
        if cell is None:
            # No measurement for this cell - only ever true of `route`, which
            # has never been called for real. Model it: the prompt, plus
            # ROUTER_MAX_TOKENS of output.
            prompt = models.build_prompt(task, kind)
            cell = models._price(
                tier, tokens_in=len(prompt) // 4,
                tokens_out=models._max_tokens_for(kind),
            )
            unpriced += 1
        total += cell
    return total, unpriced


def main():
    ap = argparse.ArgumentParser(
        description="Record the calls a full replay needs but the cache lacks.")
    ap.add_argument("--no-route", action="store_true",
                    help="skip llm_router's classification calls")
    ap.add_argument("--max-spend", type=float, default=1.00,
                    help="refuse to start if the estimate exceeds this (default $1)")
    ap.add_argument("--go", action="store_true",
                    help="actually make the calls. Without this, plan only.")
    args = ap.parse_args()

    tasks = load_tasks()
    ladder = models.LADDER
    calls = missing_calls(tasks, want_route=not args.no_route)

    by_kind = {}
    for tier, task, temp, idx, kind in calls:
        label = f"{kind}  tier={tier}  domain={task['domain']}  temp={temp}"
        by_kind[label] = by_kind.get(label, 0) + 1

    unit = unit_costs(ladder)
    est, unpriced = estimate(calls, unit)

    print(f"ladder    {ladder}   mode {models.MODE}")
    print(f"tasks     {len(tasks)}  "
          f"(math {sum(1 for t in tasks if t['domain'] == 'math')}, "
          f"code {sum(1 for t in tasks if t['domain'] == 'code')})")
    print(f"k         SELF_CONSISTENCY_K = {policies.SELF_CONSISTENCY_K}, "
          f"so {policies.SELF_CONSISTENCY_K - 1} extra samples per maths task per "
          f"samplable tier")
    print(f"samplable {', '.join(t for t in models.TIERS if models.MODELS[t]['accepts_temperature']) or '(none)'}"
          f"   - tiers that reject a temperature make no sample calls at all")
    print()

    if not calls:
        print("Nothing missing. A full replay has everything it needs.")
        return

    print(f"missing   {len(calls)} calls")
    for label, n in sorted(by_kind.items()):
        print(f"            {n:>4}  {label}")
    print(f"estimate  ${est:.4f}"
          + (f"   ({unpriced} modelled from the price table, never measured)"
             if unpriced else ""))
    print(f"          output length varies per draw, so treat this as +/- 25%.")

    if est > args.max_spend:
        sys.exit(f"\nRefusing: estimate ${est:.4f} exceeds --max-spend "
                 f"${args.max_spend:.2f}.")

    if not args.go:
        print("\nPlan only - nothing was called. Add --go to spend.")
        print("Set ROUTER_MODE=real first, or this replays and records nothing.")
        return

    if models.MODE != "real":
        sys.exit(f"\nROUTER_MODE is {models.MODE!r}. Refusing: only real mode "
                 f"can record a response that was not already there.")

    print(f"\nRecording {len(calls)} calls...", file=sys.stderr)
    spent = 0.0
    for i, (tier, task, temp, idx, kind) in enumerate(calls, 1):
        r = models.call(tier, task, temperature=temp, sample_idx=idx, kind=kind)
        spent += r.cost_usd
        if i % 25 == 0 or i == len(calls):
            print(f"  [{i}/{len(calls)}]  spent ${spent:.4f}", file=sys.stderr)

    print(f"\nrecorded {len(calls)} calls for ${spent:.4f} "
          f"(estimated ${est:.4f})")
    if models.truncated_ids:
        # The deduped set, not a raw count: this script records each response
        # once, but the set is what stays correct if that ever stops being true.
        print(f"!! {len(models.truncated_ids)} response(s) hit max_tokens. A "
              f"truncated answer grades as WRONG, so it is a MISSING "
              f"measurement rather than a capability result:")
        for task_id, tier, kind, idx in sorted(models.truncated_ids):
            print(f"     {task_id:<16} {tier:<10} {kind} sample={idx}")
        print("   Raising models.MAX_TOKENS re-charges every cached response "
              "(SHIP_PLAN.md section 1). Exclude the task instead.")
    print("\nEverything replays free from here:")
    print(f"  ROUTER_MODE=replay ROUTER_LADDER={ladder} python3 run_eval.py")


if __name__ == "__main__":
    main()
