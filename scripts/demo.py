#!/usr/bin/env python3
"""Replay the three routing scenarios that matter, from committed real data.

    python scripts/demo.py

No API key, no account, no spend. Every response below was paid for once
against the real DeepSeek and Anthropic APIs and committed to
`cache/raw_calls.wide.jsonl`, so this reproduces the exact traces field for
field on any machine.

The three scenarios are chosen because together they are the argument:

    1. verified at the cheap rung   the cascade's win: ~80x saved
    2. escalated                    the cascade's cost: it paid twice
    3. the perfect verifier         what a cascade does when it can check exactly

Run it with ROUTER_MODE=real and a key to re-derive them from scratch; the
cost of doing so is printed at the end.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Must be set before `models` is imported: the ladder is built at module scope.
os.environ.setdefault("ROUTER_LADDER", "wide")
os.environ.setdefault("ROUTER_MODE", "replay")
os.environ.setdefault("ROUTER_ALLOW_CODE_EXEC", "1")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


SCENARIOS = [
    {
        "title": "1. The cascade's win - verified at the cheap rung",
        "query": (
            "Let f(x) = x^3 - 3x + 1. Find the sum of the squares of all real "
            "roots. Give the final answer in \\boxed{}."
        ),
        "domain": "math",
        "tests": None,
        "kwargs": {"self_consistency_k": 3, "agreement_threshold": 1.0},
        "point": (
            "Three independent draws from DeepSeek all gave 6, which is right. "
            "Unanimous agreement, so the cascade accepted and never called "
            "Opus 5. Pricing Opus 5 for an answer of the same length puts it "
            "at ~$0.0084 against this run's $0.000315 - about 27x. That is a "
            "projection, not a measurement: Opus was never called, and it "
            "tends to write longer answers, so 27x is a conservative floor."
        ),
    },
    {
        "title": "2. The cascade's cost - it disagreed with itself, so it paid twice",
        "query": (
            "Let N be the number of ordered triples (a,b,c) of positive "
            "integers with a*b*c = 2310 and a<b<c. Compute N. Answer in "
            "\\boxed{}."
        ),
        "domain": "math",
        "tests": None,
        "kwargs": {"self_consistency_k": 4, "agreement_threshold": 1.0},
        "point": (
            "DeepSeek agreed with itself only 75% of the time, so verification "
            "rejected and the cascade escalated. Note it paid for BOTH rungs: "
            "this is the case where a cascade is more expensive than routing "
            "straight to the top, and it is why the price ratio decides "
            "whether cascading is worth it at all."
        ),
    },
    {
        "title": "3. The perfect verifier - caller supplied the tests",
        "query": (
            "Write a Python function called dedupe_sorted(xs) that removes "
            "duplicates from a sorted list, preserving order, without using "
            "set()."
        ),
        "domain": "code",
        "tests": [
            "assert dedupe_sorted([1,1,2,3,3,3]) == [1,2,3]",
            "assert dedupe_sorted([]) == []",
            "assert dedupe_sorted([5]) == [5]",
        ],
        "kwargs": {"allow_code_execution": True},
        "point": (
            "Tests were executed, not sampled. Exact verification, zero "
            "verification cost, and the cheap rung was provably sufficient - "
            "Opus 5 priced at the same answer length would have been ~81x "
            "this run's $0.000031. This is the cascade at its strongest, and "
            "it is only available because the CALLER brought the tests. Most "
            "production traffic cannot, which is what sweep_degraded.py "
            "prices."
        ),
    },
]


def main() -> int:
    from router_agent.config import RouterConfig
    from router_agent.engine import route
    import models

    mode = os.environ["ROUTER_MODE"]
    print()
    print("=" * 74)
    print("  LLM routing - three live traces, replayed from committed real data")
    print(f"  ladder=wide (DeepSeek v4-flash -> Opus 5)   mode={mode}")
    print("=" * 74)

    total_attributed = 0.0
    failures = 0

    for s in SCENARIOS:
        print()
        print(s["title"])
        print("-" * 74)
        print(f"  query: {s['query'][:120]}")
        if s["tests"]:
            print(f"  tests: {len(s['tests'])} asserts supplied by the caller")
        print()

        cfg = RouterConfig(
            mode=mode, ladder="wide", policy="cascade",
            max_cost_usd=0.50, **s["kwargs"],
        )
        try:
            out = route(s["query"], cfg=cfg, domain=s["domain"], tests=s["tests"])
        except KeyError as exc:
            print("  SKIPPED - no cached response for this scenario.")
            print(f"  {str(exc).splitlines()[0]}")
            failures += 1
            continue

        d = out.to_dict()
        for e in d["trace"]:
            print(f"    {e['node']:<9} {e['detail']}")

        total_attributed += d["cost_usd"]
        print()
        print(f"    answered by  {d['final_model']}")
        print(f"    verified     {d['verified']}  ({d['verifier']})")
        print(f"    cost         ${d['cost_usd']:.6f}"
              f"   backend ${d['backend_cost_usd']:.6f}")
        print()
        print(f"    -> {s['point']}")

    print()
    print("=" * 74)
    print(f"  attributed cost of the three queries : ${total_attributed:.6f}")
    print(f"  actually spent by this run           : "
          f"${0.0 if mode != 'real' else total_attributed:.6f}")
    print(f"  backend calls made                   : {models.call_stats['backend']}")
    print("=" * 74)
    print()
    print("  `cost` is what serving these queries would cost in production.")
    print("  In replay mode the run itself spends nothing - the responses were")
    print("  bought once and committed. That is the same distinction run_eval.py")
    print("  draws between attributed cost and what leaves your account.")
    print()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
