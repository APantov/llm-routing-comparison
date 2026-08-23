#!/usr/bin/env python3
"""The MCP server, demonstrated end to end against a live client session.

    python scripts/demo_mcp.py

No API key, no account, no spend. Every model response below was bought once
against the real DeepSeek and Anthropic APIs and committed, so this reproduces
field for field on any machine.

`demo.py` shows what the ROUTER does. This shows what the SERVER does - the
three MCP primitives, in the order a client meets them:

    1. discovery   what the server advertises, and which calls cost money
    2. resources   the benchmark's findings, read-only and addressable
    3. tools       a verdict, a projection, and a routed answer
    4. the pause   human approval inside the escalation loop

It runs over a real stdio subprocess through the official MCP client, not
in-process, so what you see is what any MCP client would get.

TWO SERVERS, and the reason is the point of ROUTER_K. Self-consistency samples
are cached per sample index, so `k` is pinned at start-up to what the responses
were actually bought under - k=3 for the query that verifies at the cheap rung,
k=4 for the one whose fourth draw disagrees and triggers the escalation. Asking
a cache for a sample nobody purchased is a miss, not a cheaper answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

W = 74

# The two queries the committed cache can serve, and what each is here to show.
BS = chr(92)  # written this way because a stray shell can eat a backslash
VERIFIES = (
    "Let f(x) = x^3 - 3x + 1. Find the sum of the squares of all real "
    "roots. Give the final answer in " + BS + "boxed{}."
)
ESCALATES = (
    "Let N be the number of ordered triples (a,b,c) of positive integers "
    "with a*b*c = 2310 and a<b<c. Compute N. Answer in " + BS + "boxed{}."
)

_spent = {"attributed": 0.0, "backend": 0.0}


def rule(n: int, title: str) -> None:
    print(f"\n{'=' * W}\n  {n}. {title}\n{'=' * W}")


def note(text: str) -> None:
    print(f"\n  -> {text}")


@contextlib.asynccontextmanager
async def server(**env):
    """One stdio server, held open for as many calls as the caller makes.

    Held open deliberately. The approval checkpoint lives in the server's
    memory, so a client that starts a fresh process per call - which is what
    `mcp_call.py` does - can never resume what the previous one paused.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "router_agent.mcp_server"],
        # The server reads ROUTER_* at import time, so the ladder and mode are
        # chosen here and cannot change under a running process.
        env={**os.environ, "PYTHONIOENCODING": "utf-8",
             "ROUTER_LADDER": "wide", "ROUTER_MODE": "replay", **env},
        cwd=str(REPO),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            yield sess


async def call(sess, tool: str, **args) -> dict:
    result = await sess.call_tool(tool, args)
    payload = result.structured_content
    if payload is None:
        raise SystemExit(f"{tool} returned no structured content: {result.content}")
    if "cost_usd" in payload:
        _spent["attributed"] += payload["cost_usd"]
        _spent["backend"] += payload.get("backend_cost_usd", 0.0)
    return payload


async def act_one() -> None:
    async with server(ROUTER_K="3", ROUTER_AGREEMENT="1.0") as sess:
        rule(1, "Discovery - what the server advertises")
        tools = (await sess.list_tools()).tools
        print("\n  tools")
        for t in sorted(tools, key=lambda x: x.name):
            a = t.annotations
            spends = "free" if (a and a.read_only_hint) else "SPENDS"
            print(f"    {t.name:<18} {spends}")
        res = (await sess.list_resources()).resources
        print("\n  resources")
        for r in sorted(res, key=lambda x: str(x.uri)):
            print(f"    {str(r.uri)}")
        print("\n  prompts")
        for p in (await sess.list_prompts()).prompts:
            print(f"    {p.name}")
        note("A server whose argument is that model calls should be priced "
             "before\n     they are made ought to say which of ITS OWN calls "
             "spend. Those\n     hints are static, so they describe the worst "
             "case: route_query is\n     free in replay, but a client that "
             "cached the tool list cannot know\n     which mode the server was "
             "restarted in.")

        rule(2, "A resource - the three ladders, and what each one measured")
        chunks = list((await sess.read_resource("routing://ladders")).contents)
        d = json.loads(chunks[0].text)
        # One line per ladder would truncate `claude`, whose three rung ids run
        # to 61 characters. A demo that clips a model id is asking to be
        # misread, so the rungs get their own line.
        for name, spec in sorted(d["ladders"].items()):
            f = spec["finding"]
            print(f"\n  {name:<9} {f['effective_ratio']:>6}x   -> {f['verdict']}")
            print(f"            {' -> '.join(r['model'] for r in spec['rungs'])}")
        note("Non-monotonic in the price ratio. deepseek at 3.1x says cascade,\n"
             "     claude at 6.5x says route, wide at 46x says cascade - claude "
             "sits\n     BETWEEN the other two and is the one that does NOT want "
             "a cascade,\n     so no threshold separates them. This repository "
             "set out to find\n     that threshold and measured that none "
             "exists.")

        rule(3, "A tool - the same question, two ladders, opposite answers")
        for ladder in ("wide", "claude"):
            r = (await call(sess, "explain_routing", ladder=ladder))["ratio"]
            print(f"\n  {ladder:<8} verdict={r['verdict']:<9} "
                  f"cascade vs always-best "
                  f"{r['cascade_vs_always_best_pct']:+.1f}%")
            print(f"           measured from {r['economics_source']}, "
                  f"n={r['economics_n']}, simulated={r['economics_simulated']}")
        note("This is the finding, and the router READS it rather than "
             "assuming\n     it. There used to be a constant here and two of "
             "its three verdicts\n     were backwards. A ladder with no "
             "frontier run gets no verdict.")

        rule(4, "A free tool - price the options before spending anything")
        est = await call(sess, "estimate_cost", query=VERIFIES, domain="math")
        for e in est["estimates"]:
            print(f"    {e['policy']:<18} ${e['min_usd']:.6f} - ${e['max_usd']:.6f}")
        note("No model was called to produce that. A cascade's floor is above\n"
             "     always_cheap because it also pays to verify, on every query,\n"
             "     including the ones it was always going to accept.")

        rule(5, "Routing a query - verified at the cheap rung")
        out = await call(sess, "route_query", query=VERIFIES, domain="math")
        for e in out["trace"]:
            print(f"    {e['node']:<10} {e['detail']}")
        print(f"\n    answered by  {out['final_model']} ({out['final_tier']})")
        print(f"    verified     {out['verified']}  via {out['verifier']}")
        print(f"    cost         ${out['cost_usd']:.6f}   "
              f"backend ${out['backend_cost_usd']:.6f}")
        note("Three draws from DeepSeek agreed, so the cascade accepted and "
             "never\n     called Opus 5 - about 27x cheaper. `verified` is the "
             "VERIFIER'S\n     OPINION, never a correctness claim: serving has "
             "no ground truth,\n     which is why the payload carries "
             "`verified_meaning` beside it.")
        print(f"\n    verified_meaning: {out['verified_meaning']}")


async def act_two() -> None:
    """A second server, because k and the approval threshold are start-up state."""
    async with server(ROUTER_K="4", ROUTER_AGREEMENT="1.0",
                      ROUTER_APPROVAL_USD="0.001") as sess:
        rule(6, "The pause - human approval inside the escalation loop")
        print("\n  server restarted with ROUTER_K=4 ROUTER_APPROVAL_USD=0.001")

        for approve in (False, True):
            out = await call(sess, "route_query", query=ESCALATES, domain="math")
            assert out["stop_reason"] == "awaiting_approval", out["stop_reason"]
            i = out["interrupted"]
            print(f"\n  {'-' * (W - 2)}")
            print(f"  PAUSED  {i['question']}")
            print(f"          spent so far ${i['spent_so_far_usd']:.4f}   "
                  f"thread {out['thread_id']}")
            print(f"          client answers: "
                  f"{'APPROVE' if approve else 'DENY'}")

            done = await call(sess, "resume_routing",
                              thread_id=out["thread_id"], approved=approve)
            print(f"          -> {done['stop_reason']}, "
                  f"answered by {done['final_model']}, "
                  f"{done['escalations']} escalation(s), "
                  f"${done['cost_usd']:.6f}")

        note("The graph SUSPENDED mid-cascade and waited. One keystroke "
             "separates\n     $0.0008 from $0.0135 on the same query. Approval "
             "is asked per\n     escalation, so a three-rung ladder asks twice, "
             "and the checkpoint\n     lives in this server process - both "
             "calls had to reach the same\n     running server.")


async def main() -> int:
    print()
    print("=" * W)
    print("  The llm-routing MCP server, over a real stdio client session")
    print("  ladder=wide (DeepSeek v4-flash -> Opus 5)   mode=replay")
    print("=" * W)

    await act_one()
    await act_two()

    print()
    print("=" * W)
    print(f"  attributed cost of everything above : ${_spent['attributed']:.6f}")
    print(f"  actually spent by this run          : ${_spent['backend']:.6f}")
    print("=" * W)
    print("\n  The first is what serving these queries would cost in "
          "production.\n  The second is what left the account. They are "
          "separate on purpose,\n  and in replay the second is always zero.\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        raise SystemExit(130)
