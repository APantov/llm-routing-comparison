#!/usr/bin/env python3
"""Smoke-test the MCP server without a client.

Registration is where MCP servers fail silently: a decorator typo means the
tool simply is not there, and the server still starts. This exercises the same
code paths a client would - list, call, read - and prints what an MCP client
would see.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Run from scripts/, import from the repo root.
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

os.environ.setdefault("ROUTER_LADDER", "wide")
os.environ.setdefault("ROUTER_MODE", "replay")

EXPECTED_TOOLS = {"route_query", "estimate_cost", "compare_policies", "explain_routing"}
EXPECTED_RESOURCES = {
    "routing://ladders",
    "routing://findings/probe",
    "routing://findings/verifiers",
    "routing://policies",
}


async def main() -> int:
    from router_agent.mcp_server import mcp

    problems: list[str] = []

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    print(f"tools ({len(names)}):")
    for t in sorted(tools, key=lambda x: x.name):
        n = len(t.description or "")
        print(f"  {t.name:<20} {n:>4} chars of description")
        if n < 120:
            problems.append(f"{t.name}: description too thin for a model to route on")
    if names != EXPECTED_TOOLS:
        problems.append(f"tools mismatch: {names ^ EXPECTED_TOOLS}")

    resources = await mcp.list_resources()
    uris = {str(r.uri) for r in resources}
    print(f"\nresources ({len(uris)}):")
    for u in sorted(uris):
        print(f"  {u}")
    if uris != EXPECTED_RESOURCES:
        problems.append(f"resources mismatch: {uris ^ EXPECTED_RESOURCES}")

    prompts = await mcp.list_prompts()
    print(f"\nprompts ({len(prompts)}):")
    for p in prompts:
        print(f"  {p.name}")

    # Exercise the free tools for real. `estimate_cost` and `explain_routing`
    # call no model, so this stays free in any mode.
    print("\ncalling explain_routing(deepseek):")
    r = await mcp.call_tool("explain_routing", {"ladder": "deepseek"})
    if r.is_error:
        problems.append("explain_routing returned an error")
    else:
        v = r.structured_content["ratio"]
        print(f"  verdict={v['verdict']}  "
              f"cascade vs always-best {v['cascade_vs_always_best_pct']:+}%  "
              f"from {v['economics_source']}")
        # Checks that the verdict is TRACEABLE, not that it is a particular
        # word. This used to assert `route` for deepseek, from a
        # mock-era constant the measurement later contradicted - the real
        # frontier for that ladder says cascade, by 4.4%.
        if v["verdict"] not in ("route", "cascade"):
            problems.append(f"explain_routing gave no usable verdict: {v['verdict']!r}")
        if v.get("economics_source") != "frontier.deepseek.jsonl":
            problems.append(
                f"the verdict should come from that ladder's own frontier run, "
                f"got {v.get('economics_source')!r}")
        if v.get("economics_simulated") is not False:
            problems.append("the served verdict is not backed by real responses")

    print("\ncalling estimate_cost:")
    r = await mcp.call_tool("estimate_cost", {"query": "What is 17 times 23?"})
    if r.is_error:
        problems.append("estimate_cost returned an error")
    else:
        for e in r.structured_content["estimates"]:
            print(f"  {e['policy']:<18} ${e['min_usd']:.6f} - ${e['max_usd']:.6f}")

    print("\nreading routing://findings/probe:")
    chunks = list(await mcp.read_resource("routing://findings/probe"))
    payload = json.loads(chunks[0].content)
    if "routable_pct" in payload:
        print(f"  routable={payload['routable_pct']}%  "
              f"ceiling={payload['ceiling_over_cheap_pct']}%  n={payload['n']}")
    else:
        print(f"  {payload.get('error', 'unexpected payload')}")

    print()
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1
    print("OK: MCP server registers and responds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
