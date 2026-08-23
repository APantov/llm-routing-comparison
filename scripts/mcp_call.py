#!/usr/bin/env python3
"""Call one MCP tool or resource from the terminal, as a client would.

    python scripts/mcp_call.py --list
    python scripts/mcp_call.py explain_routing ladder=claude
    python scripts/mcp_call.py estimate_cost query="prove sqrt(2) is irrational"
    python scripts/mcp_call.py --resource routing://findings/probe

`check_mcp_server.py` answers "is the server correct". This answers "what does
it say", which is the question you actually have while building against it.

Piping JSON-RPC into the server by hand does not work, and the way it fails is
worth knowing: the server treats stdin EOF as shutdown and exits without
draining its queue, so `echo '...' | python -m router_agent.mcp_server` prints
the initialize reply, silently drops the tool call, and exits 0. A client holds
the pipe open until the reply arrives. This one does that.

Mode is inherited from the environment, so the same command can be free or not:

    ROUTER_MODE=replay   answers from committed responses, cannot spend  (default)
    ROUTER_MODE=mock     fabricated answers, offline, no key
    ROUTER_MODE=real     calls the provider and BILLS YOU

Real mode is the one to be deliberate about. `cost_usd` is a production
projection and is printed in every mode; `backend_cost_usd` is what actually
left the account, and it is zero in every mode but `real`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mcp_call.py",
        description="Call one tool or resource on the llm-routing MCP server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python scripts/mcp_call.py --list
  python scripts/mcp_call.py explain_routing ladder=wide
  python scripts/mcp_call.py estimate_cost query="prove sqrt(2) is irrational"
  python scripts/mcp_call.py --resource routing://ladders

  # a real DeepSeek call, through MCP, that spends money:
  ROUTER_MODE=real ROUTER_LADDER=deepseek \\
      python scripts/mcp_call.py route_query query="What is 17 * 23?"

arguments are key=value. Values are parsed as JSON when they parse, so
  max_cost_usd=0.01   is a number
  policies='["cascade","always_cheap"]'   is a list
and anything else is a string.
""",
    )
    p.add_argument("tool", nargs="?", help="tool name, e.g. route_query")
    p.add_argument("args", nargs="*", metavar="key=value")
    p.add_argument("--list", action="store_true",
                   help="list tools, resources and prompts, then exit")
    p.add_argument("--resource", metavar="URI",
                   help="read a resource instead of calling a tool")
    p.add_argument("--raw", action="store_true",
                   help="print the whole payload, not the summary")
    return p.parse_args(argv)


def first_sentence(text: str) -> str:
    """The first whole sentence of a description, or all of it if it has none.

    Splitting on `.` alone would cut `deepseek at 3.11x` in half, so a sentence
    ends at a period followed by whitespace or by nothing - which a decimal
    point never is.
    """
    text = " ".join(text.split())
    match = re.search(r"\.(?:\s|$)", text)
    return text[:match.start() + 1] if match else text


def entry(label: str, text: str, label_width: int) -> str:
    """One `label   description` row, wrapped rather than cut.

    The previous version took `description[:78]` and appended a full stop, so a
    description longer than that came back chopped mid-word and wearing a
    period - `compare what each return.` reads as a finished sentence and is
    not one. Truncation that cannot be seen is worse than a wrapped line: this
    listing is the first thing anyone runs against the server, and it is the
    only place most callers ever read what a tool does.
    """
    width = min(max(shutil.get_terminal_size((100, 24)).columns, 60), 100)
    gutter = 2 + label_width + 1
    # Wrapped against the space left AFTER the label column, not the full
    # width, or the first line overruns by exactly the gutter.
    lines = textwrap.wrap(text, width=width - gutter) or [""]
    return "\n".join(
        [f"  {label:<{label_width}} {lines[0]}"]
        + [" " * gutter + line for line in lines[1:]]
    )


def parse_tool_args(pairs: list[str]) -> dict:
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"arguments must be key=value - got {pair!r}")
        key, _, value = pair.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def summarise(tool: str, payload: dict) -> None:
    """Print the few fields that carry the point, per tool.

    Falls through to raw JSON for anything unrecognised, so a tool added later
    still prints something useful rather than nothing.
    """
    if "error" in payload:
        print(f"  {payload['error']}: {payload.get('detail', '')}")
        return

    if tool == "route_query":
        print(f"  answered by  {payload['final_model']} ({payload['final_tier']})")
        print(f"  verified     {payload['verified']}  via {payload['verifier']}")
        print(f"  cost         ${payload['cost_usd']:.6f}   "
              f"backend ${payload['backend_cost_usd']:.6f}")
        print(f"  escalations  {payload['escalations']}   "
              f"stop: {payload['stop_reason']}")
        for e in payload["trace"]:
            print(f"    {e['node']:<10} {e['detail']}")
        print(f"\n  verified means: {payload['verified_meaning']}")
        answer = (payload.get("answer") or "").strip()
        if answer:
            print(f"\n  answer: {answer[:600]}")
    elif tool == "estimate_cost":
        print(f"  {payload['domain']} on {payload['ladder']}: "
              f"{payload['rungs']['cheap']} -> {payload['rungs']['expensive']}")
        for e in payload["estimates"]:
            print(f"    {e['policy']:<18} ${e['min_usd']:.6f} - ${e['max_usd']:.6f}")
    elif tool == "compare_policies":
        for row in payload["results"]:
            if "error" in row:
                print(f"    {row['policy']:<18} {row['error']}")
                continue
            print(f"    {row['policy']:<18} {row['answered_by']:<20} "
                  f"verified={str(row['verified']):<5} ${row['cost_usd']:.6f}  "
                  f"calls={row['n_calls']}")
        print(f"  cheapest: {payload['cheapest_policy']}")
        print(f"\n  {payload['caveat']}")
    elif tool == "explain_routing":
        r = payload["ratio"]
        if not r.get("known"):
            print(f"  no verdict for this ladder: {r.get('why', 'no frontier run')}")
            return
        print(f"  ladder    {r['ladder']}  ({r['rungs']})")
        print(f"  verdict   {r['verdict']}")
        print(f"  cascade vs always-best, at matched accuracy   "
              f"{r['cascade_vs_always_best_pct']:+}%")
        print(f"  measured from {r['economics_source']}, n={r['economics_n']}, "
              f"simulated={r['economics_simulated']}")
    else:
        print(json.dumps(payload, indent=2))


async def run(ns: argparse.Namespace) -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # Read .env into the environment before reporting what the run is, so the
    # banner states what the SERVER will load rather than what this shell
    # happens to export. Not cosmetic: which ladder is loaded is the thing that
    # flips the verdict in this repository, and a banner that says `claude`
    # while the server serves `wide` is worse than no banner. `load_dotenv`
    # never overwrites an existing variable, so an inline
    # `ROUTER_MODE=real ...` still wins over the file.
    from llm_routing.models import load_dotenv
    load_dotenv()

    mode = os.environ.get("ROUTER_MODE", "replay")
    ladder = os.environ.get("ROUTER_LADDER", "claude")
    print(f"server: python -m router_agent.mcp_server  "
          f"[ladder={ladder} mode={mode}]")
    if mode == "real":
        print("  !! real mode: this call goes to the provider and is billed.")

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "router_agent.mcp_server"],
        # The server reads ROUTER_* at import time, so it must inherit the
        # environment this was launched with - that is the whole interface for
        # choosing a ladder and a mode from the command line.
        env={**os.environ, "ROUTER_MODE": mode, "PYTHONIOENCODING": "utf-8"},
        cwd=str(REPO),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print(f"        {info.server_info.name} {info.server_info.version}\n")

            if ns.list:
                tools = await session.list_tools()
                print("tools:")
                for t in sorted(tools.tools, key=lambda x: x.name):
                    print(entry(t.name, first_sentence(t.description or ""), 18))
                res = await session.list_resources()
                print("\nresources:")
                for r in sorted(res.resources, key=lambda x: str(x.uri)):
                    print(entry(str(r.uri), first_sentence(r.description or ""), 30))
                prompts = await session.list_prompts()
                print("\nprompts:")
                for p in prompts.prompts:
                    print(f"  {p.name}")
                return 0

            if ns.resource:
                chunks = list((await session.read_resource(ns.resource)).contents)
                print(chunks[0].text)
                return 0

            if not ns.tool:
                print("nothing to do: name a tool, or pass --list / --resource",
                      file=sys.stderr)
                return 2

            result = await session.call_tool(ns.tool, parse_tool_args(ns.args))
            if result.is_error:
                for block in result.content:
                    print(getattr(block, "text", block), file=sys.stderr)
                return 1
            payload = result.structured_content
            print(f"{ns.tool}:")
            if ns.raw or payload is None:
                print(json.dumps(payload, indent=2) if payload else
                      "\n".join(getattr(b, "text", "") for b in result.content))
            else:
                summarise(ns.tool, payload)
            return 0


def main() -> int:
    ns = parse_args()
    os.environ.setdefault("ROUTER_MODE", "replay")
    try:
        return asyncio.run(run(ns))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
