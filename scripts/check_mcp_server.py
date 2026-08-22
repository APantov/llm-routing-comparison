#!/usr/bin/env python3
"""Smoke-test the MCP server, in-process and then over the real transport.

Two failure modes, and only one of them is visible in-process.

Registration is where MCP servers fail silently: a decorator typo means the
tool simply is not there, and the server still starts. Phase 1 exercises the
same code paths a client would - list, call, read - and prints what an MCP
client would see.

Phase 2 exists because phase 1 cannot see the other one. On stdio, stdout IS
the JSON-RPC channel: one stray `print` anywhere beneath a tool corrupts the
frame and the client drops the response, while every in-process test still
passes. That is not hypothetical here - `response_cache` warns about stale
cache keys on its first load, which is triggered by the first `route_query`
and by nothing else, so the server listed its tools perfectly and then
returned a mangled answer to the first real call. Phase 2 launches the server
as a subprocess, speaks JSON-RPC to it by hand, and asserts every line of its
stdout parses. Warnings belong on stderr; this is what enforces it.

Phase 2 runs twice, because there are now two ways to speak MCP and the SDK
serves both. Revision 2026-07-28 deleted the initialize handshake: a modern
client sends its protocol version in `_meta` on every request and negotiates
through `server/discover` instead. Older clients still handshake. Exercising
one leg tests half of what ships - and the leg this check used to pin,
2025-06-18, had gone two revisions stale without ever failing, which is how a
compatibility check rots: silently, and green.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
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

# The two that call no model. Their annotations have to say so: a server whose
# argument is that model calls should be priced before they are made is not
# entitled to leave a client guessing which of its own calls spend.
READ_ONLY_TOOLS = {"estimate_cost", "explain_routing"}

# The stateless revision, and the last one that still had a handshake. Named
# here rather than left as whatever was current when this was written, since
# that is precisely how the old pin drifted out from under the check.
STATELESS_VERSION = "2026-07-28"
HANDSHAKE_VERSION = "2025-11-25"


def check_stdout_is_pure_jsonrpc(*, stateless: bool) -> list[str]:
    """Launch the server for real and assert nothing but JSON-RPC reaches stdout.

    `stateless` picks the dialect: 2026-07-28, where every request carries its
    own protocol version in `_meta` and there is no handshake, or the older
    initialize flow that shipping clients still send. Both have to serve, and
    both have to keep stdout clean - a stray `print` does not care which one
    is on the wire.

    Hand-written JSON-RPC rather than the client SDK on purpose: the SDK logs a
    malformed line and carries on, which is precisely the silence being tested
    for. Reading the raw pipe is what makes the corruption visible.

    `route_query` is the call that matters. It is the first thing to touch the
    response cache, so it is the first thing that can print - the free tools
    load nothing and would pass either way. The query need not be a cached one:
    the cache is read before the miss is discovered, so the error path proves
    the point just as well and keeps this independent of what the cache holds.
    """
    problems: list[str] = []
    label = "stateless" if stateless else "handshake"
    # id 3 is the tools/call in both dialects, so the drain below reads the
    # same either way.
    if stateless:
        meta = {
            "io.modelcontextprotocol/protocolVersion": STATELESS_VERSION,
            "io.modelcontextprotocol/clientCapabilities": {},
            "io.modelcontextprotocol/clientInfo": {
                "name": "check_mcp_server", "version": "0",
            },
        }
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "server/discover",
             "params": {"_meta": meta}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list",
             "params": {"_meta": meta}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                "name": "route_query", "arguments": {"query": "What is 2+2?"},
                "_meta": meta,
            }},
        ]
    else:
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": HANDSHAKE_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "check_mcp_server", "version": "0"},
            }},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                "name": "route_query", "arguments": {"query": "What is 2+2?"},
            }},
        ]
    proc = subprocess.Popen(
        [sys.executable, "-m", "router_agent.mcp_server"],
        cwd=REPO, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )

    # stdin stays OPEN until the last reply is in. The server treats EOF as
    # shutdown and exits without draining its queue, so closing it up front -
    # what subprocess.run does - loses the tools/call reply and the check would
    # pass by never testing anything.
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(
        target=lambda: ([lines.put(ln) for ln in proc.stdout], lines.put("")),
        daemon=True,
    ).start()
    for msg in requests:
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    replies: dict[int, dict] = {}
    lineno = 0

    def drain(until_id: int | None) -> bool:
        """Read stdout, checking every line. True if `until_id` arrived."""
        nonlocal lineno
        while True:
            try:
                line = lines.get(timeout=180).strip()
            except queue.Empty:
                problems.append("the server stopped responding over stdio")
                return False
            if not line:
                return until_id is None or until_id in replies
            lineno += 1
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                problems.append(
                    f"stdout line {lineno} is not JSON-RPC, so it corrupts "
                    f"the stream: {line[:120]!r}. Send it to stderr instead."
                )
                continue
            if msg.get("id") is not None:
                replies[msg["id"]] = msg
            if until_id is not None and until_id in replies:
                return True

    drain(until_id=3)

    # Then read to EOF. Stopping at the reply would be the easy mistake: the
    # child's stdout is block-buffered when it is a pipe, so a stray `print`
    # can be flushed at exit, arriving AFTER the reply it corrupts the stream
    # for. Checking only up to the reply lets exactly that regression pass.
    proc.stdin.close()
    drain(until_id=None)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    stderr = proc.stderr.read()

    if 3 not in replies:
        problems.append(
            f"{label}: no reply to tools/call - the server started but "
            f"could not serve"
        )
    else:
        print(f"  [{label}] tools/call route_query answered over stdio")

    if not stateless:
        if 1 not in replies:
            problems.append("handshake: the server never answered initialize")
        else:
            result = replies[1].get("result", {})
            info = result.get("serverInfo", {})
            print(f"  [{label}] {info.get('name')} {info.get('version')} "
                  f"on {result.get('protocolVersion')}")
    else:
        # Identity, which with the handshake gone arrives stamped on every result
        # instead of once at the start. A caller that never learns which server
        # answered cannot attribute a cost report to anything.
        stamp = (replies.get(3, {}).get("result") or {}).get("_meta", {})
        info = stamp.get("io.modelcontextprotocol/serverInfo", {})
        if not info.get("name"):
            problems.append(
                "stateless: results carry no serverInfo, so the caller never "
                "learns which server answered"
            )
        else:
            print(f"  [{label}] {info.get('name')} {info.get('version')} "
                  f"stamped on the result")

        if "result" not in replies.get(1, {}):
            problems.append(
                "stateless: server/discover did not answer, which is the only "
                "way to negotiate now that initialize is gone"
            )
        else:
            offered = replies[1]["result"].get("supportedVersions", [])
            print(f"  [{label}] server/discover offers "
                  f"{', '.join(offered) or 'nothing'}")
            if STATELESS_VERSION not in offered:
                problems.append(
                    f"stateless: server/discover does not offer "
                    f"{STATELESS_VERSION}, got {offered}"
                )

        # The cache hints, read where a client actually reads them. On the wire
        # rather than in-process because the SDK fills them at the boundary: the
        # constructor argument can be right and the served result still wrong.
        listed = replies.get(2, {}).get("result") or {}
        if not listed.get("ttlMs"):
            problems.append(
                f"stateless: tools/list carries ttlMs={listed.get('ttlMs')!r}, "
                f"telling every client never to cache a list that is fixed at "
                f"import and cannot change"
            )
        elif listed.get("cacheScope") != "public":
            problems.append(
                f"stateless: tools/list cacheScope is "
                f"{listed.get('cacheScope')!r}, but this tool surface is "
                f"identical for every caller"
            )
        else:
            print(f"  [{label}] tools/list ttlMs={listed['ttlMs']} "
                  f"cacheScope={listed['cacheScope']}")
    if stderr.strip():
        # Not a failure. Warnings on stderr are the correct outcome, and
        # showing them here is how a reader sees where they went.
        print(f"  stderr (correctly, not stdout): "
              f"{stderr.strip().splitlines()[0][:90]}")
    return problems


async def main() -> int:
    from router_agent.mcp_server import mcp

    problems: list[str] = []

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    print(f"tools ({len(names)}):")
    for t in sorted(tools, key=lambda x: x.name):
        n = len(t.description or "")
        free = t.name in READ_ONLY_TOOLS
        print(f"  {t.name:<20} {n:>4} chars of description  "
              f"{'free' if free else 'spends'}")
        if n < 120:
            problems.append(f"{t.name}: description too thin for a model to route on")
        if t.annotations is None:
            problems.append(
                f"{t.name}: no annotations, so a client cannot tell whether "
                f"calling it spends money"
            )
        elif t.annotations.read_only_hint is not free:
            problems.append(
                f"{t.name}: read_only_hint is "
                f"{t.annotations.read_only_hint!r}, but this tool "
                f"{'calls no model' if free else 'calls models'}"
            )
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

    print(f"\nphase 2 - over the real stdio transport, both dialects:")
    problems += check_stdout_is_pure_jsonrpc(stateless=True)
    problems += check_stdout_is_pure_jsonrpc(stateless=False)

    print()
    if problems:
        for p in problems:
            print(f"FAIL: {p}", file=sys.stderr)
        return 1
    print("OK: MCP server registers, responds, and keeps stdout clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
