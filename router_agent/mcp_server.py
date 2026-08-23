"""MCP server: the routing engine, exposed as tools any MCP client can call.

Worth exposing over MCP rather than leaving as a library because an agent that
calls models is already routing, implicitly and usually badly - one model in its
config, used for everything. A `route_query` tool makes that explicit, priced and
auditable, and reports what it spent.

All three MCP primitives, each for a distinct reason:

    tools      actions with side effects and cost - routing, comparing
    resources  the benchmark's findings, read-only, addressable by URI
    prompts    a workflow that walks a client through choosing a policy

`route_query` can pause: with ROUTER_APPROVAL_USD set, an escalation dearer
than that suspends the graph and comes back `awaiting_approval` with a
`thread_id`, and `resume_routing` carries the human's answer back in. The
checkpoint lives in this process, so both calls have to reach the same running
server - a client that spawns one per call cannot resume.

Run it:

    llm-router-mcp                          # stdio, the usual transport
    python -m router_agent.mcp_server

Register it with an MCP client - the Claude Desktop / Claude Code form:

    {
      "mcpServers": {
        "llm-routing": {
          "command": "llm-router-mcp",
          "env": {"ROUTER_LADDER": "wide", "ROUTER_MODE": "replay"}
        }
      }
    }

`ROUTER_MODE=replay` is the safe default to register with: the server answers
from committed real responses and cannot spend money. Switch to `real` with an
API key when you want it to serve arbitrary queries.
"""

from __future__ import annotations

import json
import os
from typing import Any

from mcp.server.caching import CacheHint
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

INSTRUCTIONS = """\
Cost-aware LLM routing, backed by a benchmark in the same repository.

Use `route_query` when you want an answer produced at the lowest rung that can
be verified to produce it, rather than paying top-rung prices by default.

Two things to understand before reading its output:

* `verified` is a VERIFIER'S OPINION, never a correctness claim. Serving has no
  ground truth. Read `verified_meaning` alongside it - it says exactly what was
  and was not measured.
* `cost_usd` is what serving the query would cost in production. In `replay` -
  the default - the run itself spends nothing; `backend_cost_usd` is what
  actually left the account. A query nobody paid for returns
  `error: no_cached_response` rather than an invented answer.

Call `explain_routing` first if you are choosing a policy. Whether cascading is
the right architecture is a measured property of the ladder you loaded, and on
that ladder it may be the wrong answer. Do not infer it from the price ratio
between the rungs: the three measured ladders are non-monotonic in that ratio
- `claude` sits between the other two and is the one that does not want a
cascade - so no threshold rule separates them.
"""

# What a client may cache, and for how long. Every list this server serves is
# fixed at import time by the decorators below - five tools, four resources,
# one prompt - so a client re-listing on each connection is asking again for
# something that cannot have changed. Left unset, the SDK sends
# `ttlMs=0, cacheScope=private`, which tells a client exactly the opposite.
#
# `public` throughout because nothing here varies by caller: the tool surface
# is identical for everyone, and the resources serve committed benchmark
# artefacts rather than per-user state. `resources/read` gets the shorter hour
# because it is the only one that reads the filesystem, so it is the only one a
# fresh benchmark run can invalidate under a running server.
_LIST_TTL_MS = 24 * 60 * 60 * 1000
_READ_TTL_MS = 60 * 60 * 1000
CACHE_HINTS = {
    "server/discover": CacheHint(ttl_ms=_LIST_TTL_MS, scope="public"),
    "tools/list": CacheHint(ttl_ms=_LIST_TTL_MS, scope="public"),
    "prompts/list": CacheHint(ttl_ms=_LIST_TTL_MS, scope="public"),
    "resources/list": CacheHint(ttl_ms=_LIST_TTL_MS, scope="public"),
    "resources/templates/list": CacheHint(ttl_ms=_LIST_TTL_MS, scope="public"),
    "resources/read": CacheHint(ttl_ms=_READ_TTL_MS, scope="public"),
}

# Which calls cost money. A server whose whole argument is that model calls
# should be priced before they are made ought to say which of its OWN calls
# spend, and until now it did not.
#
# These are static, so they describe the WORST case. `route_query` and
# `compare_policies` spend nothing in replay - the default - but a client that
# cached the tool list has no way to know which mode the server was started in,
# and a hint reading "free" on a server since restarted in `real` mode is worse
# than no hint at all.
#
# MCP has no "costs money" hint, so the mapping is: calling a model is a
# real-world side effect, hence not read-only; nothing is destroyed, hence not
# destructive; and a cascade resamples, so the same query need not come back
# with the same answer or the same bill, hence not idempotent. `destructive`
# and `idempotent` are meaningful only when `read_only` is false, which is why
# FREE omits them rather than asserting something the spec would ignore.
FREE = ToolAnnotations(read_only_hint=True, open_world_hint=False)
SPENDS = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=False,
    open_world_hint=True,
)

mcp = MCPServer(
    name="llm-routing",
    title="Cost-aware LLM routing",
    version="0.2.0",
    instructions=INSTRUCTIONS,
    cache_hints=CACHE_HINTS,
)


def _cfg(**overrides):
    from router_agent.config import RouterConfig
    return RouterConfig.from_env(**overrides)


def _ladder() -> str:
    return os.environ.get("ROUTER_LADDER", "claude")


# Replay can only serve prompts that were paid for. Both tools that call the
# graph can hit it, and a caller who reads one wording here and a different one
# there has to work out whether they are the same condition. They are.
REPLAY_NOTE = (
    "This server is in replay mode - the default - which can only serve "
    "prompts that were actually paid for (the 417 benchmark tasks). Set "
    "ROUTER_MODE=real with an API key to serve arbitrary queries."
)


def _summarised(out) -> dict[str, Any]:
    """An outcome as a tool result.

    The trace is valuable but long, and a client that wanted every token of it
    can read the events itself. Summarise by default - and in ONE place, so
    `route_query` and `resume_routing` cannot drift into returning differently
    shaped results for the two halves of the same run.
    """
    result = out.to_dict()
    result["trace"] = [
        {"node": e["node"], "detail": e["detail"]} for e in result["trace"]
    ]
    return result


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    annotations=SPENDS,
    description=(
        "Route one query through a cost-aware cascade and return the answer "
        "with a full cost and verification trace. Call this instead of going "
        "straight to an expensive model when you want the cheapest rung that "
        "can be verified to answer. Returns the answer, which model produced "
        "it, whether verification accepted it, what it would cost in "
        "production, and a step-by-step trace of every routing decision. "
        "NOTE: `verified` is a verifier's opinion, not a correctness claim - "
        "read `verified_meaning` with it."
    ),
)
def route_query(
    query: str,
    domain: str = "auto",
    policy: str = "cascade",
    verifier: str = "auto",
    max_cost_usd: float | None = None,
    tests: list[str] | None = None,
) -> dict[str, Any]:
    """
    Args:
        query: The question or task to answer.
        domain: auto | math | code | general. `auto` infers from the text.
        policy: cascade | predictive | always_cheap | always_expensive.
            `cascade` answers cheaply, verifies, and escalates only on
            failure. `predictive` guesses once from the query text and commits.
        verifier: auto | self_consistency | tests | none. `auto` picks
            self-consistency, or caller-supplied tests when available and
            execution is enabled.
        max_cost_usd: Hard ceiling for this query, enforced before each
            escalation rather than after the fact.
        tests: Python assert statements the answer must pass. Enables the
            exact verifier - but only when code execution is explicitly
            enabled server-side, since running them executes model-generated
            code.
    """
    from llm_routing import models
    from router_agent.engine import route

    cfg = _cfg(
        policy=policy, domain=domain, verifier=verifier,
        max_cost_usd=max_cost_usd,
    )
    # ReplayMiss, not KeyError: a genuine KeyError raised inside the graph is a
    # bug in this server, and returning it to the caller as "no cached response"
    # would describe a defect as a configuration issue.
    try:
        out = route(query, cfg=cfg, tests=tests)
    except models.ReplayMiss as exc:
        return {
            "error": "no_cached_response",
            "detail": REPLAY_NOTE,
            "raw": str(exc)[:300],
        }

    return _summarised(out)


@mcp.tool(
    annotations=SPENDS,
    description=(
        "Finish a run that `route_query` paused for human approval. A "
        "route_query returning `stop_reason: awaiting_approval` has NOT "
        "answered yet - it is holding at the cheap rung with the escalation "
        "unpaid, and its `interrupted` payload names the model and the "
        "projected cost waiting on a decision. Pass that run's `thread_id` "
        "with approved=true to authorise the escalation and the spend, or "
        "approved=false to decline it and finalise on the answer already in "
        "hand. The whole point of the pause is that a human decides how their "
        "money is spent: ask them, and do not answer on their behalf."
    ),
)
def resume_routing(thread_id: str, approved: bool) -> dict[str, Any]:
    """
    Args:
        thread_id: From a `route_query` result whose stop_reason was
            `awaiting_approval`.
        approved: True authorises the escalation and the cost it projected.
            False declines it; the run finalises with `approval_denied` on
            the rung it already paid for.
    """
    from llm_routing import models
    from router_agent.engine import resume
    from router_agent.graph import compiled

    # Only `cascade` reaches the escalate node - the one-shot policies return
    # "accept" from the router edge - so a paused run is always a cascade, and
    # the default config reports its policy correctly without the caller
    # having to echo back what they started it with.
    cfg = _cfg()
    if cfg.require_approval_above_usd is None:
        return {
            "error": "approval_not_enabled",
            "detail": (
                "This server was started without ROUTER_APPROVAL_USD, so no "
                "run can pause and there is nothing to resume. Set it and "
                "restart the server - 0 asks before every escalation."
            ),
        }

    # Look the thread up before invoking. LangGraph answers an unknown one
    # with a bare KeyError('config') from inside the pregel loop, and catching
    # that around the call would swallow a genuine KeyError from a node too -
    # the same distinction route_query draws for ReplayMiss.
    saver = compiled(with_checkpointer=True).checkpointer
    if saver.get_tuple({"configurable": {"thread_id": thread_id}}) is None:
        return {
            "error": "no_suspended_run",
            "detail": (
                f"No paused run under thread_id {thread_id!r}. Checkpoints are "
                "held in memory for the life of this server process, so a run "
                "paused before a restart is gone and the query has to be sent "
                "again. A client that starts a fresh server per call can never "
                "resume one."
            ),
        }

    try:
        out = resume(thread_id, approved=approved, cfg=cfg)
    except models.ReplayMiss as exc:
        return {
            "error": "no_cached_response",
            "detail": REPLAY_NOTE,
            "raw": str(exc)[:300],
        }

    return _summarised(out)


@mcp.tool(
    annotations=FREE,
    description=(
        "Project what each routing policy would cost for a query WITHOUT "
        "calling any model. Free and instant. Use this to decide whether a "
        "query is worth routing before you spend anything, or to show a user "
        "the cost of their options. Returns a per-policy USD range plus the "
        "benchmark's recommendation for the loaded ladder."
    ),
)
def estimate_cost(query: str, domain: str = "auto") -> dict[str, Any]:
    """
    Args:
        query: The question to price.
        domain: auto | math | code | general.
    """
    from router_agent.engine import estimate
    return estimate(query, _cfg(domain=domain))


@mcp.tool(
    annotations=SPENDS,
    description=(
        "Run the same query under several routing policies and compare what "
        "each returned and spent. This is the repository's central experiment, "
        "run live on one query. Use it to show, concretely, whether cascading "
        "beats paying for the top model on this ladder. Costs real money in "
        "real mode - it runs every policy named."
    ),
)
def compare_policies(
    query: str,
    policies: list[str] | None = None,
    domain: str = "auto",
) -> dict[str, Any]:
    """
    Args:
        query: The question to route under each policy.
        policies: Which to compare. Defaults to cascade, predictive,
            always_cheap and always_expensive.
        domain: auto | math | code | general.
    """
    from llm_routing import models
    from router_agent.engine import route

    names = policies or ["always_cheap", "predictive", "cascade", "always_expensive"]
    rows = []
    for name in names:
        try:
            out = route(query, cfg=_cfg(policy=name, domain=domain))
        except models.ReplayMiss:
            rows.append({"policy": name, "error": "no cached response (replay mode)"})
            continue
        rows.append({
            "policy": name,
            "answered_by": out.final_model,
            "tier": out.final_tier,
            "verified": out.verified,
            "verifier": out.verifier,
            "cost_usd": round(out.cost_usd, 6),
            "escalations": out.escalations,
            "n_calls": len(out.calls),
            "stop_reason": out.stop_reason,
            "answer_preview": (out.answer or "")[:200],
        })

    priced = [r for r in rows if "cost_usd" in r]
    cheapest = min(priced, key=lambda r: r["cost_usd"]) if priced else None
    return {
        "query": query,
        "ladder": _ladder(),
        "results": rows,
        "cheapest_policy": cheapest["policy"] if cheapest else None,
        "caveat": (
            "One query is an anecdote, not a measurement. The repository's "
            "conclusions come from 417 tasks - 209 of them held out for "
            "evaluation - with paired significance tests; a single comparison "
            "can go either way by chance."
        ),
    }


@mcp.tool(
    annotations=FREE,
    description=(
        "Explain when cascade routing beats predictive routing, using this "
        "repository's measurements. Call this BEFORE choosing a policy: "
        "whether cascading is cheaper is a property of the specific ladder, "
        "and it is NOT predictable from the price ratio between the rungs - "
        "the measured ladders are non-monotonic in it, so no ratio threshold "
        "separates them. The deciding term is what verification "
        "costs on that ladder. Returns the measured verdict for a ladder, the "
        "two-arm probe cross-tab, and how much of the benchmark is real versus "
        "simulated. A ladder with no frontier run gets no verdict rather than "
        "a guess."
    ),
)
def explain_routing(ladder: str | None = None) -> dict[str, Any]:
    """
    Args:
        ladder: claude | deepseek | wide. Defaults to the loaded ladder.
    """
    from router_agent import findings
    return findings.summary(ladder or _ladder())


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource(
    "routing://ladders",
    name="Model ladders",
    description="The model ladders available, with prices and price ratios.",
    mime_type="application/json",
)
def ladders_resource() -> str:
    from llm_routing import models
    from router_agent import findings

    out = {}
    for name, ids in models.LADDERS.items():
        rungs = []
        for model_id in ids:
            spec = models.MODEL_SPECS[model_id]
            rungs.append({
                "model": model_id,
                "provider": spec["provider"],
                "price_in_per_mtok": spec["price_in"],
                "price_out_per_mtok": spec["price_out"],
                "accepts_temperature": spec["accepts_temperature"],
                "verifiable_by_self_consistency": spec["accepts_temperature"],
            })
        out[name] = {
            "rungs": rungs,
            "finding": findings.ratio_verdict(name),
        }
    return json.dumps({
        "loaded": _ladder(),
        "ladders": out,
        "note": (
            "A rung that refuses a temperature cannot be resampled, so "
            "self-consistency verification is unavailable there. On the "
            "claude and wide ladders that applies to every rung above the "
            "bottom one."
        ),
    }, indent=2)


@mcp.resource(
    "routing://findings/probe",
    name="Two-arm probe",
    description=(
        "The real cheap-vs-expensive cross-tab, recomputed from the committed "
        "probe data. The measurement that says whether routing has anything "
        "to decide."
    ),
    mime_type="application/json",
)
def probe_resource() -> str:
    from router_agent import findings
    probe = findings.load_probe()
    if probe is None:
        return json.dumps({"error": "results.probe.jsonl not found"}, indent=2)
    payload = probe.to_dict()
    payload["how_to_read"] = {
        "routable": (
            "cheap wrong, expensive right - the ONLY cell where escalating "
            "pays. The ceiling on what any router can add."
        ),
        "both_fail": (
            "neither rung solves it. Escalating here spends the expensive "
            "rung and still returns a wrong answer."
        ),
        "rescue_rate": (
            "of the cheap rung's failures, the fraction escalating actually "
            "fixes. The number a cascade lives on."
        ),
    }
    return json.dumps(payload, indent=2)


@mcp.resource(
    "routing://findings/verifiers",
    name="Verifier transfer",
    description=(
        "Which of the benchmark's verifiers survive the move into production, "
        "and what it costs when the perfect one does not."
    ),
    mime_type="application/json",
)
def verifiers_resource() -> str:
    from router_agent import findings
    return json.dumps({
        "verifiers": findings.VERIFIER_TRANSFER,
        "degradation_experiment": findings.DEGRADATION_NOTE,
        "headline": (
            "Self-consistency transfers unchanged because it never consults "
            "an answer key. Running tests does not, because a served query "
            "carries none. The cascade's production economics are governed by "
            "which verifier you can actually obtain."
        ),
    }, indent=2)


@mcp.resource(
    "routing://policies",
    name="Policies",
    description="The routing policies this server can run, and what each does.",
    mime_type="application/json",
)
def policies_resource() -> str:
    return json.dumps({
        "cascade": {
            "does": "answer at the cheapest rung, verify, escalate on failure",
            "pays": "the cheap call and verification on EVERY query",
            "buys": "the chance to skip an expensive call",
            "wins_when": (
                "the top rung is genuinely better than the bottom one AND "
                "verification is cheap on this ladder. Measured most accurate "
                "on all three ladders; cheaper at matched accuracy on two of "
                "three (deepseek -4.4%, wide -83.1%, claude +11.7% DEARER)."
            ),
            "not_predictable_from_price_ratio": (
                "The measured ladders are non-monotonic in the price ratio: "
                "claude has the higher ratio of the two close ladders and is "
                "the one where cascading costs more, because verification is "
                "expensive there. Call explain_routing for the loaded ladder "
                "rather than applying a ratio threshold."
            ),
        },
        "predictive": {
            "does": "guess difficulty from the query text, commit to one rung",
            "pays": "exactly one call, no verification",
            "buys": "no wasted cheap call",
            "loses_when": "it misroutes - and it never finds out that it did",
            "note": (
                "This is predictive routing the architecture, served on query "
                "text alone. Do not read the benchmark's old `predictive` row "
                "as a measurement of it: that policy routed on a shipped "
                "difficulty label which was constant across the maths half, "
                "making it always_expensive there, and it has since been "
                "deleted. The benchmark now measures this architecture with "
                "`llm_router` and `routellm`."
            ),
        },
        "always_cheap": {"does": "always the bottom rung", "role": "cost floor"},
        "always_expensive": {
            "does": "always the top rung",
            "role": "quality ceiling, and the baseline cascading must beat",
        },
    }, indent=2)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@mcp.prompt(
    name="choose_routing_policy",
    description=(
        "Walk through choosing a routing policy for a workload, using this "
        "repository's measurements rather than intuition."
    ),
)
def choose_routing_policy(workload: str = "", monthly_queries: str = "") -> str:
    """
    Args:
        workload: What the queries look like - domain, difficulty, volume.
        monthly_queries: Rough monthly query count, if known.
    """
    return f"""\
Help me choose an LLM routing policy.

My workload: {workload or "(not described yet - ask me)"}
Monthly volume: {monthly_queries or "(unknown)"}

Work through this in order, using the `llm-routing` MCP server:

1. Call `explain_routing` to get the measured verdict for the ladder in use.
   Do NOT reason from the price ratio between the rungs: this repository set
   out to find a ratio threshold and measured that none exists. The ladder
   with the higher ratio of the two close ones (`claude`, 6.5x) is the one
   where cascading costs MORE, while the lower-ratio `deepseek` at 3.11x comes
   out cheaper. The deciding term is what VERIFICATION costs on the ladder -
   a cascade pays for the cheap call and for verification on every query,
   including the ones it was always going to accept, and on `claude` that
   means five samples from the cheap rung on every maths query.

2. Read `routing://findings/probe`. If the `routable` fraction is near zero,
   the cheap and expensive models succeed and fail on the same queries and NO
   router can help - the correct action is to pick one model, not to route.

3. Read `routing://findings/verifiers`. Ask whether my workload can obtain a
   real verifier. If answers can be checked automatically - tests, a schema, a
   database lookup - a cascade is strong. If correctness is a matter of
   judgement, the only available verifier is self-consistency, which costs k
   extra calls and is a proxy.

4. Call `estimate_cost` on two or three representative queries of mine to get
   concrete numbers, then multiply by my volume.

5. Give me a recommendation with the reasoning, and say plainly which parts
   rest on measurement and which on assumption about my workload.
"""


def main() -> None:
    # Declare which task ids are live before the first cache read.
    #
    # `response_cache` cannot tell a duplicate that invalidates a paired
    # comparison from one on a task the set no longer contains, so left to
    # itself it hedges with a note on stderr - and on stdio that note is the
    # first thing a client's log shows, on every single call. The server can
    # answer the question instead: it has the task set on disk. `cli.py` and
    # `demo.py` already do exactly this, for exactly this reason.
    #
    # Tolerant of a missing task set: a clone that has not run build_taskset
    # yet should still start, and the note is the correct behaviour then.
    from llm_routing import paths, response_cache

    if paths.TASKSET.exists():
        try:
            response_cache.LIVE_TASK_IDS = {
                json.loads(line)["id"]
                for line in paths.TASKSET.open(encoding="utf-8")
                if line.strip()
            }
        except (OSError, ValueError, KeyError):
            pass  # a malformed task set is build_taskset's problem, not the server's

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
