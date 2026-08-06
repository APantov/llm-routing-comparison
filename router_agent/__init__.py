"""A cost-aware LLM routing service, built on the benchmark next door.

The repository root is an experiment: it measures *when* cascade routing beats
predictive routing, and finds that the answer is governed by the price ratio
between rungs and by the quality of the verifier. This package is the other
half - the deployable thing that implements those findings and can be pointed
at a real query.

The two halves share one substrate on purpose. `router_agent` does not open its
own HTTP connections, keep its own price table, or define its own notion of a
model tier: it calls `models.call`, exactly as `policies.py` does. That is what
buys three properties which are otherwise very hard to get in a serving layer:

  * **Cost accounting that matches the benchmark.** Same verified price tables,
    same tokenizer-asymmetry handling, same arithmetic. A dollar figure the
    router reports means the same thing as a dollar figure in the paper tables.
  * **Replay.** `ROUTER_MODE=replay` serves the whole agent from the 318 real
    responses committed under `cache/`. The demo runs end to end, against real
    model output, with no API key and at zero cost.
  * **One place where money is spent.** Auditing what a run costs means reading
    one function, not grepping a package.

Layout, in dependency order:

    config.py      what to run: ladder, mode, policy, verifier, budget
    live.py        the bridge - an arbitrary query becomes a task dict
    verifiers.py   verification WITHOUT ground truth, which serving requires
    findings.py    the measured results, loaded from the committed real data
    state.py       the LangGraph state and its reducers
    nodes.py       one function per node, each independently testable
    graph.py       the cyclic graph: answer -> verify -> escalate -> answer
    engine.py      the façade the CLI and the MCP server both call
    cli.py         `llm-router "query"`
    mcp_server.py  the same engine, exposed over MCP

Import cost is deliberate: `import router_agent` pulls in nothing heavy, so the
research core keeps its bare-interpreter property. LangGraph is imported by
`graph.py` and the MCP SDK by `mcp_server.py`, both at first use.
"""

__version__ = "0.2.0"

__all__ = ["RouterConfig", "RouteOutcome", "route", "__version__"]


def __getattr__(name):
    """Lazily re-export the façade.

    Kept lazy so that `import router_agent` does not drag in LangGraph. The MCP
    server and the CLI want the façade; the test suite frequently wants only
    `live` or `verifiers`, and should not pay for a graph compile to get them.
    """
    if name in ("RouterConfig",):
        from router_agent.config import RouterConfig
        return RouterConfig
    if name in ("RouteOutcome", "route"):
        from router_agent.engine import RouteOutcome, route
        return {"RouteOutcome": RouteOutcome, "route": route}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
