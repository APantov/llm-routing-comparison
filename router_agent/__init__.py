"""A cost-aware LLM routing service, built on the benchmark next door.

`llm_routing` measures *when* cascade routing beats predictive routing; this
package is the deployable half that implements the findings.

The two share one substrate on purpose. `router_agent` opens no HTTP connections,
keeps no price table and defines no notion of a model tier: it calls
`models.call`, exactly as `policies.py` does. Three properties follow:

  * **Cost accounting that matches the benchmark.** Same price tables, same
    tokenizer-asymmetry handling, same arithmetic - so a dollar figure here
    means what a dollar figure in the tables means.
  * **Replay.** `ROUTER_MODE=replay` serves the whole agent from the 5,075 real
    responses under `cache/`: end to end, no API key, zero cost.
  * **One place where money is spent** - one function to audit, not a package.

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

`import router_agent` pulls in nothing heavy: LangGraph is imported by
`graph.py` and the MCP SDK by `mcp_server.py`, both at first use, and this
package re-exports nothing so importing it cannot pull either in by accident.
Import what you need from the module that defines it:

    from router_agent.config import RouterConfig
    from router_agent.engine import route
"""

__version__ = "0.2.0"
