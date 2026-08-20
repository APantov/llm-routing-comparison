"""The routing graph. Pure wiring - every decision lives in `nodes.py`.

                        ┌───────────┐
              START ───►│  classify │   domain, starting rung, verifier
                        └─────┬─────┘
                              ▼
                        ┌───────────┐
                   ┌───►│  answer   │   the only node that always spends
                   │    └─────┬─────┘
                   │          ▼
                   │    ┌───────────┐
                   │    │  verify   │   the cascade's fixed cost
                   │    └─────┬─────┘
                   │          │ route_after_verify
                   │    ┌─────┴──────┬──────────────┐
                   │  escalate     accept          stop
                   │    │            │              │
                   │    ▼            ▼              ▼
                   │ ┌──────────┐  ┌────────────────────┐
                   │ │ escalate │  │      finalize      │──► END
                   │ └────┬─────┘  └────────────────────┘
                   │      │ route_after_escalate         ▲
                   └──────┴──────────── stop ────────────┘
                        answer

**Why a graph and not a `while` loop.** The `escalate` -> `answer` edge is a
cycle, and it is what makes LangGraph earn its place:

  * **Checkpointing across the cycle** - state persists at every step, so a run
    that pauses mid-cascade resumes on the rung it stopped at instead of
    re-paying for the cheap call.
  * **The interrupt is inside the loop.** Approval is needed *before an
    escalation*, mid-iteration. A `while` loop would hand-roll suspend/resume;
    here `interrupt()` suspends and `Command(resume=...)` continues.
  * **The trace is structural** - every lap appends to reducer-backed channels,
    so "why did this cost 40 cents" is answered by reading state.

Compiled once per (checkpointer, approval) pair and cached: compilation is not
free and the shape never varies with the query.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from router_agent import nodes
from router_agent.state import RouteState


def build_graph() -> StateGraph:
    """Assemble the uncompiled graph."""
    g = StateGraph(RouteState)

    g.add_node("classify", nodes.classify)
    g.add_node("answer", nodes.answer)
    g.add_node("verify", nodes.verify)
    g.add_node("escalate", nodes.escalate)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "classify")
    g.add_edge("classify", "answer")
    g.add_edge("answer", "verify")

    g.add_conditional_edges(
        "verify",
        nodes.route_after_verify,
        {"accept": "finalize", "escalate": "escalate", "stop": "finalize"},
    )
    # The cycle. `escalate` either bumps the rung and sends control back to
    # `answer`, or refuses (budget, approval, top of ladder) and terminates.
    g.add_conditional_edges(
        "escalate",
        nodes.route_after_escalate,
        {"answer": "answer", "stop": "finalize"},
    )
    g.add_edge("finalize", END)
    return g


@lru_cache(maxsize=8)
def compiled(with_checkpointer: bool = False):
    """Compile once and reuse.

    A checkpointer is required for the human-approval interrupt - LangGraph
    needs somewhere to persist the suspended run - and is pure overhead
    without it, so it is opt-in.

    `InMemorySaver` is correct for a CLI and for a single-process MCP server:
    the checkpoint has to outlive one graph invocation, not one process. A
    multi-process deployment wants a durable saver (`SqliteSaver`,
    `PostgresSaver`) and nothing else about this file changes.
    """
    graph = build_graph()
    if not with_checkpointer:
        return graph.compile()

    from langgraph.checkpoint.memory import InMemorySaver
    return graph.compile(checkpointer=InMemorySaver())


# The recursion limit LangGraph enforces per run. A three-rung ladder visits at
# most classify + 3x(answer, verify, escalate) + finalize = 11 steps, so 25 is
# generous headroom that still stops a routing bug from looping forever.
RECURSION_LIMIT = 25
