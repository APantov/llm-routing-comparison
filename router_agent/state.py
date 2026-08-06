"""The graph's state, and the reducers that make a cyclic graph safe.

A cascade is a loop: answer, verify, escalate, answer again. Loops are where
naive state handling goes wrong, because two different things want two
different merge rules:

  * **Position** - which rung are we on, what is the current answer. The newest
    value replaces the old one. No reducer; last write wins.
  * **History** - what did this run cost, which rungs did it call, what
    happened at each step. Every pass round the loop ADDS to these. They need
    `operator.add`, or the second lap silently erases the first lap's spend.

Getting that wrong is not a crash, it is a wrong number: a cascade that
escalated twice would report only the final rung's cost and look cheaper than
it was. Since the entire subject of this repository is what routing costs, that
is the specific bug most worth designing against.

So the rule for reading this file: **anything `Annotated[..., operator.add]` is
cumulative across the whole run; everything else is the current position.**
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class CallRecord(TypedDict):
    """One model call a policy was charged for."""
    tier: str
    model: str
    kind: str          # "answer" | "verify"
    tokens_in: int
    tokens_out: int
    cost_usd: float
    """What this call costs a policy - charged whether or not it was served
    from the cache, because in production there is no cross-run cache."""
    backend_cost_usd: float
    """What this call actually sent to a provider: `cost_usd` on a real call,
    0.0 when the response came from the cache. The two differ on any run that
    partially replays, which is most real runs after the first."""
    latency_s: float


class Event(TypedDict):
    """One step of the trace, in order.

    The trace is a first-class output rather than logging. A router that cannot
    explain why it escalated is not auditable, and "why did this query cost 40
    cents" is the question a cost-control product exists to answer.
    """
    node: str
    detail: str
    data: dict


class RouteState(TypedDict, total=False):
    # --- input: written once by the caller, never by a node ----------------
    query: str
    task: dict
    config: dict
    """`RouterConfig.to_dict()`. Carried in state rather than closed over so
    that a checkpointed run resumes with the settings it started with, not with
    whatever the process happens to hold now."""

    # --- position: last write wins -----------------------------------------
    rung_index: int
    """Index into `models.TIERS`. The cascade starts at 0; a predictive route
    may start higher, which is precisely the difference between the two
    architectures."""

    answer: str | None
    verified: bool
    confidence: float | None
    verifier_used: str
    escalations: int
    awaiting_approval: bool
    approved: bool | None
    stop_reason: str
    """Why the run ended: verified | exhausted_ladder | budget_exceeded |
    max_escalations | approval_denied | unverified_final."""

    # --- history: cumulative, via reducers ---------------------------------
    calls: Annotated[list[CallRecord], operator.add]
    cost_usd: Annotated[float, operator.add]
    latency_s: Annotated[float, operator.add]
    events: Annotated[list[Event], operator.add]


def initial_state(query: str, task: dict, config: dict, start_rung: int = 0) -> RouteState:
    """A fully-populated starting state.

    Every reducer-backed key is seeded explicitly. LangGraph would default a
    missing `Annotated[float, operator.add]` channel, but an explicit 0.0 means
    the first `+=` cannot depend on that behaviour - and it makes the state
    readable in a checkpoint dump.
    """
    return {
        "query": query,
        "task": task,
        "config": config,
        "rung_index": start_rung,
        "answer": None,
        "verified": False,
        "confidence": None,
        "verifier_used": "",
        "escalations": 0,
        "awaiting_approval": False,
        "approved": None,
        "stop_reason": "",
        "calls": [],
        "cost_usd": 0.0,
        "latency_s": 0.0,
        "events": [],
    }


def event(node: str, detail: str, **data: Any) -> Event:
    return {"node": node, "detail": detail, "data": data}
