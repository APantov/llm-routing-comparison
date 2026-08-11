"""One function per graph node.

Each node takes a `RouteState` and returns a *partial* state - only the keys it
changed. LangGraph merges those with the reducers in `state.py`, so a node that
spends money returns `{"cost_usd": <what it just spent>}` and never
`{"cost_usd": <running total>}`. Returning a total would double-count on the
second lap of the cascade loop.

Nodes are plain functions with no LangGraph imports, which is deliberate: the
routing logic can be unit-tested by calling them with a dict, and the graph in
`graph.py` is left as pure wiring. The one exception is `escalate`, which needs
`interrupt()` for the human-approval gate; it imports it lazily so that the
rest of this module stays importable without LangGraph installed.
"""

from __future__ import annotations

from llm_routing import models
from router_agent import live, pricing, verifiers
from router_agent.config import RouterConfig
from router_agent.state import RouteState, event


def _cfg(state: RouteState) -> RouterConfig:
    return RouterConfig(**state["config"])


def _tier_at(index: int) -> str:
    """Clamp an index onto the loaded ladder.

    Clamping rather than raising: `escalate` is what enforces the top of the
    ladder, and a node that crashes on an out-of-range index would turn a
    routing bug into a stack trace in front of a user.
    """
    return models.TIERS[min(index, len(models.TIERS) - 1)]


# ---------------------------------------------------------------------------

def classify(state: RouteState) -> dict:
    """Decide the domain and which rung to start on.

    This node is where the two architectures diverge, and the difference is one
    integer:

        cascade     start at rung 0 and let verification decide
        predictive  guess from the query text and commit

    Everything downstream is shared. That is what makes their costs
    comparable - they differ in the starting rung and in whether the verifier
    is allowed to send them back round the loop, not in the model, the prompt,
    or the accounting.
    """
    cfg = _cfg(state)
    task = state["task"]
    ladder_top = len(models.TIERS) - 1

    if cfg.policy == "always_expensive":
        start, why = ladder_top, "policy pins the top rung"
    elif cfg.policy == "always_cheap":
        start, why = 0, "policy pins the bottom rung"
    elif cfg.policy == "predictive":
        hard = live.predict_is_hard_live(task)
        start = ladder_top if hard else 0
        why = (
            f"predictive heuristic says {'HARD' if hard else 'EASY'} from the "
            f"query text alone (no difficulty label exists for a live query)"
        )
    else:  # cascade
        start, why = 0, "cascade always starts at the cheapest rung"

    verifier = verifiers.select(task, cfg)

    return {
        "rung_index": start,
        "verifier_used": verifier,
        "events": [
            event(
                "classify",
                f"domain={task['domain']}, start={_tier_at(start)}, "
                f"verifier={verifier}",
                domain=task["domain"],
                start_tier=_tier_at(start),
                reason=why,
                verifier=verifier,
                policy=cfg.policy,
            )
        ],
    }


def answer(state: RouteState) -> dict:
    """Call the model at the current rung. The only node that always spends."""
    tier = _tier_at(state["rung_index"])
    task = state["task"]

    r, backend_cost = pricing.call_tracked(tier, task)
    spec = models.MODELS[tier]

    return {
        "answer": r.text,
        "cost_usd": r.cost_usd,
        "latency_s": r.latency_s,
        "calls": [{
            "tier": tier, "model": spec["id"], "kind": "answer",
            "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
            "cost_usd": r.cost_usd,
            "backend_cost_usd": backend_cost,
            "latency_s": r.latency_s,
        }],
        "events": [event(
            "answer", f"{tier} ({spec['id']}) answered",
            tier=tier, model=spec["id"], cost_usd=round(r.cost_usd, 6),
            tokens_out=r.tokens_out,
        )],
    }


def verify(state: RouteState) -> dict:
    """Ask the verifier whether the current answer is good enough to return.

    The cost this node reports is the cascade's fixed overhead: it is paid on
    every query, including the ones that were always going to be accepted. That
    is the term which makes cascading lose below a ~3x price ratio, so it is
    accounted separately from the answer call rather than folded into it.
    """
    cfg = _cfg(state)
    tier = _tier_at(state["rung_index"])
    name = state.get("verifier_used") or verifiers.select(state["task"], cfg)

    try:
        check = verifiers.run(name, state["task"], state["answer"], tier, cfg)
    except PermissionError as exc:
        # The `tests` verifier is gated behind allow_code_execution. Refusing to
        # execute is correct, but refusing to answer would not be - fall back to
        # the proxy verifier and record the downgrade in the trace.
        check = verifiers.run("self_consistency", state["task"], state["answer"], tier, cfg)
        name = "self_consistency"
        check.detail = {**check.detail, "downgraded_from": "tests", "reason": str(exc)}

    calls = []
    if check.cost_usd > 0:
        calls.append({
            "tier": tier, "model": models.MODELS[tier]["id"], "kind": "verify",
            "tokens_in": 0, "tokens_out": 0,
            "cost_usd": check.cost_usd,
            "backend_cost_usd": check.backend_cost_usd,
            "latency_s": check.latency_s,
        })

    conf = "" if check.confidence is None else f", confidence={check.confidence:.2f}"
    return {
        "verified": check.accepted,
        "confidence": check.confidence,
        "verifier_used": name,
        "answer": check.answer_text,
        "cost_usd": check.cost_usd,
        "latency_s": check.latency_s,
        "calls": calls,
        "events": [event(
            "verify",
            f"{name} -> {'ACCEPT' if check.accepted else 'REJECT'}{conf}",
            verifier=name, accepted=check.accepted,
            confidence=check.confidence, cost_usd=round(check.cost_usd, 6),
            **check.detail,
        )],
    }


def escalate(state: RouteState) -> dict:
    """Move up one rung, if the budget and the human both allow it.

    Three gates, checked in this order and for this reason:

    1. **Ladder** - is there a rung above this one at all.
    2. **Budget** - would the next call breach the ceiling. Checked with a
       PROJECTED cost, before spending, because a check afterwards is a receipt
       rather than a budget.
    3. **Human** - if `require_approval_above_usd` is set and the projection
       exceeds it, pause the graph and wait.

    Cheapest check first: never interrupt a human to approve a call that the
    budget was going to refuse anyway.
    """
    cfg = _cfg(state)
    current = state["rung_index"]
    top = len(models.TIERS) - 1

    if current >= top:
        return {
            "stop_reason": "exhausted_ladder",
            "events": [event(
                "escalate", "already at the top rung; nothing to escalate to",
                tier=_tier_at(current),
            )],
        }

    nxt = current + 1
    next_tier = _tier_at(nxt)
    task = state["task"]

    projected = pricing.estimate_call_cost(task, next_tier)
    projected += pricing.estimate_verification_cost(
        task, next_tier, state.get("verifier_used", "self_consistency"),
        cfg.self_consistency_k,
    )
    spent = state.get("cost_usd", 0.0)

    if spent + projected > cfg.max_cost_usd:
        return {
            "stop_reason": "budget_exceeded",
            "events": [event(
                "escalate",
                f"refused: {next_tier} projected at ${projected:.4f}, spent "
                f"${spent:.4f}, ceiling ${cfg.max_cost_usd:.4f}",
                refused=True, projected_usd=round(projected, 6),
                spent_usd=round(spent, 6), ceiling_usd=cfg.max_cost_usd,
            )],
        }

    if (
        cfg.require_approval_above_usd is not None
        and projected > cfg.require_approval_above_usd
        and not state.get("approved")
    ):
        # Dynamic interrupt. Needs a checkpointer and a thread id; engine.py
        # supplies both, and raises a clear error if approval is requested
        # without them rather than letting this fail deep inside LangGraph.
        from langgraph.types import interrupt

        decision = interrupt({
            "kind": "escalation_approval",
            "question": (
                f"Escalate to {next_tier} ({models.MODELS[next_tier]['id']})? "
                f"Projected additional cost ${projected:.4f}."
            ),
            "from_tier": _tier_at(current),
            "to_tier": next_tier,
            "projected_usd": round(projected, 6),
            "spent_so_far_usd": round(spent, 6),
        })

        approved = decision is True or (
            isinstance(decision, str) and decision.strip().lower()
            in ("y", "yes", "approve", "approved", "ok", "true")
        ) or (isinstance(decision, dict) and bool(decision.get("approved")))

        if not approved:
            return {
                "stop_reason": "approval_denied",
                "awaiting_approval": False,
                "events": [event(
                    "escalate", f"human declined escalation to {next_tier}",
                    approved=False, to_tier=next_tier,
                )],
            }

    return {
        "rung_index": nxt,
        "escalations": state.get("escalations", 0) + 1,
        "verified": False,
        "awaiting_approval": False,
        "approved": None,
        "events": [event(
            "escalate", f"{_tier_at(current)} -> {next_tier}",
            from_tier=_tier_at(current), to_tier=next_tier,
            projected_usd=round(projected, 6),
        )],
    }


def finalize(state: RouteState) -> dict:
    """Assemble the terminal reason. Spends nothing.

    `escalate` records its own refusals (budget, approval, top of ladder)
    because only it knows them. The remaining reasons are inferred here, since
    a conditional edge returns a route name and cannot write to state.
    """
    cfg = _cfg(state)
    reason = state.get("stop_reason")
    if not reason:
        if state.get("verified"):
            reason = "verified"
        elif cfg.policy in ("predictive", "always_cheap", "always_expensive"):
            # These never escalate, so an unverified answer is the intended
            # terminal state rather than a failure to climb.
            reason = "verified" if state.get("verified") else "one_shot_unverified"
        elif state.get("escalations", 0) >= cfg.max_escalations:
            reason = "max_escalations"
        elif state["rung_index"] >= len(models.TIERS) - 1:
            reason = "exhausted_ladder"
        else:
            reason = "unverified_final"
    return {
        "stop_reason": reason,
        "events": [event(
            "finalize", f"done: {reason}",
            stop_reason=reason,
            final_tier=_tier_at(state["rung_index"]),
            total_cost_usd=round(state.get("cost_usd", 0.0), 6),
            escalations=state.get("escalations", 0),
        )],
    }


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def route_after_verify(state: RouteState) -> str:
    """accept | escalate | stop. The cascade's whole decision, in one place."""
    cfg = _cfg(state)

    if state.get("verified"):
        return "accept"

    # Policies that route once never escalate, whatever the verifier said. This
    # is what makes `predictive` a one-shot router rather than a cascade with a
    # different starting rung.
    if cfg.policy in ("predictive", "always_cheap", "always_expensive"):
        return "accept"

    if state.get("escalations", 0) >= cfg.max_escalations:
        return "stop"
    if state["rung_index"] >= len(models.TIERS) - 1:
        return "stop"
    return "escalate"


def route_after_escalate(state: RouteState) -> str:
    """Did the escalation actually happen, or was it refused?"""
    return "stop" if state.get("stop_reason") else "answer"
