"""The façade. One call in, one auditable outcome out.

`route()` is what the CLI and the MCP server both use. It owns the things
neither of them should each reinvent: configuring the response cache, running
the graph, and turning terminal state into a result object whose vocabulary is
honest about what was and was not measured.

The most important line in this file is in `RouteOutcome`: there is no
`correct` field. Serving has no ground truth, so the strongest claim available
is `verified` - a verifier's opinion. Naming it `correct` would be the exact
error the benchmark exists to argue against.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from llm_routing import models
from llm_routing import response_cache
from router_agent import findings, live, pricing
from router_agent.config import RouterConfig
from router_agent.state import initial_state


@dataclass
class RouteOutcome:
    """The result of routing one query."""

    query: str
    answer: str | None
    domain: str

    verified: bool
    """The verifier accepted this answer. NOT a correctness claim - see the
    module docstring, and `verified_meaning` for the caveat in words."""

    confidence: float | None
    verifier: str
    final_tier: str
    final_model: str

    cost_usd: float
    """What this query would cost in production. On a cache hit the caller is
    still charged in full, exactly as in `run_eval.py`: `cost_usd` answers
    "what would this cost to serve", and in production there is no cache of
    somebody else's identical call."""

    backend_cost_usd: float
    """What this run actually sent to a provider, summed per call.

    Zero in mock and replay. In real mode it is `cost_usd` minus whatever was
    already in the response cache, so the two diverge on any run that partially
    replays - which is most real runs after the first. Reporting `cost_usd`
    here whenever a run touched a backend at all would bill cached calls as
    money spent."""

    latency_s: float
    escalations: int
    stop_reason: str
    calls: list = field(default_factory=list)
    events: list = field(default_factory=list)
    # Pessimistic defaults, and deliberately not the real ones: an outcome that
    # was never populated by `_outcome_from_state` describes itself as
    # fabricated rather than claiming a measurement it cannot vouch for.
    simulated: bool = True
    mode: str = "mock"
    ladder: str = ""
    policy: str = ""
    thread_id: str | None = None
    interrupted: Any = None
    """Set when the graph paused for human approval. Resume with `resume()`."""

    @property
    def verified_meaning(self) -> str:
        """What `verified` does and does not claim, in words.

        Reads the outcome, not just the verifier name: a verifier that could
        not run must not be described as having agreed or disagreed. The
        distinction between "the model disagreed with itself" and "agreement
        was never measured" is exactly the kind of thing a serving layer
        blurs, and blurring it here would undercut the whole argument.
        """
        if self.verifier == "none":
            return "nothing was verified; the answer is returned as-is"

        if self.verifier == "tests":
            if self.verified:
                return (
                    "the answer passed the caller-supplied tests - exact, "
                    "within the limits of those tests"
                )
            return "the answer failed the caller-supplied tests"

        if self.verifier == "self_consistency":
            if self.confidence is None:
                return (
                    "agreement could NOT be measured (the rung refuses a "
                    "temperature, or no cached sample was available), so the "
                    "answer is unverified rather than disagreed-with"
                )
            if self.verified:
                return (
                    "the model agreed with itself across independent draws - "
                    "a proxy for correctness, not a measurement of it"
                )
            return (
                "the model disagreed with itself across independent draws, "
                "which is evidence of unreliability, not proof of error"
            )

        return "unknown verifier"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "domain": self.domain,
            "verified": self.verified,
            "verified_meaning": self.verified_meaning,
            "confidence": self.confidence,
            "verifier": self.verifier,
            "final_tier": self.final_tier,
            "final_model": self.final_model,
            "cost_usd": round(self.cost_usd, 6),
            "backend_cost_usd": round(self.backend_cost_usd, 6),
            "latency_s": round(self.latency_s, 3),
            "escalations": self.escalations,
            "stop_reason": self.stop_reason,
            "n_calls": len(self.calls),
            "calls": self.calls,
            "trace": self.events,
            "simulated": self.simulated,
            "mode": self.mode,
            "ladder": self.ladder,
            "policy": self.policy,
            "thread_id": self.thread_id,
            "interrupted": self.interrupted,
        }


# ---------------------------------------------------------------------------

def _configure(cfg: RouterConfig) -> None:
    """Point the shared response cache at the right file.

    `response_cache.configure(mode, ladder)` selects one file per ladder so
    that a DeepSeek response can never be replayed as a Claude one. Serving
    calls it for the same reason the evaluation does.
    """
    response_cache.configure(cfg.mode, cfg.ladder)


def _outcome_from_state(
    final: dict, cfg: RouterConfig, thread_id: str | None
) -> RouteOutcome:
    tier = models.TIERS[min(final.get("rung_index", 0), len(models.TIERS) - 1)]
    calls = final.get("calls", [])

    outcome = RouteOutcome(
        query=final["query"],
        answer=final.get("answer"),
        domain=final["task"]["domain"],
        verified=bool(final.get("verified")),
        confidence=final.get("confidence"),
        verifier=final.get("verifier_used", ""),
        final_tier=tier,
        final_model=models.MODELS[tier]["id"],
        cost_usd=final.get("cost_usd", 0.0),
        # Summed per call rather than inferred from the run's mode. A real run
        # that partially replays pays for some calls and reads the rest from
        # the cache, so "did this run touch a backend at all" is the wrong
        # question - each call answers it for itself.
        backend_cost_usd=sum(c.get("backend_cost_usd", 0.0) for c in calls),
        latency_s=final.get("latency_s", 0.0),
        escalations=final.get("escalations", 0),
        stop_reason=final.get("stop_reason", ""),
        calls=calls,
        events=final.get("events", []),
        simulated=cfg.mode == "mock",
        mode=cfg.mode,
        ladder=cfg.ladder,
        policy=cfg.policy,
        thread_id=thread_id,
    )

    # A paused run surfaces as `__interrupt__` in the returned state. Reporting
    # it explicitly beats returning a half-finished answer that looks complete.
    #
    # Checked HERE rather than in `route`, because `resume` needs it just as
    # much and used not to have it: approval is per-escalation, so on a
    # three-rung ladder the first approval walks the graph straight into the
    # second interrupt. `resume` reported that as a finished run with an empty
    # `stop_reason`, and the CLI's `while out.interrupted` loop believed it and
    # printed a mid-cascade answer as final. Two rungs never showed it.
    if "__interrupt__" in final:
        interrupts = final["__interrupt__"]
        outcome.interrupted = interrupts[0].value if interrupts else None
        outcome.stop_reason = "awaiting_approval"

    return outcome


def route(
    query: str,
    cfg: RouterConfig | None = None,
    tests: list[str] | None = None,
    thread_id: str | None = None,
    **overrides,
) -> RouteOutcome:
    """Route one query through the graph and return an auditable outcome.

    When `cfg.require_approval_above_usd` is set, the run may pause: the
    returned outcome carries `interrupted` and a `thread_id`, and `resume()`
    continues it.
    """
    cfg = cfg or RouterConfig.from_env(**overrides)
    _configure(cfg)

    task = live.synthesize_task(query, cfg.domain, tests)
    needs_checkpointer = cfg.require_approval_above_usd is not None

    from router_agent.graph import RECURSION_LIMIT, compiled
    app = compiled(with_checkpointer=needs_checkpointer)

    tid = thread_id or uuid.uuid4().hex[:12]
    run_config: dict = {"recursion_limit": RECURSION_LIMIT}
    if needs_checkpointer:
        run_config["configurable"] = {"thread_id": tid}

    state = initial_state(query, task, cfg.to_dict())

    final = app.invoke(state, config=run_config)

    return _outcome_from_state(
        final, cfg, tid if needs_checkpointer else None
    )


def resume(thread_id: str, approved: bool, cfg: RouterConfig | None = None) -> RouteOutcome:
    """Continue a run that paused for human approval."""
    cfg = cfg or RouterConfig.from_env()
    if cfg.require_approval_above_usd is None:
        raise ValueError(
            "resume() needs the same config the run started with, including "
            "require_approval_above_usd; otherwise the graph compiles without "
            "a checkpointer and the suspended run cannot be found"
        )
    _configure(cfg)

    from langgraph.types import Command
    from router_agent.graph import RECURSION_LIMIT, compiled

    app = compiled(with_checkpointer=True)
    run_config = {
        "recursion_limit": RECURSION_LIMIT,
        "configurable": {"thread_id": thread_id},
    }
    final = app.invoke(Command(resume=approved), config=run_config)
    return _outcome_from_state(final, cfg, thread_id)


def estimate(query: str, cfg: RouterConfig | None = None,
             tests: list[str] | None = None) -> dict:
    """Project the cost of every policy for this query. Calls no model."""
    cfg = cfg or RouterConfig.from_env()
    task = live.synthesize_task(query, cfg.domain, tests)
    policies = ["always_cheap", "always_expensive", "predictive", "cascade"]
    return {
        "query": query,
        "domain": task["domain"],
        "ladder": cfg.ladder,
        "rungs": {t: models.MODELS[t]["id"] for t in models.TIERS},
        "estimates": [pricing.estimate_policy_cost(task, p, cfg) for p in policies],
        "recommended_policy": findings.ratio_verdict(cfg.ladder),
        "basis": (
            "projection from the verified price tables and output-token counts "
            "measured on the two-arm probe (maths 650 tok, code 55 tok). "
            "The `general` domain is an assumption, not a measurement."
        ),
    }
