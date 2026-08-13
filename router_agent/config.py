"""What to run, and the defaults the benchmark argues for.

Every default in `RouterConfig` is either a measurement from this repository or
a decision recorded next to the code that implements it. Where a default came
from a number, the number is cited. Where it is a judgement call, it says so.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Literal

Policy = Literal["cascade", "predictive", "always_cheap", "always_expensive"]
Verifier = Literal["auto", "self_consistency", "tests", "none"]
Domain = Literal["auto", "math", "code", "general"]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a number - got {raw!r}")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer - got {raw!r}")


def _env_opt_float(name: str) -> float | None:
    """A float if the variable is set, else None.

    Distinct from `_env_float` because for the approval gate "unset" and "zero"
    are different states: unset means never ask, zero means ask before every
    escalation. A sentinel default cannot express that.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a number - got {raw!r}")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class RouterConfig:
    """One routing run's settings.

    Constructed from the environment by `from_env`, then overridden per call.
    The MCP server builds one of these per tool invocation, so a client can ask
    for a different policy or budget without restarting the server.
    """

    # --- what to route with -------------------------------------------------

    policy: Policy = "cascade"
    """
    `cascade` by default, and the benchmark is why: it is the most accurate
    policy measured on every ladder, and on two of three it is also cheaper at
    matched accuracy.

    It is NOT unconditionally cheaper, and the default does not pretend
    otherwise. Measured, cascade against always-best at matched accuracy:

        deepseek   3.11x ratio     -4.4%   cheaper
        claude      6.5x ratio    +11.7%   DEARER
        wide       46.4x ratio    -83.1%   much cheaper

    Note the middle row, and note that it is not ordered by price ratio. What
    decides the sign is what verification costs on the ladder, not the price
    gap - on `claude` the cheap rung is haiku and the maths half draws five
    samples from it. A rule of the form "cascade above ~3x" gets `claude` and
    `deepseek` backwards, which is why no such rule appears anywhere in this
    package. Call findings.ratio_verdict for the ladder actually loaded; it
    reads that ladder's committed frontier and declines to guess when it has
    none.
    """

    verifier: Verifier = "auto"
    """
    `auto` picks by domain: tests when the caller supplied them, otherwise
    self-consistency. See verifiers.select for the full table and for why the
    perfect verifier is usually unavailable in production.
    """

    domain: Domain = "auto"
    """`auto` infers from the query. See live.infer_domain."""

    # --- how hard to verify -------------------------------------------------

    self_consistency_k: int = field(
        default_factory=lambda: _env_int("ROUTER_K", 5)
    )
    """
    Samples drawn to estimate agreement. Mirrors policies.SELF_CONSISTENCY_K
    (DECISION #2). Cost is linear in k and it is the single largest lever on
    what a cascade spends: at k=5 a verified maths answer costs 5 cheap calls
    plus the greedy one.
    """

    agreement_threshold: float = field(
        default_factory=lambda: _env_float("ROUTER_AGREEMENT", 0.8)
    )
    """
    Fraction of samples that must agree for the cheap answer to be accepted.
    Mirrors policies.AGREEMENT_THRESHOLD (DECISION #3). 1.0 accepts only
    unanimous answers and escalates far more.
    """

    # --- what it is allowed to spend ---------------------------------------

    max_cost_usd: float = field(
        default_factory=lambda: _env_float("ROUTER_MAX_COST_USD", 0.50)
    )
    """
    Hard ceiling for one query, enforced before each escalation rather than
    after the fact. 50 cents is roughly 60x the measured cost of a level-5
    maths answer on the `wide` ladder ($0.0078/call), so it binds on runaway
    loops and not on ordinary traffic.
    """

    max_escalations: int = field(
        default_factory=lambda: _env_int("ROUTER_MAX_ESCALATIONS", 3)
    )
    """
    Belt to the budget's braces. A ladder has at most three rungs, so any run
    escalating more than that has a bug in its routing predicate, and a bounded
    graph is easier to reason about than one that relies on cost to terminate.
    """

    require_approval_above_usd: float | None = field(
        default_factory=lambda: _env_opt_float("ROUTER_APPROVAL_USD")
    )
    """
    Human-in-the-loop gate. When set, an escalation whose projected cost
    exceeds this pauses the graph and waits for approval instead of spending.
    None disables it; 0 asks before every escalation.

    This is not decoration. On the `wide` ladder the top rung is ~46x the
    bottom one effectively, so a single escalation is the entire cost decision,
    and "ask before spending 46x" is a real product requirement.
    """

    # --- safety -------------------------------------------------------------

    allow_code_execution: bool = field(
        default_factory=lambda: _env_bool("ROUTER_ALLOW_CODE_EXEC", False)
    )
    """
    Gates the `tests` verifier, which runs MODEL-GENERATED CODE in a
    subprocess.

    Off by default, and it should stay off unless the process is sandboxed. In
    the benchmark this risk is bounded - the code is written against MBPP+
    problems and executed on a machine the author controls. In serving it is
    not: the query comes from a user, the code comes from a model, and running
    it is arbitrary code execution as a service.

    graders.grade_run_asserts shells out to `[sys.executable, path]`. There is
    no container, no seccomp filter, and no network policy. Turn this on only
    where that sentence is acceptable.
    """

    # --- substrate ----------------------------------------------------------

    ladder: str = field(
        default_factory=lambda: os.environ.get("ROUTER_LADDER", "claude")
    )
    mode: str = field(
        default_factory=lambda: os.environ.get("ROUTER_MODE", "mock")
    )

    def __post_init__(self):
        if not 0.0 < self.agreement_threshold <= 1.0:
            raise ValueError(
                f"agreement_threshold must be in (0, 1] - got "
                f"{self.agreement_threshold}"
            )
        if self.self_consistency_k < 1:
            raise ValueError(
                f"self_consistency_k must be >= 1 - got {self.self_consistency_k}"
            )
        if self.max_cost_usd <= 0:
            raise ValueError(f"max_cost_usd must be > 0 - got {self.max_cost_usd}")
        if self.max_escalations < 0:
            raise ValueError(
                f"max_escalations must be >= 0 - got {self.max_escalations}"
            )

    @classmethod
    def from_env(cls, **overrides) -> "RouterConfig":
        """Environment first, explicit arguments last.

        Overrides whose value is None are dropped, so a caller can pass through
        an optional argument without having to branch on it.
        """
        clean = {k: v for k, v in overrides.items() if v is not None}
        return cls(**clean)

    def to_dict(self) -> dict:
        return asdict(self)
