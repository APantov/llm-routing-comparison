"""Cost projection, so the budget guard can refuse a call before making it.

A cascade decides whether to escalate *before* it knows what the next rung will
cost. Checking spend after the fact is not a budget, it is a receipt - so this
module estimates the cost of a call that has not happened yet.

The estimate is deliberately built on `models._price`, the same function the
benchmark bills with. Only the token counts are predicted; the arithmetic and
the price table are shared. That keeps one price table in the repository, which
is what makes a dollar figure from the router comparable to a dollar figure
from `run_eval.py`.

Output lengths are calibrated on measurement, not on a guess. The two-arm
probe metered 318 real responses and found the modelled 80-120 tokens in the
original cost estimate to be wrong by roughly 6x on hard maths - which is why
that probe cost $0.92 against a $0.44 projection. The constants below are the
measured values.
"""

from __future__ import annotations

from llm_routing import models

# Mean OUTPUT tokens per answer, measured on the `wide` ladder during the
# two-arm probe (docs/RESULTS.md §1, "What the probe cost").
#
#   maths  650  level-5 MATH500 problems produce long derivations
#   code    55  MBPP+ solutions are short functions
#
# `general` is NOT measured - there is no general-domain task in the benchmark.
# 400 is a deliberate over-estimate so the budget guard errs toward refusing a
# call rather than toward overspending, and it is labelled as an assumption
# everywhere it surfaces.
MEAN_OUTPUT_TOKENS = {"math": 650, "code": 55, "general": 400}

# Characters per token for the input side. A rough constant rather than a
# tokenizer call: the point is to size a budget check, and being 20% out on the
# input side moves the estimate very little, because output tokens dominate
# every price in the table by a factor of 2 to 5.
CHARS_PER_TOKEN = 4.0


def call_tracked(tier: str, task: dict, **kwargs):
    """`models.call`, plus how much of its cost actually left the account.

    Returns `(response, backend_cost_usd)`.

    Two things have to be true for a call to have cost real money, and
    conflating them is easy:

    * **It was not served from the response cache.** `models.call` returns the
      same ModelResponse either way and charges the policy in full - `cost_usd`
      answers "what would this cost in production", and production has no
      cross-run cache. Only a miss reaches a provider.
    * **The mode is `real`.** `call_stats["backend"]` counts everything the
      cache did not serve, which in mock mode is a *fabricated* response.
      Fabrication is not spend. Omitting this check makes mock runs report
      money they never spent - which is the failure this repository exists to
      avoid, appearing inside the tool that measures it.
    """
    before = models.call_stats["backend"]
    r = models.call(tier, task, **kwargs)
    reached_provider = models.call_stats["backend"] > before
    backend = r.cost_usd if (reached_provider and models.MODE == "real") else 0.0
    return r, backend


def estimate_tokens(task: dict, tier: str) -> tuple[int, int]:
    """(tokens_in, tokens_out) for one answer call at `tier`."""
    prompt = models.build_prompt(task, kind="answer")
    raw_in = max(1, int(len(prompt) / CHARS_PER_TOKEN))

    # The same tokenizer asymmetry the benchmark models: Claude 4.7 and later
    # emit roughly 30% more tokens for identical text, so the rungs of the
    # `claude` ladder disagree about how long the same prompt is. Ignoring it
    # would under-price exactly the rung a cascade escalates to.
    factor = models.MODELS[tier].get("tokenizer_factor", 1.0)
    tokens_in = int(raw_in * factor)

    tokens_out = MEAN_OUTPUT_TOKENS.get(task.get("domain"), 400)
    return tokens_in, int(tokens_out * factor)


def estimate_call_cost(task: dict, tier: str) -> float:
    """Projected USD for one answer call. Billed with the benchmark's own table."""
    tokens_in, tokens_out = estimate_tokens(task, tier)
    return models._price(tier, tokens_in, tokens_out)


def estimate_verification_cost(task: dict, tier: str, verifier: str, k: int) -> float:
    """Projected USD to verify one answer at `tier`.

    This is the cost a comparison against predictive routing must not forget.
    A cascade pays it on every query, including the ones it was always going to
    accept. It is the fixed cost that decides whether cascading is cheaper on
    a ladder - the price ratio does not, and the measured ladders are not
    monotonic in it (see findings.ratio_verdict).
    """
    if verifier == "self_consistency":
        if not models.MODELS[tier]["accepts_temperature"]:
            return 0.0  # cannot resample; verifier returns unverified for free
        # k-1 further draws: index 0 is the greedy answer already paid for.
        return (k - 1) * estimate_call_cost(task, tier)
    # `tests` runs a subprocess and `none` does nothing. Neither calls a model.
    return 0.0


def estimate_policy_cost(task: dict, policy: str, cfg) -> dict:
    """Project what each policy would spend on this query, without calling.

    Powers the `estimate_cost` MCP tool. Every figure is a projection from the
    price table and the measured token counts above - nothing here calls a
    model, so it is free and instant.

    The cascade figure is a RANGE, because its cost depends on an outcome it
    has not observed yet:

        best case   the cheap answer verifies    cheap + verification
        worst case  it escalates every rung      every rung + verification

    Reporting the expected value instead would need P(escalate), which is a
    property of the traffic rather than of the policy - so both ends are
    reported and the caller can weight them with their own escalation rate.
    """
    tiers = models.TIERS
    cheap, expensive = tiers[0], tiers[-1]
    verifier = cfg.verifier if cfg.verifier != "auto" else "self_consistency"

    cheap_cost = estimate_call_cost(task, cheap)
    expensive_cost = estimate_call_cost(task, expensive)

    if policy == "always_cheap":
        return {"policy": policy, "min_usd": cheap_cost, "max_usd": cheap_cost}
    if policy == "always_expensive":
        return {"policy": policy, "min_usd": expensive_cost, "max_usd": expensive_cost}
    if policy == "predictive":
        # One call, one rung, no verification. Which rung is chosen depends on
        # the heuristic, so the range spans both.
        return {"policy": policy, "min_usd": cheap_cost, "max_usd": expensive_cost}

    # cascade
    verify_cost = estimate_verification_cost(task, cheap, verifier, cfg.self_consistency_k)
    best = cheap_cost + verify_cost
    worst = best
    for tier in tiers[1:]:
        worst += estimate_call_cost(task, tier)
        worst += estimate_verification_cost(
            task, tier, verifier, cfg.self_consistency_k
        )
    return {
        "policy": policy,
        "min_usd": best,
        "max_usd": worst,
        "note": (
            "best case: the cheap answer verifies. worst case: it escalates "
            "every rung. Both include verification, which a cascade pays on "
            "every query whether or not it escalates."
        ),
    }
