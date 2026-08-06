"""
Model client. Three modes, selected by the ROUTER_MODE environment variable.

  MOCK   - no API key, no spend. Simulates a cheap and an expensive model with
           correlated success rates. Runs the ENTIRE pipeline end to end before
           any money is spent. This is the default.

  REAL   - live API calls. Needs whichever key the selected ladder's providers
           use; _client() names the missing one if it is unset.

  REPLAY - serve every call from cache/raw_calls.<ladder>.jsonl and refuse to
           touch the network. A miss is an error, not a fetch. This is what makes
           the repo reproducible by someone with no API key, and what makes every
           sweep after the first paid run cost nothing.

EVERY call goes through response_cache first, in all three modes. Read the module
docstring there before changing anything here: the cache is not a speed
optimisation, it is what makes the paired statistics valid. Without it,
always_cheap and cascade would be compared on different draws from the same
model, and the oracle would bound a set of responses nobody else received.

Every call returns a ModelResponse carrying tokens, latency and cost. Cost
accounting is half the point of the project and retrofitting it is painful.

    python3 run_eval.py                              # mock
    ROUTER_MODE=real   python3 run_eval.py --limit 10
    ROUTER_MODE=replay python3 run_eval.py
"""

import hashlib
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import response_cache


# ---------------------------------------------------------------------------
# .env loading
#
# Hand-rolled rather than `pip install python-dotenv`, because mock mode is meant
# to run on a bare interpreter with nothing installed, and a secrets loader is
# twenty lines. Adding a dependency that only matters in real mode would cost the
# repo its "clone it and it runs" property for everyone who never spends a cent.
#
# PRECEDENCE: a real environment variable always beats the file. That ordering is
# not arbitrary - it means CI, a one-off `ANTHROPIC_API_KEY=... python3 ...`, and a
# temporarily exported key all override .env without anyone having to remember to
# edit it back.
#
# The file is gitignored; .env.example is committed in its place and holds no
# secrets.
# ---------------------------------------------------------------------------

def load_dotenv(path=None):
    """Read KEY=VALUE lines from .env into os.environ, without overwriting.

    Deliberately forgiving about format, because the failure mode of a strict
    parser here is a confusing crash while holding a secret. Accepts a leading
    `export `, ignores blank lines and `#` comments, and strips one layer of
    matching quotes.

    Returns the names of the keys it set. NEVER the values - nothing in this repo
    should be able to print a secret by accident.
    """
    path = Path(path) if path else Path(__file__).parent / ".env"
    if not path.exists():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        # Real environment wins. See PRECEDENCE above.
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


DOTENV_LOADED = load_dotenv()

MODE = os.environ.get("ROUTER_MODE", "mock")  # "mock" | "real" | "replay"
if MODE not in ("mock", "real", "replay"):
    raise SystemExit(f"ROUTER_MODE must be mock, real or replay - got {MODE!r}")

# Seed for the whole mock world. Every mock outcome is a pure hash of
# (seed, task, tier, temperature, sample_idx), so a run is reproducible byte for
# byte, and a sweep over seeds gives the run-to-run variance of the mock. That
# variance is the only honest context for reading a two-point accuracy gap at
# n=100.
MOCK_SEED = int(os.environ.get("MOCK_SEED", "0"))

# ---------------------------------------------------------------------------
# DECISION #1: the model ladder.
#
# The ladder is SELECTED, not hard-coded, because the price ratio between its
# rungs is the single largest driver of every result in this repo. A wider ratio
# makes a wasted cheap call cheaper and therefore flatters every escalating
# policy. Any conclusion here is ratio-dependent, and the only way to say that
# honestly is to be able to change the ratio and re-run:
#
#     ROUTER_LADDER=claude   python3 run_eval.py    # 1x / 3x / 5x  (default)
#     ROUTER_LADDER=deepseek python3 run_eval.py    # 1x / 3.1x, one provider
#     ROUTER_LADDER=wide     python3 run_eval.py    # 1x / 36x, cross-provider
#
# All prices are per million tokens, input / output, at LIST or standard rates.
# Verified 2026-07-30 against platform.claude.com/docs/en/about-claude/pricing
# and api-docs.deepseek.com/quick_start/pricing.
#
# PROMOTIONAL PRICING IS DELIBERATELY IGNORED. Sonnet 5 has introductory $2/$10
# pricing running to 2026-08-31; the standard $3/$15 is used instead. Billing a
# rung at a rate that expires would mean a run reproduced in September did not
# match a run from August, and an experiment whose cost axis expires is not
# reproducible. DeepSeek's page likewise says prices may be adjusted, so the
# figures below are stamped with the date they were checked.
#
# THE CHEAP RUNG IS NOT A FREE CHOICE. verify_math samples it at temperature 0.8,
# and that sampling IS the self-consistency signal - without it there is nothing
# to disagree. Among current Claude models only Haiku 4.5 still accepts a
# temperature: it was removed on Opus 4.7+ and returns a 400 on Sonnet 5. Both
# DeepSeek rungs accept it, which is why the deepseek ladder can do something the
# claude ladder cannot - verify at EVERY rung rather than only the bottom one.
# ---------------------------------------------------------------------------

PROVIDERS = {
    # base_url None means the SDK default. Both providers speak the Anthropic
    # wire format, so one client class serves both and there is no second SDK.
    "anthropic": {
        "base_url": None,
        "key_env": "ANTHROPIC_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/anthropic",
        "key_env": "DEEPSEEK_API_KEY",
        # !! SILENT REMAPPING. DeepSeek's Anthropic-format endpoint maps any
        # unrecognised model name onto deepseek-v4-flash, and maps names starting
        # with `claude-opus` onto deepseek-v4-pro. It does NOT error.
        #
        # That is the most dangerous behaviour in this file: a typo in a model id
        # would quietly return answers from a different, cheaper model, the run
        # would look completely normal, and the cost table would be wrong. Guarded
        # by _check_model_ids() at import.
        "remaps_unknown_models": True,
    },
}

# One entry per model, so a model's price and API contract are stated once and a
# ladder is just an ordering over them.
MODEL_SPECS = {
    "claude-haiku-4-5-20251001": {
        # Dated id rather than the `claude-haiku-4-5` alias: an alias can be
        # repointed, and this run needs to be reproducible.
        "provider": "anthropic",
        "price_in": 1.00, "price_out": 5.00,
        "accepts_temperature": True,
        # Pre-4.6 model: thinking is off unless explicitly enabled.
        "thinking_on_by_default": False,
        # Previous tokenizer. See TOKENIZER ASYMMETRY below.
        "tokenizer_factor": 1.00,
        "mock": {"base": 0.86, "spread": 0.36},
        "mock_tokens_out": 80, "mock_latency_s": 0.4,
    },
    "claude-sonnet-5": {
        "provider": "anthropic",
        "price_in": 3.00, "price_out": 15.00,
        # temperature, top_p and top_k all return a 400 at non-default values.
        "accepts_temperature": False,
        # Adaptive thinking is ON by default on Sonnet 5, unlike Sonnet 4.6.
        "thinking_on_by_default": True,
        "tokenizer_factor": 1.30,
        "mock": {"base": 0.94, "spread": 0.24},
        "mock_tokens_out": 100, "mock_latency_s": 0.6,
    },
    "claude-opus-5": {
        # Opus 5 ids carry no date suffix - never append one.
        "provider": "anthropic",
        "price_in": 5.00, "price_out": 25.00,
        # Sampling parameters were removed on Opus 4.7+; sending temperature at
        # all returns a 400. This is why call() cannot pass it unconditionally.
        "accepts_temperature": False,
        # Thinking is ON by default on Opus 5, unlike Opus 4.8 and earlier where
        # omitting the field meant off. Disabled deliberately so the comparison
        # isolates capability rather than reasoning-time compute. Disabling is
        # valid only at effort `high` or below; `high` is the default.
        "thinking_on_by_default": True,
        "tokenizer_factor": 1.30,
        "mock": {"base": 0.97, "spread": 0.16},
        "mock_tokens_out": 120, "mock_latency_s": 0.9,
    },
    "deepseek-v4-flash": {
        "provider": "deepseek",
        # Cache-MISS input price. DeepSeek's context cache drops input to $0.0028,
        # a 50x reduction, but that discount applies to repeated prefixes and this
        # evaluation sends a distinct prompt per task, so the miss price is the
        # honest one to model.
        "price_in": 0.14, "price_out": 0.28,
        # Fully supported, range 0.0-2.0. This is what lets the deepseek ladder
        # run a self-consistency verifier at any rung.
        "accepts_temperature": True,
        # Thinking mode is the DEFAULT on both DeepSeek rungs, so it has to be
        # disabled explicitly, exactly as on Opus 5.
        "thinking_on_by_default": True,
        # Unknown tokenizer relative to Claude's. 1.0 is a placeholder meaning
        # "not modelled", not a measurement - see TOKENIZER ASYMMETRY.
        "tokenizer_factor": 1.00,
        "mock": {"base": 0.84, "spread": 0.38},
        "mock_tokens_out": 90, "mock_latency_s": 0.5,
    },
    "deepseek-v4-pro": {
        "provider": "deepseek",
        "price_in": 0.435, "price_out": 0.87,
        "accepts_temperature": True,
        "thinking_on_by_default": True,
        "tokenizer_factor": 1.00,
        "mock": {"base": 0.93, "spread": 0.26},
        "mock_tokens_out": 110, "mock_latency_s": 0.8,
    },
}

# Named ladders, cheapest rung first. Two or three rungs; more would need tier
# names this repo does not define, and _build_ladder says so rather than guessing.
LADDERS = {
    # The default. Three rungs from one provider at 1x / 3x / 5x list price.
    # Three rather than two because with only two rungs a cascade has exactly one
    # decision to make, so "the cascade won" cannot be separated from "there were
    # only two options". Three is the smallest ladder on which a rung can be
    # skipped.
    "claude": ["claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-opus-5"],

    # One provider, one tokenizer, both rungs accepting a temperature. The
    # cleanest ladder in the repo: it removes the tokenizer confound entirely and
    # allows verification above the bottom rung. Its ratio is 3.1x, NARROWER than
    # the claude ladder, which makes it the pessimistic case for escalation.
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro"],

    # The extreme-ratio case, and the reason the ladder is configurable at all.
    # DeepSeek's cheap rung against Claude's top rung is roughly 36x on input and
    # 89x on output, so a wasted cheap call costs almost nothing. If the cascade's
    # advantage does not widen here, the "results are ratio-dependent" caveat is
    # weaker than it looks - which would be a finding.
    #
    # Cross-provider, so it is also the only ladder that needs two API keys, and
    # the only one where a capability difference is confounded with a provider
    # difference. Read its numbers accordingly.
    "wide": ["deepseek-v4-flash", "claude-opus-5"],
}

LADDER = os.environ.get("ROUTER_LADDER", "claude")
if LADDER not in LADDERS:
    raise SystemExit(
        f"ROUTER_LADDER must be one of {', '.join(sorted(LADDERS))} - got {LADDER!r}"
    )

# Tier names by ladder depth. The names are positional labels rather than
# capability claims: `cheap` means "bottom rung of whatever ladder is loaded".
_TIER_NAMES = {2: ["cheap", "expensive"], 3: ["cheap", "mid", "expensive"]}


def _build_ladder(name):
    ids = LADDERS[name]
    if len(ids) not in _TIER_NAMES:
        raise SystemExit(
            f"ladder {name!r} has {len(ids)} rungs; this repo names only "
            f"{sorted(_TIER_NAMES)} of them. Add names to _TIER_NAMES first."
        )
    tiers = _TIER_NAMES[len(ids)]
    models = {}
    for tier, model_id in zip(tiers, ids):
        if model_id not in MODEL_SPECS:
            raise SystemExit(f"ladder {name!r} refers to unknown model {model_id!r}")
        models[tier] = {"id": model_id, **MODEL_SPECS[model_id]}
    # Prices must ascend, or the words "cheap" and "expensive" are lies and every
    # cost-ordered policy in policies.py silently does the wrong thing.
    prices = [models[t]["price_in"] for t in tiers]
    if prices != sorted(prices):
        raise SystemExit(
            f"ladder {name!r} is not in ascending price order: {prices}. "
            f"The tier names and the oracle's cost ordering both depend on it."
        )
    return models, tiers


MODELS, TIERS = _build_ladder(LADDER)


def _check_model_ids():
    """Fail loudly on a provider that silently substitutes models.

    See PROVIDERS["deepseek"]. An endpoint that answers an unknown model name with
    a different model rather than an error can invalidate a whole paid run without
    producing a single warning, so the ids are checked against MODEL_SPECS before
    anything is sent. This is cheap insurance against a typo costing real money and
    producing plausible, wrong numbers.
    """
    for tier, cfg in MODELS.items():
        provider = PROVIDERS[cfg["provider"]]
        if provider.get("remaps_unknown_models") and cfg["id"] not in MODEL_SPECS:
            raise SystemExit(
                f"tier {tier!r} uses id {cfg['id']!r} on a provider that silently "
                f"remaps unknown model names instead of erroring. Verify the id "
                f"against the provider's model list before running."
            )


_check_model_ids()

# Configured only now, because the cache filenames carry the ladder name and the
# ladder is not known until it has been built and validated.
response_cache.configure(MODE, LADDER)


def ladder_summary():
    """One line per rung, for a run header. Ratios are what matter, so show them."""
    base_in = MODELS[TIERS[0]]["price_in"]
    base_eff = base_in * MODELS[TIERS[0]]["tokenizer_factor"]
    out = [f"ladder={LADDER}  rungs={len(TIERS)}"]
    for tier in TIERS:
        m = MODELS[tier]
        eff = m["price_in"] * m["tokenizer_factor"]
        out.append(
            f"  {tier:<10} {m['id']:<28} {m['provider']:<10} "
            f"${m['price_in']}/${m['price_out']} per MTok  "
            f"list {m['price_in'] / base_in:.1f}x  effective {eff / base_eff:.1f}x"
            f"{'  temp OK' if m['accepts_temperature'] else '  no temp'}"
        )
    return out


# ---------------------------------------------------------------------------
# TOKENIZER ASYMMETRY - why the effective price ratio is not the list ratio.
#
# Claude 4.7 and later use a newer tokenizer that produces roughly 30% more
# tokens for the same text than the tokenizer used by Claude Sonnet 4.6 and
# earlier (platform.claude.com/docs/en/about-claude/pricing, checked 2026-07-30).
# Sonnet 5 and Opus 5 are on the new tokenizer; Haiku 4.5 is on the old one. So on
# the claude ladder the boundary runs between the bottom rung and everything above
# it.
#
# The rungs therefore do not merely charge different prices for the same token
# count - they disagree about how many tokens the same prompt IS. On identical
# text the upper rungs bill about 1.3x the input tokens, making the claude
# ladder's effective input ratios roughly 1x / 3.9x / 6.5x rather than 1x / 3x /
# 5x. `ladder_summary()` prints both.
#
# `tokenizer_factor` models that in mock mode only. In real mode token counts come
# back from the API already correct. The exact factor is content-dependent and
# 1.30 is the documented approximation, so it is a modelled constant like every
# other mock parameter rather than a measurement.
#
# THE DEEPSEEK ENTRIES USE 1.00, WHICH MEANS "NOT MODELLED", NOT "THE SAME".
# DeepSeek does not publish a comparison against Claude's tokenizer and this
# project has not measured one, so on the `wide` ladder the cross-provider token
# counts are the least trustworthy numbers in the repo. A real run fixes this for
# free, because then the counts come from the APIs.
#
# Consequence worth stating plainly: on the claude ladder this asymmetry makes
# escalation more expensive than the price table suggests, which works AGAINST the
# cascade and in favour of routing cheap.
# ---------------------------------------------------------------------------

# Truncation is the expensive failure here: a cut-off math answer loses its
# \boxed{} and the grader scores a correct answer as wrong, which reads as a
# capability result instead of a bug.
#
# Was 2048, on the reasoning that replies are short with thinking disabled.
# MEASURED on 6 August 2026 against 118 real responses on the `wide` ladder, and
# 2048 was not enough once the maths half moved to MATH500 level 5:
#
#   code            mean  55 tokens out
#   maths           mean 650 tokens out
#   at the 2048 cap 2 of 118 calls (math-422 on cheap, math-103 on expensive)
#
# Both truncations were inspected rather than assumed. Neither was a degenerate
# loop: both were coherent level-5 derivations that simply ran long, and the
# Opus one was a few hundred tokens short of its \boxed{}. So the cap was
# binding on real work, not catching pathology.
#
# 1.7% sounds ignorable and is not, because the truncations are not randomly
# distributed. They land on the HARDEST tasks, which are exactly the ones that
# decide the routable fraction, and they bias it directionally: a truncated
# cheap answer reads as "the cheap rung failed" and inflates `routable`.
#
# 4096 leaves headroom over the observed maths tail. Raising it invalidates the
# response cache, since max_tokens is part of the cache key - see
# response_cache._KEY_FIELDS. That is why this number is worth getting right
# once rather than tuning later.
MAX_TOKENS = 4096

# The LLM-as-router call (policies.policy_llm_router) is a one-word
# classification, so it gets its own tiny cap. This is the number that decides
# whether "an LLM call would defeat the purpose" holds: at 8 output tokens the
# router call cannot cost more than a rounding error, and the point of the policy
# is to measure that rather than assert it.
ROUTER_MAX_TOKENS = 8


@dataclass
class ModelResponse:
    text: str
    tier: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    cost_usd: float


def _price(tier: str, tokens_in: int, tokens_out: int) -> float:
    m = MODELS[tier]
    return (tokens_in / 1e6) * m["price_in"] + (tokens_out / 1e6) * m["price_out"]


# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------

# Mock tuning, read per MODEL rather than per tier position. p_correct for a model
# is base - spread * difficulty_pct, and both constants live in that model's
# MODEL_SPECS entry.
#
# Per model, not per tier, for a reason that matters once ladders are selectable:
# `cheap` means a different model on each ladder, so a skill table keyed on the
# tier name would claim DeepSeek's flash rung and Claude's Haiku rung are equally
# capable. Keyed on the model id, each ladder gets its own honest mock world and
# two ladders sharing a model share its behaviour.
#
# EVERY ONE OF THESE NUMBERS IS STIPULATED. They are chosen so the bottom rung
# fails often enough for routing to matter and the rungs stay separated; the
# realised failure rate is a consequence of them plus the task mix, and run_eval
# prints it as the pilot gate. Read it there rather than trusting a comment.
MOCK_SKILL = {tier: MODELS[tier]["mock"] for tier in TIERS}
MOCK_TOKENS_OUT = {tier: MODELS[tier]["mock_tokens_out"] for tier in TIERS}
MOCK_LATENCY_S = {tier: MODELS[tier]["mock_latency_s"] for tier in TIERS}

# How much of a task's outcome is a shared property of the task rather than
# tier-specific luck. 1.0 = perfectly nested failure (anything the expensive
# model gets wrong, the cheap model also gets wrong); 0.0 = independent.
# Real tiers from one provider are strongly but not perfectly correlated.
MOCK_FAILURE_CORRELATION = 0.75

# ---------------------------------------------------------------------------
# Mock skill of the LLM-as-router classifier (policies.policy_llm_router).
#
# READ THIS BEFORE QUOTING ANY llm_router ACCURACY FROM MOCK MODE.
#
# There is no way to simulate "the cheap model reads the question and judges its
# difficulty" without deciding in advance how good that judgement is. So the mock
# router is an oracle on the mock's own latent difficulty, corrupted at rate
# 1 - MOCK_ROUTER_SKILL. Its accuracy in mock mode is therefore a RESTATEMENT OF
# THIS CONSTANT and measures nothing. Raising it would make llm_router look
# better with no change to any real capability.
#
# What IS measurable in mock mode, and what the policy exists to measure, is the
# COST AND LATENCY of the extra round trip: those come from the price table and
# the token counts rather than from this constant. policies.py DECISION #4
# rejected LLM routing on the grounds that it "would add a full round trip and
# defeat the purpose". That is a quantitative claim, and this is what tests it.
# ---------------------------------------------------------------------------
MOCK_ROUTER_SKILL = 0.70

# The mock router's notion of "hard": the upper half of the within-domain
# difficulty percentile. Matches how MOCK_SKILL drives p_correct, so the router
# is judging the same latent quantity the mock models are failing on.
MOCK_ROUTER_HARD_PCT = 0.5


def _wrong_answer(truth: str, rng: random.Random) -> str:
    r"""A plausible wrong answer, drawn from a small spread.

    Must SCATTER rather than be constant. verify_math accepts the cheap answer
    when self-consistency samples agree, so a mock that always returned the same
    wrong string would make every wrong answer look unanimously confident and the
    math cascade would never escalate.

    Note the flip side, because it inflates the math result: scattering across
    distinct values means majority voting recovers the truth whenever 2 of 5
    samples are right. Real models cluster on the same wrong answer far more
    often, so mock-mode self-consistency is stronger than the literature's.

    MATH500 answers are not all integers (\frac{14}{3}, 3\sqrt{13}, 6-5i), so
    the integer path is a fast case rather than the only one.
    """
    delta = rng.randint(1, 9)
    try:
        return str(int(truth) + delta)
    except ValueError:
        return f"{truth}+{delta}"


def _draw(*parts) -> random.Random:
    """A Random seeded by hashing its arguments.

    Everything stochastic in the mock goes through here, so a mock outcome is a
    PURE FUNCTION of its inputs. No global RNG state, therefore no dependence on
    how many calls happened to precede this one.
    """
    key = "|".join(str(p) for p in (MOCK_SEED, *parts))
    return random.Random(hashlib.md5(key.encode()).hexdigest())


def _mock_tokens_in(tier: str, prompt: str) -> int:
    """Modelled input token count for one tier on one prompt.

    Roughly four characters per token, scaled by the tier's tokenizer. See
    TOKENIZER ASYMMETRY: the same prompt is a different number of tokens on the
    two tiers, and ignoring that understates what escalation costs.
    """
    return max(20, int(len(prompt) / 4 * MODELS[tier]["tokenizer_factor"]))


def _mock_route_call(tier: str, prompt: str, task: dict, sample_idx: int) -> ModelResponse:
    """Simulate the LLM-as-router classification call.

    See MOCK_ROUTER_SKILL. The label is fabricated from a constant; the tokens,
    latency and price are not, and they are the part worth reading.
    """
    difficulty = task.get("difficulty_pct", 0.5)
    truth_hard = difficulty >= MOCK_ROUTER_HARD_PCT
    agrees = _draw(
        task["id"], MODELS[tier]["id"], "router", sample_idx
    ).random() < MOCK_ROUTER_SKILL
    said_hard = truth_hard if agrees else not truth_hard

    tokens_in = _mock_tokens_in(tier, prompt)
    tokens_out = 3  # "HARD" or "EASY", plus the stop
    return ModelResponse(
        text="HARD" if said_hard else "EASY",
        tier=tier,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        # Shorter than an answer call because there is almost nothing to
        # generate, but still a full network round trip - which is the cost the
        # policy exists to measure. A modelled constant, like every other mock
        # latency in this file.
        latency_s=0.35,
        cost_usd=_price(tier, tokens_in, tokens_out),
    )


def _mock_call(
    tier: str, prompt: str, task: dict, temperature: float, sample_idx: int,
    kind: str = "answer",
) -> ModelResponse:
    """Simulate a model.

    Success is driven by task difficulty so that cheap and expensive models FAIL
    ON THE SAME HARD TASKS. That correlation matters: if failures were
    independent, escalation would look far better than any real cascade achieves.

    The correlation is produced by a SHARED latent draw per task, blended with a
    tier-specific draw.
    """
    if kind == "route":
        return _mock_route_call(tier, prompt, task, sample_idx)

    # difficulty_pct is a within-domain percentile, so it means the same thing
    # for a math task and a code task. difficulty_proxy does not.
    difficulty = task.get("difficulty_pct", 0.5)
    cfg = MOCK_SKILL[tier]
    p_correct = max(0.05, min(0.99, cfg["base"] - cfg["spread"] * difficulty))

    # Keyed on sample_idx, NOT on a fresh random nonce. Self-consistency samples
    # still need to differ from each other, but they must differ REPRODUCIBLY:
    # sample 3 of a given task is always the same sample 3. An unseeded global
    # RNG here made every mock run different, which on a project whose headline
    # is a two-point gap meant the headline was noise.
    #
    # Keying on sample_idx also makes the mock invariant to CALL ORDER, so a
    # given (task, tier, temperature, sample_idx) is the same response whatever
    # ran before it. `--limit 10` therefore reproduces the first ten tasks of the
    # full run, and running one policy alone gives the same answers as running
    # all of them.
    #
    # Two policies are exempt, and for a good reason rather than a bug:
    # random_matched and routellm are CALIBRATED on the task set being run, so a
    # 10-task run sets their escalation rate and score threshold from 10 tasks
    # and can route a task differently than the full run does. The model
    # responses are still identical; the policy's choice of which one to ask is
    # what moved. See policies.calibrate_random_rates.
    # Keyed on the MODEL ID, not the tier name. Two ladders both have a rung
    # called `cheap`, and keying on the name would make DeepSeek's flash rung draw
    # the same luck as Claude's Haiku rung while being scored against a different
    # p_correct - which would silently make cross-ladder comparisons nonsense.
    shared = _draw(task["id"], "shared", temperature, sample_idx).random()
    rng = _draw(task["id"], MODELS[tier]["id"], temperature, sample_idx)
    draw = MOCK_FAILURE_CORRELATION * shared + (1 - MOCK_FAILURE_CORRELATION) * rng.random()
    correct = draw < p_correct

    if task["domain"] == "math":
        truth = task["grader_payload"]["answer"]
        answer = truth if correct else _wrong_answer(truth, rng)
        text = f"Reasoning... the answer is $\\boxed{{{answer}}}$"
    else:
        # A "correct" mock answer must be code that actually passes the shipped
        # asserts, so it has to be the reference solution stored by
        # build_taskset.py. Without _ref_code every code task would fail and the
        # whole code domain would silently read 0%.
        ref = task.get("_ref_code")
        if correct and not ref:
            raise KeyError(
                f"task {task['id']} has no _ref_code; rebuild with build_taskset.py"
            )
        text = "```python\n" + (ref if correct else "def _wrong(): pass") + "\n```"

    tokens_in = _mock_tokens_in(tier, prompt)
    # Modelled output lengths, all well under a real reply, which is why run_eval
    # warns against quoting mock cost RATIOS rather than absolute mock dollars.
    tokens_out = MOCK_TOKENS_OUT[tier]
    latency = MOCK_LATENCY_S[tier]

    return ModelResponse(
        text=text,
        tier=tier,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_s=latency,
        cost_usd=_price(tier, tokens_in, tokens_out),
    )


# ---------------------------------------------------------------------------
# Real implementation
# ---------------------------------------------------------------------------

_CLIENTS = {}
truncated_calls = 0


def _client(provider: str):
    """One client per provider, reused for the whole run.

    Constructing a client per call spins up a fresh HTTP connection pool and throws
    it away, and this evaluation makes hundreds of calls.

    Both providers speak the Anthropic wire format, so the only difference is the
    base URL and which environment variable holds the key. That is why adding
    DeepSeek needed no second SDK and no provider abstraction beyond this function.

    The key is checked BEFORE the SDK is imported, deliberately. Both can be
    missing at once on a fresh clone, and "your key is not set" is the more useful
    of the two messages to lead with - it is the one a reader is more likely to be
    confused by, and `pip install` is obvious once mentioned.
    """
    if provider not in _CLIENTS:
        cfg = PROVIDERS[provider]
        key = os.environ.get(cfg["key_env"])
        if not key:
            env_file = Path(__file__).parent / ".env"
            where = (f"{env_file.name} exists but has no {cfg['key_env']} line"
                     if env_file.exists() else f"no {env_file.name} file found")
            raise SystemExit(
                f"\n{cfg['key_env']} is not set, and ladder {LADDER!r} needs it for "
                f"the {provider!r} provider.\n"
                f"  ({where})\n\n"
                f"  Easiest fix - copy the template and fill it in:\n"
                f"      cp .env.example .env      # then edit .env\n\n"
                f"  Or set it for one command:\n"
                f"      macOS/Linux   export {cfg['key_env']}=sk-...\n"
                f"      PowerShell    $env:{cfg['key_env']}=\"sk-...\"\n"
                f"      cmd.exe       set {cfg['key_env']}=sk-...\n\n"
                f"  Or pick a ladder that does not need this provider:\n"
                f"      ROUTER_LADDER=claude    needs ANTHROPIC_API_KEY only\n"
                f"      ROUTER_LADDER=deepseek  needs DEEPSEEK_API_KEY only\n"
            )

        try:
            from anthropic import Anthropic
        except ImportError:
            raise SystemExit(
                "\nThe `anthropic` package is not installed, and real mode needs it.\n"
                "  pip install -r requirements.txt\n\n"
                "  It is needed only for real mode. Mock and replay run on the\n"
                "  standard library alone, which is why it is not imported at the\n"
                "  top of this file.\n"
            )

        kwargs = {"api_key": key}
        if cfg["base_url"]:
            kwargs["base_url"] = cfg["base_url"]
        _CLIENTS[provider] = Anthropic(**kwargs)
    return _CLIENTS[provider]


def _real_call(
    tier: str, prompt: str, task: dict, temperature: float, sample_idx: int,
    kind: str = "answer",
) -> ModelResponse:
    # sample_idx does not pin a real model: sampling at temperature > 0 is
    # stochastic and cannot be reproduced. What pins it is the response cache.
    # The FIRST draw for a given sample_idx is stored, and every later reader
    # gets that one. So real mode is reproducible after the fact rather than by
    # construction, which is the strongest guarantee a hosted API allows.
    global truncated_calls
    cfg = MODELS[tier]

    # The two tiers have genuinely different API contracts, so the request is
    # built per-model rather than shared. Sending temperature to Opus 5 is a 400;
    # omitting `thinking` on Opus 5 silently turns thinking ON.
    kwargs = {
        "model": cfg["id"],
        "max_tokens": _max_tokens_for(kind),
        "messages": [{"role": "user", "content": prompt}],
    }
    if cfg["accepts_temperature"]:
        kwargs["temperature"] = temperature
    if cfg["thinking_on_by_default"]:
        kwargs["thinking"] = {"type": "disabled"}

    t0 = time.time()
    msg = _client(cfg["provider"]).messages.create(**kwargs)
    latency = time.time() - t0

    # A truncated reply is the dangerous failure: it grades as WRONG rather than
    # as an error, so it silently deflates the accuracy of whichever tier hit the
    # cap. Count it loudly instead of letting it pass as a result.
    if msg.stop_reason == "max_tokens":
        truncated_calls += 1
        cap = "ROUTER_MAX_TOKENS" if kind == "route" else "MAX_TOKENS"
        consequence = (
            "the routing decision falls back to EASY"
            if kind == "route"
            else "this will grade as wrong"
        )
        print(
            f"  !! TRUNCATED at max_tokens: {task['id']} on {tier} ({kind}). "
            f"{consequence}. Raise models.{cap}.",
            file=sys.stderr,
        )

    text = "".join(b.text for b in msg.content if b.type == "text")
    return ModelResponse(
        text=text,
        tier=tier,
        # Real token counts, so no tokenizer_factor here - the API has already
        # applied whichever tokenizer the model uses.
        tokens_in=msg.usage.input_tokens,
        tokens_out=msg.usage.output_tokens,
        latency_s=latency,
        cost_usd=_price(tier, msg.usage.input_tokens, msg.usage.output_tokens),
    )


# ---------------------------------------------------------------------------

PROMPTS = {
    # \boxed{} is the standard MATH answer protocol. Asking for "a plain number"
    # would be wrong here: most level 3-5 answers are fractions, radicals or
    # tuples.
    "math": (
        "Solve this problem. Show brief reasoning, then give the final answer "
        "in \\boxed{{}}.\n\n{q}"
    ),
    # The tests go in the prompt. Without them the model has to guess the
    # function name the asserts will call, and failures then measure
    # name-guessing rather than difficulty. This is the standard MBPP protocol,
    # and it does not leak: the asserts are the specification, not the answer.
    "code": (
        "Write a Python function for this task. Return ONLY a python code "
        "block, no explanation.\n\n{q}\n\nYour code should pass these tests:\n{tests}"
    ),
    # The LLM-as-router prompt. Deliberately austere:
    #   - it shows the QUESTION ONLY, never the tests or the answer, so it sees
    #     exactly what predict_is_hard sees and the comparison is fair;
    #   - it forbids reasoning, because a router that thinks before routing is
    #     just a slow expensive model and the whole premise is that it is cheap;
    #   - one word out, so ROUTER_MAX_TOKENS can be 8 and the cost is bounded.
    "route": (
        "You are a difficulty classifier for a model router. Answer with "
        "exactly one word, EASY or HARD, and nothing else.\n\n"
        "HARD means a small fast model would probably get this wrong and it "
        "should be sent to a larger model.\n\nProblem:\n{q}"
    ),
}


def build_prompt(task: dict, kind: str = "answer") -> str:
    """The exact text sent to the model.

    This is what the cache is keyed on, so it must be a pure function of the task
    and the kind. Anything that varied per run - a timestamp, a nonce, a dict
    that iterates differently - would silently make every call a cache miss.
    """
    if kind == "route":
        return PROMPTS["route"].format(q=task["prompt"])
    if task["domain"] == "code":
        tests = "\n".join(task["grader_payload"]["tests"])
        return PROMPTS["code"].format(q=task["prompt"], tests=tests)
    return PROMPTS["math"].format(q=task["prompt"])


def _max_tokens_for(kind: str) -> int:
    return ROUTER_MAX_TOKENS if kind == "route" else MAX_TOKENS


# How many times the policies asked for a call, how many were served from the
# cache, and how many actually reached a backend. The last is the number that
# costs money; the first is the number the cost table is built from. See
# response_cache's docstring for why those differ on purpose.
call_stats = {"requested": 0, "from_cache": 0, "backend": 0}


def reset_call_stats():
    for k in call_stats:
        call_stats[k] = 0


def call(
    tier: str, task: dict, temperature: float = 0.0, sample_idx: int = 0,
    kind: str = "answer",
) -> ModelResponse:
    """One model call, served from the response cache when it has been seen.

    sample_idx identifies WHICH draw this is for a given (task, tier,
    temperature). Callers that sample repeatedly - only the math verifier, so far
    - must pass a distinct index per sample, or every sample is the same cache
    entry and self-consistency becomes trivially unanimous.

    kind selects the prompt template: "answer" solves the task, "route" asks the
    cheap model to classify its difficulty. It is not in the cache key directly,
    because the prompt text already differs and the prompt IS in the key.

    Returns the full ModelResponse on a cache hit, cost included. The caller is
    charged either way - see the response_cache docstring. Deduplication changes
    what the harness spends, not what a policy costs.
    """
    prompt = build_prompt(task, kind)

    def keyfor(mode):
        return response_cache.make_key(
            mode=mode, model=MODELS[tier]["id"], prompt=prompt,
            temperature=temperature, sample_idx=sample_idx,
            max_tokens=_max_tokens_for(kind),
            mock_seed=MOCK_SEED if mode == "mock" else None,
        )

    # Replay prefers real responses and falls back to mock ones, so a mock cache
    # can be used to exercise the replay path itself without an API key. Real and
    # mock entries never collide, because `mode` is in the hash.
    candidates = [keyfor("real"), keyfor("mock")] if MODE == "replay" else [keyfor(MODE)]
    key = candidates[0]

    call_stats["requested"] += 1
    for k in candidates:
        rec = response_cache.get(k)
        if rec is not None:
            call_stats["from_cache"] += 1
            return ModelResponse(
                text=rec["text"], tier=rec["tier"], tokens_in=rec["tokens_in"],
                tokens_out=rec["tokens_out"], latency_s=rec["latency_s"],
                cost_usd=rec["cost_usd"],
            )

    if MODE == "replay":
        raise KeyError(
            f"replay mode: no cached response for {task['id']} tier={tier} "
            f"temp={temperature} sample={sample_idx} kind={kind}\n"
            f"  key={key}\n"
            f"  Replay never calls a backend. Either the cache is incomplete, or "
            f"a prompt or parameter changed since it was populated.\n"
            f"  Repopulate with: ROUTER_MODE=real python3 run_eval.py"
        )

    call_stats["backend"] += 1
    backend = _mock_call if MODE == "mock" else _real_call
    r = backend(tier, prompt, task, temperature, sample_idx, kind)

    # Written immediately rather than batched: a real run that dies halfway
    # through must keep every response it paid for.
    response_cache.put(key, {
        # Not part of the key. Stored so the file can be grepped and audited by a
        # human, and so a cache entry can be traced back to a task.
        "task_id": task["id"], "domain": task["domain"], "kind": kind,
        "mode": MODE, "mock_seed": MOCK_SEED if MODE == "mock" else None,
        "model": MODELS[tier]["id"], "temperature": temperature,
        "sample_idx": sample_idx, "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        **asdict(r),
    })
    return r
