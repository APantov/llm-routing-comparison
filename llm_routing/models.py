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

EVERY call goes through response_cache first, in all three modes. It is not a
speed optimisation but what makes the paired statistics valid - read its module
docstring before changing anything here.

Every call returns a ModelResponse carrying tokens, latency and cost.

    python -m llm_routing.run_eval                              # mock
    ROUTER_MODE=real   python -m llm_routing.run_eval --limit 10
    ROUTER_MODE=replay python -m llm_routing.run_eval
"""

import hashlib
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from llm_routing import paths
from llm_routing import response_cache


# .env loading, hand-rolled rather than python-dotenv: mock mode runs on a bare
# interpreter with nothing installed, and a dependency that only matters in real
# mode would cost that for everyone who never spends a cent.
#
# PRECEDENCE: a real environment variable always beats the file, so CI and a
# one-off `ANTHROPIC_API_KEY=... python ...` override .env without an edit-and-undo.
# The file is gitignored; .env.example is committed in its place.

def load_dotenv(path=None):
    """Read KEY=VALUE lines from .env into os.environ, without overwriting.

    Forgiving about format, because a strict parser here crashes confusingly
    while holding a secret: accepts a leading `export `, ignores blanks and `#`
    comments, strips one layer of matching quotes.

    Returns the names of the keys it set, NEVER the values.
    """
    path = Path(path) if path else paths.ENV_FILE
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
# (seed, task, tier, temperature, sample_idx), so a run reproduces byte for byte
# and a sweep over seeds gives the mock's run-to-run variance - the only honest
# context for reading a two-point accuracy gap.
MOCK_SEED = int(os.environ.get("MOCK_SEED", "0"))

# Replay serves from the real cache. When a key is missing it can either raise
# (the default) or fall back to the mock cache for the same key.
#
# Defaults OFF: it is otherwise a silent route from a fabricated response into a
# file labelled as a measurement. Left on, it can serve all 240 of the maths
# cascade's self-consistency samples from the mock cache into a results.jsonl
# whose every row says `simulated: false`. Off, a missing sample raises ReplayMiss
# and run_eval drops the policy.
#
# Turning it on is supported and does not lie: ModelResponse.simulated tracks
# which cache served each call, so a row built from even one fabricated response
# is stamped `simulated: true`.
REPLAY_FALLBACK_TO_MOCK = os.environ.get(
    "ROUTER_REPLAY_FALLBACK", "0") in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# DECISION #1: the model ladder.
#
# The ladder is SELECTED, not hard-coded, because the price ratio between its
# rungs is the single largest driver of every result in this repo. A wider ratio
# makes a wasted cheap call cheaper and therefore flatters every escalating
# policy. Any conclusion here is ratio-dependent, and the only way to say that
# honestly is to be able to change the ratio and re-run:
#
#     ROUTER_LADDER=claude   python -m llm_routing.run_eval    # 1x / 3x / 5x  (default)
#     ROUTER_LADDER=deepseek python -m llm_routing.run_eval    # 1x / 3.1x, one provider
#     ROUTER_LADDER=wide     python -m llm_routing.run_eval    # 1x / 36x, cross-provider
#
# All prices are per million tokens, input / output, at LIST or standard rates.
# Verified against platform.claude.com/docs/en/about-claude/pricing
# and api-docs.deepseek.com/quick_start/pricing.
#
# PROMOTIONAL PRICING IS DELIBERATELY IGNORED. Sonnet 5 has introductory $2/$10
# pricing that runs out; the standard $3/$15 is used instead. Billing a rung at
# a rate that expires would mean a run reproduced later did not match an earlier
# one, and an experiment whose cost axis expires is not reproducible. DeepSeek's
# page likewise says prices may be adjusted, so the figures below are pinned
# here rather than read from a provider at run time.
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
        "assumed_tokens_out": 80,
    },
    "claude-sonnet-5": {
        "provider": "anthropic",
        "price_in": 3.00, "price_out": 15.00,
        # temperature, top_p and top_k all return a 400 at non-default values.
        "accepts_temperature": False,
        # Adaptive thinking is ON by default on Sonnet 5, unlike Sonnet 4.6.
        "thinking_on_by_default": True,
        "tokenizer_factor": 1.30,
        "assumed_tokens_out": 100,
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
        "assumed_tokens_out": 120,
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
        "assumed_tokens_out": 90,
    },
    "deepseek-v4-pro": {
        "provider": "deepseek",
        "price_in": 0.435, "price_out": 0.87,
        "accepts_temperature": True,
        "thinking_on_by_default": True,
        "tokenizer_factor": 1.00,
        "assumed_tokens_out": 110,
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
# earlier (platform.claude.com/docs/en/about-claude/pricing).
# Sonnet 5 and Opus 5 are on the new tokenizer; Haiku 4.5 is on the old one. So on
# the claude ladder the boundary runs between the bottom rung and everything above
# it.
#
# So the rungs do not merely charge different prices for the same token count -
# they disagree about how many tokens the same prompt IS. On identical text the
# upper rungs bill about 1.3x the input tokens, making the claude ladder's
# effective input ratios roughly 1x / 3.9x / 6.5x rather than 1x / 3x / 5x.
# `ladder_summary()` prints both. That makes escalation more expensive than the
# price table suggests, which works AGAINST the cascade.
#
# `tokenizer_factor` models this in MOCK MODE ONLY; real runs get correct counts
# back from the API. 1.30 is the documented approximation, so it is a modelled
# constant rather than a measurement.
#
# THE DEEPSEEK ENTRIES USE 1.00, MEANING "NOT MODELLED", NOT "THE SAME". No
# comparison against Claude's tokenizer is published or measured here, so on the
# `wide` ladder the cross-provider token counts are the least trustworthy numbers
# in the repo. A real run fixes it for free.
# ---------------------------------------------------------------------------

# Truncation is the expensive failure here: a cut-off math answer loses its
# \boxed{} and the grader scores a correct answer as wrong, which reads as a
# capability result instead of a bug.
#
# MEASURED against 118 real responses on the `wide` ladder. A 2048 cap - the
# obvious choice, since replies are short with thinking disabled - is not enough
# once the maths half is MATH500 level 5:
#
#   code            mean  55 tokens out
#   maths           mean 650 tokens out
#   at a 2048 cap   2 of 118 calls (math-422 on cheap, math-103 on expensive)
#
# Both truncations were inspected rather than assumed. Neither was a degenerate
# loop: both were coherent level-5 derivations that simply ran long, and the Opus
# one was a few hundred tokens short of its \boxed{}. So the cap binds on real
# work rather than catching pathology.
#
# 1.7% sounds ignorable and is not, because the truncations are not randomly
# distributed. They land on the HARDEST tasks, which are exactly the ones that
# decide the routable fraction, and they bias it directionally: a truncated cheap
# answer reads as "the cheap rung failed" and inflates `routable`.
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


class ReplayMiss(KeyError):
    """Replay was asked for a response that was never recorded.

    Its own type so a caller can tell "this policy has no cached data" apart
    from a genuine KeyError raised by policy code, which must still crash. It
    subclasses KeyError so any existing `except KeyError` keeps working.

    A miss is not always an error. A cache populated by one run legitimately
    lacks the calls a DIFFERENT policy would have made - the two-arm probe
    recorded always_cheap and always_expensive and nothing else, so
    llm_router's classification call has never existed in it. Distinguishing
    that from a bug is the whole reason this type exists; see run_eval.run.
    """


@dataclass
class ModelResponse:
    text: str
    tier: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    cost_usd: float
    # Was this text FABRICATED, or did a model produce it?
    #
    # Carried per response rather than derived from MODE, because in replay the
    # two are not the same thing: the run mode says where responses were looked
    # up, and this says what was found. A replay that reads mostly real
    # responses and two mock ones is not a real result, and MODE cannot see the
    # difference. Whoever consumes a run aggregates this rather than re-deriving
    # it - see run_eval.run.
    simulated: bool = False
    # Did this response hit max_tokens mid-answer?
    #
    # Carried for the same reason as `simulated`, and found the same way: the
    # detection lived in _real_call only, so REPLAY - the mode everyone else
    # reproduces the numbers in - could not see it, and three truncated
    # responses in the committed cache were being graded as capability failures.
    # A truncated answer is not a wrong answer, it is a missing measurement, and
    # the two must not be indistinguishable downstream. See docs/LIMITATIONS.md.
    truncated: bool = False


def _price(tier: str, tokens_in: int, tokens_out: int) -> float:
    m = MODELS[tier]
    return (tokens_in / 1e6) * m["price_in"] + (tokens_out / 1e6) * m["price_out"]


# ---------------------------------------------------------------------------
# Mock implementation
# ---------------------------------------------------------------------------

# Synthetic p_correct, derived from LADDER POSITION and nothing else.
#
# There is deliberately no per-model skill table. All three ladders have 5,075
# real responses, so a stipulated number claiming a named model is 0.86-good
# would be an invented capability estimate sitting next to measured ones. Mock
# proves the PLUMBING works before money is spent; it does not stand in for a
# result.
#
# These are tuned to run_eval's own pilot gate, not to any model: the bottom rung
# lands near 35% failure at mean difficulty, inside the gate's 20-55% band, so a
# cascade has something to escalate and the plumbing run is not drowned in a
# warning about the mock.
_MOCK_P_BOTTOM, _MOCK_P_TOP, _MOCK_SPREAD = 0.80, 0.95, 0.30


def _mock_skill():
    """base/spread per tier, evenly spaced across the loaded ladder."""
    n = len(TIERS)
    step = (_MOCK_P_TOP - _MOCK_P_BOTTOM) / (n - 1) if n > 1 else 0.0
    return {t: {"base": _MOCK_P_BOTTOM + i * step, "spread": _MOCK_SPREAD}
            for i, t in enumerate(TIERS)}


MOCK_SKILL = _mock_skill()

# Mock response shape. Flat, because nothing downstream of a mock run is a
# measurement: `cost_usd` on a mock row is modelled from these and is labelled
# MODELLED rather than attributed, and latency is modelled in every mode
# (docs/LIMITATIONS.md).
MOCK_TOKENS_OUT = {tier: 100 for tier in TIERS}
MOCK_LATENCY_S = {tier: 0.5 for tier in TIERS}

# NOT A MOCK PARAMETER, despite sitting beside them. `assumed_tokens_out` is the
# reply length `cascade_routing` assumes when pricing a call it has not made yet:
# `policies._est_cost` and `policies._default_lambda` read it in EVERY mode, real
# included, so it decides which tasks that policy escalates in a paid run.
# Changing a value here moves runs/results.*.jsonl and runs/frontier.*.jsonl,
# which is why it stays per model and at its measured-run values. Naming it
# MOCK_TOKENS_OUT, as it once was, invites exactly that edit.
ASSUMED_TOKENS_OUT = {tier: MODELS[tier]["assumed_tokens_out"] for tier in TIERS}

# How much of a task's outcome is shared across rungs rather than tier-specific
# luck. 1.0 = perfectly nested failure, 0.0 = independent. 0.5 is a round number
# chosen so the mock cross-tab has all four cells populated; it is not calibrated
# against the measured correlation, which is in runs/routable.<ladder>.txt.
MOCK_FAILURE_CORRELATION = 0.5

# The mock LLM-as-router (policies.policy_llm_router) is A COIN FLIP, and 0.5
# says so rather than inventing a skill level: simulating "the cheap model judges
# this question's difficulty" means deciding in advance how good that judgement
# is, so any value above 0.5 makes the policy's mock accuracy a restatement of
# this constant. What mock CAN measure here is the cost and latency of the extra
# round trip. Its accuracy is measured on real responses, in docs/RESULTS.md.
MOCK_ROUTER_SKILL = 0.5

# What the mock router calls "hard": the upper half of the within-domain
# difficulty percentile, the same latent quantity MOCK_SKILL fails on.
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

    # Keyed on sample_idx, NOT a fresh nonce. Self-consistency samples must
    # differ from each other but REPRODUCIBLY - sample 3 is always sample 3 - and
    # an unseeded RNG here makes every mock run different, which on a two-point
    # headline means the headline is noise.
    #
    # It also makes the mock invariant to CALL ORDER, so `--limit 10` reproduces
    # the first ten tasks of the full run and one policy alone gives the same
    # answers as all of them.
    #
    # random_matched and routellm are exempt by design: both CALIBRATE on the task
    # set being run, so a 10-task run sets their rate and threshold from 10 tasks.
    # The responses are identical; the policy's choice of which to ask is what
    # moved. See policies.calibrate_random_rates.
    # Keyed on the MODEL ID, not the tier name: two ladders both have a `cheap`
    # rung, and keying on the name would make DeepSeek's flash draw the same luck
    # as Claude's Haiku while scored against a different p_correct.
    # Correlate the rungs by SHARING a draw with probability c, not by blending
    # two draws. Blending averages two uniforms, which concentrates the result
    # near 0.5 and makes the realised failure rate differ from p_correct - at
    # c=0.5 a stated 0.80 realises as about 0.92. A mixture keeps the marginal
    # uniform, so MOCK_SKILL means what it says.
    shared = _draw(task["id"], "shared", temperature, sample_idx).random()
    rng = _draw(task["id"], MODELS[tier]["id"], temperature, sample_idx)
    pick = _draw(task["id"], "corr", MODELS[tier]["id"], temperature, sample_idx)
    draw = shared if pick.random() < MOCK_FAILURE_CORRELATION else rng.random()
    correct = draw < p_correct

    # SERVING ONLY - a live query, which has no ground truth to perturb.
    #
    # The mock simulates a model by corrupting the known answer with probability
    # 1 - p_correct. A user's query has no known answer, so it CANNOT simulate
    # correctness here; saying so beats a KeyError three frames down.
    #
    # It can still simulate SELF-AGREEMENT - a capable rung converges across
    # draws, a weak rung scatters - so self-consistency verification is exercised
    # for real on obviously-fake content, marked so it cannot be read as a
    # model's opinion.
    if task.get("_live"):
        token = "A" if correct else rng.choice(["B", "C", "D"])
        note = "[MOCK - simulated response, ROUTER_MODE=mock. Not a real answer.]"
        if task["domain"] == "math":
            text = f"{note} Reasoning... the answer is $\\boxed{{{token}}}$"
        elif task["domain"] == "code":
            text = f"# {note}\n```python\ndef solution():\n    return {token!r}\n```"
        else:
            text = f"{note} The answer is {token}."
        tokens_in = _mock_tokens_in(tier, prompt)
        tokens_out = MOCK_TOKENS_OUT[tier]
        return ModelResponse(
            text=text, tier=tier, tokens_in=tokens_in, tokens_out=tokens_out,
            latency_s=MOCK_LATENCY_S[tier],
            cost_usd=_price(tier, tokens_in, tokens_out),
        )

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


def _note_truncation(task, tier, kind, sample_idx, *, from_cache):
    """Record a response that hit the cap, once per distinct response.

    Warns on first sight only. Replay serves the same truncated response to
    every policy that asks for it, and a warning per policy would suggest the
    problem is nine times bigger than it is.
    """
    # Counted on EVERY serve, so a caller can detect "did this row consume a
    # truncated response" by watching the counter across one policy call. The
    # set below is deduped and is what gets reported; the two answer different
    # questions and conflating them would mark only the first policy to ask.
    call_stats["truncated"] += 1
    ident = (task["id"], tier, kind, sample_idx)
    if ident in truncated_ids:
        return
    truncated_ids.add(ident)
    cap = "ROUTER_MAX_TOKENS" if kind == "route" else "MAX_TOKENS"
    consequence = (
        "the routing decision falls back to EASY"
        if kind == "route"
        else "this grades as WRONG and is not a capability result"
    )
    where = "in the cache" if from_cache else "at max_tokens"
    print(
        f"  !! TRUNCATED {where}: {task['id']} on {tier} ({kind}, sample "
        f"{sample_idx}). {consequence}.\n"
        f"     Raising models.{cap} re-charges every cached response - see "
        f"docs/ARCHITECTURE.md (standing invariants). Exclude the task instead.",
        file=sys.stderr,
    )


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
            env_file = paths.ENV_FILE
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
                '  pip install -e ".[real]"\n\n'
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
    truncated = msg.stop_reason == "max_tokens"
    if truncated:
        _note_truncation(task, tier, kind, sample_idx, from_cache=False)

    text = "".join(b.text for b in msg.content if b.type == "text")
    return ModelResponse(
        text=text,
        truncated=truncated,
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
    # The LLM-as-router prompt, austere on purpose: QUESTION ONLY (so it sees no
    # more than any other pre-call router), no reasoning (a router that thinks is
    # a slow expensive model), one word out (so ROUTER_MAX_TOKENS can be 8).
    "route": (
        "You are a difficulty classifier for a model router. Answer with "
        "exactly one word, EASY or HARD, and nothing else.\n\n"
        "HARD means a small fast model would probably get this wrong and it "
        "should be sent to a larger model.\n\nProblem:\n{q}"
    ),
    # SERVING ONLY - never reached by the evaluation. No row in taskset.jsonl has
    # domain "general", so this template cannot move a number in the experiment.
    # It exists because router_agent serves arbitrary queries, and sending those
    # through the math template would demand a \boxed{} for "summarise this
    # email".
    "general": "{q}",
    # Code with no caller-supplied tests. Same reasoning: serving only. The
    # evaluation always has asserts, because the asserts ARE the MBPP
    # specification, so this branch is unreachable from build_taskset output.
    "code_untested": (
        "Write Python for this task. Return ONLY a python code block, no "
        "explanation.\n\n{q}"
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
        tests = "\n".join(task.get("grader_payload", {}).get("tests", []))
        if not tests:
            # Serving only - see PROMPTS["code_untested"]. Every task from
            # build_taskset.py carries asserts, so this is unreachable from the
            # evaluation.
            return PROMPTS["code_untested"].format(q=task["prompt"])
        return PROMPTS["code"].format(q=task["prompt"], tests=tests)
    if task["domain"] == "general":
        return PROMPTS["general"].format(q=task["prompt"])
    return PROMPTS["math"].format(q=task["prompt"])


def _max_tokens_for(kind: str) -> int:
    return ROUTER_MAX_TOKENS if kind == "route" else MAX_TOKENS


def is_truncated(record: dict) -> bool:
    r"""Was this stored response cut off at the cap?

    ONE definition of the rule, because more than one analysis grades straight
    off the cache file and each would otherwise have to re-derive it. A response
    that hit the cap has no `\boxed{}` - it stopped mid-derivation - so every
    caller that grades a record has to be able to tell "the model was wrong"
    apart from "the model was interrupted", and they are not the same fact.

    The `truncated` field is the answer when it is there. It is NOT there on
    records written before the field existed, which includes all three
    truncations currently on disk, so the fallback is the load-bearing branch
    rather than a courtesy.

    The fallback is EXACT rather than heuristic: `max_tokens` is part of the
    cache key, so any record a lookup can reach was recorded under the cap now
    in force, and `tokens_out == cap` can only mean the cap bound.
    """
    flag = record.get("truncated")
    if flag is not None:
        return bool(flag)
    return record.get("tokens_out", 0) >= _max_tokens_for(record.get("kind", "answer"))


def is_reachable(record: dict, task: dict) -> bool:
    """Would the response cache actually serve this stored record today?

    For analyses that read `cache/raw_calls.<ladder>.jsonl` directly instead of
    going through `response_cache.get`. Those bypass the key, so they see rows
    the experiment cannot: **orphans recorded under a parameter that has since
    changed.**

    `max_tokens` is the one that bites, and it is invisible in the record - it
    is in the key hash and nowhere else. When MAX_TOKENS went 2048 -> 4096 every
    2048-capped response was stranded: still on disk, still `mode: real`, still
    gradeable, and permanently unreachable. `math-96` has two cheap greedy draws
    on disk for that reason, and they disagree - the 2048 one finished with a
    \\boxed{} at 1614 tokens, the 4096 one ran long and was cut off.

    Rather than guess at a token count, this recomputes the key the cache would
    build today and compares. Exact by construction, and it stays correct if a
    different key field changes next.
    """
    kind = record.get("kind", "answer")
    try:
        key = response_cache.make_key(
            mode=record["mode"],
            model=record["model"],
            prompt=build_prompt(task, kind),
            temperature=record["temperature"],
            sample_idx=record["sample_idx"],
            max_tokens=_max_tokens_for(kind),
            mock_seed=record.get("mock_seed"),
        )
    except (KeyError, TypeError):
        # A record too old to carry the fields the key needs cannot be shown to
        # be reachable, so it is not treated as such.
        return False
    return key == record.get("key")


# Requested / served-from-cache / reached-a-backend. The last costs money, the
# first builds the cost table; response_cache's docstring says why they differ.
#
# served_real and served_mock split served responses by what produced the text.
# The split only carries information in replay - which is the mode that needs it.
call_stats = {"requested": 0, "from_cache": 0, "backend": 0,
              "served_real": 0, "served_mock": 0, "truncated": 0}

# Which responses hit the cap, as (task_id, tier, kind, sample_idx). A set rather
# than a count because replay serves the same truncated response to every policy
# that asks for it, so the count inflates with the number of policies while the
# number of damaged MEASUREMENTS does not. The set is what to report.
truncated_ids = set()

# Dollars that actually left the account: summed over calls that reached a
# backend, and only those. NOT what any policy is charged - a cache hit returns
# its full cost_usd and the policy pays it, because that is production cost.
#
# A spend cap must read THIS one. Capping on attributed cost would abort a free
# replay ($0.00 spent, several dollars attributed) and under-count a real run.
backend_spend_usd = 0.0

# Hard per-process spend cap, REAL MODE ONLY.
#
# It lives next to the one line that adds to `backend_spend_usd`, not in each
# entry point that can spend: a cap wrapping run_eval's policy loop misses the
# llm_router pre-pass and estimator fitting, and a cap the spender can walk
# around is not a cap.
#
# Mock and replay are exempt by construction - their cost_usd is modelled and no
# card is charged - so `real` is the only mode where this counter is money.
#
#     ROUTER_MAX_SPEND_USD=8 ROUTER_MODE=real python -m llm_routing.run_eval
MAX_SPEND_USD = float(os.environ.get("ROUTER_MAX_SPEND_USD", "5.0"))


class SpendCapExceeded(RuntimeError):
    """Raised by call() when real spend crosses MAX_SPEND_USD.

    An EXCEPTION rather than a stop-and-return, and that is the point of it. A
    cap that printed "stopping early" and returned the rows it had would have
    main() write those to results.jsonl and report on them - and a half-measured
    task set is indistinguishable from a complete one once it is a file. Every
    policy not yet reached simply scores lower, and nothing downstream can tell
    that from a real result.

    Nothing paid for is lost when it fires. response_cache.put writes each
    response as it arrives, so an aborted run is fully replayable and a re-run
    with a higher cap pays only for what it did not reach.
    """


def reset_call_stats():
    global backend_spend_usd
    for k in call_stats:
        call_stats[k] = 0
    truncated_ids.clear()
    backend_spend_usd = 0.0


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

    # Real and mock entries never collide, because `mode` is in the hash, so
    # asking for both is safe - but replay asks for the mock one only when the
    # fallback is explicitly enabled. See REPLAY_FALLBACK_TO_MOCK.
    if MODE == "replay":
        candidates = [keyfor("real")]
        if REPLAY_FALLBACK_TO_MOCK:
            candidates.append(keyfor("mock"))
    else:
        candidates = [keyfor(MODE)]
    key = candidates[0]

    call_stats["requested"] += 1
    for k in candidates:
        rec = response_cache.get(k)
        if rec is not None:
            call_stats["from_cache"] += 1
            # The record's own `mode`, written when it was stored, rather than
            # which candidate key matched. Authoritative for entries already on
            # disk, and it stays correct if the lookup order ever changes.
            simulated = rec.get("mode") == "mock"
            call_stats["served_mock" if simulated else "served_real"] += 1
            truncated = is_truncated(rec)
            if truncated:
                _note_truncation(task, tier, kind, sample_idx, from_cache=True)
            return ModelResponse(
                text=rec["text"], tier=rec["tier"], tokens_in=rec["tokens_in"],
                tokens_out=rec["tokens_out"], latency_s=rec["latency_s"],
                cost_usd=rec["cost_usd"], simulated=simulated,
                truncated=bool(truncated),
            )

    if MODE == "replay":
        raise ReplayMiss(
            f"replay mode: no cached response for {task['id']} tier={tier} "
            f"temp={temperature} sample={sample_idx} kind={kind}\n"
            f"  key={key}\n"
            f"  Replay never calls a backend. Either the cache is incomplete, or "
            f"a prompt or parameter changed since it was populated.\n"
            f"  Repopulate with: ROUTER_MODE=real python -m llm_routing.run_eval\n"
            f"  Or serve the gap from the mock cache with "
            f"ROUTER_REPLAY_FALLBACK=1 - which is not a result, and every row "
            f"it touches is stamped simulated: true."
        )

    global backend_spend_usd
    # Checked here, immediately before the only line in the program that can
    # charge a card, so no caller can reach a backend without passing it.
    if MODE == "real" and backend_spend_usd > MAX_SPEND_USD:
        raise SpendCapExceeded(
            f"spend cap hit: ${backend_spend_usd:.4f} of ${MAX_SPEND_USD:.2f} "
            f"reached a backend.\n"
            f"  Stopped before calling {MODELS[tier]['id']} for {task['id']} "
            f"({kind}, sample {sample_idx}).\n"
            f"  Every response paid for is already in "
            f"{response_cache.PATH.name} - nothing is lost, and a re-run pays "
            f"only for what it did not reach.\n"
            f"  Raise the cap for one run once you know why the estimate was "
            f"low:\n"
            f"      ROUTER_MAX_SPEND_USD={max(1, int(backend_spend_usd * 2))} "
            f"ROUTER_MODE=real python3 ...\n"
            f"  If you do not know why, do not raise it - something is calling "
            f"more than you think."
        )
    call_stats["backend"] += 1
    backend = _mock_call if MODE == "mock" else _real_call
    r = backend(tier, prompt, task, temperature, sample_idx, kind)
    # One place rather than inside each backend, so a new backend cannot forget.
    r.simulated = MODE == "mock"
    call_stats["served_mock" if r.simulated else "served_real"] += 1
    # The only line in the codebase that adds to real spend. Here rather than in
    # each backend for the same reason as the line above, and after the call
    # rather than before it because a call that raised was not billed.
    backend_spend_usd += r.cost_usd

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
