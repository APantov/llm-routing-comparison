"""
Model client. Two modes:

  MOCK  - no API key, no spend. Simulates a cheap and an expensive model with
          realistic, correlated success rates. Lets you run the ENTIRE pipeline
          end to end before spending a cent. Use this until everything works.

  REAL  - actual API calls. Flip MODE to "real" and set your key.

Every call returns a ModelResponse carrying tokens, latency and cost, because
cost accounting is half the point of the project and retrofitting it is painful.
"""

import hashlib
import os
import random
import sys
import time
from dataclasses import dataclass

MODE = os.environ.get("ROUTER_MODE", "mock")  # "mock" or "real"

# Seed for the whole mock world. Every mock outcome is a pure hash of
# (seed, task, tier, temperature, sample_idx), so a run is reproducible byte for
# byte and a SWEEP over seeds gives you the run-to-run variance of the mock -
# which is the only honest way to read a 2-point accuracy difference at n=100.
MOCK_SEED = int(os.environ.get("MOCK_SEED", "0"))

# ---------------------------------------------------------------------------
# DECISION #1 (yours): the model pair. Prices per million tokens, in / out,
# checked against the pricing page 2026-07-29. The RATIO drives the whole
# economics: 5x here. A larger ratio flatters the cascade, so any result from
# this file has to be reported as ratio-dependent, not absolute.
#
# The CHEAP tier is not a free choice. verify_math samples it at temperature
# 0.8 - that sampling IS self-consistency, and without it there is nothing to
# disagree. Among current models only Haiku 4.5 still accepts temperature;
# it was removed on Opus 4.7+ and is rejected as non-default on Sonnet 5.
# Picking a 5-series cheap model would silently break the math verifier.
#
# Sonnet 5 was the alternative expensive tier at a 3x ratio. Rejected for the
# smaller ratio, and because its introductory $2/$10 pricing runs to
# 2026-08-31, which would make the billed ratio 2x and the run harder to
# reproduce later at list price.
# ---------------------------------------------------------------------------
MODELS = {
    "cheap": {
        # Dated ID rather than the `claude-haiku-4-5` alias: an alias can be
        # repointed, and this run needs to be reproducible.
        "id": "claude-haiku-4-5-20251001",
        "price_in": 1.00,
        "price_out": 5.00,
        "accepts_temperature": True,
        # Pre-4.6 model: thinking is off unless explicitly enabled.
        "thinking_on_by_default": False,
    },
    "expensive": {
        # Opus 5 IDs carry no date suffix - never append one.
        "id": "claude-opus-5",
        "price_in": 5.00,
        "price_out": 25.00,
        # Sampling params were removed on Opus 4.7+; sending temperature at
        # all returns a 400. This is why call() cannot pass it unconditionally.
        "accepts_temperature": False,
        # Thinking is ON by default on Opus 5 (unlike Opus 4.8/4.7, where
        # omitting the field meant off). We disable it deliberately - see
        # DECISION C - so the comparison isolates capability rather than
        # reasoning-time compute. Valid only at effort `high` or below;
        # the default effort is `high`, so no effort field is needed.
        "thinking_on_by_default": True,
    },
}

# With thinking disabled, replies are short. 2048 is generous headroom rather
# than a tuned value, because truncation is the expensive failure here: a
# cut-off math answer loses its \boxed{} and the grader scores a correct
# answer as wrong, which reads as a capability result instead of a bug.
MAX_TOKENS = 2048


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

# Mock tuning. p_correct = base - spread * difficulty_pct, per tier.
# Chosen so the cheap model fails roughly a third of the time (the band where
# routing decisions actually matter) and the tiers stay clearly separated.
MOCK_SKILL = {
    "cheap": {"base": 0.86, "spread": 0.36},
    "expensive": {"base": 0.97, "spread": 0.16},
}

# How much of a task's outcome is a shared property of the task rather than
# tier-specific luck. 1.0 = perfectly nested failure (anything the expensive
# model gets wrong, the cheap model also gets wrong); 0.0 = independent.
# Real tiers from one provider are strongly but not perfectly correlated.
MOCK_FAILURE_CORRELATION = 0.75


def _wrong_answer(truth: str, rng: random.Random) -> str:
    """A plausible wrong answer, drawn from a small spread.

    Must SCATTER, not be constant. verify_math accepts the cheap answer when
    self-consistency samples agree, so a mock that always returned the same
    wrong string would make every wrong answer look unanimously confident and
    the math cascade would never escalate.

    MATH500 answers are not all integers (\\frac{14}{3}, 3\\sqrt{13}, 6-5i),
    so the integer path is a fast case, not the only one.
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
    how many calls preceded this one.
    """
    key = "|".join(str(p) for p in (MOCK_SEED, *parts))
    return random.Random(hashlib.md5(key.encode()).hexdigest())


def _mock_call(
    tier: str, prompt: str, task: dict, temperature: float, sample_idx: int
) -> ModelResponse:
    """Simulate a model.

    Success is driven by task difficulty so that cheap and expensive models
    FAIL ON THE SAME HARD TASKS (correlated failure). That correlation matters:
    if failures were independent, escalation would look far better than it is.

    Correlation is produced by a SHARED latent draw per task, blended with a
    tier-specific draw. Without the shared term, escalating would fix failures
    at a rate no real cascade achieves.
    """
    # difficulty_pct is a within-domain percentile, so it means the same thing
    # for a math task and a code task. difficulty_proxy does not.
    difficulty = task.get("difficulty_pct", 0.5)
    cfg = MOCK_SKILL[tier]
    p_correct = max(0.05, min(0.99, cfg["base"] - cfg["spread"] * difficulty))

    # sample_idx, NOT a fresh random nonce. Self-consistency samples still need
    # to differ from each other, but they must differ REPRODUCIBLY: sample 3 of
    # a given task is always the same sample 3.
    #
    # This used to be `nonce = random.random()` off the unseeded global RNG,
    # which made every mock run different. Three identical repeats of the
    # cascade over 25 math tasks scored 20 / 23 / 22 - a 12-point swing from
    # nothing but RNG state, on a project whose headline is a 2-point gap.
    #
    # Keying on sample_idx also makes the mock invariant to CALL ORDER, so
    # `--limit 10` reproduces the first 10 tasks of the full run exactly, and
    # running one policy alone gives the same answers as running all five.
    shared = _draw(task["id"], "shared", temperature, sample_idx).random()
    rng = _draw(task["id"], tier, temperature, sample_idx)
    draw = MOCK_FAILURE_CORRELATION * shared + (1 - MOCK_FAILURE_CORRELATION) * rng.random()
    correct = draw < p_correct

    if task["domain"] == "math":
        truth = task["grader_payload"]["answer"]
        answer = truth if correct else _wrong_answer(truth, rng)
        text = f"Reasoning... the answer is $\\boxed{{{answer}}}$"
    else:
        # A "correct" mock answer must be code that actually passes the shipped
        # asserts, so it has to be the reference solution stored by
        # build_taskset.py. If _ref_code is missing, every code task fails and
        # the whole code domain silently reads 0% - which is exactly the bug
        # this line used to have.
        ref = task.get("_ref_code")
        if correct and not ref:
            raise KeyError(
                f"task {task['id']} has no _ref_code; rebuild with build_taskset.py"
            )
        text = "```python\n" + (ref if correct else "def _wrong(): pass") + "\n```"

    tokens_in = max(20, len(prompt) // 4)
    tokens_out = 120 if tier == "expensive" else 80
    latency = 0.9 if tier == "expensive" else 0.4

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

_CLIENT = None
truncated_calls = 0


def _client():
    """One client for the whole run.

    Constructing an Anthropic() per call spins up a fresh HTTP client and
    throws away the connection pool; this eval makes ~700 calls.
    """
    global _CLIENT
    if _CLIENT is None:
        from anthropic import Anthropic  # pip install anthropic

        _CLIENT = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    return _CLIENT


def _real_call(
    tier: str, prompt: str, task: dict, temperature: float, sample_idx: int
) -> ModelResponse:
    # sample_idx is unused here: a real model at temperature > 0 is stochastic
    # by nature and cannot be pinned. It exists so the mock and real backends
    # share one signature, and as a reminder that real-mode results are NOT
    # reproducible the way mock results now are.
    global truncated_calls
    cfg = MODELS[tier]

    # The two tiers have genuinely different API contracts, so the request is
    # built per-model rather than shared. Sending temperature to Opus 5 is a
    # 400; omitting `thinking` on Opus 5 silently turns thinking ON.
    kwargs = {
        "model": cfg["id"],
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    if cfg["accepts_temperature"]:
        kwargs["temperature"] = temperature
    if cfg["thinking_on_by_default"]:
        kwargs["thinking"] = {"type": "disabled"}

    t0 = time.time()
    msg = _client().messages.create(**kwargs)
    latency = time.time() - t0

    # A truncated reply is the dangerous failure: it grades as WRONG rather
    # than as an error, so it silently deflates the accuracy of whichever tier
    # hit the cap. Count it loudly instead of letting it pass as a result.
    if msg.stop_reason == "max_tokens":
        truncated_calls += 1
        print(
            f"  !! TRUNCATED at max_tokens: {task['id']} on {tier}. "
            f"This will grade as wrong. Raise MAX_TOKENS.",
            file=sys.stderr,
        )

    text = "".join(b.text for b in msg.content if b.type == "text")
    return ModelResponse(
        text=text,
        tier=tier,
        tokens_in=msg.usage.input_tokens,
        tokens_out=msg.usage.output_tokens,
        latency_s=latency,
        cost_usd=_price(tier, msg.usage.input_tokens, msg.usage.output_tokens),
    )


# ---------------------------------------------------------------------------

PROMPTS = {
    # \boxed{} is the standard MATH answer protocol. Asking for "a plain
    # number" would be wrong here: most level 3-5 answers are fractions,
    # radicals or tuples.
    "math": (
        "Solve this problem. Show brief reasoning, then give the final answer "
        "in \\boxed{{}}.\n\n{q}"
    ),
    # The tests go in the prompt. Without them the model has to guess the
    # function name the asserts will call, and failures measure name-guessing
    # rather than difficulty. This is the standard MBPP protocol, and it does
    # not leak: the asserts are the specification, not the answer.
    "code": (
        "Write a Python function for this task. Return ONLY a python code "
        "block, no explanation.\n\n{q}\n\nYour code should pass these tests:\n{tests}"
    ),
}


def build_prompt(task: dict) -> str:
    if task["domain"] == "code":
        tests = "\n".join(task["grader_payload"]["tests"])
        return PROMPTS["code"].format(q=task["prompt"], tests=tests)
    return PROMPTS["math"].format(q=task["prompt"])


def call(
    tier: str, task: dict, temperature: float = 0.0, sample_idx: int = 0
) -> ModelResponse:
    """One model call.

    sample_idx identifies WHICH draw this is for a given (task, tier,
    temperature). Callers that sample repeatedly - only the math verifier, so
    far - must pass a distinct index per sample, or the mock will hand back the
    same answer every time and self-consistency will be trivially unanimous.
    """
    prompt = build_prompt(task)
    if MODE == "mock":
        return _mock_call(tier, prompt, task, temperature, sample_idx)
    return _real_call(tier, prompt, task, temperature, sample_idx)
