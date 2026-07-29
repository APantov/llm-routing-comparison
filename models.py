"""
Model client. Three modes:

  MOCK   - no API key, no spend. Simulates a cheap and an expensive model with
           realistic, correlated success rates. Lets you run the ENTIRE pipeline
           end to end before spending a cent. Use this until everything works.

  REAL   - actual API calls. Flip MODE to "real" and set your key.

  REPLAY - serve every call from cache/raw_calls.jsonl and refuse to touch the
           network. A miss is an error, not a fetch. This is what makes the repo
           reproducible by someone with no API key, and it is what makes every
           sweep after the first paid run cost nothing.

EVERY call goes through response_cache first, in all three modes. Read the
module docstring there before changing anything here: the cache is not a speed
optimisation, it is what makes the paired statistics valid. Without it,
always_cheap and cascade would be compared on different draws from the same
model, and policy_oracle would bound a set of responses nobody else received.

Every call returns a ModelResponse carrying tokens, latency and cost, because
cost accounting is half the point of the project and retrofitting it is painful.
"""

import hashlib
import os
import random
import sys
import time
from dataclasses import dataclass, asdict

import response_cache

MODE = os.environ.get("ROUTER_MODE", "mock")  # "mock" | "real" | "replay"
if MODE not in ("mock", "real", "replay"):
    raise SystemExit(f"ROUTER_MODE must be mock, real or replay - got {MODE!r}")
response_cache.configure(MODE)

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

# The LLM-as-router call (policies.policy_llm_router) is a one-word
# classification, so it gets its own tiny cap. This is the number that decides
# whether "an LLM call would defeat the purpose" is true: at 8 output tokens the
# router call cannot cost more than a rounding error, and the point of the
# policy is to measure that rather than assert it.
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

# ---------------------------------------------------------------------------
# Mock skill of the LLM-as-router classifier (policies.policy_llm_router).
#
# READ THIS BEFORE QUOTING ANY llm_router ACCURACY NUMBER FROM MOCK MODE.
#
# There is no way to simulate "Haiku reads the question and judges difficulty"
# without deciding in advance how good that judgement is. So the mock router is
# an oracle on the mock's own latent difficulty, corrupted at rate
# 1 - MOCK_ROUTER_SKILL. Its accuracy in mock mode is therefore a RESTATEMENT OF
# THIS CONSTANT, not a measurement of anything. 0.70 is a placeholder chosen to
# sit between the predictive heuristic and the oracle; it is not evidence.
#
# What IS measurable in mock mode, and what the policy exists to measure, is the
# COST AND LATENCY of the extra round trip - because those come from the price
# table and the token counts, not from this constant. policies.py Decision #4
# rejected LLM routing on the grounds that it "would add a full round trip and
# defeat the purpose". That is a quantitative claim and this is what tests it.
# ---------------------------------------------------------------------------
MOCK_ROUTER_SKILL = 0.70

# The mock router's notion of "hard": the upper half of the within-domain
# difficulty percentile. Matches how MOCK_SKILL drives p_correct, so the router
# is judging the same latent quantity the mock models are failing on.
MOCK_ROUTER_HARD_PCT = 0.5


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


def _mock_route_call(tier: str, prompt: str, task: dict, sample_idx: int) -> ModelResponse:
    """Simulate the LLM-as-router classification call.

    See MOCK_ROUTER_SKILL. The label is fabricated from a constant; the tokens,
    latency and price are not, and they are the part worth reading.
    """
    difficulty = task.get("difficulty_pct", 0.5)
    truth_hard = difficulty >= MOCK_ROUTER_HARD_PCT
    agrees = _draw(task["id"], tier, "router", sample_idx).random() < MOCK_ROUTER_SKILL
    said_hard = truth_hard if agrees else not truth_hard

    tokens_in = max(20, len(prompt) // 4)
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

    Success is driven by task difficulty so that cheap and expensive models
    FAIL ON THE SAME HARD TASKS (correlated failure). That correlation matters:
    if failures were independent, escalation would look far better than it is.

    Correlation is produced by a SHARED latent draw per task, blended with a
    tier-specific draw. Without the shared term, escalating would fix failures
    at a rate no real cascade achieves.
    """
    if kind == "route":
        return _mock_route_call(tier, prompt, task, sample_idx)

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
    tier: str, prompt: str, task: dict, temperature: float, sample_idx: int,
    kind: str = "answer",
) -> ModelResponse:
    # sample_idx does not pin a real model - sampling at temperature > 0 is
    # stochastic and cannot be reproduced. What pins it is the response cache:
    # the FIRST draw for a given sample_idx is stored and every later reader gets
    # that one. So real mode is reproducible after the fact rather than by
    # construction, which is the strongest guarantee a hosted API allows.
    global truncated_calls
    cfg = MODELS[tier]

    # The two tiers have genuinely different API contracts, so the request is
    # built per-model rather than shared. Sending temperature to Opus 5 is a
    # 400; omitting `thinking` on Opus 5 silently turns thinking ON.
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
    msg = _client().messages.create(**kwargs)
    latency = time.time() - t0

    # A truncated reply is the dangerous failure: it grades as WRONG rather
    # than as an error, so it silently deflates the accuracy of whichever tier
    # hit the cap. Count it loudly instead of letting it pass as a result.
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

    This is what the cache is keyed on, so it must be a pure function of the
    task and the kind. Anything that varies per run (a timestamp, a nonce, a
    dict that iterates differently) would silently make every call a cache miss.
    """
    if kind == "route":
        return PROMPTS["route"].format(q=task["prompt"])
    if task["domain"] == "code":
        tests = "\n".join(task["grader_payload"]["tests"])
        return PROMPTS["code"].format(q=task["prompt"], tests=tests)
    return PROMPTS["math"].format(q=task["prompt"])


def _max_tokens_for(kind: str) -> int:
    return ROUTER_MAX_TOKENS if kind == "route" else MAX_TOKENS


# How many times each policy asked for a call, how many were served from the
# cache, and how many actually reached a backend. The last one is the number
# that costs money; the first is the number the cost table is built from. Read
# response_cache's docstring for why those are different on purpose.
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
    temperature). Callers that sample repeatedly - only the math verifier, so
    far - must pass a distinct index per sample, or every sample is the same
    cache entry and self-consistency becomes trivially unanimous.

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

    # Replay prefers real responses and falls back to mock ones, so that a mock
    # cache can be used to exercise replay itself without an API key. Real and
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
            f"  Repopulate with: ROUTER_MODE=real python run_eval.py"
        )

    call_stats["backend"] += 1
    backend = _mock_call if MODE == "mock" else _real_call
    r = backend(tier, prompt, task, temperature, sample_idx, kind)

    # Written immediately, not batched: a real run that dies halfway through
    # must keep every response it paid for.
    response_cache.put(key, {
        # Not part of the key. Stored so the file can be grepped and audited by
        # a human, and so a cache entry can be traced back to a task.
        "task_id": task["id"], "domain": task["domain"], "kind": kind,
        "mode": MODE, "mock_seed": MOCK_SEED if MODE == "mock" else None,
        "model": MODELS[tier]["id"], "temperature": temperature,
        "sample_idx": sample_idx, "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        **asdict(r),
    })
    return r
