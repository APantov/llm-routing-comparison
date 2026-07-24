"""
The five policies.

Each takes a task and returns a PolicyResult. Everything runs with placeholder
values so you can SEE the pipeline work immediately, but the three values marked
DECISION are yours to set and defend in an interview. Change them, re-run, watch
the numbers move. That is how you learn what they actually do.
"""

from dataclasses import dataclass, field
from collections import Counter

import models
from graders import grade, extract_answer

# ---------------------------------------------------------------------------
# DECISION #2: self-consistency sample count for the math verifier.
# Higher k = better failure detection, linearly more cost. k=5 is a common
# starting point. You must be able to say why you chose yours.
# ---------------------------------------------------------------------------
SELF_CONSISTENCY_K = 5

# ---------------------------------------------------------------------------
# DECISION #3: the escalation threshold.
# Fraction of self-consistency samples that must agree for the cheap answer to
# be ACCEPTED. Below this, escalate. 1.0 = only unanimous answers accepted.
# Expect the interview question "what happens just below your threshold?"
# ---------------------------------------------------------------------------
AGREEMENT_THRESHOLD = 0.8


@dataclass
class PolicyResult:
    task_id: str
    policy: str
    correct: bool
    cost_usd: float
    latency_s: float
    escalated: bool = False
    calls: list = field(default_factory=list)


@dataclass
class Verdict:
    """What a verifier hands back.

    `answer_text` is the response the cascade should ACTUALLY GRADE if it
    accepts. It is not always the response that was passed in: a verifier that
    paid for extra samples has a better answer available than the one it was
    handed, and returning it is free. See verify_math.
    """
    accepted: bool
    answer_text: str
    cost_usd: float
    latency_s: float


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------

def verify_code(task, response_text):
    """FREE AND PERFECT. Run the shipped asserts.

    This is the ideal case and almost no real problem looks like it. Nothing
    extra is sampled, so the answer to grade is the one that came in.
    """
    passed = grade(task, response_text)
    return Verdict(accepted=passed, answer_text=response_text, cost_usd=0.0, latency_s=0.0)


def verify_math(task, response_text):
    """PROXY. No ground truth at runtime, so sample the cheap model k times at
    temperature > 0 and check whether the answers agree.

    The logic: a model with a stable internal answer returns it repeatedly; a
    model that is guessing scatters. Agreement is a proxy for confidence, and
    it is a NOISY one. Measuring how much worse this is than verify_code is
    the most interesting result in the project.

    Returns the PLURALITY answer, not the greedy one it was handed. Two
    distinct things come out of the same k samples and it would be wasteful to
    take only one:

      - the ESCALATION signal, how much the samples agree
      - a better ANSWER, the majority vote, which is what self-consistency was
        published for in the first place

    This function used to compute the first and throw away the second: it paid
    5x the cheap-tier price for the samples and then let the cascade grade
    sample 1. Note the consequence for interpretation - the accepted math
    answer is now a self-consistency answer, so `cascade` bundles two
    mechanisms, majority voting for accuracy and agreement for escalation.
    That is what AutoMix-style systems do, and it has to be said out loud
    rather than left implicit, because it means the math cascade is no longer
    comparable to a bare single cheap call.

    Ties go to the greedy sample: it is inserted first and most_common breaks
    ties on insertion order. Arbitrary, but deterministic and defensible.
    """
    answers = [extract_answer(response_text)]
    cost = 0.0
    latency = 0.0
    # sample_idx starts at 1 because index 0 is the greedy call the cascade
    # already made. Distinct indices are what make the samples differ in mock
    # mode; passing the same one k times would produce k identical answers and
    # unanimous agreement on every task.
    for i in range(1, SELF_CONSISTENCY_K):
        r = models.call("cheap", task, temperature=0.8, sample_idx=i)
        answers.append(extract_answer(r.text))
        cost += r.cost_usd
        latency += r.latency_s

    counts = Counter(a for a in answers if a is not None)
    if not counts:
        # Every sample was unparseable. Escalate, and there is no better answer
        # to offer than the one we came in with.
        return Verdict(False, response_text, cost, latency)

    top, top_count = counts.most_common(1)[0]
    # Denominator is len(answers), NOT the number of parseable ones, so
    # unparseable samples count against agreement rather than being ignored.
    # Deliberate: a model that cannot produce a readable answer is not one to
    # be confident in. Worth knowing that it conflates "the model disagreed
    # with itself" with "the parser failed".
    agreement = top_count / len(answers)

    # Rewrap as \boxed{} so grade() takes its primary parse path on a value
    # that is already normalised, rather than the last-line fallback.
    return Verdict(
        accepted=agreement >= AGREEMENT_THRESHOLD,
        answer_text=f"\\boxed{{{top}}}",
        cost_usd=cost,
        latency_s=latency,
    )


VERIFIERS = {"code": verify_code, "math": verify_math}


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def policy_always(task, tier):
    r = models.call(tier, task)
    return PolicyResult(
        task_id=task["id"],
        policy=f"always_{tier}",
        correct=grade(task, r.text),
        cost_usd=r.cost_usd,
        latency_s=r.latency_s,
        calls=[tier],
    )


def policy_cascade(task):
    """Cheap -> verify -> escalate on failure.

    Never misroutes an easy query. Double-pays every time it escalates.
    """
    cheap = models.call("cheap", task)
    v = VERIFIERS[task["domain"]](task, cheap.text)

    cost = cheap.cost_usd + v.cost_usd
    latency = cheap.latency_s + v.latency_s

    if v.accepted:
        return PolicyResult(
            task_id=task["id"], policy="cascade",
            # v.answer_text, not cheap.text. For code they are the same thing.
            # For math they are not: the verifier bought k samples and the
            # majority vote among them is the better answer.
            correct=grade(task, v.answer_text),
            cost_usd=cost, latency_s=latency,
            escalated=False, calls=["cheap"],
        )

    exp = models.call("expensive", task)
    return PolicyResult(
        task_id=task["id"], policy="cascade",
        correct=grade(task, exp.text),
        cost_usd=cost + exp.cost_usd,
        latency_s=latency + exp.latency_s,
        escalated=True, calls=["cheap", "expensive"],
    )


# ---------------------------------------------------------------------------
# DECISION #4: the predictive heuristic.
# Route ONCE, up front, using only features available before you call anything.
# Deliberately NOT a trained model and NOT an LLM call - an LLM call would add
# a full round trip and defeat the purpose. Keep it simple and defensible.
#
# The rule is DOMAIN-AWARE because the available signal is wildly asymmetric,
# and that asymmetry is a result rather than an inconvenience:
#
#   math - MATH500 ships a human-assigned `level` (1-5) alongside the question.
#          Legitimate under the leak test (shipped with the question, not
#          derived from the answer) and strongly predictive. But FLATTERING:
#          production traffic does not arrive labelled with its difficulty, so
#          treat the math half as an optimistic upper bound.
#
#   code - nothing in an MBPP prompt predicts difficulty. Measured against the
#          reference-solution line count on this taskset: prompt chars r=+0.28,
#          words +0.15, keyword hits +0.13, assert chars +0.01, n_asserts
#          -0.02. The hard part of a coding task lives in the solution, not in
#          the question. Prompt length is used because it is the best of a bad
#          set, and it is reported as near-random, not dressed up.
#
# That is the second asymmetry in the project: predictive routing needs a
# difficulty signal up front, cascade does not because it looks at the answer.
# ---------------------------------------------------------------------------

# MATH500 level at or above which we pay for the expensive model.
# 5 selects 24/60 tasks; 4 would select 40/60.
PREDICTIVE_HARD_LEVEL = 5

# Prompt-length cutoff for code, in characters. Calibration, not a discovered
# signal: 100 flags 14/40 (35%), which puts the code half at roughly the same
# escalation rate as the math half (40%) so the two are cost-comparable.
PREDICTIVE_CODE_CHARS = 100


def predict_is_hard(task) -> bool:
    # Reads ONLY from predict_features, which build_taskset.py populates with
    # question-derived values. Not from difficulty_proxy, which for code is the
    # reference solution's line count - information you do not have before
    # answering. Keeping the allowed inputs in their own field is what stops
    # that leak from creeping back in.
    f = task["predict_features"]
    if task["domain"] == "math":
        return f["level"] >= PREDICTIVE_HARD_LEVEL
    return f["prompt_chars"] >= PREDICTIVE_CODE_CHARS


def policy_predictive(task):
    tier = "expensive" if predict_is_hard(task) else "cheap"
    r = models.call(tier, task)
    return PolicyResult(
        task_id=task["id"], policy="predictive",
        correct=grade(task, r.text),
        cost_usd=r.cost_usd, latency_s=r.latency_s,
        calls=[tier],
    )


def policy_oracle(task):
    """Hindsight-optimal: try both, keep the cheap one if it worked.

    Not deployable. It exists to bound how good ANY router could be. Without it
    you cannot tell whether a small gap means your routers are bad or means
    routing cannot help much on this task set.

    Cost is charged as the cheapest CORRECT option, which is what a perfect
    router would actually have paid.
    """
    cheap = models.call("cheap", task)
    if grade(task, cheap.text):
        return PolicyResult(
            task_id=task["id"], policy="oracle", correct=True,
            cost_usd=cheap.cost_usd, latency_s=cheap.latency_s, calls=["cheap"],
        )
    exp = models.call("expensive", task)
    return PolicyResult(
        task_id=task["id"], policy="oracle",
        correct=grade(task, exp.text),
        cost_usd=exp.cost_usd, latency_s=exp.latency_s, calls=["expensive"],
    )


POLICIES = {
    "always_cheap": lambda t: policy_always(t, "cheap"),
    "always_expensive": lambda t: policy_always(t, "expensive"),
    "predictive": policy_predictive,
    "cascade": policy_cascade,
    "oracle": policy_oracle,
}
