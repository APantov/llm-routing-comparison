"""
The policies.

Each takes a task and returns a PolicyResult. The values marked DECISION are
yours to set and defend in an interview. Change them, re-run, watch the numbers
move. That is how you learn what they actually do.

    always_cheap        one cheap call, always
    always_expensive    one expensive call, always
    predictive          hand-written heuristic picks a tier up front
    llm_router          the cheap model itself picks the tier, then answers
    random_matched      coin flip at predictive's own escalation rate
    random_50           coin flip at 50/50
    cascade             cheap -> verify -> escalate on failure
    cascade_degraded    cascade with a deliberately damaged verifier (code only)
    oracle              hindsight-optimal, not deployable

The three additions worth reading the comments on:

  cascade_degraded  is the experiment, not a variant. It moves verifier quality
                    inside a single domain, which is the only way to stop
                    verifier quality being confounded with math-vs-code.

  random_matched    is the null hypothesis. A router that picks 38 tasks at
                    random also gains accuracy - it just pays for it. Without
                    this, predictive's 89% against always_cheap's 74% does not
                    establish that predict_is_hard has any skill at all.

  llm_router        exists to test Decision #4's own claim that an LLM routing
                    call "would defeat the purpose". That is a cost claim, and
                    cost is measurable.
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

    "Perfect" means perfect WITH RESPECT TO THE SPEC, and the spec is the
    asserts. A solution that passes the shipped tests and is otherwise bad is
    accepted, correctly, by both this verifier and the grader.
    """
    passed = grade(task, response_text)
    return Verdict(accepted=passed, answer_text=response_text, cost_usd=0.0, latency_s=0.0)


# ---------------------------------------------------------------------------
# DECISION #5: the verifier corruption rate. THIS IS THE MANIPULATED VARIABLE.
#
# The project's stated contribution is that verifier quality is what varies.
# Until now it had exactly two levels of that variable and they were perfectly
# confounded with task domain:
#
#     perfect verifier <-> code <-> MBPP <-> run asserts <-> $0 verification
#     proxy   verifier <-> math <-> MATH500 <-> exact match <-> 4 extra calls
#
# Every row differs in five ways at once, so "the code cascade wins and the math
# cascade doesn't" cannot be attributed to verifier quality. It can only be
# attributed to "code is different from math", which is not a finding.
#
# cascade_degraded fixes that by corrupting verify_code on the CODE domain.
# Same tasks, same models, same grader, same prompts, same cost structure - only
# verifier fidelity moves. Sweeping p from 0 to 1 gives a curve from a perfect
# verifier to a coin flip, with everything else held fixed, which is what turns
# an observation into a measurement.
#
# p is the probability that the verifier IGNORES the test result and returns a
# coin flip instead. So the effective error rate is p/2, not p, and p=1.0 is a
# verifier with zero information (AUC 0.5) rather than an inverted one.
# ---------------------------------------------------------------------------
VERIFIER_CORRUPTION = 0.0

# Which realisation of the corruption this is.
#
# At n=40 a single draw per corruption level is NOT a curve. One task is 2.5
# points of accuracy, and the binomial standard deviation at these rates is
# around 5 points, so a six-point single-draw sweep can easily show a rise where
# the mechanism predicts a fall. Sweeping this seed and averaging is what turns
# the sweep from an anecdote into an estimate, and it costs nothing because the
# model responses are all cache hits.
VERIFIER_CORRUPTION_SEED = 0


def verify_code_degraded(task, response_text):
    """verify_code, damaged on purpose at rate VERIFIER_CORRUPTION.

    Deterministic given (task, corruption rate, mock seed): the corruption is
    drawn through models._draw like everything else stochastic in this project,
    so a sweep is reproducible and two policies asked the same question at the
    same p get the same answer. A bare random.random() here would make the whole
    sweep un-rerunnable, which is the failure this codebase already fixed once.

    Note which way the errors cost you, because they are not symmetric:

      false REJECT of a correct answer -> escalate anyway. Money wasted,
                                          accuracy unharmed.
      false ACCEPT of a wrong answer   -> ship the wrong answer. Accuracy lost,
                                          money saved.

    Since the cheap model is right on most code tasks, false rejects dominate at
    low p, so cost rises faster than accuracy falls. That asymmetry is the
    engineering answer the sweep is for: it tells you the minimum verifier
    quality at which cascading still pays.
    """
    true_verdict = grade(task, response_text)
    accepted = true_verdict
    if VERIFIER_CORRUPTION > 0.0:
        # One RNG stream per (task, p). Two draws from it: the first decides
        # whether the verifier bothered to look at the tests, the second is the
        # coin it flips when it didn't.
        #
        # The stream is keyed on p, so the set of corrupted tasks at p=0.25 is
        # NOT a superset of the set at p=0.10. That is deliberate. Nesting the
        # draws would make each sweep point a refinement of the last and the
        # curve would look smoother than the evidence supports; keying on p
        # makes each point an independent realisation, so a monotonic curve is
        # a real result rather than an artefact of shared randomness.
        rng = models._draw(
            task["id"], "verifier_corrupt", VERIFIER_CORRUPTION, VERIFIER_CORRUPTION_SEED
        )
        if rng.random() < VERIFIER_CORRUPTION:
            accepted = rng.random() < 0.5
    return Verdict(accepted=accepted, answer_text=response_text, cost_usd=0.0, latency_s=0.0)


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

# Used only by cascade_degraded. Math keeps its honest proxy verifier so that if
# the policy is ever run on math, it is still the CODE verifier that is being
# damaged and the manipulation stays inside one domain.
DEGRADED_VERIFIERS = {"code": verify_code_degraded, "math": verify_math}


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


def _cascade(task, verifiers, name):
    """Cheap -> verify -> escalate on failure.

    Never misroutes an easy query. Double-pays every time it escalates.

    Parameterised on the verifier map rather than hard-coded, so that
    cascade_degraded is the SAME control loop with a different verifier. That is
    the whole point: if the degraded policy had its own copy of this function,
    a difference between the two curves could be a difference in the loop, and
    the experiment would be measuring the wrong thing.
    """
    cheap = models.call("cheap", task)
    v = verifiers[task["domain"]](task, cheap.text)

    cost = cheap.cost_usd + v.cost_usd
    latency = cheap.latency_s + v.latency_s

    if v.accepted:
        return PolicyResult(
            task_id=task["id"], policy=name,
            # v.answer_text, not cheap.text. For code they are the same thing.
            # For math they are not: the verifier bought k samples and the
            # majority vote among them is the better answer.
            correct=grade(task, v.answer_text),
            cost_usd=cost, latency_s=latency,
            escalated=False, calls=["cheap"],
        )

    exp = models.call("expensive", task)
    return PolicyResult(
        task_id=task["id"], policy=name,
        correct=grade(task, exp.text),
        cost_usd=cost + exp.cost_usd,
        latency_s=latency + exp.latency_s,
        escalated=True, calls=["cheap", "expensive"],
    )


def policy_cascade(task):
    return _cascade(task, VERIFIERS, "cascade")


def policy_cascade_degraded(task):
    """The cascade with a verifier of tunable quality. Code domain only.

    Registered as a first-class policy rather than bolted onto a sweep script,
    because it is the experiment. See DECISION #5 above, and sweep_degraded.py
    for the curve. At VERIFIER_CORRUPTION = 0 this is byte-for-byte the same
    policy as `cascade` on code, which is the sweep's own control.
    """
    return _cascade(task, DEGRADED_VERIFIERS, "cascade_degraded")


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


# ---------------------------------------------------------------------------
# DECISION #6: the random baselines. THE MISSING NULL HYPOTHESIS.
#
# predictive routes 38/100 tasks to the expensive tier and scores 89.0% against
# always_cheap's 74.0%. That gap is uninterpretable on its own, because a router
# that picks 38 tasks AT RANDOM also gains accuracy - it just spends money to do
# it. Without this baseline there is no evidence that predict_is_hard has any
# skill at all, only evidence that spending more helps.
#
# This is not a general methodological nicety. The project plan's Decision #1
# defends the heuristic router with "you can say where yours sits between random
# and optimal", and random did not exist. The written defence rested on a
# baseline that was never implemented.
#
# The escalation rate is matched PER DOMAIN, not globally, because predictive's
# rate differs by domain (math 24/60 = 40%, code 14/40 = 35%) and a global match
# would compare a router that spends unevenly against one that spends evenly.
# The rates are read off predict_is_hard at import time rather than hard-coded,
# so they cannot drift when PREDICTIVE_HARD_LEVEL changes.
#
# One draw is not a baseline. RANDOM_SEED picks which draw goes into
# results.jsonl for pairing; random_baseline.py reports the mean and spread over
# many seeds, which is the number to quote.
# ---------------------------------------------------------------------------
RANDOM_SEED = 0

# Filled by run_eval / random_baseline via calibrate_random_rates(). Defaults are
# the measured rates on the shipped 100-task set, so the policy is usable
# without calibration but says so if it was not calibrated.
RANDOM_MATCHED_RATES = {"math": 0.40, "code": 0.35}
_rates_calibrated = False


def calibrate_random_rates(tasks):
    """Set random_matched's escalation rate to predictive's realised rate.

    Measured on the tasks actually being run, not assumed, so that --limit and
    --domain runs stay cost-matched instead of silently comparing a 35% router
    against a 40% one.
    """
    global RANDOM_MATCHED_RATES, _rates_calibrated
    rates = {}
    for domain in ("math", "code"):
        sub = [t for t in tasks if t["domain"] == domain]
        if sub:
            rates[domain] = sum(predict_is_hard(t) for t in sub) / len(sub)
    RANDOM_MATCHED_RATES = rates or RANDOM_MATCHED_RATES
    _rates_calibrated = True
    return RANDOM_MATCHED_RATES


def _coin(task, rate, label):
    """Reproducible per-task coin at the given rate.

    Through models._draw for the same reason everything else is: an unseeded
    random() would make the baseline un-rerunnable, and a baseline you cannot
    re-run cannot be compared against on a paired basis.
    """
    return models._draw(task["id"], label, RANDOM_SEED, round(rate, 6)).random() < rate


def _policy_random(task, rate, name):
    tier = "expensive" if _coin(task, rate, name) else "cheap"
    r = models.call(tier, task)
    return PolicyResult(
        task_id=task["id"], policy=name,
        correct=grade(task, r.text),
        cost_usd=r.cost_usd, latency_s=r.latency_s,
        # escalated is False by definition: this is a one-shot router, so it
        # never escalates. Which tier it picked is in `calls`.
        calls=[tier],
    )


def policy_random_matched(task):
    rate = RANDOM_MATCHED_RATES.get(task["domain"], 0.4)
    return _policy_random(task, rate, "random_matched")


def policy_random_50(task):
    """A fixed 50/50 flip. A second anchor, and a sanity check on the first:
    random_50 should cost more than random_matched and score no better than the
    extra spend buys."""
    return _policy_random(task, 0.5, "random_50")


# ---------------------------------------------------------------------------
# DECISION #7: LLM-as-router. Testing Decision #4's own rejection of it.
#
# Decision #4 rejects this outright: "Deliberately NOT a trained model and NOT
# an LLM call - an LLM call would add a full round trip and defeat the purpose."
#
# The latency half of that is true and this policy measures it. The COST half is
# quantitatively wrong at this project's own prices. A classification call is
# ~200 tokens in and ~3 out on Haiku, which is roughly a tenth of a full cheap
# call and a fiftieth of an expensive one. It does not defeat the purpose on
# cost by any reading.
#
# Worth an hour precisely because it converts a design comment into a number,
# and because LLM-as-router is what production teams ship first.
#
# IN MOCK MODE THE ACCURACY OF THIS POLICY IS NOT A MEASUREMENT. The mock
# classifier's skill is the constant models.MOCK_ROUTER_SKILL. Its COST and
# LATENCY overhead are real arithmetic on the price table, and those are the
# numbers this policy exists to produce.
# ---------------------------------------------------------------------------

# The routing call's own cost and latency, recorded separately from the answer
# call. Without this the overhead is buried inside the policy total and the
# claim it is meant to test cannot be read off the report.
ROUTER_CALL_COST = []
ROUTER_CALL_LATENCY = []


def policy_llm_router(task):
    """Ask the cheap model whether the task is hard, then route on the answer.

    The router call is charged to the policy, which is the entire point: a
    routing method that hides its own overhead is not being compared fairly
    against one that has none.

    An unparseable reply routes CHEAP. Failing open to the expensive tier would
    let a broken parser quietly turn this into always_expensive and read as a
    capability result.
    """
    r = models.call("cheap", task, kind="route")
    ROUTER_CALL_COST.append(r.cost_usd)
    ROUTER_CALL_LATENCY.append(r.latency_s)
    said_hard = "HARD" in r.text.strip().upper()

    tier = "expensive" if said_hard else "cheap"
    ans = models.call(tier, task)
    return PolicyResult(
        task_id=task["id"], policy="llm_router",
        correct=grade(task, ans.text),
        cost_usd=r.cost_usd + ans.cost_usd,
        latency_s=r.latency_s + ans.latency_s,
        calls=["router", tier],
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
    "random_matched": policy_random_matched,
    "random_50": policy_random_50,
    "predictive": policy_predictive,
    "llm_router": policy_llm_router,
    "cascade": policy_cascade,
    "cascade_degraded": policy_cascade_degraded,
    "oracle": policy_oracle,
}

# Which domains a policy is defined on. None means all of them.
#
# cascade_degraded is code-only BY DESIGN. Its purpose is to vary verifier
# fidelity while holding the domain fixed, so running it on math - where the
# verifier is already a proxy and the corruption would compound with it - would
# reintroduce exactly the confound it exists to remove.
POLICY_DOMAINS = {
    "cascade_degraded": ("code",),
}
