"""
The routing policies.

Each takes a task and returns a PolicyResult. The values marked DECISION are the
tuneable choices the experiment rests on; changing one and re-running is the
intended way to see what it does.

The two architectures this repository compares:

  PREDICTIVE (one-shot)   inspect the query, pick a tier, commit. Pays once.
                          Misroutes silently and never finds out.

    llm_router          the cheap model itself picks the tier, then answers
    routellm            a pretrained learned router scores the prompt

  CASCADING               answer, verify, escalate on failure. Never misroutes
                          an easy query. Double-pays on every escalation.

    cascade             cheap -> verify -> escalate, over every rung
    cascade_routing     routing and cascading unified under one lambda
    cascade_degraded    cascade with a deliberately damaged verifier (code only)

  Fixed and bounds:

    always_cheap        one cheap call, always
    always_expensive    one expensive call, always
    random_matched      coin flip at llm_router's own escalation rate
    oracle              hindsight-optimal, not deployable

Three of them exist for reasons that are not obvious from the name:

  cascade_degraded  is the experiment, not a variant. It moves verifier quality
                    inside a single domain, which is the only way to stop
                    verifier quality from being confounded with math-vs-code.

  random_matched    is the null hypothesis. A router that escalates the same
                    fraction of tasks AT RANDOM also gains accuracy - it just
                    pays for it. Without this baseline, a gap between a router
                    and always_cheap establishes only that spending more helps,
                    not that the router has any skill.

  llm_router        exists to test the claim recorded in DECISION #4 that an LLM
                    routing call "would defeat the purpose". That is a cost
                    claim, and cost is measurable.

A hand-written `predictive` policy was removed on 8 August 2026. See the
DECISION #4 tombstone below: it was not a router, it was a constant.
"""

from dataclasses import dataclass, field
from collections import Counter

import models
from graders import grade, extract_answer

# ---------------------------------------------------------------------------
# DECISION #2: self-consistency sample count for the math verifier.
# Higher k means better failure detection at linearly more cost. k=5 is the
# common starting point in the self-consistency literature.
# ---------------------------------------------------------------------------
SELF_CONSISTENCY_K = 5

# ---------------------------------------------------------------------------
# DECISION #3: the escalation threshold.
# The fraction of self-consistency samples that must agree for the cheap answer
# to be ACCEPTED. Below this, escalate. 1.0 accepts only unanimous answers.
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
    # Which tiers this policy actually paid for, in order. This is the field to
    # read for "did the expensive model get involved", because `escalated` is
    # meaningful only for the cascades - a one-shot router never escalates by
    # definition, whichever tier it picked.
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

# Every verifier takes (task, response_text, tier) and returns a Verdict. `tier`
# is which model produced the response, and it matters more than it looks:
# self-consistency has to resample THE SAME MODEL that gave the answer, so
# verifying a mid-tier answer costs mid-tier samples. Verification therefore gets
# rapidly more expensive as a cascade climbs the ladder, which is one of the real
# economic reasons deep cascades are rarer than shallow ones.

def verify_code(task, response_text, tier="cheap"):
    """FREE AND PERFECT. Run the shipped asserts.

    This is the ideal case, and almost no real problem looks like it. Nothing
    extra is sampled, so the answer to grade is the one that came in, and `tier`
    is irrelevant - running tests costs the same whoever wrote the code.

    "Perfect" means perfect WITH RESPECT TO THE SPEC, and the spec is the
    asserts. A solution that passes the shipped tests and is otherwise bad is
    accepted, correctly, by both this verifier and the grader.
    """
    passed = grade(task, response_text)
    return Verdict(accepted=passed, answer_text=response_text, cost_usd=0.0, latency_s=0.0)


def _self_consistency(task, response_text, tier="cheap"):
    """Sample `tier` k times and return the plurality answer.

    Returns (answer, agreement, cost, latency). `answer` is None when every
    sample was unparseable.

    Only the cheap tier accepts a temperature (see models.MODELS), so sampling
    any higher tier would return k identical greedy answers and report unanimous
    agreement on every task - a verifier that always accepts. Callers asking for
    a higher tier get `None` agreement rather than that silent lie; see
    verify_math.

    Shared by verify_math, the oracle and cascade_routing, deliberately. The
    oracle has to be able to reach every action a deployable policy can reach,
    and majority voting over k samples is one of them - see policy_oracle.

    sample_idx starts at 1 because index 0 is the greedy call the caller already
    made. Distinct indices are what make the samples differ in mock mode; passing
    the same one k times would produce k identical answers and unanimous
    agreement on every task.

    Ties go to the greedy sample: it is inserted first and most_common breaks
    ties on insertion order. Arbitrary, but deterministic and defensible.
    """
    if not models.MODELS[tier]["accepts_temperature"]:
        return None, None, 0.0, 0.0

    answers = [extract_answer(response_text)]
    cost = 0.0
    latency = 0.0
    for i in range(1, SELF_CONSISTENCY_K):
        r = models.call(tier, task, temperature=0.8, sample_idx=i)
        answers.append(extract_answer(r.text))
        cost += r.cost_usd
        latency += r.latency_s

    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return None, 0.0, cost, latency
    top, top_count = counts.most_common(1)[0]
    # Denominator is len(answers), NOT the number of parseable ones, so
    # unparseable samples count against agreement rather than being ignored.
    # Deliberate: a model that cannot produce a readable answer is not one to be
    # confident in. Worth knowing that it conflates "the model disagreed with
    # itself" with "the parser failed".
    return top, top_count / len(answers), cost, latency


def verify_math(task, response_text, tier="cheap"):
    """PROXY. There is no ground truth at runtime, so sample the model k times at
    temperature > 0 and check whether the answers agree.

    The logic: a model with a stable internal answer returns it repeatedly; a
    model that is guessing scatters. Agreement is a proxy for confidence, and it
    is a NOISY one. Measuring how much worse this is than verify_code is the most
    interesting result the project is set up to produce.

    Returns the PLURALITY answer rather than the greedy one it was handed. Two
    distinct things come out of the same k samples and it would be wasteful to
    use only one:

      - the ESCALATION signal, how much the samples agree
      - a better ANSWER, the majority vote, which is what self-consistency was
        published for in the first place

    Note the consequence for interpretation: the accepted math answer is a
    self-consistency answer, so `cascade` bundles two mechanisms - majority
    voting for accuracy and agreement for escalation. That is what AutoMix-style
    systems do, and it has to be said out loud rather than left implicit, because
    it means the math cascade is not comparable to a bare single cheap call.
    """
    top, agreement, cost, latency = _self_consistency(task, response_text, tier)

    if agreement is None:
        # This tier cannot be sampled, so there is no agreement signal to read.
        # REJECT rather than accept: an unverifiable answer is exactly the case
        # where a cascade should keep going, and accepting here would turn "no
        # verifier available" into "verifier said yes".
        return Verdict(False, response_text, cost, latency)

    if top is None:
        # Every sample was unparseable. Escalate, and there is no better answer
        # to offer than the one that came in.
        return Verdict(False, response_text, cost, latency)

    # Rewrapped as \boxed{} so grade() takes its primary parse path on a value
    # that is already normalised, rather than the last-line fallback.
    return Verdict(
        accepted=agreement >= AGREEMENT_THRESHOLD,
        answer_text=f"\\boxed{{{top}}}",
        cost_usd=cost,
        latency_s=latency,
    )


# ---------------------------------------------------------------------------
# DECISION #5: the verifier corruption rate. THIS IS THE MANIPULATED VARIABLE.
#
# The project's stated contribution is that verifier quality is what varies.
# With only the two natural verifiers, that variable has exactly two levels and
# they are perfectly confounded with task domain:
#
#     perfect verifier <-> code <-> MBPP    <-> run asserts <-> $0 verification
#     proxy   verifier <-> math <-> MATH500 <-> exact match <-> k-1 extra calls
#
# Every row differs in five ways at once, so "the code cascade wins and the math
# cascade does not" cannot be attributed to verifier quality. It can only be
# attributed to "code is different from math", which is not a finding.
#
# cascade_degraded fixes that by corrupting verify_code on the CODE domain. Same
# tasks, same models, same grader, same prompts, same cost structure - only
# verifier fidelity moves. Sweeping p from 0 to 1 gives a curve from a perfect
# verifier to a coin flip with everything else held fixed, which is what turns an
# observation into a measurement.
#
# p is the probability that the verifier IGNORES the test result and returns a
# coin flip instead. So the effective error rate is p/2 rather than p, and p=1.0
# is a verifier with zero information (AUC 0.5) rather than an inverted one.
# ---------------------------------------------------------------------------
VERIFIER_CORRUPTION = 0.0

# Which realisation of the corruption this is.
#
# At n=40 a single draw per corruption level is NOT a curve. One task is 2.5
# points of accuracy, and the binomial standard deviation at these rates is
# around 5 points, so a six-point single-draw sweep can easily show a rise where
# the mechanism predicts a fall. Sweeping this seed and averaging is what turns
# the sweep from an anecdote into an estimate, and it costs nothing because every
# model response is a cache hit.
VERIFIER_CORRUPTION_SEED = 0


def verify_code_degraded(task, response_text, tier="cheap"):
    """verify_code, damaged on purpose at rate VERIFIER_CORRUPTION.

    Deterministic given (task, corruption rate, corruption seed, mock seed): the
    corruption is drawn through models._draw like everything else stochastic in
    this project, so a sweep is reproducible and two policies asked the same
    question at the same p get the same answer.

    Note which way the two error types cost, because they are not symmetric:

      false REJECT of a correct answer -> escalate anyway. Money wasted,
                                          accuracy unharmed.
      false ACCEPT of a wrong answer   -> ship the wrong answer. Accuracy lost,
                                          money saved.

    Since the cheap model is right on most code tasks, false rejects dominate at
    low p, so cost rises faster than accuracy falls. That asymmetry is the
    engineering answer the sweep exists for: it says what the minimum verifier
    quality is at which cascading still pays.
    """
    accepted = grade(task, response_text)
    if VERIFIER_CORRUPTION > 0.0:
        # One RNG stream per (task, p). Two draws from it: the first decides
        # whether the verifier bothered to look at the tests, the second is the
        # coin it flips when it did not.
        #
        # The stream is keyed on p, so the set of corrupted tasks at p=0.25 is
        # NOT a superset of the set at p=0.10. That is deliberate. Nesting the
        # draws would make each sweep point a refinement of the last and the
        # curve would look smoother than the evidence supports; keying on p makes
        # each point an independent realisation, so a monotonic curve is a real
        # result rather than an artefact of shared randomness.
        rng = models._draw(
            task["id"], "verifier_corrupt", VERIFIER_CORRUPTION, VERIFIER_CORRUPTION_SEED
        )
        if rng.random() < VERIFIER_CORRUPTION:
            accepted = rng.random() < 0.5
    return Verdict(accepted=accepted, answer_text=response_text, cost_usd=0.0, latency_s=0.0)


VERIFIERS = {"code": verify_code, "math": verify_math}

# Used only by cascade_degraded. Math keeps its honest proxy verifier so that if
# the policy is ever run on math, it is still the CODE verifier being damaged and
# the manipulation stays inside one domain.
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


def _verdict_is_predetermined(task, tier, verifier) -> bool:
    """True when this verifier must reject `tier` whatever it answers.

    Currently one case, and it comes from an API contract rather than from
    anything about the task: the self-consistency verifier needs to resample the
    model at a temperature, and only some rungs accept one. On a rung that does
    not, there is no agreement signal, verify_math rejects, and the answer that
    was paid for is discarded unread.

    Deliberately narrow. It asks "is the verdict knowable without the answer",
    not "do I expect a rejection" - a cascade that skipped rungs it merely
    expected to fail would be a predictive router wearing a cascade's clothes,
    and the whole point of the comparison is that the two are different.
    """
    if verifier is not verify_math:
        return False
    return not models.MODELS[tier]["accepts_temperature"]


def _cascade(task, verifiers, name, chain=("cheap", "expensive")):
    """Walk a ladder of tiers: answer, verify, escalate on rejection.

    Never misroutes an easy query, because it looks at the answer before
    deciding. Pays for every rung it climbs, so an escalation is strictly more
    expensive than having routed correctly the first time.

    `chain` is the ladder, cheapest first. The two-tier default keeps `cascade`
    default exists only so a caller can deliberately skip rungs; policy_cascade
    passes the full models.TIERS ladder.

    Parameterised on the verifier map rather than hard-coded, so cascade_degraded
    is the SAME control loop with a different verifier. That is the whole point:
    if the degraded policy had its own copy of this loop, a difference between
    the two curves could be a difference in the loop, and the experiment would be
    measuring the wrong thing.

    On the last rung there is nothing left to escalate to, so no verifier is run
    at all. That is not a shortcut - verifying an answer you cannot act on is
    pure cost, and on math it would be four extra samples for no decision.

    UNVERIFIABLE MIDDLE RUNGS ARE SKIPPED, and the reason is worth stating because
    it is a real cost that was measured rather than a hypothetical. On the claude
    ladder the mid rung (Sonnet 5) does not accept a temperature, so verify_math
    cannot sample it and correctly refuses to accept an answer it cannot check.
    The consequence, before this guard existed: on every maths escalation the
    cascade paid for a mid-tier answer whose rejection was already certain, then
    escalated anyway. Measured on the shipped task set that was 25 out of 25
    escalations and about 20% of the cascade's maths spend, bought for nothing.

    Skipping such a rung is strictly better - identical answers, identical
    accuracy, less money - so it is not a tuning choice. A rung is only skipped
    when something above it remains; the top rung is always reachable, because at
    that point the answer is taken on trust rather than verified.
    """
    cost = 0.0
    latency = 0.0
    calls = []
    verifier = verifiers[task["domain"]]

    for i, tier in enumerate(chain):
        last = i == len(chain) - 1
        # Would this rung's verdict be decided before its answer is even seen?
        # _self_consistency reports no signal on a rung it cannot sample, and
        # verify_math turns that into a rejection. Paying for a rejection that is
        # already certain is waste, so step over it.
        if not last and _verdict_is_predetermined(task, tier, verifier):
            continue
        r = models.call(tier, task)
        cost += r.cost_usd
        latency += r.latency_s
        calls.append(tier)

        if i == len(chain) - 1:
            return PolicyResult(
                task_id=task["id"], policy=name,
                correct=grade(task, r.text),
                cost_usd=cost, latency_s=latency,
                escalated=i > 0, calls=calls,
            )

        v = verifiers[task["domain"]](task, r.text, tier)
        cost += v.cost_usd
        latency += v.latency_s
        # The verifier's own samples are charged to the tier it sampled, so the
        # call list stays an honest record of what was bought.
        if v.cost_usd > 0.0:
            calls.extend([tier] * (SELF_CONSISTENCY_K - 1))

        if v.accepted:
            # v.answer_text, not r.text. For code they are the same thing. For
            # math they are not: the verifier bought k samples and the majority
            # vote among them is the better answer.
            return PolicyResult(
                task_id=task["id"], policy=name,
                correct=grade(task, v.answer_text),
                cost_usd=cost, latency_s=latency,
                escalated=i > 0, calls=calls,
            )

    raise AssertionError("unreachable: the last rung always returns")


def policy_cascade(task):
    """The cascade over whatever ladder is loaded.

    It walks every rung of models.TIERS rather than jumping bottom to top, so on a
    three-rung ladder it can stop somewhere that is neither the floor nor the
    ceiling. There is no separate two-rung and three-rung policy: the ladder is the
    variable, and `ROUTER_LADDER=deepseek` versus `ROUTER_LADDER=claude` is how the
    "does an extra rung pay?" question gets asked.

    The extra rung is not free, and the direction is not obvious in advance:

      + a task the bottom rung fails but a middle rung handles costs 1x + 3x
        instead of 1x + 5x;
      - a task that needs the top rung now pays for a wasted middle call on the
        way, plus a second round of verification;
      - on math, that second verification is k middle-rung samples at the middle
        rung's price, which is the largest single cost an extra rung adds.
    """
    return _cascade(task, VERIFIERS, "cascade", chain=tuple(models.TIERS))


def policy_cascade_degraded(task):
    """The cascade with a verifier of tunable quality. Code domain only.

    Registered as a first-class policy rather than bolted onto a sweep script,
    because it is the experiment. See DECISION #5 above, and sweep_degraded.py
    for the curve. At VERIFIER_CORRUPTION = 0 this is the same policy as
    `cascade` on code, which is the sweep's own control.
    """
    return _cascade(task, DEGRADED_VERIFIERS, "cascade_degraded")


# ---------------------------------------------------------------------------
# DECISION #4: the hand-written predictive heuristic. RETRACTED 8 August 2026.
#
# The numbering is kept and the slot left empty on purpose. DECISIONS #1-#9 are
# cited by number in README.md, docs/NOTES.md and docs/WALKTHROUGH.md, and
# renumbering would silently rewrite the record. This is a tombstone.
#
# WHAT IT WAS. `predict_is_hard` routed math on MATH500's shipped `level >= 5`
# and code on `prompt_chars >= 100`, choosing one tier up front and committing.
#
# WHY IT IS GONE. build_taskset.MIN_MATH_LEVEL was raised to 5 on 6 August, which
# makes `predict_features.level` CONSTANT at 5 across all 60 math tasks. The
# predicate therefore returned True for every one of them: on 60% of the task set
# the policy was not a router, it was `always_expensive` spelled differently.
# build_taskset.py:54-61 recorded that consequence when the filter was raised; it
# never reached this file, the report, or the README.
#
# It was measured at 86.0% accuracy against `random_matched`'s 88.0% and
# `llm_router`'s 88.0% on the held-out half - BELOW the coin flip it was supposed
# to beat - and its frontier sweep drew two attainable points rather than a curve,
# because a threshold sweep over a constant has nowhere to go. The published
# reading of that sweep, "predictive contributes no point to the frontier,
# LLMRouterBench reproduced on real data", was an arithmetic consequence of the
# level filter and is retracted. See STATUS.md section 2.3.
#
# THE CLAIM IT MADE, which DECISION #7 exists to test and run_eval still prints:
# an LLM routing call "would add a full round trip and defeat the purpose". The
# latency half is true. The cost half is quantitatively wrong at this project's
# own prices, and `llm_router` measures by how much.
#
# WHAT REPLACED IT. Predictive routing did not go away - it is one of the two
# architectures this repository exists to compare. It is now represented by its
# two real implementations, `llm_router` and `routellm`, neither of which reads a
# difficulty label that no production query carries. That was always the deeper
# problem with this heuristic: `level` is written by someone who has already
# solved the problem. The asymmetry it was meant to illustrate survives it -
# predictive routing needs a difficulty signal up front, and a cascade does not,
# because it looks at the answer.
#
# `predict_features` stays in the task set. It is the leak-discipline artifact
# (build_taskset.py populates it with question-derived values ONLY, keeping the
# reference solution's line count out of reach of any router), and
# router_agent/live.py reads `prompt_chars` from it.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DECISION #6: the random baseline. THE NULL HYPOTHESIS.
#
# A gap between a router and always_cheap is uninterpretable on its own, because
# a router that escalates the same NUMBER of tasks at random also gains accuracy
# - it just spends money to do it. Without this baseline there is no evidence
# that any router has skill, only evidence that spending more helps.
#
# THE ANCHOR IS `llm_router`, changed 8 August 2026 from the deleted `predictive`
# heuristic (DECISION #4). It is the only member of the predictive family that
# runs in every mode with no external dependency: `routellm` sits out whenever
# its score cache is stale, and anchoring the null to a policy that may not run
# is a bad structural bet.
#
# Reading the rate costs nothing. The routing decisions go through
# models.call(..., kind="route"), which is response_cache-backed, so the pre-pass
# hits exactly the entries policy_llm_router hits moments later. In replay and
# mock it is free by construction; in real mode the policy pays for those calls
# anyway and the cache deduplicates.
#
# TWO DISCLOSURES, both of which run_eval prints:
#   - random_matched does NOT pay the router call, so it is cheaper than
#     llm_router by exactly mean(ROUTER_CALL_COST). The LLM-as-router overhead
#     block reports that number.
#   - in mock mode the anchor derives from models.MOCK_ROUTER_SKILL, so the null
#     is fabricated along with everything else. The SIMULATED banner covers it.
#
# The escalation rate is matched PER DOMAIN, not globally, because the router's
# rate differs by domain and a global match would compare a router that spends
# unevenly against one that spends evenly. Measured at run time on the tasks
# actually being run, so --limit and --domain runs stay cost-matched.
#
# ONE ANCHORED NULL CANNOT SERVE A WHOLE FAMILY. Policies that spend differently
# from llm_router - routellm on a fixed threshold, and both cascades - need a
# null at THEIR OWN spend. run_eval.routing_skill computes that analytically from
# the always_cheap -> always_expensive chord. This policy stays as the printed
# empirical check that the chord is not a fiction.
#
# One draw is not a baseline. RANDOM_SEED picks which draw goes into
# results.jsonl for pairing; frontier.py sweeps the rate across its whole range,
# which is the stronger comparison and replaced a dedicated seed-sweep script.
# ---------------------------------------------------------------------------
RANDOM_SEED = 0

# Set by calibrate_random_rates(), which run_eval, frontier and routellm_router
# all call before running anything. The defaults are a fallback so the policy is
# usable uncalibrated; every entry point calibrates.
RANDOM_MATCHED_RATES = {"math": 0.40, "code": 0.35}


def calibrate_random_rates(tasks, decisions=None):
    """Set random_matched's escalation rate to llm_router's realised rate.

    `decisions` maps task id -> True when the router said HARD, as returned by
    llm_router_decisions(). Passing None keeps the declared defaults rather than
    guessing, which is the same "sit out rather than invent" rule routellm and
    cascade_routing follow; the caller is expected to say so out loud.
    """
    global RANDOM_MATCHED_RATES
    if decisions is None:
        return RANDOM_MATCHED_RATES
    rates = {}
    for domain in ("math", "code"):
        sub = [t for t in tasks if t["domain"] == domain]
        if sub:
            rates[domain] = sum(
                bool(decisions.get(t["id"], False)) for t in sub) / len(sub)
    RANDOM_MATCHED_RATES = rates or RANDOM_MATCHED_RATES
    return RANDOM_MATCHED_RATES


def _coin(task, rate, label):
    """Reproducible per-task coin at the given rate.

    Through models._draw for the same reason everything else is: an unseeded
    random() would make the baseline un-rerunnable, and a baseline that cannot be
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
        # escalated stays False by definition: this is a one-shot router, so it
        # never escalates. Which tier it picked is in `calls`.
        calls=[tier],
    )


def policy_random_matched(task):
    """The null hypothesis: escalate at llm_router's own rate, but at random.

    There used to be a second `random_50` anchor at a fixed 50/50 rate. It was
    removed rather than kept: frontier.py sweeps this policy's rate from 0 to 1, so
    a 50/50 flip is one point on a curve the frontier already draws, and having it
    as a named policy implied it carried information the curve does not.
    """
    rate = RANDOM_MATCHED_RATES.get(task["domain"], 0.4)
    return _policy_random(task, rate, "random_matched")


# ---------------------------------------------------------------------------
# DECISION #7: LLM-as-router. Testing DECISION #4's own rejection of it.
#
# DECISION #4 rejects an LLM routing call on the grounds that it "would add a
# full round trip and defeat the purpose".
#
# The latency half of that is true, and this policy measures it. The COST half is
# quantitatively wrong at this project's own prices: a classification call is a
# couple of hundred tokens in and about three out on the cheap tier, which is a
# small fraction of a full cheap answer call and a much smaller fraction of an
# expensive one. run_eval prints the exact ratio.
#
# Worth implementing precisely because it converts a design comment into a
# number, and because LLM-as-router is what production teams tend to ship first.
#
# IN MOCK MODE THE ACCURACY OF THIS POLICY IS NOT A MEASUREMENT. The mock
# classifier's skill is the constant models.MOCK_ROUTER_SKILL, and the mock
# router judges the mock's own latent difficulty, so it will tend to outperform
# every honest router here. Its COST and LATENCY overhead are real arithmetic on
# the price table, and those are the numbers this policy exists to produce.
# ---------------------------------------------------------------------------

# The routing call's own cost and latency, recorded separately from the answer
# call. Without this the overhead is buried inside the policy total and the claim
# it is meant to test cannot be read off the report.
ROUTER_CALL_COST = []
ROUTER_CALL_LATENCY = []


def _said_hard(text) -> bool:
    """Parse the router's reply. Anything unrecognised routes CHEAP.

    Extracted so the policy and the rate pre-pass cannot drift apart. If they
    read the same reply differently, random_matched would be calibrated to a
    rate llm_router never realised, and the null would stop being cost-matched
    without anything failing.
    """
    return "HARD" in text.strip().upper()


def llm_router_decisions(tasks):
    """{task_id: said_hard} for DECISION #6's rate anchor. Costs nothing extra.

    Every call here is the same (tier, prompt, kind) tuple policy_llm_router
    makes, so response_cache serves this pass and the policy from one draw. In
    replay a missing entry raises ReplayMiss, which the caller catches and
    degrades from - it does not invent a decision.

    Deliberately does NOT append to ROUTER_CALL_COST / ROUTER_CALL_LATENCY. The
    policy appends on its own pass, and double-counting here would inflate the
    overhead figure the report prints to test DECISION #4's cost claim.
    """
    return {t["id"]: _said_hard(models.call("cheap", t, kind="route").text)
            for t in tasks}


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

    tier = "expensive" if _said_hard(r.text) else "cheap"
    ans = models.call(tier, task)
    return PolicyResult(
        task_id=task["id"], policy="llm_router",
        correct=grade(task, ans.text),
        cost_usd=r.cost_usd + ans.cost_usd,
        latency_s=r.latency_s + ans.latency_s,
        calls=["router", tier],
    )


# ---------------------------------------------------------------------------
# DECISION #8: RouteLLM's pretrained router.
#
# A published learned router, trained on Chatbot Arena preference data, with its
# threshold calibrated to predictive's expensive-call rate so the comparison is
# cost-matched. See routellm_router.py for which of RouteLLM's variants is used
# and why, and for the score cache that makes it a one-time cost.
#
# This policy makes NO routing call of its own at serving time on the bert path:
# the score is a local forward pass, cached to disk. So unlike llm_router it adds
# no round trip, which is the argument for a learned router over an LLM one.
#
# It is registered only when real scores exist. There is deliberately no
# fallback: a router that guesses when its model is missing is not a router, and
# a row labelled `routellm` that was produced by a coin flip would be the worst
# kind of number in this repo.
# ---------------------------------------------------------------------------

def policy_routellm(task):
    import routellm_router

    tier = "expensive" if routellm_router.routes_expensive(task) else "cheap"
    r = models.call(tier, task)
    return PolicyResult(
        task_id=task["id"], policy="routellm",
        correct=grade(task, r.text),
        cost_usd=r.cost_usd, latency_s=r.latency_s,
        calls=[tier],
    )


# ---------------------------------------------------------------------------
# DECISION #9: CASCADE ROUTING - the unified policy.
#
# The whole repo is framed as "cascade vs predictive", and the literature's answer
# to that framing is that it is a false choice. Dekoninck, Baader and Vechev,
# "A Unified Approach to Routing and Cascading for LLMs" (arXiv:2410.10347, ICML
# 2025) prove that routing and cascading are both special cases of one strategy,
# derive the optimal form, and report that the unified version beats either one
# alone by a large margin - up to 8% on RouterBench.
#
# Their central conclusion is also, independently, this project's thesis:
#
#   "we identify good quality estimators as the critical factor for the success
#    of model selection paradigms"
#
# and, more precisely, that routing needs good EX-ANTE quality estimation (can I
# predict this model will do well?) while cascading needs good POST-HOC quality
# estimation (was that answer any good?). The verifier is this repo's post-hoc
# estimator, and it is a good one on the code half. So the repo is in a position
# to test their claim empirically on objectively-graded tasks, which their
# RouterBench experiments could only do by injecting synthetic Gaussian noise
# into a quality signal - and sweep_degraded.py does exactly that.
#
# The EX-ANTE side is where this repository comes up empty, which is a result
# rather than a gap in the implementation. `predict_is_hard` held that slot until
# 8 August 2026 and turned out to be a constant (DECISION #4). `_Q_EXANTE` now
# carries a domain prior and an empty feature slot, so cascade_routing runs with
# a deliberately uninformative ex-ante term. Read its numbers as "the unified
# strategy with only the post-hoc half working", which is the honest description
# and is the paper's own prediction about what happens next.
#
# THE STRATEGY. Every model i gets a quality estimate q_i and a cost estimate
# c_i. A single parameter lambda prices quality against money, and the strategy
# maximises the tradeoff
#
#     tau_i = q_i - lambda * c_i
#
# choosing at each step between stopping with the best answer in hand and paying
# for another tier. Sweeping lambda from 0 to infinity walks the policy from
# always_expensive down to always_cheap, tracing a whole cost-quality curve
# rather than sitting at one operating point. frontier.py does that sweep.
#
# WHY THIS IS NOT JUST `cascade`. Two differences, both of which matter:
#
#   1. It need not start at the bottom. A task the ex-ante estimator flags as hard
#      goes straight to a higher tier, skipping a cheap call that would have been
#      thrown away. `cascade` always pays for that call.
#   2. It need not climb one rung at a time. If the post-hoc signal says the
#      answer is badly wrong, it can jump to the top rather than pay for the
#      middle on the way.
#
# WHAT IS IMPLEMENTED HERE is the paper's GREEDY variant: at each step it
# compares stopping against continuing to the single best next tier, rather than
# taking an expectation over every remaining subset of the ladder. The paper
# evaluates that variant too and reports it costs 0.5% to 1.3% against the full
# version, mostly in low-noise settings. It is chosen here for legibility - the
# full version needs a variance estimate per model per step - and the gap is
# stated rather than hidden.
# ---------------------------------------------------------------------------

# Lagrange multiplier on cost, in quality per dollar.
#
# It CANNOT be a fixed number, and getting this wrong is instructive. lambda
# multiplies dollars, so its useful magnitude is set by the ladder's absolute
# prices - and the ladders here differ by more than two orders of magnitude
# (a top-rung Claude call is around a thousandth of a dollar; a top-rung DeepSeek
# call is around a hundred-thousandth). A lambda tuned on the claude ladder makes
# cost effectively free on the deepseek ladder, so the policy climbs to the top
# rung on every task and quietly becomes always_expensive while still being
# reported as a router.
#
# So it is expressed in a scale-free unit and converted: LAMBDA_QUALITY_PER_TOP_CALL
# is how much accuracy, in probability, one top-rung call is deemed to be worth.
# 0.5 means "paying for the best model is worth it if it adds 50 accuracy points",
# which is deliberately stingy and puts the default in the interesting region
# rather than at either extreme.
#
# frontier.py sweeps lambda across four orders of magnitude, so the default only
# decides the single operating point that lands in results.jsonl.
LAMBDA_QUALITY_PER_TOP_CALL = 0.5


def _default_lambda():
    """Convert the scale-free setting into dollars for the loaded ladder."""
    top = models.TIERS[-1]
    nominal = models._price(
        top,
        # A nominal prompt and reply, so the conversion depends on the price table
        # rather than on any particular task.
        tokens_in=500,
        tokens_out=models.MOCK_TOKENS_OUT[top],
    )
    return LAMBDA_QUALITY_PER_TOP_CALL / nominal if nominal > 0 else 0.0


CASCADE_ROUTING_LAMBDA = _default_lambda()

# Calibrated estimator tables, filled by fit_estimators() from the CALIBRATION
# split only. Empty means uncalibrated, and the policy refuses to run rather than
# invent a quality estimate - the same rule routellm follows.
#
#   _Q_EXANTE[(tier, domain, feat)]  P(tier correct | what is knowable pre-call)
#   _Q_POSTHOC[(domain, accepted)]   P(answer correct | verifier verdict)
#   _Q_RESCUE[(tier, from_tier)]     P(tier correct | from_tier was wrong)
#
# _Q_EXANTE was keyed on (tier, predict_is_hard(task)) until 8 August 2026. With
# `level` constant across the math half, that flag was really reading DOMAIN -
# `hard=True` pooled all math plus long code, `hard=False` was short code only.
# The two rows differed, and run_eval printed that as evidence the flag carried
# signal. It was a false positive: the rows differed because the domains differ.
#
# So the key is now (tier, domain, EXANTE_FEATURE(task)), with the feature slot
# empty by default. That is the honest state of this repository - Dekoninck et
# al. identify a good EX-ANTE quality estimator as what routing needs, and this
# repo does not have one. Recording the absence in the shape of the table beats
# filling it with a constant. EXANTE_FEATURE is where a real one plugs in.
#
# The third table is the one that makes escalation decisions honest. The value of
# climbing a rung is not "how good is the next model" but "how good is the next
# model ON THE TASKS THIS ONE JUST FAILED", and correlated failure means those are
# very different numbers. Measuring it rather than assuming independence is what
# stops the policy from over-escalating.
_Q_EXANTE = {}
_Q_POSTHOC = {}
_Q_RESCUE = {}
ESTIMATORS_FITTED = False

# Optional pre-call feature for the ex-ante estimate: callable(task) -> hashable,
# or None for "this repository has no ex-ante signal". Set it and both
# fit_estimators and policy_cascade_routing pick it up; they must agree, which is
# why they both go through _exante_key.
EXANTE_FEATURE = None


def _exante_key(tier, task):
    feat = EXANTE_FEATURE(task) if EXANTE_FEATURE is not None else None
    return (tier, task["domain"], feat)


def fit_estimators(tasks):
    """Fit the quality estimators on `tasks`, which must be the CALIBRATION split.

    Uses ground truth, which is exactly what a calibration split is for. Fitting
    these on the evaluation tasks and then reporting on the same tasks would make
    cascade_routing's numbers a measure of its own hindsight.

    Free in every mode: every model response involved is a cache hit after the
    first policy has run.
    """
    global ESTIMATORS_FITTED
    _Q_EXANTE.clear()
    _Q_POSTHOC.clear()
    _Q_RESCUE.clear()

    correct = {}  # (tier, task_id) -> bool
    for tier in models.TIERS:
        for t in tasks:
            correct[(tier, t["id"])] = grade(t, models.call(tier, t).text)

    # Ex-ante: how often each tier is right, given only what is knowable before
    # calling anything. With EXANTE_FEATURE unset that is the domain prior and
    # nothing more, which is the true state of this repo's ex-ante signal. Set
    # EXANTE_FEATURE and the buckets split further; if the resulting rows come
    # out equal, the feature carries nothing and the policy will ignore it.
    buckets = {}
    for tier in models.TIERS:
        for t in tasks:
            buckets.setdefault(_exante_key(tier, t), []).append(
                correct[(tier, t["id"])])
    for key, vals in buckets.items():
        _Q_EXANTE[key] = sum(vals) / len(vals)

    # Post-hoc: how much the verifier's verdict actually tells you. The gap
    # between the accepted and rejected rows IS the verifier's quality, which is
    # the quantity this whole repo manipulates. Under a corrupted verifier the two
    # rows converge and cascade_routing stops trusting the signal, which is the
    # mechanism the paper predicts and sweep_degraded measures.
    for domain in ("math", "code"):
        sub = [t for t in tasks if t["domain"] == domain]
        buckets = {True: [], False: []}
        for t in sub:
            cheap = models.call("cheap", t)
            v = VERIFIERS[domain](t, cheap.text, "cheap")
            buckets[bool(v.accepted)].append(grade(t, v.answer_text))
        for accepted, vals in buckets.items():
            if vals:
                _Q_POSTHOC[(domain, accepted)] = sum(vals) / len(vals)

    # Rescue: P(tier correct | from_tier wrong). The conditional that correlated
    # failure makes necessary.
    for from_tier in models.TIERS:
        failed = [t for t in tasks if not correct[(from_tier, t["id"])]]
        for tier in models.TIERS:
            if tier == from_tier or not failed:
                continue
            _Q_RESCUE[(tier, from_tier)] = (
                sum(correct[(tier, t["id"])] for t in failed) / len(failed)
            )

    ESTIMATORS_FITTED = True
    return {"exante": dict(_Q_EXANTE), "posthoc": dict(_Q_POSTHOC),
            "rescue": dict(_Q_RESCUE)}


def _est_cost(tier, task):
    """Cost estimate for one call, from the price table and the prompt length.

    Pre-call and therefore leak-free: it uses the prompt, which is available, and
    a modelled output length, which is not knowable in advance for a real model.
    The paper suggests exactly this - tokenise the input and assume an average
    output length.
    """
    prompt = models.build_prompt(task)
    tokens_in = models._mock_tokens_in(tier, prompt)
    return models._price(tier, tokens_in, models.MOCK_TOKENS_OUT[tier])


def policy_cascade_routing(task, lam=None):
    """Routing and cascading as one strategy. See DECISION #9.

    At each step, compare the value of stopping with the answer in hand against
    the value of paying for the best remaining tier, using tau = q - lambda*c.
    Stop when stopping wins.
    """
    if not ESTIMATORS_FITTED:
        raise RuntimeError(
            "policies.fit_estimators(calibration_tasks) has not been called.\n"
            "  cascade_routing needs calibrated quality estimates and will not\n"
            "  invent them: an uncalibrated quality estimator is a coin flip with\n"
            "  a decimal point."
        )
    lam = CASCADE_ROUTING_LAMBDA if lam is None else lam
    domain = task["domain"]

    remaining = list(models.TIERS)
    cost = 0.0
    latency = 0.0
    calls = []
    best_text = None
    best_q = 0.0
    last_tier = None

    while remaining:
        # Value of each candidate next tier. Quality is the ex-ante estimate on
        # the first step, and the RESCUE conditional afterwards, because by then
        # the relevant question is whether this tier fixes what the last one got
        # wrong.
        options = []
        for tier in remaining:
            q_exante = _Q_EXANTE.get(_exante_key(tier, task), 0.5)
            if last_tier is None:
                q_new = q_exante
            else:
                q_new = _Q_RESCUE.get((tier, last_tier), q_exante)
            # Combined quality if we pay for this tier: we keep what we have, and
            # the new tier saves the remaining probability mass of being wrong.
            q_combined = best_q + (1.0 - best_q) * q_new
            c = _est_cost(tier, task)
            options.append((q_combined - lam * c, tier, q_combined))

        best_option = max(options)
        tau_continue, next_tier, q_after = best_option
        tau_stop = best_q - lam * 0.0  # cost already sunk, so it drops out

        if best_text is not None and tau_stop >= tau_continue:
            break

        r = models.call(next_tier, task)
        cost += r.cost_usd
        latency += r.latency_s
        calls.append(next_tier)
        best_text = r.text
        last_tier = next_tier
        remaining.remove(next_tier)

        # Post-hoc update. Only worth buying if there is somewhere left to go:
        # verifying the top rung is pure cost, exactly as in _cascade.
        if remaining:
            v = VERIFIERS[domain](task, r.text, next_tier)
            cost += v.cost_usd
            latency += v.latency_s
            if v.cost_usd > 0.0:
                calls.extend([next_tier] * (SELF_CONSISTENCY_K - 1))
            best_text = v.answer_text
            # The calibrated meaning of that verdict, NOT the verdict itself. A
            # perfect verifier pushes this to 1.0 or 0.0 and the policy behaves
            # like a cascade; a corrupted one pushes both rows towards the base
            # rate and the policy stops escalating on it, falling back towards a
            # pure router. That transition is the paper's claim, made mechanical.
            best_q = _Q_POSTHOC.get((domain, bool(v.accepted)), q_after)
        else:
            best_q = q_after

    return PolicyResult(
        task_id=task["id"], policy="cascade_routing",
        correct=grade(task, best_text),
        cost_usd=cost, latency_s=latency,
        escalated=len(set(calls)) > 1,
        calls=calls,
    )


def policy_oracle(task):
    """Hindsight-optimal. Not deployable.

    It exists to bound how good ANY router could be. Without it, a small gap
    between routers cannot be read: it might mean the routers are bad, or it
    might mean routing cannot help much on this task set.

    A bound is only a bound if it can reach every action the policies it bounds
    can reach. The oracle therefore enumerates the SAME action space the
    deployable policies have, in cost order:

        1. one greedy answer from each tier on the ladder
        2. majority vote over k cheap samples, math only

    Action 2 is the one that is easy to forget, and omitting it is a real error
    rather than a simplification: `cascade` on math accepts a self-consistency
    answer, so an oracle without action 2 can be BEATEN by the cascade on
    accuracy - at which point it is not bounding anything. It is math-only
    because it is only on math that a policy actually buys those samples; the code
    verifier samples nothing.

    Every rung of models.TIERS is enumerated, so adding a tier automatically
    widens the bound rather than silently leaving the new tier outside it.

    Accounting:
      - if any action is correct, the oracle is correct and pays the CHEAPEST
        correct action, which is what a perfect router would have paid;
      - if none is correct, the oracle is wrong and pays the cheapest action
        overall, since a perfect router that knew every option failed would not
        pay for an expensive one.

    Both halves matter: the first makes it an accuracy ceiling, the second makes
    it a cost floor.
    """
    # (cost, latency, text_to_grade, calls) for each reachable action.
    candidates = []
    cheap = None
    for tier in models.TIERS:
        r = models.call(tier, task)
        if tier == "cheap":
            cheap = r
        candidates.append((r.cost_usd, r.latency_s, r.text, [tier]))

    if task["domain"] == "math" and cheap is not None:
        top, _agreement, sc_cost, sc_latency = _self_consistency(task, cheap.text, "cheap")
        sc_text = f"\\boxed{{{top}}}" if top is not None else cheap.text
        candidates.append((
            cheap.cost_usd + sc_cost,
            cheap.latency_s + sc_latency,
            sc_text,
            ["cheap"] * SELF_CONSISTENCY_K,
        ))

    candidates.sort(key=lambda c: c[0])
    for cost, latency, text, calls in candidates:
        if grade(task, text):
            return PolicyResult(
                task_id=task["id"], policy="oracle", correct=True,
                cost_usd=cost, latency_s=latency, calls=calls,
            )

    cost, latency, _text, calls = candidates[0]
    return PolicyResult(
        task_id=task["id"], policy="oracle", correct=False,
        cost_usd=cost, latency_s=latency, calls=calls,
    )


POLICIES = {
    # One always_<rung> per rung of the loaded ladder, generated rather than
    # listed, so a two-rung ladder does not carry a dead always_mid row and a
    # ladder gaining a rung does not need an edit here.
    **{f"always_{tier}": (lambda t, _tier=tier: policy_always(t, _tier))
       for tier in models.TIERS},
    "random_matched": policy_random_matched,
    "routellm": policy_routellm,
    "llm_router": policy_llm_router,
    "cascade": policy_cascade,
    "cascade_routing": policy_cascade_routing,
    "cascade_degraded": policy_cascade_degraded,
    "oracle": policy_oracle,
}

# Which domains a policy is defined on. An absent entry means all of them.
#
# cascade_degraded is code-only BY DESIGN. Its purpose is to vary verifier
# fidelity while holding the domain fixed, so running it on math - where the
# verifier is already a proxy and the corruption would compound with it - would
# reintroduce exactly the confound it exists to remove.
POLICY_DOMAINS = {
    "cascade_degraded": ("code",),
}

# Policies that need fit_estimators() called on a calibration split before they
# can run. Like routellm, they sit out rather than guess.
NEEDS_ESTIMATORS = ("cascade_routing",)
