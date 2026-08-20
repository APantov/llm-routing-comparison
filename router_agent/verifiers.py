"""Verification without ground truth, which is the only kind serving has.

The benchmark uses two verifiers and the contrast between them is the whole
experiment:

    verify_code   run the shipped asserts        free, and perfect
    verify_math   self-consistency over k draws  costs k calls, and is a proxy

Exactly one survives the move into production.

**`verify_math` transfers unchanged.** Self-consistency asks the model the same
question k times and measures self-agreement; it never consults an answer key,
because there isn't one. Everything the benchmark measured applies directly.

**`verify_code` does not.** "Run the tests" is free and perfect only because
MBPP+ ships them, and a user asking for a function supplies no asserts. This is
the biggest gap between the experiment and the product, and `cascade_degraded`
is what priced it: corrupt `verify_code` by a controlled amount and watch
cascade quality fall, which is the curve you slide down when you lose the tests.

So: the cascade's production economics are governed by which verifier you can
actually obtain, and for most workloads that is the proxy.

Each verifier returns a `Check`, which has no `correct` field. A verifier's
opinion is `accepted`, and conflating the two is how a serving layer starts
reporting accuracy it never measured.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from llm_routing import models
from llm_routing.graders import extract_answer

from router_agent import pricing
from router_agent.config import RouterConfig


@dataclass
class Check:
    """A verifier's verdict on one candidate answer."""

    accepted: bool
    """Whether the cascade should stop here. NOT a correctness claim."""

    answer_text: str
    """
    The answer to return if accepted, which is not always the one passed in.
    A self-consistency verifier that paid for k samples has a better answer
    available than the greedy draw it was handed - the plurality one - and
    returning it costs nothing. Mirrors policies.Verdict.answer_text.
    """

    confidence: float | None
    """
    Agreement ratio in [0, 1] where the method produces one, else None.
    `tests` returns None rather than 1.0: passing tests is a pass/fail signal,
    and reporting it as a probability would invite averaging it against
    agreement ratios, which are a different quantity.
    """

    method: str
    cost_usd: float = 0.0
    """What verification costs a policy - charged even on a cache hit."""

    backend_cost_usd: float = 0.0
    """The part of `cost_usd` that actually reached a provider. Zero in mock
    and replay; on a real run it is whichever draws were not already cached."""

    latency_s: float = 0.0
    detail: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Answer extraction, per domain
# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)
_WS = re.compile(r"\s+")


def _canonical_answer(text: str, domain: str) -> str | None:
    """Reduce a response to the string whose repetition means "agreement".

    Self-consistency is only as good as this function. Compare raw responses
    and nothing ever agrees, because the prose around the answer varies on
    every draw; compare too loosely and everything agrees.
    """
    if not text or not text.strip():
        return None

    if domain == "math":
        # The \boxed{} protocol, and the same extractor the benchmark grades
        # with - so a served maths answer is canonicalised exactly as an
        # evaluated one is.
        return extract_answer(text)

    if domain == "code":
        m = _CODE_BLOCK.search(text)
        body = m.group(1) if m else text
        # Whitespace-insensitive, comment-insensitive. Two draws that differ
        # only in indentation or in a comment are the same program, and
        # counting them as disagreement would make the verifier escalate on
        # formatting noise.
        lines = []
        for line in body.splitlines():
            stripped = line.split("#")[0].strip()
            if stripped:
                lines.append(stripped)
        return _WS.sub(" ", " ".join(lines)) or None

    # general: exact agreement on free text is hopeless - two correct
    # summaries of the same email share almost no substrings. Normalising case
    # and whitespace is the most that can be justified without a similarity
    # model, and the docstring for verify_self_consistency says plainly that
    # the signal is weak here.
    return _WS.sub(" ", text.strip().lower()) or None


# ---------------------------------------------------------------------------
# The verifiers
# ---------------------------------------------------------------------------

def verify_none(task, response_text, tier, cfg) -> Check:
    """Accept everything. The degenerate case, and a real baseline.

    This is what predictive routing does: commit to one model and never find
    out whether it was right. It is included as a verifier rather than special
    cased so that `predictive` and `cascade` differ in exactly one component,
    which is what makes their costs comparable.
    """
    return Check(
        accepted=True,
        answer_text=response_text,
        confidence=None,
        method="none",
        detail={"note": "no verification performed; answer accepted as-is"},
    )


def verify_tests(task, response_text, tier, cfg) -> Check:
    """Execute caller-supplied tests. The perfect verifier, when obtainable.

    Free in the sense that matters - no model calls - and exact. It is also the
    verifier most workloads cannot have, because it requires the caller to know
    the tests before seeing the answer.

    SECURITY: this runs model-generated code in a subprocess with no sandbox.
    Gated behind `cfg.allow_code_execution`, off by default. See
    RouterConfig.allow_code_execution.
    """
    tests = task.get("grader_payload", {}).get("tests")
    if not tests:
        raise ValueError(
            "the `tests` verifier needs tests; pass them with the query or "
            "select a different verifier"
        )
    if not cfg.allow_code_execution:
        raise PermissionError(
            "the `tests` verifier executes model-generated code in a "
            "subprocess and is disabled by default. Enable it only in a "
            "sandboxed environment, with allow_code_execution=True or "
            "ROUTER_ALLOW_CODE_EXEC=1."
        )

    # Caller-supplied tests are always an assert list, so `run_asserts` is the
    # right grader. `grade_test_program` is the MBPP+ path and expects a
    # complete self-contained program under `test_program`, which a serving
    # caller has no way to supply.
    from llm_routing.graders import grade_run_asserts

    try:
        passed = grade_run_asserts(response_text, task["grader_payload"])
    except Exception as exc:  # a broken test program is not a wrong answer
        return Check(
            accepted=False,
            answer_text=response_text,
            confidence=None,
            method="tests",
            detail={"error": f"{type(exc).__name__}: {exc}"},
        )

    return Check(
        accepted=bool(passed),
        answer_text=response_text,
        confidence=None,
        method="tests",
        detail={"n_tests": len(tests), "passed": bool(passed)},
    )


def verify_self_consistency(task, response_text, tier, cfg) -> Check:
    """Resample k times and measure agreement. The proxy verifier.

    Mirrors policies._self_consistency, including the two decisions in it that
    are easy to get wrong:

    * **Distinct sample indices.** `models.call` keys its cache partly on
      `sample_idx`, so drawing k samples at the same index returns the same
      cached response k times and reports unanimous agreement on everything.
    * **Unparseable draws count against agreement.** The denominator is k, not
      the number of parseable draws. A model that cannot produce a readable
      answer is not one to be confident in. It does conflate "disagreed with
      itself" with "the parser failed", which is a known wart rather than a
      hidden one.

    One thing here is NOT inherited from the benchmark. `policies` returns
    `None` agreement when a tier refuses a temperature, because in the
    experiment that would silently produce k identical greedy draws and a
    verifier that always accepts. Serving cannot return "unknown" and stop, so
    this escalates instead: unverifiable is treated as unverified. On the
    `claude` and `wide` ladders the top rungs reject `temperature` outright,
    so this branch is reached in normal operation, not only in theory.
    """
    spec = models.MODELS[tier]
    if not spec["accepts_temperature"]:
        return Check(
            accepted=False,
            answer_text=response_text,
            confidence=None,
            method="self_consistency",
            detail={
                "note": (
                    f"{spec['id']} does not accept a temperature, so it cannot "
                    f"be resampled; treating as unverified"
                ),
                "unverifiable": True,
            },
        )

    domain = task["domain"]
    # (canonical form, the text that produced it). The text is kept so the
    # plurality answer can actually be returned - see below.
    draws: list[tuple[str | None, str]] = [
        (_canonical_answer(response_text, domain), response_text)
    ]
    cost = 0.0
    backend_cost = 0.0
    latency = 0.0

    # Index 0 is the greedy call the caller already made and passed in.
    for i in range(1, cfg.self_consistency_k):
        try:
            r, call_backend_cost = pricing.call_tracked(
                tier, task, temperature=0.8, sample_idx=i
            )
            backend_cost += call_backend_cost
        except KeyError as exc:
            # Replay mode, and this draw was never paid for: the two-arm probe
            # bought one greedy call per task and no temperature samples, so a
            # cascade replayed over probe data reaches this on the first extra
            # draw.
            #
            # Escalating is the honest response - verification did not happen, so
            # the answer is unverified, and that is what a cascade escalates on.
            # Accepting would invent confidence that was never measured.
            return Check(
                accepted=False,
                answer_text=response_text,
                confidence=None,
                method="self_consistency",
                cost_usd=cost,
                backend_cost_usd=backend_cost,
                latency_s=latency,
                detail={
                    "note": (
                        "replay mode has no cached sample for this draw, so "
                        "agreement could not be measured; treating as "
                        "unverified"
                    ),
                    "unverifiable": True,
                    "draws_available": len(draws),
                    "cache_miss": str(exc).split("\n")[0],
                },
            )
        draws.append((_canonical_answer(r.text, domain), r.text))
        cost += r.cost_usd
        latency += r.latency_s

    answers = [c for c, _ in draws]
    counts = Counter(a for a in answers if a is not None)
    if not counts:
        return Check(
            accepted=False, answer_text=response_text, confidence=0.0,
            method="self_consistency", cost_usd=cost,
            backend_cost_usd=backend_cost, latency_s=latency,
            detail={"note": "no draw produced a parseable answer", "k": len(answers)},
        )

    top, top_count = counts.most_common(1)[0]
    agreement = top_count / len(answers)

    # Return the text of a draw that produced the plurality answer, not the
    # greedy one, when they differ. Same rationale as policies.verify_math:
    # having paid for the samples, the better answer is free. The first
    # matching draw wins, and since the greedy draw is inserted first, a greedy
    # answer that is already in the plurality is preferred - which keeps the
    # common case byte-identical to not resampling at all.
    best_text = next(text for canon, text in draws if canon == top)

    return Check(
        accepted=agreement >= cfg.agreement_threshold,
        answer_text=best_text,
        confidence=agreement,
        method="self_consistency",
        cost_usd=cost,
        backend_cost_usd=backend_cost,
        latency_s=latency,
        detail={
            "k": len(answers),
            "agreement": round(agreement, 4),
            "threshold": cfg.agreement_threshold,
            "plurality_answer": top if domain == "math" else None,
            "distinct_answers": len(counts),
        },
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

VERIFIERS: dict[str, Callable] = {
    "none": verify_none,
    "tests": verify_tests,
    "self_consistency": verify_self_consistency,
}


def select(task: dict, cfg: RouterConfig) -> str:
    """Resolve `verifier="auto"` to a concrete verifier name.

    The table, and the reason for each row:

        one-shot policy                         -> none
            `predictive`, `always_cheap` and `always_expensive` commit to a
            rung and never escalate, so verification could not change what they
            return. Running it anyway would charge them the cascade's fixed
            cost while giving them none of its benefit - and since the cost
            comparison between cascading and routing is the entire subject of
            this repository, that would invalidate the one number that matters.
            This is the fixed term that decides whether a cascade is cheaper
            on a given ladder; it belongs only to policies that can act on it.
        code + caller tests + execution allowed -> tests
            The perfect verifier. Take it whenever it is on offer.
        code + tests but execution disabled     -> self_consistency
            Falling back rather than refusing, because the safe default must
            still answer the query. The trace records the downgrade.
        anything else                           -> self_consistency
            The proxy. What most production traffic actually gets.

    An explicit `cfg.verifier` always wins, including asking a one-shot policy
    to verify. That combination is not nonsense - it measures confidence
    without acting on it - but it has to be requested rather than inferred.
    """
    if cfg.verifier != "auto":
        return cfg.verifier

    if cfg.policy in ("predictive", "always_cheap", "always_expensive"):
        return "none"

    has_tests = bool(task.get("grader_payload", {}).get("tests"))
    if has_tests and cfg.allow_code_execution:
        return "tests"
    return "self_consistency"


def run(name: str, task, response_text: str, tier: str, cfg: RouterConfig) -> Check:
    try:
        fn = VERIFIERS[name]
    except KeyError:
        raise ValueError(
            f"unknown verifier {name!r}; expected one of {sorted(VERIFIERS)}"
        )
    return fn(task, response_text, tier, cfg)
