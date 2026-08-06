"""The bridge: an arbitrary query becomes something the benchmark can run.

Everything in the research core takes a *task dict* - a row of `taskset.jsonl`
carrying a prompt, a domain, and a `grader_payload` holding the right answer.
That shape is what `models.call` keys its cache on, what `graders.grade` marks
against, and what `policies.py` routes.

A served query has a prompt and nothing else. No answer, no difficulty label,
no asserts. This module manufactures the smallest task dict that the existing
machinery will accept, so that serving reuses the benchmark's model client,
price table, response cache and cost accounting rather than reimplementing
them.

--------------------------------------------------------------------------
The asymmetry this module exists to make visible
--------------------------------------------------------------------------

Two things the benchmark's policies read simply do not exist in production, and
pretending otherwise is how a routing result stops replicating:

**1. Ground truth.** `graders.grade` needs the answer. Serving has none, so
nothing here reports correctness - only `verified`, which is a verifier's
opinion. `verifiers.py` is where that distinction is enforced.

**2. The difficulty label.** `policies.predict_is_hard` routes maths on
`predict_features["level"] >= 5`. That level is MATH500's own annotation,
shipped with the dataset and written by a human who had already solved the
problem. A user's query arrives with no such field and never will.

That second one is worth stating plainly, because it flatters the benchmark:
**the predictive router measured in this repo has an advantage its deployed
counterpart cannot have.** Whatever `predictive` scores in `results.jsonl` is
therefore an upper bound on what a deployed predictive router would score. The
serving heuristic below is honest about being weaker; it reads only the query
text, which is all a real router ever has.

This cuts in the cascade's favour and it is not a thumb on the scale - it is a
real structural difference between the two architectures. A cascade needs no
difficulty label, because it finds out by trying. That is precisely the
property the benchmark's framing is about, and it is the reason `cascade` is
the default policy in `config.py`.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

# Loaded lazily inside functions that need them. Keeping module import free of
# the research core means `live` can be imported by tooling that only wants
# infer_domain, and by the test suite before any ladder is configured.


# ---------------------------------------------------------------------------
# Domain inference
# ---------------------------------------------------------------------------

# Deliberately crude, deliberately transparent, and deliberately NOT a model
# call. A learned classifier here would be a second routing decision hidden
# inside the first one: it would cost money, it would need its own evaluation,
# and its errors would be attributed to the router. The benchmark has a whole
# policy (`llm_router`) measuring what happens when you ask a model to classify
# before answering, and on mock data it is not competitive.
#
# The cost of being wrong is small and bounded: the domain picks a prompt
# template and a default verifier, not a model. A maths question misread as
# `general` still gets answered, just without the \boxed{} protocol and
# verified by agreement over free text rather than over extracted answers.

# Matched on WORD BOUNDARIES, not as substrings. Substring matching looks
# harmless and is not: "api" is inside "capital", so "What is the capital of
# France?" classified as code. Every marker here is short enough to hide inside
# an ordinary English word, so the boundaries are load-bearing.
_CODE_MARKERS = (
    "def", "class", "import", "return", "python", "javascript", "function",
    "regex", "sql", "algorithm", "code", "compile", "debug", "refactor",
    "unit test", "stack trace", "traceback", "api", "typescript", "rust",
    "script", "variable", "array", "recursion",
)

_MATH_MARKERS = (
    "solve", "prove", "compute", "evaluate", "simplify", "integral",
    "derivative", "polynomial", "equation", "theorem", "probability",
    "matrix", "modulo", "factorial", "geometry", "algebra", "calculus",
    "integrate", "differentiate", "sum", "product",
)


def _count_markers(text: str, markers: tuple[str, ...]) -> int:
    """How many distinct markers appear as whole words."""
    return sum(
        1 for m in markers
        if re.search(r"\b" + re.escape(m) + r"\b", text)
    )

# LaTeX, arithmetic operators between digits, and other notation that is
# effectively conclusive for maths.
_MATH_NOTATION = re.compile(
    r"\\[a-zA-Z]+\{|\$[^$]+\$|\b\d+\s*[+\-*/^]\s*\d+|\\frac|\\sqrt|\\int|\\sum"
)

_CODE_FENCE = re.compile(r"```")


def infer_domain(query: str, tests: Iterable[str] | None = None) -> str:
    """Guess the domain from the query text alone.

    Returns "code", "math" or "general". `tests` being present is conclusive:
    a caller who supplied asserts is asking for code, whatever the prose says.
    """
    if tests:
        return "code"

    q = query.lower()

    if _CODE_FENCE.search(query):
        return "code"

    # Notation beats keywords. "Write a function to compute the determinant"
    # hits a maths keyword but is plainly a coding task; "evaluate $\int_0^1
    # x^2 dx$" carries notation no coding request would.
    has_notation = bool(_MATH_NOTATION.search(query))
    code_hits = _count_markers(q, _CODE_MARKERS)
    math_hits = _count_markers(q, _MATH_MARKERS)

    if code_hits and code_hits >= math_hits:
        return "code"
    if has_notation or math_hits:
        return "math"
    return "general"


# ---------------------------------------------------------------------------
# Task synthesis
# ---------------------------------------------------------------------------

def live_task_id(query: str, domain: str) -> str:
    """A stable id derived from the content, not from a counter or a clock.

    This is load-bearing rather than cosmetic. `response_cache` stores the task
    id on every record, and `models.call` writes one cache entry per distinct
    (model, prompt, temperature, sample_idx, max_tokens). A uuid or a timestamp
    here would make every run of the same query look like a different task in
    the cache file, which defeats both auditing and replay.

    Truncated to 12 hex characters: collision-irrelevant at serving volumes,
    and short enough to read in a trace.
    """
    h = hashlib.sha256(f"{domain}\x00{query}".encode("utf-8")).hexdigest()
    return f"live-{h[:12]}"


def synthesize_task(
    query: str,
    domain: str = "auto",
    tests: list[str] | None = None,
) -> dict:
    """Build the minimal task dict `models.call` will accept.

    The result deliberately does NOT carry a `grader_payload["answer"]`. Every
    consumer in this package must therefore treat correctness as unknown, and
    `graders.grade` would raise rather than silently mark a served answer
    against a missing key. That is the intended failure mode: it is better for
    a serving path to crash than to invent a verdict.
    """
    if not query or not query.strip():
        raise ValueError("query is empty")

    if domain == "auto":
        domain = infer_domain(query, tests)
    if domain not in ("math", "code", "general"):
        raise ValueError(
            f"domain must be one of math, code, general, auto - got {domain!r}"
        )

    task = {
        "id": live_task_id(query, domain),
        "domain": domain,
        "prompt": query.strip(),
        # No `grader`, and no answer. Serving has no ground truth; see the
        # module docstring.
        "grader_payload": {},
        # What a deployed router is allowed to look at: the query, and nothing
        # derived from having already solved it. Note the absence of `level` -
        # see the module docstring for why that matters more than it looks.
        "predict_features": {
            "prompt_chars": len(query.strip()),
            "n_asserts": len(tests or []),
        },
        # Marks every record this task writes into the response cache, so a
        # served call can never be mistaken for an evaluation call when the
        # cache file is read back.
        "_live": True,
    }

    if tests:
        task["grader"] = "test_program"
        task["grader_payload"] = {"tests": list(tests)}

    return task


# ---------------------------------------------------------------------------
# Serving-mode difficulty heuristic
# ---------------------------------------------------------------------------

# Chosen to match policies.PREDICTIVE_CODE_CHARS, which the benchmark
# calibrated to put the code half near the maths half's escalation rate. Reused
# rather than re-tuned so that the serving router and the measured one differ
# in exactly one respect - the missing difficulty label - and not in two.
_LONG_PROMPT_CHARS = 100

_HARD_MARKERS = (
    "prove", "derive", "optimi", "why does", "explain why", "trade-off",
    "tradeoff", "design", "architect", "concurren", "distributed",
    "race condition", "deadlock", "complexity", "asymptotic",
)


def predict_is_hard_live(task: dict) -> bool:
    """The deployable predictive heuristic: query text only.

    Weaker than `policies.predict_is_hard`, on purpose and unavoidably. The
    benchmark's version reads MATH500's shipped `level`; this one cannot,
    because a user's question does not come with a difficulty annotation.

    Anyone reporting a number from this function should not compare it to the
    `predictive` row of `results.jsonl` - that row was produced with an oracle
    feature. See the module docstring.
    """
    features = task.get("predict_features", {})

    # If a real level is present the caller has an evaluation task, not a live
    # one, and the benchmark's own predicate is the right thing to use.
    if "level" in features:
        from policies import predict_is_hard
        return predict_is_hard(task)

    prompt = task["prompt"].lower()
    if any(m in prompt for m in _HARD_MARKERS):
        return True
    return features.get("prompt_chars", 0) >= _LONG_PROMPT_CHARS
