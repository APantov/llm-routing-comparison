"""
Objective graders. There is no LLM judge anywhere in this project.

Every task is scored deterministically: no judge to calibrate, no golden set to
hand-label, byte-for-byte reproducible verdicts. That requirement is the biggest
reason the two domains are what they are - MATH500 answers compare as normalised
strings, and MBPP solutions execute against shipped asserts.

Two graders are registered:

    exact_match_str   math  - compare the normalised final answer
    run_asserts       code  - execute the candidate against MBPP's asserts
"""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Generated code runs in a subprocess. The timeout is the only thing standing
# between an accidental infinite loop and a hung evaluation run.
CODE_TIMEOUT_S = 10

# The MBPP+ path gets its own budget rather than a multiple of the one above,
# because the two are sized against different things. CODE_TIMEOUT_S guards
# against generated code that loops forever; this one has to fit the SHIPPED
# TEST SUITE, which is a measured quantity and much larger than it looks.
#
# codeplus-599 is the worst case by a wide margin. Its reference solution is a
# three-line `sum(range(1, number+1))`, but the expanded suite calls it on
# inputs up to 100,000,007 across 90 cases - about 1.6 BILLION interpreted loop
# iterations for one task, ~14s on a fast machine where the other 356 tasks
# average well under a second.
#
# The old budget was CODE_TIMEOUT_S * 3, and its comment claimed the tests ran
# "about 0.1s each" and that the headroom existed "for pathological generated
# code rather than for the tests themselves". That was false: at 30s the
# repo's own REFERENCE solution had 2.1x headroom, against 5.9x for the
# next-slowest task and 18x for the remaining 351. A timeout returns False,
# which is indistinguishable from a wrong answer, so a slow runner would not
# have reported "too slow" - it would have reported the reference solution
# failing its own tests, and every accuracy number in the repo with it.
#
# 90s restores ~6x on the worst case, which is the headroom the rest of the set
# already had.
TEST_PROGRAM_TIMEOUT_S = 90

# Memoisation of grade(), which is pure - the same response cannot grade two
# ways. Two hot paths grade twice: the code cascade grades inside verify_code and
# again after acceptance, and the degradation sweep replays the same cheap
# responses at every corruption level and repeat (tens of thousands of subprocess
# launches for a few hundred distinct answers).
#
# It assumes candidate code is deterministic, which holds here because MBPP
# references are pure functions and the mock emits the reference or a stub. Code
# that read the clock or called random() would have its flakiness hidden rather
# than surfaced, so the memo can be switched off:
#
#     ROUTER_GRADE_MEMO=0 python -m llm_routing.run_eval
MEMO_ENABLED = os.environ.get("ROUTER_GRADE_MEMO", "1") not in ("0", "false", "no")
_memo = {}


def _last_boxed(text: str):
    r"""Return the contents of the LAST \boxed{...}, or None.

    Brace-matched rather than regex-matched: MATH answers nest freely
    (\boxed{\frac{1}{2}}), and a non-greedy regex stops at the first inner
    closing brace while a greedy one swallows trailing prose.
    """
    idx = text.rfind(r"\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None  # unbalanced braces


def _canon_latex(s):
    r"""Give \frac, \sqrt, ^ and _ arguments their braces, so \sqrt5 == \sqrt{5}
    and x^2 == x^{2}.

    LaTeX lets a single-token argument drop its braces, and models use both
    forms interchangeably within one answer. MATH500's stored answers do too:
    `\frac{270}7` is in the data as written, and `8n^2 + 4n + 1` is math-481's
    shipped answer while `8n^{2}+4n+1` is an equally natural way to write it.

    Canonicalising TOWARDS the braced form is the safe direction. Stripping
    braces instead would merge `2^{10}` with `2^10`, and those are genuinely
    different in LaTeX - `2^10` is 2^1 followed by a 0. Adding braces to a
    single token cannot merge anything that was not already identical, so
    `2^{10}` and `2^10` correctly stay apart.

    Only bare alphanumerics get braces here. `x^\pi` is left alone: deciding
    where a backslash command ends needs a parser, and `^\circ` is already
    stripped by normalize_math_answer before this runs.

    The superscript case was missed by the first grader fix, which caught
    \sqrt and \frac but not ^ and _. See docs/LIMITATIONS.md.
    """
    s = re.sub(r"\\sqrt(?!\{)(\\?[A-Za-z0-9])", r"\\sqrt{\1}", s)
    # Two passes so \frac{270}7 and \frac3{4} are both reached.
    for _ in range(2):
        s = re.sub(r"\\frac(?!\{)(\\?[A-Za-z0-9])", r"\\frac{\1}", s)
        s = re.sub(r"(\\frac\{[^{}]*\})(?!\{)(\\?[A-Za-z0-9])", r"\1{\2}", s)
    s = re.sub(r"([\^_])(?!\{)([A-Za-z0-9])", r"\1{\2}", s)
    return s


def normalize_math_answer(s):
    r"""Light normalisation so formatting differences are not scored as wrong
    answers.

    Deliberately conservative. It collapses notation that is universally
    equivalent (\dfrac vs \frac, spacing, \left/\right) and does NOT attempt
    algebraic equivalence: 1/2 and 0.5 stay different, because deciding they
    are the same is a solver's job rather than a grader's.

    This produces ONE canonical string. Equivalences that are structural rather
    than notational - \pm, set ordering, units, `LHS = answer` - are handled by
    answer_variants() instead, because they need more than a rewrite.
    """
    if s is None:
        return None
    s = s.strip()
    # \$ before $: MATH500 writes currency as \$18.90, and stripping the dollar
    # alone leaves the escaping backslash behind. That bug graded a correct
    # `18.90` as wrong.
    s = s.replace(r"\$", "").replace("$", "")
    s = s.replace(r"\!", "").replace(r"\,", "").replace(r"\;", "").replace(r"\ ", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = re.sub(r"\^\{?\\circ\}?", "", s)                 # 90^\circ -> 90
    s = s.replace(r"\%", "").replace("%", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)           # \text{cm} -> cm
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s)
    s = _canon_latex(s)
    s = s.rstrip(".")
    if re.fullmatch(r"-?[\d,]+", s):                     # 1,000 -> 1000, but not (6,31,-1)
        s = s.replace(",", "")
    s = re.sub(r"^0+(\d)", r"\1", s)                     # 007 -> 7
    return s


def _split_top(s, sep=","):
    """Split on `sep`, ignoring separators nested inside brackets."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [p for p in out if p]


def _expand_pm(part):
    r"""`3\pm2\sqrt2` -> the two values it stands for.

    Only the first \pm is expanded. A second one would mean four values and a
    convention about whether the signs are linked, and nothing in MATH500 needs
    it - guessing there would risk accepting a wrong answer.
    """
    i = part.find(r"\pm")
    if i == -1:
        return [part]
    lhs, rhs = part[:i], part[i + 3:]
    if not rhs:
        return [part]
    return [f"{lhs}+{rhs}", f"{lhs}-{rhs}"] if lhs else [rhs, f"-{rhs}"]


# Answers whose elements are UNORDERED. A parenthesised tuple is deliberately
# absent: (6,31,-1) is a coordinate, and sorting it would accept a wrong answer.
def _as_unordered_set(s):
    r"""Canonical form for a set answer, or None if it is not one.

    Handles \{a,b\} and a bare `a,b` list. Expands \pm inside each element
    first, so `\{1\pm\sqrt5,-2\}` and `\{-2,1+\sqrt5,1-\sqrt5\}` agree.
    """
    inner, braced = s, False
    if s.startswith(r"\{") and s.endswith(r"\}"):
        inner, braced = s[2:-2], True
    elif s.startswith("{") and s.endswith("}"):
        inner, braced = s[1:-1], True

    parts = _split_top(inner)
    # A bare `1\pm\sqrt{19}` is a two-element set with no comma in it, so the
    # \pm has to be enough on its own to qualify.
    if len(parts) < 2 and r"\pm" not in inner:
        return None
    if not braced and s.startswith("(") and s.endswith(")"):
        return None  # ordered tuple

    expanded = []
    for p in parts:
        expanded.extend(_expand_pm(p))
    if len(expanded) < 2:
        return None
    return "SET(" + "|".join(sorted(_canon_latex(e) for e in expanded)) + ")"


_INEQUALITY = re.compile(
    r"^(-?[\d.]+)(<|\\le|\\leq)([A-Za-z]|\\[A-Za-z]+)(<|\\le|\\leq)(-?[\d.]+)$"
)


def _as_interval(s):
    r"""`3<\lambda\le4` -> `(3,4]`, so a range reads the same either way."""
    m = _INEQUALITY.match(s)
    if not m:
        return None
    lo, lo_op, _var, hi_op, hi = m.groups()
    return f"{'(' if lo_op == '<' else '['}{lo},{hi}{')' if hi_op == '<' else ']'}"


def answer_variants(s):
    r"""Every form of `s` that means the same thing, as a set of strings.

    Two answers match when their variant sets intersect. Variants are only ever
    ADDED, never substituted, so the plain normalised string is always present
    and nothing that matched before can stop matching.

    Each rule below was put here by a specific real failure in the two-arm
    probe, where the grader scored a correct answer as wrong:

      \pm expansion     `1\pm\sqrt{19}` vs `1+\sqrt{19}, 1-\sqrt{19}`
      set ordering      `\{1\pm\sqrt5,-2\}` vs `\{-2,1+\sqrt5,1-\sqrt5\}`
      trailing units    `\frac{270}7\text{ degrees}` vs `\frac{270}{7}`
      `LHS = answer`    `3R^2` vs `AF^2+BF^2+CF^2 = 3R^2`
      interval notation `(3,4]` vs `3<\lambda\le4`

    What it still refuses to do is algebra. `1/2` and `0.5` remain different,
    and so do `2\sqrt{113}` and `4\sqrt{29}` - which are genuinely different
    numbers that a laxer grader might have been tempted to reconcile.
    """
    if s is None:
        return frozenset()
    out = {s}

    # `AF^2+BF^2+CF^2=3R^2` -> `3R^2`. Only for a single top-level `=`, so an
    # answer that IS an equation is left alone.
    parts = _split_top(s, "=")
    if len(parts) == 2 and all(parts):
        out.add(parts[1])

    # Trailing unit word: `4\sqrt{29}feet` -> `4\sqrt{29}`. Requires something
    # non-alphabetic to remain, so `even` or `\text{yes}` survives intact.
    for v in list(out):
        m = re.match(r"^(.*?[\d}\)])([A-Za-z]{2,})$", v)
        if m:
            out.add(m.group(1))

    for v in list(out):
        as_set = _as_unordered_set(v)
        if as_set:
            out.add(as_set)
        interval = _as_interval(v)
        if interval:
            out.add(interval)

    return frozenset(out)


def extract_answer(text: str):
    r"""Pull the model's final answer out of free-form reasoning.

    Permissive about format, strict about position. The answer is returned as a
    STRING rather than a number, because MATH500 answers are fractions,
    radicals, complex numbers and tuples as often as they are integers.

    Prefers the last \boxed{}, which the prompt asks for. Falls back to the
    last non-empty line, so a model that ignores the format instruction is
    penalised for being wrong rather than for being untidy.
    """
    if text is None:
        return None
    boxed = _last_boxed(text)
    if boxed is not None:
        return normalize_math_answer(boxed)
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None
    last = re.sub(r"^.*?(?:answer(?:\s+is)?|answer)\s*[:=]?\s*", "", lines[-1], flags=re.I)
    return normalize_math_answer(last)


def grade_exact_match_str(response: str, payload: dict) -> bool:
    got = extract_answer(response)
    if got is None:
        return False
    want = normalize_math_answer(payload["answer"])
    if got == want:
        return True
    # Structural equivalences - see answer_variants. Checked only after the
    # plain comparison fails, so the common case stays a string compare.
    return bool(answer_variants(got) & answer_variants(want))


def _strip_code_fences(text: str) -> str:
    if text is None:
        return ""
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return fence.group(1) if fence else text


def grade_run_asserts(response: str, payload: dict) -> bool:
    """Execute the candidate solution against MBPP's assert list.

    Runs in a subprocess with a timeout so an infinite loop in generated code
    cannot hang the evaluation. This is the free, perfect verifier the code
    domain gets and the math domain does not.
    """
    code = _strip_code_fences(response)
    parts = [code, payload.get("setup", "") or ""]
    parts.extend(payload["tests"])
    program = "\n".join(parts)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        # Explicit encoding: MBPP prompts and solutions contain non-ASCII, and
        # the default encoding differs between Windows and Linux, which would
        # make the grader's verdict platform-dependent.
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                timeout=CODE_TIMEOUT_S,
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except OSError:
            return False


def grade_test_program(response: str, payload: dict) -> bool:
    """Execute the candidate against a self-contained test PROGRAM.

    This is the MBPP+ path. Where `run_asserts` is handed a list of assert
    statements to append, this is handed a complete program that defines its own
    comparison helper and loops over a hundred or so input cases. The candidate
    code goes first, the program follows, and a zero exit status means every case
    passed.

    Why not reduce MBPP+ to an assert list and reuse the other grader: the
    expanded suites compare floats with `np.allclose` rather than `==`, so
    flattening them would change what "correct" means. A benchmark modified to fit
    the harness is no longer the benchmark.

    Consequence worth stating plainly: these programs `import numpy`, so grading
    the code half now needs numpy installed. See scripts/provenance/fetch_mbppplus.py.
    """
    code = _strip_code_fences(response)
    program = code + "\n" + payload["test_program"]

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                # See TEST_PROGRAM_TIMEOUT_S: sized against the slowest
                # shipped suite, not against a multiple of the assert budget.
                timeout=TEST_PROGRAM_TIMEOUT_S,
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except OSError:
            return False


GRADERS = {
    "exact_match_str": grade_exact_match_str,
    "run_asserts": grade_run_asserts,
    "test_program": grade_test_program,
}


def grade(task: dict, response: str) -> bool:
    """Score one response against one task. Memoised; see MEMO_ENABLED."""
    if not MEMO_ENABLED:
        return GRADERS[task["grader"]](response, task["grader_payload"])
    # Keyed on the task id AND the response, because the payload (the asserts,
    # the expected answer) belongs to the task. The response is hashed rather
    # than stored, which keeps the memo small when responses are long.
    key = (
        task["id"],
        task["grader"],
        hashlib.sha1((response or "").encode("utf-8")).hexdigest(),
    )
    if key in _memo:
        return _memo[key]
    verdict = GRADERS[task["grader"]](response, task["grader_payload"])
    _memo[key] = verdict
    return verdict
