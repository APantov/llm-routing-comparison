"""
Objective graders. There is no LLM judge anywhere in this project.

Every task is scored deterministically, which buys three things:

  - no judge to calibrate, and no judge bias to argue about
  - no golden set to hand-label
  - byte-for-byte reproducible verdicts

That requirement is the single biggest reason the two task domains were chosen
as they were: MATH500 answers can be compared as normalised strings, and MBPP
solutions can be executed against shipped asserts.

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

# ---------------------------------------------------------------------------
# Memoisation of grade().
#
# grade(task, response) is a pure function, so grading the same response twice
# cannot produce two different verdicts. Two places do exactly that, and both
# are hot:
#
#   1. The code cascade grades once inside verify_code and once again in the
#      cascade after acceptance, launching the subprocess grader twice per task
#      for a verdict that cannot differ.
#   2. The degradation sweep replays the same cheap responses at every
#      corruption level, for every repeat. Un-memoised that is tens of
#      thousands of subprocess launches to re-derive a few hundred distinct
#      answers.
#
# The assumption underneath: candidate code is deterministic. It holds here
# because MBPP reference solutions are pure functions and the mock emits either
# the reference solution or a stub. A real model that returned code reading the
# clock or calling random() would have its flakiness hidden by this memo rather
# than surfaced, so the memo can be switched off:
#
#     ROUTER_GRADE_MEMO=0 python3 run_eval.py
# ---------------------------------------------------------------------------
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


def normalize_math_answer(s):
    r"""Light normalisation so formatting differences are not scored as wrong
    answers.

    Deliberately conservative. It collapses notation that is universally
    equivalent (\dfrac vs \frac, spacing, \left/\right) and does NOT attempt
    algebraic equivalence: 1/2 and 0.5 stay different, because deciding they
    are the same is a solver's job rather than a grader's.
    """
    if s is None:
        return None
    s = s.strip()
    s = s.replace("$", "").replace(r"\!", "").replace(r"\,", "").replace(r"\;", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = re.sub(r"\^\{?\\circ\}?", "", s)                 # 90^\circ -> 90
    s = s.replace(r"\%", "").replace("%", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)           # \text{cm} -> cm
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)  # \frac34 -> \frac{3}{4}
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    if re.fullmatch(r"-?[\d,]+", s):                     # 1,000 -> 1000, but not (6,31,-1)
        s = s.replace(",", "")
    s = re.sub(r"^0+(\d)", r"\1", s)                     # 007 -> 7
    return s


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
    return got == normalize_math_answer(payload["answer"])


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
    the code half now needs numpy installed. See fetch_mbppplus.py.
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
                # The expanded suites run far more cases than the originals, so
                # they get more room. Measured at about 0.1s each on the shipped
                # tasks; the headroom is for pathological generated code rather
                # than for the tests themselves.
                timeout=CODE_TIMEOUT_S * 3,
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
