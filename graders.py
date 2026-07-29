"""
Objective graders. No LLM judge anywhere in this project.

Every task is scored deterministically, which means:
  - no judge calibration problem
  - no golden-set hand-labelling
  - results are reproducible byte-for-byte

This is the single biggest reason the task domains were chosen as they were.
"""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CODE_TIMEOUT_S = 10

# ---------------------------------------------------------------------------
# Memoisation of grade().
#
# grade(task, response) is a pure function, so grading the same response twice
# can only produce the same verdict. Two places do exactly that, both of them
# hot:
#
#   1. The code cascade grades once inside verify_code and once again in the
#      cascade after acceptance, launching the subprocess grader twice per task
#      for a verdict that cannot differ.
#   2. The degradation sweep replays the same 40 cheap responses at 6 corruption
#      levels x N repeats. Un-memoised that is tens of thousands of subprocess
#      launches to re-derive a few hundred distinct answers.
#
# The assumption this rests on: candidate code is deterministic. It holds here
# because MBPP reference solutions are pure functions and the mock emits either
# the reference solution or a stub. If a real model ever returns code that reads
# the clock or calls random(), this memo would hide the flakiness rather than
# surface it - so it can be switched off.
#
#     ROUTER_GRADE_MEMO=0 python3 run_eval.py
# ---------------------------------------------------------------------------
MEMO_ENABLED = os.environ.get("ROUTER_GRADE_MEMO", "1") not in ("0", "false", "no")
_memo = {}
memo_stats = {"lookups": 0, "hits": 0}


def extract_final_int(text: str):
    """Pull the model's final integer answer out of free-form reasoning.

    Deliberately permissive about format and strict about position: we take
    the LAST number in the response, because models reason forward and state
    the answer at the end. A stricter parser would penalise formatting rather
    than correctness, which would contaminate the routing signal we care about.
    """
    if text is None:
        return None
    cleaned = text.replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not nums:
        return None
    val = nums[-1]
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def grade_exact_match_int(response: str, payload: dict) -> bool:
    got = extract_final_int(response)
    if got is None:
        return False
    return got == payload["answer"]


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
    return None  # unbalanced


def extract_answer(text: str):
    r"""Pull the model's final answer out of free-form reasoning.

    Same philosophy as extract_final_int - permissive about format, strict
    about position - but the answer is a STRING, because MATH500 answers are
    fractions, radicals, complex numbers and tuples, not just integers.

    Prefers the last \boxed{}, which the prompt asks for. Falls back to the
    last non-empty line so that a model which ignores the format instruction
    is penalised for being wrong, not for being untidy.
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


def normalize_math_answer(s: str) -> str:
    r"""Light normalisation so that formatting differences are not scored as
    wrong answers. Deliberately conservative: it collapses notation that is
    universally equivalent (\dfrac vs \frac, spacing, \left/\right) and does
    NOT attempt algebraic equivalence. 1/2 and 0.5 stay different, because
    deciding they are the same is a solver's job, not a grader's.
    """
    if s is None:
        return None
    s = s.strip()
    s = s.replace("$", "").replace(r"\!", "").replace(r"\,", "").replace(r"\;", "")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = re.sub(r"\^\{?\\circ\}?", "", s)          # 90^\circ -> 90
    s = s.replace(r"\%", "").replace("%", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)    # \text{cm} -> cm
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)  # \frac34 -> \frac{3}{4}
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    if re.fullmatch(r"-?[\d,]+", s):             # 1,000 -> 1000, but not (6,31,-1)
        s = s.replace(",", "")
    s = re.sub(r"^0+(\d)", r"\1", s)             # 007 -> 7
    return s


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

    Runs in a subprocess with a timeout so an infinite loop in generated
    code cannot hang the eval run. This is the free, perfect verifier that
    the code domain gets and the math domain does not.
    """
    code = _strip_code_fences(response)
    parts = [code, payload.get("setup", "") or ""]
    parts.extend(payload["tests"])
    program = "\n".join(parts)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        path.write_text(program)
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                timeout=CODE_TIMEOUT_S,
            )
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


GRADERS = {
    "exact_match_int": grade_exact_match_int,
    "exact_match_str": grade_exact_match_str,
    "run_asserts": grade_run_asserts,
}


def grade(task: dict, response: str) -> bool:
    if not MEMO_ENABLED:
        return GRADERS[task["grader"]](response, task["grader_payload"])
    # Keyed on the task id AND the response, because the payload (the asserts,
    # the expected answer) belongs to the task. Hashing the response rather than
    # storing it keeps the memo small when responses are long.
    key = (
        task["id"],
        task["grader"],
        hashlib.sha1((response or "").encode("utf-8")).hexdigest(),
    )
    memo_stats["lookups"] += 1
    if key in _memo:
        memo_stats["hits"] += 1
        return _memo[key]
    verdict = GRADERS[task["grader"]](response, task["grader_payload"])
    _memo[key] = verdict
    return verdict
