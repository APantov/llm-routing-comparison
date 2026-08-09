"""Prove the graders can score known-good answers.

A grader that cannot score the reference answer is broken, and every number
downstream of it is meaningless. This is the repo's regression gate: run it after
any change to graders.py, to build_taskset.py, or to the taskset schema.

Every count must come out full. Exits non-zero otherwise, so it can be wired
into a pre-commit hook or CI.

    python3 sanity_check.py

THREE checks, and the third exists because the first two are not enough. Feeding
the ground-truth answer back into \\boxed{} only ever tests grade(GT, GT), which
passes by construction however broken the normaliser is. On 6 August 2026 this
file printed 60/60 while the maths grader was rejecting `1+\\sqrt{19},1-\\sqrt{19}`
against a ground truth of `1\\pm\\sqrt{19}` - seven such false negatives in one
100-task probe, five of them on the expensive model, costing it five accuracy
points that were reported as a capability result.

So EQUIVALENT and DISTINCT below are the real gate on graders.normalize /
answer_variants. DISTINCT matters as much as EQUIVALENT: a grader that returns
True for everything passes EQUIVALENT perfectly.
"""

import json
import sys
from pathlib import Path

from graders import grade

HERE = Path(__file__).parent

# (ground truth, a DIFFERENTLY FORMATTED but correct answer). Must grade True.
# Every pair here is a real case observed in the 6 August 2026 probe.
EQUIVALENT = [
    (r"1 \pm \sqrt{19}", r"1+\sqrt{19},\ 1-\sqrt{19}"),
    (r"3 \pm 2 \sqrt{2}", r"3+2\sqrt{2},\; 3-2\sqrt{2}"),
    (r"\{1\pm\sqrt{5},-2\}", r"\{-2,\ 1+\sqrt5,\ 1-\sqrt5\}"),
    (r"\frac{270}7\text{ degrees}", r"\frac{270}{7}"),
    (r"\$18.90", r"18.90"),
    (r"3R^2", r"AF^2+BF^2+CF^2 = 3R^2"),
    (r"(3,4]", r"3 < \lambda \le 4"),
    # Redundant braces on superscripts and subscripts. The 6 August fix caught
    # \sqrt and \frac but not ^ and _, so this class survived it; found by audit
    # on 8 August 2026. math-481's shipped answer is literally `8n^2 + 4n + 1`,
    # and `8n^{2}+4n+1` is an equally natural way for a model to write it.
    (r"8n^2 + 4n + 1", r"8n^{2}+4n+1"),
    (r"3R^2", r"3R^{2}"),
    (r"a_1", r"a_{1}"),
    # Notational cases the normaliser already handled - kept so a rewrite of it
    # cannot silently lose them.
    (r"\frac{1}{2}", r"\dfrac{1}{2}"),
    (r"90", r"90^\circ"),
    (r"\sqrt{2}", r"\sqrt2"),
]

# (ground truth, a WRONG answer that looks superficially close). Must grade
# False. This is the half that catches over-lenience.
DISTINCT = [
    (r"2\sqrt{113}", r"4\sqrt{29}"),          # 21.26 vs 21.54 - both were guessed
    (r"2\sqrt{113}", r"2\sqrt{61}"),
    (r"\frac{270}7\text{ degrees}", r"\frac{990}{7}"),
    (r"331", r"\frac{331}{3}"),
    (r"8n^2 + 4n + 1", r"\frac{13}{8n^2 + 4n + 1}"),
    (r"144", r"288"),
    (r"1 \pm \sqrt{19}", r"1+\sqrt{19}"),     # only half the answer
    (r"3 \pm 2 \sqrt{2}", r"3+2\sqrt{2},\ 3-2\sqrt{2},\ -3+2\sqrt{2},\ -3-2\sqrt{2}"),
    (r"(3,4]", r"[3,4]"),                     # wrong endpoint inclusion
    (r"\frac{1}{2}", r"0.5"),                 # algebra is not the grader's job
    (r"(6,31,-1)", r"(-1,6,31)"),             # an ordered tuple is not a set
    # The brace canonicalisation must not go the other way. `2^10` is 2^1
    # followed by a 0 in LaTeX, so stripping braces from `2^{10}` would merge
    # two genuinely different expressions. Canonicalising towards braces cannot.
    (r"2^{10}", r"2^10"),
    (r"x^2", r"x^3"),
    (r"a_1", r"a_2"),
]


def check_pairs():
    """Run EQUIVALENT and DISTINCT through the real grader."""
    failures = []

    def verdict(gt, answer):
        # A distinct id per pair: grade() memoises on it, and two pairs sharing
        # an id would return the first one's verdict for both.
        t = {
            "id": f"sanity-{hash((gt, answer)) & 0xffffffff:08x}",
            "grader": "exact_match_str",
            "grader_payload": {"answer": gt},
        }
        return grade(t, "Reasoning...\nThe final answer is $\\boxed{" + answer + "}$")

    eq_ok = 0
    for gt, answer in EQUIVALENT:
        if verdict(gt, answer):
            eq_ok += 1
        else:
            failures.append(f"  FAIL equivalent  gt={gt!r}  answer={answer!r} "
                            f"(correct answer graded WRONG)")

    di_ok = 0
    for gt, answer in DISTINCT:
        if not verdict(gt, answer):
            di_ok += 1
        else:
            failures.append(f"  FAIL distinct    gt={gt!r}  answer={answer!r} "
                            f"(WRONG answer graded correct - the grader is too lax)")

    print(f"equivalent formattings the grader accepts:      {eq_ok}/{len(EQUIVALENT)}")
    print(f"wrong answers the grader still rejects:         {di_ok}/{len(DISTINCT)}")
    return failures


def code_response(t):
    """What a model returning the reference solution would look like."""
    return "```python\n" + t["_ref_code"] + "\n```"


def math_response(t):
    r"""What a compliant model returns: reasoning, then \boxed{}."""
    return "Reasoning...\nThe final answer is $\\boxed{" + t["grader_payload"]["answer"] + "}$"


def main():
    path = HERE / "taskset.jsonl"
    if not path.exists():
        sys.exit(f"{path.name} not found. Build it first: python3 build_taskset.py")

    with path.open(encoding="utf-8") as f:
        tasks = [json.loads(l) for l in f if l.strip()]

    code = [t for t in tasks if t["domain"] == "code"]
    math = [t for t in tasks if t["domain"] == "math"]

    failures = []
    code_ok = 0
    for t in code:
        if grade(t, code_response(t)):
            code_ok += 1
        else:
            failures.append(f"  FAIL {t['id']}  (reference solution does not pass its own tests)")

    math_ok = 0
    for t in math:
        if grade(t, math_response(t)):
            math_ok += 1
        else:
            failures.append(
                f"  FAIL {t['id']}  answer={t['grader_payload']['answer']!r} "
                f"(grader cannot match the ground-truth answer)"
            )

    print(f"code reference solutions passing their asserts: {code_ok}/{len(code)}")
    print(f"math ground-truth answers the grader accepts:   {math_ok}/{len(math)}")
    failures.extend(check_pairs())

    if failures:
        print()
        print("\n".join(failures))
        sys.exit(
            f"\n{len(failures)} grader failure(s). Every accuracy number in this repo "
            f"is invalid until these are fixed."
        )
    print("\nboth graders score every reference answer correctly")


if __name__ == "__main__":
    main()
