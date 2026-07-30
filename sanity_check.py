"""Prove the graders can score known-good answers.

A grader that cannot score the reference answer is broken, and every number
downstream of it is meaningless. This is the repo's regression gate: run it after
any change to graders.py, to build_taskset.py, or to the taskset schema.

Both counts must come out full. Exits non-zero if either does not, so it can be
wired into a pre-commit hook or CI.

    python3 sanity_check.py
"""

import json
import sys
from pathlib import Path

from graders import grade

HERE = Path(__file__).parent


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
