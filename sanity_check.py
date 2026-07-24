"""Step 5: prove the graders can score known-good answers.

A grader that cannot score the reference answer is broken, and every number
downstream of it is meaningless. Run this after any change to graders.py or
to the taskset schema.
"""

import json

from graders import grade

tasks = [json.loads(l) for l in open("taskset.jsonl", encoding="utf-8")]

code = [t for t in tasks if t["domain"] == "code"]
math = [t for t in tasks if t["domain"] == "math"]


def code_response(t):
    return "```python\n" + t["_ref_code"] + "\n```"


def math_response(t):
    # Mimics what a compliant model returns: reasoning, then \boxed{}.
    return "Reasoning...\nThe final answer is $\\boxed{" + t["grader_payload"]["answer"] + "}$"


code_ok = [t for t in code if grade(t, code_response(t))]
math_ok = [t for t in math if grade(t, math_response(t))]

print(f"code refs:  {len(code_ok)}/{len(code)}")
print(f"math truth: {len(math_ok)}/{len(math)}")

for t in code:
    if t not in code_ok:
        print("  FAIL", t["id"])
for t in math:
    if t not in math_ok:
        print("  FAIL", t["id"], repr(t["grader_payload"]["answer"]))
