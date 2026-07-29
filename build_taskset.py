"""
Build a mixed task set for the cascade-router evaluation.

Two domains, deliberately chosen for different verification regimes:

  math (MATH500, levels 3-5)
               -> graded by exact match on the normalised final answer.
                  NO runtime ground truth, so the cascade needs a PROXY verifier.

  code (MBPP sanitized)
               -> graded by executing assert statements.
                  Verifier is FREE and PERFECT at runtime (just run the tests).

That asymmetry is the point of the experiment, not an accident of dataset choice.
"""

import json
import random
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "taskset.jsonl"

SEED = 20260728
N_MATH = 60
N_CODE = 40

# GSM8K was rejected: modern cheap models score in the low 90s there, so the
# cascade has nothing to route. MATH500 levels 3-5 leaves 367 candidates.
MIN_MATH_LEVEL = 3


def load_math500():
    tasks = []
    with open(DATA / "math500.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if row["level"] < MIN_MATH_LEVEL:
                continue
            tasks.append(
                {
                    "id": f"math-{i}",
                    "domain": "math",
                    "prompt": row["problem"],
                    # Answers are fractions, radicals and tuples, not just
                    # integers - hence the string grader, not extract_final_int.
                    "grader": "exact_match_str",
                    "grader_payload": {"answer": row["answer"]},
                    # MATH500 ships a 1-5 difficulty level. Unlike a reference
                    # solution's length this is shipped WITH the question, so
                    # the predictive router may use it without leaking.
                    "difficulty_proxy": row["level"],
                    "subject": row["subject"],
                    # See note on predict_features in load_mbpp.
                    "predict_features": {
                        "level": row["level"],
                        "prompt_chars": len(row["problem"]),
                    },
                }
            )
    return tasks


def load_mbpp():
    """Sanitized MBPP (427 hand-verified items), not the full 974.

    The full set contains prompts that under-specify the task - most often by
    not naming the function the asserts call. Both tiers fail those equally,
    which is cost with no routing signal and inflates the cheap-model failure
    rate that Phase 2 gates on. Note the format differs from mbpp.jsonl: a JSON
    array, 'prompt' not 'text', and 'test_imports' is a list.
    """
    with open(DATA / "sanitized-mbpp.json", encoding="utf-8") as f:
        rows = json.load(f)

    tasks = []
    for row in rows:
        tests = row.get("test_list") or []
        # task_id 1-10 are MBPP's canonical few-shot prompt examples.
        if not tests or row["task_id"] <= 10:
            continue
        tasks.append(
            {
                "id": f"code-{row['task_id']}",
                "domain": "code",
                "prompt": row["prompt"],
                "grader": "run_asserts",
                "grader_payload": {
                    "tests": tests,
                    "setup": "\n".join(row.get("test_imports") or []),
                },
                # !! LEAK for the predictive router: this is the reference
                # solution's line count, unavailable before answering. Fine
                # for stratified sampling and the mock, not fine as a router
                # feature. See RUNBOOK step 6.
                "difficulty_proxy": len(row["code"].splitlines()),
                # Everything the predictive router is allowed to see. Separated
                # into its own field on purpose: the router reads ONLY from
                # here, so the leak above cannot be reintroduced by someone
                # reaching for difficulty_proxy because it happens to be handy.
                # Every entry must be derivable from the question alone.
                "predict_features": {
                    "prompt_chars": len(row["prompt"]),
                    "n_asserts": len(tests),
                },
                # Reference solution. Used ONLY by the mock model, which
                # needs something that genuinely passes the asserts in
                # order to simulate a correct answer. Never shown to a
                # real model - see models.py.
                "_ref_code": row["code"],
            }
        )
    return tasks


def add_difficulty_pct(tasks):
    """Rank difficulty WITHIN each domain and store it as a 0-1 percentile.

    Necessary because the raw proxies are not comparable across domains:
    math counts MATH500 levels (3-5), code counts lines of reference solution
    (2-19). Any threshold expressed in raw units means something completely
    different in each domain - a rule like 'hard if proxy >= 5' classifies
    almost every code task as hard and only half the math tasks.

    TIES SHARE A PERCENTILE. Necessary since the math proxy became `level`:
    only three distinct values across 60 tasks, so ~20 tasks tie at each one.
    Ranking them by position would break those ties on file order, and the
    mock's p_correct and the predictive router's threshold would both then be
    reading sort artefacts as difficulty.
    """
    for domain in ("math", "code"):
        sub = sorted(
            [t for t in tasks if t["domain"] == domain],
            key=lambda t: t["difficulty_proxy"],
        )
        n = max(1, len(sub) - 1)
        first_rank = {}
        for rank, t in enumerate(sub):
            first_rank.setdefault(t["difficulty_proxy"], rank)
        counts = Counter(t["difficulty_proxy"] for t in sub)
        for t in sub:
            p = t["difficulty_proxy"]
            # mid-rank of the tied block, so a level sits at its centre of mass
            mid = first_rank[p] + (counts[p] - 1) / 2
            t["difficulty_pct"] = round(mid / n, 4)
    return tasks


def stratified_sample(tasks, n, rng):
    """Sample across the difficulty range rather than uniformly at random.

    A router evaluated only on mid-difficulty tasks tells you nothing:
    you need genuinely easy items (where escalating is pure waste) and
    genuinely hard ones (where staying cheap is a failure).
    """
    tasks = sorted(tasks, key=lambda t: t["difficulty_proxy"])
    buckets = 4
    per = n // buckets
    size = len(tasks) // buckets
    out = []
    for b in range(buckets):
        chunk = tasks[b * size : (b + 1) * size]
        take = per if b < buckets - 1 else n - len(out)
        out.extend(rng.sample(chunk, min(take, len(chunk))))
    return out


def main():
    rng = random.Random(SEED)
    math_tasks = stratified_sample(load_math500(), N_MATH, rng)
    code_tasks = stratified_sample(load_mbpp(), N_CODE, rng)
    tasks = math_tasks + code_tasks
    add_difficulty_pct(tasks)
    rng.shuffle(tasks)

    # newline="" so this file is byte-identical on Windows and Linux. It was
    # previously written with Python's default translation, so the shipped
    # taskset.jsonl carried CRLF while results.jsonl carried LF - the same code
    # produced different bytes on different machines, which makes any
    # hash-based regression gate impossible.
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"wrote {len(tasks)} tasks -> {OUT}")
    for domain in ("math", "code"):
        sub = [t for t in tasks if t["domain"] == domain]
        diffs = [t["difficulty_proxy"] for t in sub]
        print(
            f"  {domain:5s} n={len(sub):3d}  "
            f"difficulty_proxy min={min(diffs)} med={sorted(diffs)[len(diffs)//2]} max={max(diffs)}"
        )


if __name__ == "__main__":
    main()
