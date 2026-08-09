"""
Build the mixed task set for the routing evaluation.

Two domains, chosen for their different verification regimes:

  math (MATH500, level 5)
               -> graded by exact match on the normalised final answer.
                  There is NO ground truth available at runtime, so a cascade
                  has to fall back on a PROXY verifier.

  code (MBPP+)
               -> graded by executing the expanded evalplus suite.
                  The runtime verifier is FREE and PERFECT: just run the tests.

That asymmetry is the point of the experiment rather than an accident of dataset
choice. It is the only reason the two halves of the task set are comparable on
anything interesting.

Output is taskset.jsonl, written with LF endings on every platform so the file
is byte-identical wherever it is built.

    python3 build_taskset.py                                  # the defaults above
    python3 build_taskset.py --code mbpp --min-math-level 3   # the easier original

BOTH DEFAULTS WERE HARDENED ON 6 AUGUST 2026, for one reason: the cheap rung
solved 10 out of 10 on the original set, which leaves a router nothing to decide.
See docs/ROUTABLE_2026-07-30.md for the cross-tabulation and docs/DATASETS.md for why MBPP+
rather than a different benchmark. The originals remain one flag away, because
the point of the change is to move the failure rate, and a change you cannot undo
is a change you cannot measure.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "taskset.jsonl"

SEED = 20260728
N_MATH = 60
N_CODE = 40

# GSM8K was rejected as the math source: current cheap models score in the low
# 90s on it, which leaves a cascade almost nothing to route. MATH500 restricted
# to levels 3-5 leaves 367 candidates, which was the original setting.
#
# Raised to 5 on 6 August 2026. Levels 3-5 turned out to be no obstacle either:
# the real cheap rung went 10-for-10 on the sampled set. Level 5 alone leaves 134
# candidates, comfortably more than the 60 sampled.
#
# KNOWN COST, and it cost more than this comment expected. A single level means
# `difficulty_proxy` is constant across the maths half, so add_difficulty_pct()
# gives every maths task the same percentile and `predict_features.level` carries
# no signal at all. splits.py can only stratify maths by domain.
#
# WHAT THIS COMMENT UNDERSTATED, recorded 8 August 2026. The hand-written
# predictive policy read `level >= 5` and therefore returned True for all 60
# maths tasks: not "blind" but CONSTANT, which made it `always_expensive` on 60%
# of the task set while still being reported as a router. It scored below the
# coin flip it was meant to beat, and its frontier sweep had two attainable
# points rather than a curve. The policy was deleted (policies.py DECISION #4)
# and predictive routing is now measured with `llm_router` and `routellm`,
# neither of which reads this field.
#
# The lesson worth carrying: a consequence recorded at the place that CAUSES it
# does not reach the place that SUFFERS it. This note existed and was correct
# from 6 August; nothing downstream ever read it.
#
# Recoverable with --min-math-level 4, which restores two distinct levels. None
# of it affects the always_cheap / always_expensive probe, which is what this
# setting exists to serve: those two policies never look at difficulty.
MIN_MATH_LEVEL = 5


def load_math500(min_level=None):
    min_level = MIN_MATH_LEVEL if min_level is None else min_level
    tasks = []
    with open(DATA / "math500.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            if row["level"] < min_level:
                continue
            tasks.append(
                {
                    "id": f"math-{i}",
                    "domain": "math",
                    "prompt": row["problem"],
                    # Answers are fractions, radicals and tuples as well as
                    # integers, so the grader compares normalised strings.
                    "grader": "exact_match_str",
                    "grader_payload": {"answer": row["answer"]},
                    # MATH500 ships a 1-5 difficulty level. Unlike a reference
                    # solution's length, this arrives WITH the question, so a
                    # router may use it without leaking the answer. Passing the
                    # leak test is not the same as being useful: under
                    # MIN_MATH_LEVEL = 5 this field is constant. See DECISION #4.
                    "difficulty_proxy": row["level"],
                    "subject": row["subject"],
                    # See the note on predict_features in load_mbpp.
                    "predict_features": {
                        "level": row["level"],
                        "prompt_chars": len(row["problem"]),
                    },
                }
            )
    return tasks


def load_mbpp():
    """Sanitized MBPP (427 hand-verified items), not the full 974.

    The full set contains prompts that under-specify the task, most often by not
    naming the function the asserts will call. Both tiers fail those equally,
    which is cost with no routing signal, and it inflates the cheap-model
    failure rate the pilot gate reads.

    Note the sanitized format differs from mbpp.jsonl: it is a JSON array, the
    question field is 'prompt' rather than 'text', and 'test_imports' is a list.
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
                # This is the reference solution's line count, which is NOT
                # available before answering. Fine for stratified sampling and
                # for driving the mock's success rate; it would be a leak as a
                # router feature, which is why the router cannot reach it.
                "difficulty_proxy": len(row["code"].splitlines()),
                # Everything a router is allowed to see before calling anything,
                # in its own field on purpose: routers read ONLY from here, so the leak
                # above cannot be reintroduced by someone reaching for
                # difficulty_proxy because it happens to be nearby. Every entry
                # must be derivable from the question alone.
                "predict_features": {
                    "prompt_chars": len(row["prompt"]),
                    "n_asserts": len(tests),
                },
                # Reference solution, used ONLY by the mock model, which needs
                # something that genuinely passes the asserts in order to
                # simulate a correct answer. Never shown to a real model.
                "_ref_code": row["code"],
            }
        )
    return tasks


def load_mbppplus():
    """MBPP+ (evalplus/mbppplus): the same problems, roughly 35x more tests.

    Same 378 problems as sanitized MBPP, so the task DISTRIBUTION is unchanged and
    a swap moves exactly one variable: how thorough the tests are. That is the
    cleanest possible way to raise the cheap model's failure rate, because
    anything else - harder problems, a different domain - would move the
    distribution at the same time and confound the comparison.

    Concretely, on task 3 (`is_not_prime`) a solution that forgets the n == 1 case
    passes all four original asserts and fails the expanded suite. Under plain
    MBPP that is a point the model did not earn.

    Requires data/mbppplus.json, written once by fetch_mbppplus.py.

    The difficulty proxy stays the reference solution's line count, exactly as for
    plain MBPP, so the two sources remain comparable on that axis.
    """
    path = DATA / "mbppplus.json"
    if not path.exists():
        raise SystemExit(
            f"\n{path} not found.\n"
            f"  Build it once with:  python3 fetch_mbppplus.py\n"
            f"  (needs `pip install datasets`; the rest of the repo does not)\n"
        )
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)

    tasks = []
    for row in rows:
        tasks.append(
            {
                # Prefixed so a task id can never collide with a plain-MBPP one.
                # Two runs from different code sources must not look like the same
                # task in results.jsonl, or the paired statistics would silently
                # compare different problems.
                "id": f"codeplus-{row['task_id']}",
                "domain": "code",
                "prompt": row["prompt"],
                "grader": "test_program",
                # BOTH suites are carried, and which one is used where is the
                # whole design of this swap:
                #
                #   tests        the ORIGINAL thin asserts. models.build_prompt
                #                puts these in the prompt as the specification,
                #                exactly as for plain MBPP.
                #   test_program the EXPANDED suite. Only the grader sees it.
                #
                # So the model is shown the same specification it was shown
                # before, and only the marking gets stricter. That keeps the swap
                # a ONE-variable change. Putting the expanded suite in the prompt
                # would change the task as well as the grading - it is ten
                # kilobytes of fuzzed input/output pairs, which is both an absurd
                # prompt and a near-complete answer key.
                "grader_payload": {
                    "tests": list(row.get("test_list") or []),
                    "test_program": row["test"],
                },
                "difficulty_proxy": len(row["code"].splitlines()),
                "predict_features": {
                    "prompt_chars": len(row["prompt"]),
                    # The ORIGINAL assert count, not the expanded case count. The
                    # expanded count is a property of how evalplus fuzzed the
                    # problem rather than of the question, and a router reading it
                    # would be using information the question does not carry.
                    "n_asserts": len(row.get("test_list") or []),
                },
                "_ref_code": row["code"],
            }
        )
    return tasks


CODE_SOURCES = {"mbpp": load_mbpp, "mbppplus": load_mbppplus}


def add_difficulty_pct(tasks):
    """Rank difficulty WITHIN each domain and store it as a 0-1 percentile.

    Necessary because the raw proxies are not comparable across domains: math
    counts MATH500 levels (3-5), code counts lines of reference solution (2-26).
    A threshold expressed in raw units means something completely different in
    each domain, so a rule like "hard if proxy >= 5" would classify almost every
    code task as hard and only half the math tasks.

    TIES SHARE A PERCENTILE. This matters because the math proxy is `level`:
    only three distinct values across 60 tasks, so roughly 20 tasks tie at each
    one. Ranking them by position would break those ties on file order, and both
    the mock's success rate and any difficulty-based threshold would then be
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
            # Mid-rank of the tied block, so a level sits at its centre of mass.
            mid = first_rank[p] + (counts[p] - 1) / 2
            t["difficulty_pct"] = round(mid / n, 4)
    return tasks


def stratified_sample(tasks, n, rng):
    """Sample across the difficulty range rather than uniformly at random.

    A router evaluated only on mid-difficulty tasks reveals nothing. The set
    needs genuinely easy items, where escalating is pure waste, and genuinely
    hard ones, where staying cheap is a failure.
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


# ---------------------------------------------------------------------------
# QUARANTINE: tasks whose expected answers cannot be derived from their prompt.
#
# These are not hard tasks. They are tasks where MBPP+'s generated inputs are
# scored against whatever the MBPP reference implementation happened to return,
# including on inputs the natural-language prompt says nothing about. No model
# can pass them, and neither can a textbook-correct solution - which is how each
# one below was diagnosed, rather than by assuming that "both rungs failed"
# means "hard".
#
# This matters more than five tasks should, because they were ALL of
# always_expensive's failures on the eval split. Left in, they set the ceiling
# for every policy: `always_expensive` and `oracle` read 92% instead of 100%,
# and the code half reads 80% instead of 100%. STATUS.md's "code is now the
# harder domain in absolute terms" was this artefact. See SHIP_PLAN.md 0.1.
#
# Removal happens AFTER sampling and AFTER add_difficulty_pct, deliberately.
# Filtering the pool first would let stratified_sample draw five replacements,
# and a replacement task has no cached response, so every policy would be
# dropped by ReplayMiss and $4.40 of committed data would go unused. Filtering
# last leaves the survivors byte-identical to their entries in the previous task
# set, so the existing cache stays valid.
QUARANTINED = {
    "codeplus-119":
        "expects the XOR-fold of the reference: [1,2,3,4,5,6] -> 7, which is not "
        "in the array. A textbook solution mismatches 64 of 110 inputs.",
    "codeplus-792":
        "'count the number of lists in a given number of lists' is ambiguous; all "
        "three shipped asserts fit both len(x) and 'count sublists', and only the "
        "hidden inputs disambiguate.",
    "codeplus-771":
        "expects '' -> False. An empty expression is balanced. A textbook solution "
        "mismatches exactly 1 of 106 inputs - that one.",
    "codeplus-305":
        "the reference yields mutually inconsistent expectations for "
        "identically-shaped inputs.",
    "codeplus-630":
        "expected coordinate ordering is a reference artefact the prompt does not "
        "specify.",
}


def drop_quarantined_rows(rows, source, key="task_id", warn=True):
    """Strip quarantined tasks out of any artefact recorded before the quarantine.

    THE RULE: a quarantined task is never counted, anywhere, ever again - not in
    a rerun, not in a cross-tab, not in a figure, not in anything the agent layer
    serves. Removing them from `taskset.jsonl` is only half of that. The other
    half is the artefacts already on disk that were recorded over the old
    100-task set and still contain them:

        results.probe.jsonl   200 rows, both arms, all 5 broken tasks present
        redraw.wide.json      per-task p_hat, 5 of its 21 tasks are broken ones
        cache/raw_calls.*     their responses, harmless but reachable

    Those files are real and are deliberately kept - nothing measured gets
    deleted in this repository - so the filtering has to happen at every point
    of USE instead. This is that filter. Anything that loads one of those files
    calls it, so the rule is enforced in one place rather than remembered in
    several.

    Deleting the artefacts instead would be the other option, and it is worse:
    it destroys evidence for the quarantine itself, and the numbers in
    SHIP_PLAN.md section 0.1 could no longer be reproduced.
    """
    kept = [r for r in rows if r.get(key) not in QUARANTINED]
    dropped = len(rows) - len(kept)
    if dropped and warn:
        print(
            f"  {source}: dropped {dropped} row(s) on quarantined tasks "
            f"(recorded before the 8 August 2026 quarantine; see "
            f"build_taskset.QUARANTINED)",
            file=sys.stderr,
        )
    return kept


def drop_quarantined(tasks, keep=False):
    """Remove known-unpassable tasks, and say which and why."""
    if keep:
        print(f"  !! keeping {len(QUARANTINED)} quarantined task(s) - "
              f"the ceiling this produces is an artefact, not a measurement")
        return tasks
    kept = [t for t in tasks if t["id"] not in QUARANTINED]
    dropped = [t["id"] for t in tasks if t["id"] in QUARANTINED]
    for task_id in sorted(dropped):
        print(f"  quarantined {task_id}: {QUARANTINED[task_id]}")
    if dropped:
        print(f"  {len(dropped)} unpassable task(s) removed after sampling")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--code", default="mbppplus", choices=sorted(CODE_SOURCES),
        help="which code benchmark to use. mbppplus (default) is the same "
             "problems with the expanded evalplus suites, which are far harder "
             "to pass. mbpp is sanitized MBPP with its original thin asserts, "
             "kept reachable so the easier marking can be reproduced.",
    )
    ap.add_argument("--n-math", type=int, default=N_MATH)
    ap.add_argument("--n-code", type=int, default=N_CODE)
    ap.add_argument(
        "--min-math-level", type=int, default=MIN_MATH_LEVEL,
        help="MATH500 difficulty floor. Raise it to make the maths half harder.",
    )
    ap.add_argument(
        "--keep-quarantined", action="store_true",
        help="keep the known-unpassable tasks listed in QUARANTINED. Only for "
             "reproducing pre-8-August numbers; the ceiling it produces is an "
             "artefact of broken tests rather than a measurement.",
    )
    args = ap.parse_args()

    rng = random.Random(SEED)
    math_tasks = stratified_sample(
        load_math500(args.min_math_level), args.n_math, rng)
    code_tasks = stratified_sample(CODE_SOURCES[args.code](), args.n_code, rng)
    tasks = math_tasks + code_tasks
    add_difficulty_pct(tasks)
    rng.shuffle(tasks)
    # After the shuffle, so the surviving tasks keep exactly the difficulty_pct
    # and ordering they had before the quarantine existed.
    tasks = drop_quarantined(tasks, keep=args.keep_quarantined)

    # newline="" so this file is byte-identical on Windows and Linux. Without
    # it, Python translates \n to \r\n on Windows, the same code produces
    # different bytes on different machines, and any hash-based regression gate
    # becomes impossible.
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"wrote {len(tasks)} tasks -> {OUT}")
    print(f"  code source: {args.code}   math levels: >= {args.min_math_level}")
    for domain in ("math", "code"):
        sub = [t for t in tasks if t["domain"] == domain]
        diffs = [t["difficulty_proxy"] for t in sub]
        print(
            f"  {domain:5s} n={len(sub):3d}  "
            f"difficulty_proxy min={min(diffs)} med={sorted(diffs)[len(diffs)//2]} max={max(diffs)}"
        )


if __name__ == "__main__":
    main()
