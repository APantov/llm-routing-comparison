# Task set options

Written 30 July 2026, after the real plumbing run showed the cheap rung solving
10/10 tasks. Every dataset below was verified against the HuggingFace API on that
date: it exists, it is public, it is not gated.

---

## The two hard constraints

Any replacement has to satisfy these or it breaks the experiment rather than
improving it.

**Code half — must ship runnable tests.** The whole project rests on one domain
having a *free and perfect* verifier. That is only possible because MBPP ships
assert statements that can be executed. A code dataset without executable tests
is useless here no matter how hard it is.

**Math half — must have an exact-matchable answer.** There is no LLM judge
anywhere in this repo, deliberately, because a judge introduces a calibration
problem the project would then have to solve. Answers must be comparable as
normalised strings.

A third, softer requirement: a **shipped difficulty label** is valuable. The
predictive router used MATH500's `level` until that policy was deleted for the
field being constant; a dataset with genuine difficulty variance would restore
the option. See SHIP_PLAN.md 0.2.

---

## Code options

| dataset | n | harder? | schema fit | verdict |
|---|---|---|---|---|
| **MBPP+** `evalplus/mbppplus` | 378 | yes, via stricter tests | **identical to current** | start here |
| **BigCodeBench** `bigcode/bigcodebench` | 1140 | much | needs work | if MBPP+ is not enough |
| **LiveCodeBench** `livecodebench/code_generation_lite` | ~1K | most | needs a lot | last resort |

### MBPP+ — the cheap win  [IMPLEMENTED]

Wired up and verified on 31 July 2026. Two commands:

```bash
pip install datasets numpy
python3 fetch_mbppplus.py                       # writes data/mbppplus.json
python3 build_taskset.py --code mbppplus        # rebuild with the harder tests
python3 sanity_check.py                         # must still be full marks
```

`--code mbpp` remains the default, so nothing changes until you ask for it, and
the default taskset still rebuilds byte-identically.

Verified concretely: on task 3 (`is_not_prime`), a solution that forgets the
`n == 1` case **passes all four original MBPP asserts and fails the expanded
suite**. That is the mechanism, demonstrated rather than asserted.

Design note worth knowing: the model is shown the ORIGINAL thin asserts as its
specification, and graded against the EXPANDED suite. Both are carried in
`grader_payload`. Putting the expanded suite in the prompt would be ten kilobytes
of fuzzed input/output pairs — an absurd prompt and most of an answer key — and
would change the task rather than just the marking. Keeping the spec fixed is
what makes this a one-variable change.

**A few tasks must be dropped, and the set is platform-specific.** MBPP+'s
assertion helper only reaches `np.allclose` when the expected value is a float or
a flat sequence of floats. For a nested tuple or a complex number it falls through
to exact `==`, so the verdict turns on whether your libm matches the machine the
expectations were generated on. Task 590 (`polar_rect`, expected value
`(tuple, complex)`) passes on Linux and fails on Windows.

`fetch_mbppplus.py` therefore validates every reference solution against its own
expanded suite **on the machine that will run the evaluation**, and drops the
failures with a reason. Without that, `sanity_check.py` refuses to proceed - which
is correct but late. Re-validate an existing download without re-downloading:

```bash
python3 fetch_mbppplus.py --validate-only
```

Dropping is not tidying. The mock emits the reference as its "correct" answer, so
a task whose reference fails would have every mock-correct answer graded wrong.

**Cost: numpy becomes a grading dependency.** Every expanded test program starts
`import numpy as np` and compares floats with `np.allclose`. See `pyproject.toml`, extra `code`.

Source: <https://huggingface.co/datasets/evalplus/mbppplus> — 378 problems, the
same ones as sanitized MBPP, with roughly 35x more test cases each. The task
distribution is unchanged, so a swap moves exactly one variable: how thorough the
marking is.

### BigCodeBench — the serious step up

<https://huggingface.co/datasets/bigcode/bigcodebench>

1140 tasks, 5.6 tests each, 99% branch coverage, built around real library use.
Genuinely hard for current models.

Two costs. The `test` field is a full `unittest` class rather than a list of
asserts, so `grade_run_asserts` needs adapting. And tasks import real third-party
libraries, which means the grader's subprocess needs them installed — that would
end the repo's "runs on a bare interpreter" property for grading.

### LiveCodeBench — hardest, worst fit

<https://huggingface.co/datasets/livecodebench/code_generation_lite>

Contamination-controlled by problem release date, which is a genuinely attractive
property. But it is competitive programming: tests are stdin/stdout pairs, not
asserts, so the grader would need rewriting rather than adapting. The files are
also large. Only worth it if both options above fail.

---

## Math options

| dataset | n | harder? | difficulty label | verdict |
|---|---|---|---|---|
| **MATH-500, level 5 only** | 134 | somewhat | yes, `level` | free, try first |
| **Omni-MATH** `KbsdJames/Omni-MATH` | ~4.4K | much | yes, `difficulty` 4–10 | best real upgrade |
| **AIME 2025** `opencompass/AIME2025` | 30 | extreme | no | too small alone |

### MATH-500 level 5 — costs nothing to try

Already on disk. `MIN_MATH_LEVEL = 5` in `build_taskset.py` and rebuild. 134
problems available, against the 60 currently sampled across levels 3–5.

This is the first thing to try because it is one constant and zero download. If
the cheap rung still solves level 5 comfortably, MATH500 is finished as a source
and you move to Omni-MATH.

### Omni-MATH — the real upgrade

<https://huggingface.co/datasets/KbsdJames/Omni-MATH>
Direct file: `https://huggingface.co/datasets/KbsdJames/Omni-MATH/resolve/main/test.jsonl`

4,428 olympiad problems, single JSONL, no conversion needed. Built specifically
because "existing benchmarks like GSM8K or MATH are now being solved with high
accuracy". Fields: `problem, solution, answer, source, difficulty, domain`.

Two things to know before committing:

- **It ships a `difficulty` float, 4–10.** That would restore a predictive router's
  signal, and gives a much finer difficulty axis than MATH500's 1–5.
- **Some answers are symbolic expressions with free variables**, e.g.
  `1 + \left\lceil \frac{n}{2} \right\rceil`. Exact match on those is fragile in
  a way MATH500's mostly-numeric answers are not. Filter to answers that
  normalise cleanly, or expect a grader false-negative rate that would read as a
  capability result. `sanity_check.py` is the tool for this — it will show you
  immediately how many ground-truth answers the grader cannot match.

### AIME 2025 — a useful spike, not a base

<https://huggingface.co/datasets/opencompass/AIME2025>

30 problems, integer answers 0–999, so exact match is trivially clean. Very hard.
Too small to be the math half alone, but a good contamination-light supplement
once the main set works.

---

## Recommended path

**Step 1, free.** `MIN_MATH_LEVEL = 5`, rebuild, probe. One constant, no download.
If that alone lifts the failure rate into 20–55%, stop — you have a working task
set for the cost of one edit.

**Step 2, cheap — already implemented.** `python3 fetch_mbppplus.py` then
`python3 build_taskset.py --code mbppplus`. Same problems, far stricter marking.
Probe again.

**Step 3, only if needed.** Omni-MATH filtered to `difficulty >= 7` and to answers
that survive `normalize_math_answer`, plus BigCodeBench. This is real work: a new
loader, a grader change, and a fresh leak analysis on whatever difficulty field
you expose to the router.

Do not do all three at once. Each step changes the failure rate, and doing them
together means not knowing which one worked — the same confounding mistake the
whole `cascade_degraded` design exists to avoid.

---

## What to re-check after any swap

1. `python3 sanity_check.py` — the graders must still score reference answers at
   full marks. On a new math set this is the step that catches unmatched answer
   formats, and it exits non-zero so it cannot be skipped by accident.
2. `python3 run_eval.py --policy always_cheap --split all` in real mode — the
   probe, with its per-domain and per-difficulty-band breakdown.
3. Only then, the full run.
