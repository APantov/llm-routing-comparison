# Why these two datasets

The task set is 357 MBPP+ code problems and 60 MATH-500 level-5 maths problems.
This is why, what was rejected, and what to re-check if you swap either.

---

## The two hard constraints

Any replacement has to satisfy these or it breaks the experiment rather than
improving it.

**Code half — must ship runnable tests.** The whole project rests on one domain
having a *free and perfect* verifier, and that is only possible because MBPP
ships assert statements that can be executed. A code dataset without executable
tests is useless here no matter how hard it is.

**Maths half — must have an exact-matchable answer.** There is no LLM judge
anywhere in this repository, deliberately: a judge introduces a calibration
problem the project would then have to solve, and it would make results
irreproducible. Answers must be comparable as normalised strings.

## MBPP+ over MBPP: one variable, moved

MBPP+ is the *same problems* as sanitized MBPP with roughly 35x more test cases
each. The task distribution is unchanged, so the swap moves exactly one thing:
how thorough the marking is.

Demonstrated rather than asserted — on task 3 (`is_not_prime`), a solution that
forgets the `n == 1` case **passes all four original MBPP asserts and fails the
expanded suite**.

**The model is shown the original thin asserts as its specification, and graded
against the expanded suite.** Both are carried in `grader_payload`. Putting the
expanded suite in the prompt would be ten kilobytes of fuzzed input/output pairs
— an absurd prompt and most of an answer key — and would change the task rather
than the marking.

**A few tasks must be dropped, and the set is platform-specific.** MBPP+'s
assertion helper only reaches `np.allclose` when the expected value is a float
or a flat sequence of floats. For a nested tuple or a complex number it falls
through to exact `==`, so the verdict turns on whether your libm matches the
machine the expectations were generated on. Task 590 (`polar_rect`, expected
`(tuple, complex)`) passes on Linux and fails on Windows.

`fetch_mbppplus.py` therefore validates every reference solution against its own
expanded suite **on the machine that will run the evaluation**, and drops the
failures with a reason. Dropping is not tidying: the mock emits the reference as
its "correct" answer, so a task whose own reference fails would have every
mock-correct answer graded wrong.

```bash
python -m llm_routing.fetch_mbppplus                 # writes data/mbppplus.json
python -m llm_routing.fetch_mbppplus --validate-only # re-check without downloading
```

**Cost: numpy becomes a grading dependency.** Every expanded test program opens
with `import numpy as np`. See `pyproject.toml`, extra `code`.

Source: <https://huggingface.co/datasets/evalplus/mbppplus>

## MATH-500 level 5: the difficulty floor

Level 5 alone leaves 134 candidates, comfortably more than the 60 sampled. It
was raised from levels 3–5 on 6 August 2026 for one reason: the probe showed the
cheap rung solving 10 out of 10 at the old setting, which leaves a router
nothing to decide.

**Known cost, and it cost more than expected.** A single level makes
`difficulty_proxy` constant across the maths half, so the shipped `level` field
carries no signal at all. That is what killed the original hand-written
predictive policy — its predicate `level >= 5` was true for every maths task, so
it was `always_expensive` on that half by construction while being reported as a
router. Predictive routing is now measured with `llm_router` and `routellm`,
neither of which reads a difficulty label.

Recoverable with `--min-math-level 3`, which restores two distinct levels.

## What was rejected, and why

| dataset | why not |
|---|---|
| **GSM8K** | current cheap models score in the low 90s, leaving a cascade nothing to route |
| **BigCodeBench** (1140 tasks) | genuinely harder, but its `test` field is a full `unittest` class rather than a list of asserts, and its tasks import real third-party libraries — the grader's subprocess would need them installed, ending the "runs on a bare interpreter" property |
| **LiveCodeBench** | contamination-controlled by release date, which is attractive, but it is competitive programming: stdin/stdout pairs rather than asserts, so the grader would need rewriting rather than adapting |
| **Omni-MATH** (4.4K olympiad) | the real upgrade if MATH-500 saturates, and it ships a `difficulty` float 4–10 that would restore a predictive router's signal. Deferred because some answers are symbolic expressions with free variables, where exact match is fragile in a way MATH-500's mostly-numeric answers are not |
| **AIME 2025** (30 problems) | integer answers make exact match trivially clean, and it is very hard — but too small to be the maths half alone |

## What to re-check after any swap

1. **`python -m llm_routing.sanity_check`** — the graders must still score every
   reference answer at full marks. On a new maths set this is the step that
   catches unmatched answer formats, and it exits non-zero so it cannot be
   skipped by accident.
2. **The two-arm probe** — `run_eval --policy always_cheap --policy
   always_expensive --split all`, then `routable --real`. A new dataset changes
   the routable fraction, and if there is nothing to route there is nothing to
   measure.
3. Only then, the full run.

Change one thing at a time. Each step moves the failure rate, and doing two
together means not knowing which one worked — the same confound the
`cascade_degraded` design exists to avoid.
