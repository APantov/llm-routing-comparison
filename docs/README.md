# docs/

Two files answer "what is this and what state is it in", and both live at the
repository root: [README](../README.md) for the first, [STATUS](../STATUS.md)
for the second. Everything here is the layer below that — read when you need
the detail, not to get oriented.

## Reference

| file | read it when |
|---|---|
| [METHOD.md](METHOD.md) | you want the method: task set, ladders, policies, verifiers, the degradation experiment, and how to run it for real |
| [ARCHITECTURE.md](ARCHITECTURE.md) | you want to know how the benchmark and the serving layer fit together, and what does not survive the move between them |
| [WALKTHROUGH.md](WALKTHROUGH.md) | you want to understand the code: file by file, tracing one real task through every module |
| [EXPLAINED.md](EXPLAINED.md) | you want the plain-language version of any concept here — no prior familiarity with routing assumed |
| [NOTES.md](NOTES.md) | you want the honest list of what is wrong, unresolved, or would weaken the headline |
| [DATASETS.md](DATASETS.md) | you are choosing a different benchmark, or want to know why MBPP+ and MATH500 level 5 were the ones taken |

## The 11 August 2026 restructure

`ARCHITECTURE.md` moved here from the repository root, and `METHOD.md` is new:
it holds the long-form method that used to make up most of a 913-line README,
with every superseded figure replaced by its measured 417-task equivalent
rather than carried under a warning. The README is now a landing page.

At the same time the 16 research modules moved from the root into
`llm_routing/`, and every derived artefact into `runs/`. Commands gained a
`-m`: `python build_taskset.py` is now `python -m llm_routing.build_taskset`.
Nothing about what is measured changed, and
`python scripts/check_core_unchanged.py` proves it — mock output is
byte-identical across all three ladders either side of the move.

## Dated analyses — deleted 9 August 2026

Two 30 July snapshots used to live here: `ROUTABLE_2026-07-30.md` (why the
*routable fraction*, not the cheap-model failure rate, is the quantity that
decides whether routing has anything to do) and `SURVEY_2026-07-30.md` (an audit
of the working tree on that date).

Both were **deleted**, not moved. Every argument in them had been superseded —
`ROUTABLE`'s headline cross-tab was `routable = 0` from ten tasks, answered by
the 6 August two-arm probe, and `SURVEY` predates the task-set rebuild and the
agent layer entirely. Between them they were 68KB of prose whose every number a
reader had to be warned not to trust.

They are recoverable from git history if the *reasoning* is ever wanted:

```bash
git log --diff-filter=D --  docs/ROUTABLE_2026-07-30.md
git show <commit>^:docs/ROUTABLE_2026-07-30.md
```

For any current number, read [STATUS.md](../STATUS.md) — it is the file that is
kept current.
