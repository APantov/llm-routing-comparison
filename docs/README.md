# docs/

The three files that answer "what is this and what state is it in" live at the
repository root: [README](../README.md), [STATUS](../STATUS.md),
[ARCHITECTURE](../ARCHITECTURE.md). Everything here is the layer below that —
read when you need the detail, not to get oriented.

## Reference

| file | read it when |
|---|---|
| [WALKTHROUGH.md](WALKTHROUGH.md) | you want to understand the code: file by file, tracing one real task through every module |
| [EXPLAINED.md](EXPLAINED.md) | you want the plain-language version of any concept here — no prior familiarity with routing assumed |
| [NOTES.md](NOTES.md) | you want the honest list of what is wrong, unresolved, or would weaken the headline |
| [DATASETS.md](DATASETS.md) | you are choosing a different benchmark, or want to know why MBPP+ and MATH500 level 5 were the ones taken |

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
