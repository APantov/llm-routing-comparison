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

## Dated analyses

These two are **snapshots**, not reference. They record what was true and what
was argued on a specific date, and they are kept unedited because the reasoning
is worth more than the numbers — several of their figures have since been
superseded, and the file that superseded them says so.

| file | what it is | superseded where |
|---|---|---|
| [ROUTABLE_2026-07-30.md](ROUTABLE_2026-07-30.md) | Why the *routable fraction* — not the cheap-model failure rate — is the quantity that decides whether routing has anything to do. Still the canonical explanation of that distinction. | Its headline cross-tab (`routable = 0`, from 10 tasks) was answered by the 6 August two-arm probe: **15.0%** over 100 tasks. See [STATUS.md §1](../STATUS.md). |
| [SURVEY_2026-07-30.md](SURVEY_2026-07-30.md) | An independent audit of the working tree on 30 July: what was committed, what was at risk, what the docs claimed versus what the code did. | Most of its findings were acted on. It predates the 6 August task-set rebuild and the agent layer entirely. |

If you are reading either of these for a *number*, stop and read
[STATUS.md](../STATUS.md) instead — it is the file that is kept current.
