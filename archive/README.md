# archive/ — real data that no longer describes the current task set

Nothing in here is junk, and nothing in here should be read as a current
result. These are artefacts that were **genuinely produced** — some of them
paid for — against a task set that has since been rebuilt. They are kept
because deleting real data is not a thing this repository does, and quarantined
because leaving them in place made tools silently produce nonsense.

The rebuild that stranded them was on **6 August 2026**: the code half moved
from sanitized MBPP to MBPP+ (every id `code-*` → `codeplus-*`) and the maths
half was resampled at MATH500 level 5. Both changes were deliberate and are
documented in [DATASETS.md](../docs/DATASETS.md); the consequence is that any
artefact keyed on a task id or a prompt from before that date no longer lines
up with `taskset.jsonl`.

| file | what it is | why it was quarantined |
|---|---|---|
| `results.2026-07-30.jsonl` | 47 rows from the 30 July plumbing run. Every row is `"simulated": false` — **real model output**, and part of the $1.36 the experiment has spent. | All 5 of its task ids (`code-475`, `math-105`, …) are gone from the current task set. `stats.py` and `frontier.py` read `results.jsonl` by default and would have analysed it as though it were current. |
| `routellm_scores.2026-07-31.jsonl` | 100 RouteLLM `bert` scores, computed locally on 31 July. No API key was involved; regenerating needs torch (~1GB) but no money. | Scores are keyed on the prompt, so after the rebuild only **10 of 100** matched. `routellm_router.CALIBRATED` correctly refuses to run uncalibrated, so the policy was silently sitting out every run while the file looked present and healthy. |

## What this changed

- `stats.py` and `frontier.py` now find **no** `results.jsonl` and say so,
  instead of reporting on five tasks that no longer exist. That is the
  intended behaviour until [STATUS.md §2 step 5](../STATUS.md) — the full paid
  run — actually happens.
- The `routellm` policy still skips, exactly as before, but now it skips for a
  stated reason rather than because of a file that appeared to be fine.

## Regenerating, rather than restoring

Do not copy these back. Both are cheap to reproduce against the *current* task
set, and a reproduced artefact is worth more than a restored one:

```bash
# RouteLLM scores: local, no API key, no money. Needs torch.
pip install routellm==0.2.0
python routellm_router.py --score

# results.jsonl: this is the paid run. Read STATUS.md section 3 first.
ROUTER_MODE=real python run_eval.py
```

The one thing that was **not** stranded by the rebuild is
`cache/raw_calls.*.jsonl`. The response cache is keyed on the prompt text
rather than the task id, so every response the current task set needs is still
a hit — which is why `results.probe.jsonl` (200 rows, all real, all current)
stayed where it is, and why `scripts/demo.py` still replays for $0.00.
