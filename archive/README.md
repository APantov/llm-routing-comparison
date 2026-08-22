# archive/ — real data that no longer describes the current task set

Nothing in here is junk, and nothing in here should be read as a current
result. These are artefacts that were **genuinely produced** — some of them
paid for — against a task set that has since been rebuilt. They are kept
because deleting real data is not a thing this repository does, and quarantined
because leaving them in place made tools silently produce nonsense.

The rebuild that stranded them changed both halves at once: the code half moved
from sanitized MBPP to MBPP+ (every id `code-*` → `codeplus-*`) and the maths
half was resampled at MATH500 level 5. Both changes were deliberate and are
documented in [METHOD.md](../docs/METHOD.md#why-these-two-datasets); the consequence is that any
artefact keyed on a task id or a prompt from before the rebuild no longer lines
up with `data/taskset.jsonl`.

| file | what it is | why it was quarantined |
|---|---|---|
| `results.2026-07-30.jsonl` | 47 rows from the first plumbing run. Every row is `"simulated": false` — **real model output**, and part of the $1.36 the experiment had spent by then. | All 5 of its task ids (`code-475`, `math-105`, …) are gone from the current task set. At the time, `llm_routing/stats.py` and `llm_routing/frontier.py` both defaulted to one fixed `runs/results.jsonl`, so leaving it in place meant they would have analysed it as though it were current. |
| `routellm_scores.2026-07-31.jsonl` | 100 RouteLLM `bert` scores, computed locally before the rebuild. No API key was involved; regenerating needs torch (~1GB) but no money. | Scores are keyed on the prompt, so after the rebuild only **10 of 100** matched. `routellm_router.CALIBRATED` correctly refuses to run uncalibrated, so the policy was silently sitting out every run while the file looked present and healthy. |
| `redraw.wide.2026-08-07.json` | A **10-draw** redraw of 15 decisive `wide` tasks — paid for, and the deepest per-task estimate this project has. | Superseded as *the* redraw by `runs/redraw.wide.json`, which covers 71 tasks but only 3 draws each, and is what RESULTS reports. This file is kept rather than deleted precisely because it is **not** a subset: all 15 of its tasks appear in the current file at a third of the draws, so it holds evidence the current run cannot reproduce without spending again. Do not quote its 14.9% next to the current 13.5% — different n, different depth. |

## What this changed

- The fixed output path is gone entirely. Every writer now takes an output
  override and every reader names the ladder it wants
  (`runs/results.<ladder>.jsonl`), so there is no default file left for a
  stranded artefact to be mistaken for. All three ladders have since been
  measured against the current task set; every current number is in
  [docs/RESULTS.md](../docs/RESULTS.md).
- The `routellm` policy skips when it is uncalibrated, as before, but now it
  skips for a stated reason rather than because of a file that appeared to be
  fine.

## Regenerating, rather than restoring

Do not copy these back. Both are cheap to reproduce against the *current* task
set, and a reproduced artefact is worth more than a restored one:

```bash
# RouteLLM scores: local, no API key, no money. Needs torch.
pip install routellm==0.2.0
python -m llm_routing.routellm_router --score

# the paid run itself. Read what it cost first: ../docs/RESULTS.md
ROUTER_MODE=real python -m llm_routing.run_eval
```

The one thing that was **not** stranded by the rebuild is
`cache/raw_calls.*.jsonl`. The response cache is keyed on the prompt text
rather than the task id, so every response the current task set needs is still
a hit — which is why `runs/results.probe.jsonl` (834 rows, all real, all current)
stayed where it is, and why `scripts/demo.py` still replays for $0.00.
