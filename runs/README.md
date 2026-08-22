# runs/ — everything the analysis derives

Nothing here is an input. Every file is produced by a module in `llm_routing/`
or a script in `scripts/`, from `data/taskset.jsonl` and the responses in
`cache/`. Deleting the whole directory and regenerating it is the standard way
to check that a published figure really does come from the committed data:

```bash
ROUTER_MODE=replay python scripts/run_all_ladders.py
```

That is how the current figures were produced. 0 calls reached a backend.

The driver runs every writer below except the paid tools, which is a recent
correction: it used to skip `routable` and `scorecard`, so a from-scratch run
left two artefacts missing and `plot` quietly drew seven of the nine figures.

## One file per ladder, and why

| stem | written by | what it holds |
|---|---|---|
| `results.<ladder>.jsonl` | `run_eval` | one row per (task, policy) — the primary artefact |
| `frontier.<ladder>.jsonl` | `frontier` | cost-quality curves, achievable frontiers, AUC |
| `sweep_degraded.wide.jsonl` | `sweep_degraded` | the verifier-degradation curve. **One ladder only** — the sweep holds the ladder fixed and varies verifier fidelity inside the code domain, so the ladder is the control rather than the variable |
| `scorecard.<ladder>.{json,txt}` | `scorecard` | per-policy error attribution against the cross-tab |
| `routable.<ladder>.txt` | `routable` | the cheap/expensive cross-tab — the committed file is that run's transcript |
| `redraw.<ladder>*.json` | `scripts/provenance/redraw_decisive.py` | per-rung p̂ from multi-draw redraws — **paid for** |
| `screen.<ladder>.<pool>.json` | `scripts/provenance/redraw_decisive.py` | a screen of a candidate task pool — **paid for** |
| `triage.<ladder>.json` | `scripts/provenance/triage_both_fail.py` | evidence for a quarantine decision |

The ladder is in the filename because every analysis script used to write to one
fixed path, which was correct while there was one measured ladder and wrong the
moment there were three. A `deepseek` run once overwrote a complete
nine-policy `wide` run with 47 rows of one policy. Every writer now takes an
output override, and `tests/test_experiment.py` has a structural tripwire that
fails when a new analysis script arrives without one.

**There are no unsuffixed copies.** There used to be: the canonical `wide`
ladder was copied to `results.jsonl` and `frontier.jsonl` for readers that
wanted a default. That is a second source of truth for the same numbers, and it
went wrong in both available ways — the copy was gitignored, so a fresh clone
found no economics at all and silently fell back to a stale constant; and the
constant it fell back to had two of three ladders' verdicts backwards. Every
reader now names the ladder it wants.

## What is committed, and what is not

**Committed:** the per-ladder `results`, `frontier` and `sweep_degraded` files,
the scorecards, the cross-tabs, the redraws, the screens and the triage. The
first three are force-added past `.gitignore`, deliberately — `git add -f` is
the act that publishes a run, so nothing lands here without being meant. The redraws and screens are committed for a second reason: they cost
money, and this repository does not delete real data.

**Not committed:** `run_all_ladders.log`, which is a transcript of the last
driver run rather than a result.

A file here whose rows say `"simulated": true` is mock output: fabricated, and
meaningless about any model. Nothing in the repository can write one any more.
`models.require_measured_mode` stops a mock run before it has rows,
`run_eval.assert_measured` aborts without writing if a finished run somehow
produced any, and `scorecard`, `stats` and `plot` refuse to read such a file.
Every committed row here that carries the field says `"simulated": false`. The redraws, screens and triage records do not carry it:
the first two are written only by `redraw_decisive.py`, which exits rather than
run outside real mode — a redraw in replay would re-read one cached answer k
times and report perfect reproducibility — and the triage record is evidence
gathered from responses already in the cache.
