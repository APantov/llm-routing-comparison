# runs/ — everything the analysis derives

Nothing here is an input. Every file is produced by a module in `llm_routing/`
or a script in `scripts/`, from `data/taskset.jsonl` and the responses in
`cache/`. Deleting the whole directory and regenerating it is the standard way
to check that a published figure really does come from the committed data:

```bash
ROUTER_MODE=replay python scripts/run_all_ladders.py
```

That is how the current figures were produced. 0 calls reached a backend.

## One file per ladder, and why

| stem | written by | what it holds |
|---|---|---|
| `results.<ladder>.jsonl` | `run_eval` | one row per (task, policy) — the primary artefact |
| `frontier.<ladder>.jsonl` | `frontier` | cost-quality curves, achievable frontiers, AUC |
| `sweep_degraded.<ladder>.jsonl` | `sweep_degraded` | the verifier-degradation curve |
| `scorecard.<ladder>.{json,txt}` | `scorecard` | per-policy error attribution against the cross-tab |
| `routable.<ladder>.txt` | `routable` | the cheap/expensive cross-tab |
| `redraw.<ladder>*.json` | `scripts/redraw_decisive.py` | per-rung p̂ from multi-draw redraws — **paid for** |
| `screen.<ladder>.<pool>.json` | `scripts/redraw_decisive.py` | a screen of a candidate task pool — **paid for** |
| `triage.<ladder>.json` | `scripts/triage_both_fail.py` | evidence for a quarantine decision |

The ladder is in the filename because every analysis script used to write to one
fixed path, which was correct while there was one measured ladder and wrong the
moment there were three. On 8 August 2026 a `deepseek` run overwrote a complete
nine-policy `wide` run with 47 rows of one policy. Every writer now takes an
output override, and `tests/test_experiment.py` has a structural tripwire that
fails when a new analysis script arrives without one.

The unsuffixed names — `results.jsonl`, `frontier.jsonl`,
`sweep_degraded.jsonl` — are copies of the canonical `wide` ladder, made by
`scripts/run_all_ladders.py`, because STATUS, the test suite and
`router_agent/findings.py` read those names.

## What is committed, and what is not

**Committed:** the per-ladder `results`, `frontier` and `sweep_degraded` files,
the scorecards, the cross-tabs, the redraws, the screens and the triage. The
first three are force-added past `.gitignore`, deliberately — `git add -f` is
the act that publishes a run, so a mock run's output can never be staged by
accident. The redraws and screens are committed for a second reason: they cost
money, and this repository does not delete real data.

**Not committed:** `run_all_ladders.log`, which is a transcript of the last
driver run rather than a result.

A file here whose rows say `"simulated": true` is mock output. It is fabricated,
it means nothing about any model, and it should never have been committed —
every committed row in this directory says `"simulated": false`.
