# Engineering notes

The operational half: the rules the project runs under, and the bugs it found in
itself. None of this is needed to read the [results](RESULTS.md) — it is here for
anyone who wants to know how a benchmark that spends real money is kept honest.

---

## The quarantine rule

A `both_fail` task is either genuinely hard or broken by its own specification,
and the cross-tab cannot tell them apart. Getting it wrong is expensive both
ways: an unpassable task silently caps every policy, and deleting a
hard-but-solvable one removes the signal the experiment exists to measure.

**The bar: a task may be quarantined only if every rung's multi-draw p̂ is
exactly 0.** One greedy draw cannot establish that.

The bar earns its strictness. Redrawing all 24 `both_fail` candidates before
adjudicating cost $0.25 and rescued three that a single draw had condemned —
including `codeplus-305`, whose top rung solves it half the time, which makes it
precisely a *routable* task rather than a hopeless one. `TestQuarantine` now
enforces the rule against the historical data, and
`scripts/triage_both_fail.py` gathers the evidence and deliberately refuses to
make the call itself.

13 tasks are quarantined, each with the disputed input recorded as evidence in
`llm_routing.build_taskset.QUARANTINED`.

---

## Standing invariants

- **Never touch prompt templates, `models.MAX_TOKENS`, or `MODEL_SPECS` ids.**
  All are in the cache key. Changing one strands 5,075 responses and re-charges
  $8.51. It has happened once, for $0.39.
- **Never delete `archive/`.** It holds superseded real data that cost money.
- **A quarantined task is never counted again**, in any rerun, ladder, or figure.
  Responses are deleted, not filtered; `TestQuarantine` is the tripwire.
- **CI can never spend.** `ROUTER_MODE: mock` is hard-set and no keys are
  configured.

---

## How the money was spent

The 10 August session spent **$4.2794**: pool screen $0.0301, code-half census
$0.8804, `both_fail` redraw $0.2497, decisive redraw $0.3429, buy D $0.0000
(entirely cached), `deepseek` ladder $0.0769, `claude` ladder $2.8498.

Totals and the cross-ladder reuse saving are in [RESULTS.md §4](RESULTS.md).

There is a hard per-run spend cap in `models.call`, next to the one line that can
charge a card, overridable per run rather than by editing:

```bash
ROUTER_MAX_SPEND_USD=8 ROUTER_MODE=real python -m llm_routing.run_eval
```

Two paid tools each print a costed plan and refuse to spend without `--go`:
`scripts/redraw_decisive.py` and `scripts/record_missing.py`.

---

## Bugs this project found in itself

Kept because each one changed a published number, and because the mechanism that
caught it is now a permanent test.

**A ceiling that was not a ceiling.** The oracle chose between the cheap and
expensive answers, but the maths cascade also had majority-vote-over-5-samples
available — so the cascade scored *above* the supposed maximum, invalidating
every "fraction of headroom captured" figure. `run_eval` now prints a bound
check every run.

**A regression gate that only tested `grade(GT, GT)`.** Feeding ground truth
back into `\boxed{}` proved nothing, and seven correct answers were being graded
wrong because the normaliser could not match them. `sanity_check` now checks
equivalent formattings and near-miss wrong answers too.

**A task quarantined as unpassable that was the most valuable kind of task.**
`codeplus-305` was deleted as hopeless while a redraw file in the same commit
recorded its expensive rung solving it half the time — precisely a *routable*
task. The rule is now that every rung's multi-draw p̂ must be exactly 0, and a
test enforces it against the historical data.

**One fixed output path shared by three ladders.** A `deepseek` run overwrote a
complete nine-policy `wide` run with 47 rows of one policy. Every writer now
takes an output override, and a structural test fails when a new analysis script
arrives without one.

**A cross-tab that read empty for a fully measured ladder.** The ladder is
deliberately absent from the cache key, so a ladder's responses are not all in
its own file. Reading one file returned zero classified tasks; rows are now
matched to rungs by model, which is what the cache actually keys on.

**A quickstart that destroyed its own dataset.** `build_taskset`'s default built
a 96-task sample and overwrote the committed 417-task set, so following the
README replaced the artefact every published number is joined against. The
default is now the full pool, and it reproduces the committed file byte for
byte.

---

## The one-way arrow

`router_agent` imports `llm_routing`, never the reverse. CI has a job whose only
purpose is to keep it that way, and a second one — `scripts/check_core_unchanged.py`
— that fingerprints every mock response the task set can produce and compares it
against `llm_routing/models.py` at a git revision:

```
  claude    identical  c467911f3b282a8d
  deepseek  identical  ecda0b686b9cb668
  wide      identical  3d19dda9a9df1565

OK: 417 tasks x 3 ladders x 4 samples x 2 temperatures - byte-identical to HEAD.
```

If a future edit to the serving path leaks into the experiment, the build fails.
See [ARCHITECTURE.md](ARCHITECTURE.md).
