# STATUS — read this first

**Rewritten 10 August 2026, after the three-ladder run.** This file is true on
its own. It carries no banner telling you the text below it is wrong, because
the text below it has been corrected rather than annotated.

It replaces three documents. `SHIP_PLAN.md` (the 8 August audit) and
`next_step.md` (the execution sequence) were scaffolding for a transition that
is now finished, and both have been deleted; what they decided is recorded
below. The corrections they demanded have been applied to the bodies of the
documents that made the claims.

**Repository restructured 11 August 2026, and no number below moved.** The 16
research modules went from the repository root into `llm_routing/`, every
derived artefact into `runs/`, and the task set into `data/`; commands gained a
`-m` (`python -m llm_routing.run_eval`). `scripts/check_core_unchanged.py`
fingerprints every mock response the task set can produce and reports
byte-identical output on all three ladders either side of the move. One real
bug was fixed in passing: `build_taskset`'s default built a **96-task** set and
overwrote the committed 417-task one, so the quickstart destroyed the artefact
every result here is joined against. The default is now the full pool.

---

## 1. What this project measures

A cost-aware LLM routing benchmark, plus the serving layer whose policy it
decides. The thesis under test: **answer at the cheapest model that can be
verified correct, and escalate only when verification fails.**

Ten policies are compared over 417 tasks on three model ladders. Every response
is a real API call, committed to `cache/`, and the entire analysis regenerates
offline for $0.00 with no API key.

| | |
|---|---|
| tasks | **417** — 357 MBPP+ code, 60 MATH-500 level 5 |
| ladders | **3** — `wide` (flash→opus), `claude` (haiku→sonnet→opus), `deepseek` (flash→pro) |
| policies | **10**, including an oracle bound and a cost-matched random null |
| real responses | **5,075** |
| total spend | **$8.5145** |
| tests | **205 passing**, 1 slow test opt-in via `pytest -m slow` |

---

## 2. What the data says

### 2.1 The cascade beats always-paying-for-the-best on two ladders of three

Pre-registered comparison, exact McNemar over paired outcomes, n=209 held-out
tasks per ladder.

| ladder | cascade | always_expensive | Δ acc | p | Δ cost/task |
|---|---|---|---|---|---|
| `wide` | **95.7%** | 92.3% | +3.3% | **0.039** | **−$0.00307** |
| `claude` | **96.7%** | 92.3% | +4.3% | **0.012** | +$0.00097 |
| `deepseek` | 86.6% | 83.7% | +2.9% | 0.070 | −$0.00000 |

On `wide` the cascade is **more accurate and four times cheaper**. On `claude`
it buys accuracy at a **premium** — verification is not free when the cheap rung
is haiku and the maths half draws five samples from it.

**Do not quote "the cascade is cheaper and better" without naming the ladder.**
That is a `wide` result, and `claude` is the counter-example.

### 2.2 Accuracy differences are now detectable

This repository previously reported **0 of 8** pre-registered comparisons as
significant and summarised itself as *"cost differences between routing
architectures are measurable at n=100. Accuracy differences are not."*

At n=209 that is **4 of 8** on `wide` and on `claude`, and 1 of 8 on `deepseek`.

**The old summary is retracted.** It was a statement about a sample of 47 eval
tasks, not about routing. What bought the change was the code half going from 35
tasks to 357: the number of tasks that can distinguish two routers at all rose
from 7 to 73.

### 2.3 Predictive routing does not beat a coin flip — six of six

`random_matched` flips a coin at `llm_router`'s own escalation rate, so the
comparison holds spend roughly fixed and isolates skill.

| comparison | `wide` | `claude` | `deepseek` |
|---|---|---|---|
| `llm_router` vs `random_matched` | p=0.167 | p=0.549 | p=1.000 |
| `routellm` vs `random_matched` | p=0.454 | p=1.000 | p=0.727 |
| `cascade` vs `llm_router` | **p=0.003** | **p=0.006** | **p=0.012** |

Neither a learned router (RouteLLM's pretrained BERT) nor an LLM-as-router beats
a cost-matched coin flip on any ladder, while the cascade beats both on every
ladder. RouteLLM's frontier AUC sits at or *below* the null everywhere:
−0.0018, −0.0001, −0.0129.

The distinction is **when the decision is made**. A predictive router commits
before seeing an attempt; a cascade decides after verifying one.

### 2.4 The third ladder has almost nothing to route

| ladder | cheap rung | top rung | gap | routable | both_fail | McNemar |
|---|---|---|---|---|---|---|
| `wide` | 82.9% | 92.3% | **+9.4** | 56 | 15 | **0.000** |
| `claude` | 82.7% | 92.3% | **+9.6** | 58 | 14 | **0.000** |
| `deepseek` | 82.9% | 82.7% | **−0.2** | 15 | 56 | 1.000 |

On `deepseek` the expensive rung is **not measurably better than the cheap one**.
Fifteen of 415 tasks are routable against 56 hopeless ones, and the whole
accuracy dynamic range is 7.5%. No policy can win what is not there — which is
why `always_expensive` lands *below* a cost-matched coin flip on that ladder.

> **A confound this design cannot separate.** The project set out to test a
> **price-ratio** crossover (cascading loses at ~3×, wins at ~46×). What the data
> distinguishes is the **capability gap** between rungs, and here the two are
> confounded: the ladder with the small price ratio is also the ladder whose
> rungs are equally capable. Three ladders cannot tell them apart.
>
> The supported claim is *"cascading pays when the top rung is genuinely
> better."* The price-ratio framing remains a hypothesis, and separating them
> needs a ladder with a large price ratio and a small capability gap.

### 2.5 A fifth of the routing opportunity is noise

71 decisive tasks, 3 fresh draws at both rungs, $0.3429.

| measure | routable fraction | counts |
|---|---|---|
| observed | 13.5% | one draw per cell — what a probe publishes |
| expected | 12.2% | mean over fresh draws |
| **reproducible** | **11.3%** | cheap reliably fails **and** the top rung reliably succeeds |

**7.2% of the apparent opportunity is one model having a bad draw.** A router
credited against `observed` is paid for mass it cannot capture twice running.
This is a *lower bound* on the correction: `both_ok` and `inverted` were not
redrawn, so flakiness hidden there is still uncounted.

### 2.6 Greedy decoding is not deterministic, for either provider

Across 21 tasks with ≥5 draws at temperature 0, more than one distinct answer
came back on **76%** of tasks for `claude-opus-5` and **67%** for
`deepseek-v4-flash`.

**The previously recorded "Opus deterministic, DeepSeek not" asymmetry is
retracted.** It was scheduled to be promoted as a finding; it is backwards.

---

## 3. What each policy got right and wrong

`llm_routing/scorecard.py` joins each policy's decision against what the two rungs could
actually do. `wide` ladder, 209 tasks:

| policy | acc | $/task | rescued | no-top | missed | wasted | harmful | prec |
|---|---|---|---|---|---|---|---|---|
| `oracle` | 96.2% | 0.00050 | 25 | 2 | 0 | 0 | 0 | 100% |
| `cascade` | 95.7% | 0.00095 | 24 | 2 | 1 | 1 | 0 | 73% |
| `cascade_routing` | 95.2% | 0.00068 | 23 | 2 | 2 | 0 | 0 | 74% |
| `always_expensive` | 92.3% | 0.00402 | 27 | 0 | 0 | **166** | **8** | 13% |
| `llm_router` | 90.4% | 0.00309 | 17 | 0 | 10 | 85 | 2 | 15% |
| `routellm` | 89.0% | 0.00192 | 14 | 0 | 13 | 74 | 2 | 15% |
| `random_matched` | 87.1% | 0.00293 | 11 | 0 | 16 | 84 | 3 | 11% |
| `always_cheap` | 83.3% | 0.00005 | 0 | 0 | 27 | 0 | 0 | — |

- **rescued** — escalated a routable task, the only way to win
- **no-top** — got a routable task right *without* the top rung: cheap-rung
  self-consistency, or a middle rung
- **missed** — stayed cheap and got it wrong
- **wasted** — escalated a task the cheap rung already had
- **harmful** — escalated a task the cheap rung had right, and got it wrong

`always_expensive` escalates 201 tasks to buy 27 rescues and burns **$0.71** on
escalations that could not improve the answer, losing 8 tasks the cheap rung had
already answered. `cascade` wastes **$0.084** — eight times less — because
verification tells it which escalations are worth making.

`always_mid` is measured for the first time (on `claude`, the only three-rung
ladder): **88.5%**, solving **12 of 22** routable tasks without ever reaching
Opus. The middle rung does real work.

---

## 4. What it cost

| | |
|---|---|
| `wide` cache | $5.5878 |
| `claude` cache | $2.8498 |
| `deepseek` cache | $0.0769 |
| **total** | **$8.5145** over 5,075 real responses |

The 10 August session spent **$4.2794**: pool screen $0.0301, code-half census
$0.8804, `both_fail` redraw $0.2497, decisive redraw $0.3429, buy D $0.0000
(entirely cached), `deepseek` ladder $0.0769, `claude` ladder $2.8498.

**Cross-ladder cache reuse is worth about $1.70.** The ladder is deliberately
absent from the cache key, so `wide`'s Opus answers serve `claude`'s top rung
and `wide`'s flash answers serve `deepseek`'s bottom rung, for nothing.

**Estimates run low.** The `claude` buy came in **52% over** a $1.87 estimate;
the call count was exact (1489 vs 1491) and the per-call cost was 53% high,
because the estimate used Opus token counts as a length proxy and weaker models
write longer answers to the same question. A cheaper model is not a
proportionally cheaper call.

---

## 5. Known limitations

Ordered by how much they bound the findings.

1. **The verifier that produces the signal is not the verifier that ships.** The
   code half is graded by executing tests, which a deployed router does not
   have. The maths half — where a self-consistency verifier *would* deploy —
   carries far less routing signal. Every number here is collected under a
   verifier the product cannot have. This is the sharpest open problem in the
   repository.
2. **Verification is not uniform across rungs.** On `claude`, only haiku accepts
   a temperature, so the middle rung cannot be verified by self-consistency at
   all.
3. **Price ratio and capability gap are confounded** — see §2.4.
4. **The task set is 86% code.** Every aggregate is a code number. Per-domain
   figures are reported throughout and should be preferred.
5. **Cheap-rung failure is 19.0%** [15%, 23%], marginally below the 20% floor
   the pilot gate sets. The full MBPP+ census is *easier* than the hand-sampled
   35-task half it replaced; what it bought was ten times the discriminating
   power in absolute terms.
6. **Maths cannot discriminate one-shot routers.** `llm_router` sends every
   maths task to the expensive rung. Declared rather than measured away — the
   fix was costed at $3.2 and cut.
7. **13 tasks are quarantined as unpassable-by-specification**, each with the
   disputed input recorded as evidence in `build_taskset.QUARANTINED`. Five
   ambiguous-but-arguable tasks were deliberately **kept**, which biases against
   the routers rather than for them.
8. **`cascade_routing` is the greedy variant** of Dekoninck et al., not the full
   algorithm.

---

## 6. The quarantine rule

A `both_fail` task is either genuinely hard or broken by its own specification,
and the cross-tab cannot tell them apart. Getting this wrong is expensive in
both directions: an unpassable task silently caps every policy, and deleting a
hard-but-solvable task removes the signal the experiment exists to measure.

**The bar: a task may be quarantined only if every rung's multi-draw p̂ is
exactly 0.** One greedy draw cannot establish that.

This is not hypothetical. `codeplus-305` was quarantined on 8 August as
unpassable while `runs/redraw.wide.json` **in the same commit** recorded its
expensive rung at p̂ = 0.5. It is a task the cheap rung reliably fails and the
top rung solves half the time — precisely a *routable* task. It was reversed on
10 August, and the tripwire that now enforces the bar fires on the historical
data.

Redrawing all 24 `both_fail` candidates before adjudicating cost $0.25 and found
**two more** that were passable: `codeplus-235` (p̂ = 1.00, passes every fresh
draw) and `codeplus-301` (p̂ = 0.67). Both would have been deleted as hopeless
under the old rule.

`scripts/triage_both_fail.py` gathers the evidence and refuses to make the
decision. Its discriminator — independent prompt-conformant candidates disputing
*the same* hidden inputs — was validated against the five hand-adjudicated tasks
and reproduces their evidence sentences. Two simpler discriminators were tried
and rejected on measurement; see its docstring.

---

## 7. Standing invariants

- **Never touch prompt templates, `models.MAX_TOKENS`, or `MODEL_SPECS` ids.**
  All are in the cache key. Changing one strands 5,075 responses and re-charges
  $8.51. It has happened once, for $0.39.
- **Never delete `archive/`.** It holds superseded real data that cost money.
- **A quarantined task is never counted again**, in any rerun, ladder, or figure.
  Responses are deleted, not filtered; `TestQuarantine` is the tripwire.
- **CI can never spend.** `ROUTER_MODE: mock` is hard-set and no keys are
  configured.

---

## 8. Reproducing everything

```bash
ROUTER_MODE=replay python scripts/run_all_ladders.py   # all 3 ladders, $0.00
python -m llm_routing.stats     --results runs/results.wide.jsonl
python -m llm_routing.scorecard --results runs/results.wide.jsonl
```

The published figures were produced by **deleting every derived artefact** and
regenerating all three ladders from the response cache. 0 calls reached a
backend; 0 rows are simulated.

Analysis entry points, each runnable as `python -m llm_routing.<name>`:
`run_eval` (per-policy run), `frontier` (cost-quality curves and AUC), `stats`
(McNemar and paired bootstrap), `scorecard` (per-policy error attribution),
`routable` (the cross-tab), `sweep_degraded` (verifier-degradation curve),
`plot` (SVGs).

Paid tools, each of which prints a costed plan and refuses to spend without
`--go`: `scripts/redraw_decisive.py` (redraw or screen),
`scripts/record_missing.py` (buy the gap a replay needs).

`docs/NOTES.md` is the open-issues list and is more detailed than §5 here.

---

## 9. What is worth doing next

1. **A verifier that needs no shipped tests** — the cheap model generating its
   own tests, or self-consistency over code. This addresses limitation §5.1,
   which currently bounds every claim in this file, and is more informative than
   another ladder.
2. **A ladder with a large price ratio and a small capability gap**, to separate
   the confound in §2.4.
3. **Redraw `both_ok` and `inverted`** to turn the routable fraction from a
   lower bound into a two-sided estimate. Costed at roughly $14; not obviously
   worth it.
