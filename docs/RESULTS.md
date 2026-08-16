# Results

Every number here is measured on real models and replayable from `cache/` for
$0.00. Nothing on this page is carried over from a superseded run.

For how the benchmark is built, read [METHOD.md](METHOD.md). For what bounds
these claims, [LIMITATIONS.md](LIMITATIONS.md). For the operational rules and
the bugs the project found in itself, [ENGINEERING.md](ENGINEERING.md).

---

## 1. What was measured

Ten policies over 417 tasks on three model ladders. Every response is a real
API call, committed to `cache/`, and the entire analysis regenerates offline
for $0.00 with no API key.

| | |
|---|---|
| tasks | **417** — 357 MBPP+ code, 60 MATH-500 level 5 |
| ladders | **3** — `wide` (flash→opus), `claude` (haiku→sonnet→opus), `deepseek` (flash→pro) |
| policies | **10**, including an oracle bound and a cost-matched random null |
| real responses | **5,075** |
| total spend | **$8.5145** |
| tests | **210 passing**, plus one end-to-end reconciliation against the committed results behind `pytest -m slow` |

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

**"Cheaper and better" is a `wide` result, and `claude` is the counter-example.**
The ladder has to be named for the claim to mean anything.

### 2.2 Most of a routing benchmark does no work

**4 of 8** pre-registered comparisons are detectable at n=209 on `wide` and on
`claude`; 1 of 8 on `deepseek`.

At n=100 it was **0 of 8**, and the reason is worth more than the result. Growing
the code half from 35 tasks to 357 raised the number of tasks that can
distinguish two routers *at all* from **7 to 75** on `wide` (56 routable + 19
inverted; 76 on `claude`, 31 on `deepseek`). Every other task is one both rungs
get right or both get wrong, and contributes nothing but denominator.

**Size a routing benchmark by its discordant pairs, not by its task count.**

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

Whole task set, not the evaluation split — this is the cross-tab that says
whether a ladder has anything to route, so it uses every task. `n` differs by
ladder because a response that hit `max_tokens` is unmeasured, and a task
missing either rung cannot be placed in any cell.

| ladder | n | cheap rung | top rung | gap | routable | both_fail | McNemar |
|---|---|---|---|---|---|---|---|
| `wide` | 416 | 82.9% | 92.1% | **+9.2** | 56 | 15 | **0.000** |
| `claude` | 417 | 82.7% | 92.3% | **+9.6** | 58 | 14 | **0.000** |
| `deepseek` | 415 | 82.9% | 82.7% | **−0.2** | 15 | 56 | 1.000 |

Reproduce with `python -m llm_routing.routable --real --ladders <ladder>`; the
committed output is `runs/routable.<ladder>.txt`. **The `--real` flag is not
optional** — without it the module runs the mock ladder and prints a
plausible-looking simulated cross-tab.

> **This table is reproducible to about one task.** The code grader executes
> candidate solutions in a subprocess and treats a timeout as a failure, so a
> task whose expanded suite runs near the ceiling can grade wrong on a loaded
> machine and right on an idle one. Regenerating this cross-tab while three
> other analyses were running moved exactly one task from `both_ok` to
> `inverted` and the top rung from 92.1% to 91.8%. The cell counts that carry
> the argument — `routable` and `both_fail` — did not move, and neither did the
> McNemar p. See [LIMITATIONS.md](LIMITATIONS.md).

On `deepseek` the expensive rung is **not measurably better than the cheap one**.
Fifteen of 415 tasks are routable against 56 hopeless ones, and the whole
accuracy dynamic range is 7.5%. No policy can win what is not there — which is
why `always_expensive` lands *below* a cost-matched coin flip on that ladder.

> **A confound this design cannot separate.** What the data distinguishes is the
> **capability gap** between rungs as much as the price gap, and here the two
> move together: the ladder with the small price ratio is also the ladder whose
> rungs are equally capable. Three ladders cannot tell them apart.
>
> The supported claim is *"cascading pays when the top rung is genuinely better
> and verification is cheap."* Separating the two needs a ladder with a large
> price ratio and a small capability gap.

### 2.5 The price ratio does not decide whether to cascade

The project set out to test a price-ratio crossover: cascading should lose below
some ratio and win above it. Each ladder's own frontier run, n=209:

| ladder | effective ratio | cascade vs always-best, matched accuracy | verdict |
|---|---|---|---|
| `deepseek` | 3.11x | **−4.4%** (cheaper) | cascade |
| `claude` | 6.5x | **+11.7%** (dearer) | route |
| `wide` | 46.4x | **−83.1%** (much cheaper) | cascade |

**Not monotonic, and the middle row is why.** `claude` has the higher ratio of
the two close ladders and is the one where cascading costs more — because the
deciding term is not the price gap but **what verification costs on that
ladder**. On `claude` the cheap rung is Haiku and the maths half draws five
samples from it, and the middle rung refuses a temperature so it cannot be
verified at all. On `deepseek` both rungs accept one and the top rung is barely
better, so the cascade rarely escalates and rarely pays twice.

A router that picked its strategy from a price-ratio threshold would get
`claude` and `deepseek` backwards. This one reads the frontier instead, and
refuses to answer for a ladder that has none.

> **The `deepseek` frontier is one policy short.** `cascade_routing` is not
> replayable from that ladder's cache, so it contributes no point to the
> `deepseek` curve while it does to the other two. The run says so on every
> execution (*"not replayable from this cache, so absent from the frontier
> below"*), and it is stated here because the row above sits next to two that
> are complete. The `deepseek` verdict rests on `cascade` against
> `always_expensive`, both of which are fully measured.

### 2.6 A sixth of the routing opportunity is noise

71 decisive tasks, 3 fresh draws at both rungs, $0.3429.

| measure | routable fraction | counts |
|---|---|---|
| observed | 13.5% | one draw per cell — what a probe publishes |
| expected | 12.2% | mean over fresh draws |
| **reproducible** | **11.3%** | cheap reliably fails **and** the top rung reliably succeeds |

**The routable cell shrinks by 2.2 points, a sixth of its apparent size, once
you ask whether it reproduces.** A router credited against `observed` is paid
for mass it cannot capture twice running. (The `noise_share` field reports the
same effect per-cell at 7.2%; the sixth is the reduction in the headline
fraction, 13.5% → 11.3%.)

This is a *lower bound* on the correction: `both_ok` and `inverted` were not
redrawn, so flakiness hidden there is still uncounted.

### 2.7 Greedy decoding is not deterministic, for either provider

Across 21 tasks with ≥5 draws at temperature 0, more than one distinct answer
came back on **76%** of tasks for `claude-opus-5` and **67%** for
`deepseek-v4-flash`. Neither provider is meaningfully more stable than the
other.

This is why §2.6 exists: a benchmark that takes one draw per model per task is
partly measuring luck, and cannot tell you how much.

### 2.8 The cascade degrades smoothly with its verifier — the experiment

Everything above compares policies. This is the only part that *manipulates* a
variable: `sweep_degraded.py` corrupts the perfect code verifier by a controlled
amount, inside the code domain, holding the models, prompts, grader and tasks
fixed. `p` is the probability the verifier ignores the test result and guesses.

357 code tasks, `wide` ladder, real responses, $0.00 to run.

| corrupt `p` | 0.00 | 0.10 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|
| accuracy | **96.1%** | 95.0% | 92.2% | 93.0% | 89.9% | 89.1% |
| $/task | **0.00058** | 0.00062 | 0.00077 | 0.00094 | 0.00108 | 0.00124 |
| escalation | 17.1% | 19.0% | 26.1% | 32.2% | 43.1% | 47.3% |

Cost rises monotonically and accuracy falls across the range — seven points for
a verifier taken from perfect to a coin flip. **There is no cliff**, which is
the useful part: no threshold exists below which verifier quality stops
mattering, so it has to be budgeted for rather than assumed.

The trap this also exposes: **a cascade with a worthless verifier still looks
good on cost-per-correct.** At `p = 1.00` a coin flip sends half the traffic to
the cheap rung and saves money doing it. Only the accuracy-matched comparison
separates a good verifier from no verifier, and the sweep prints both.

### 2.9 Resampling the cheap rung does not substitute for escalating

A cascade has one move when verification fails: climb. That is a choice, and at
a 117x realized price ratio it is not an obvious one — for the price of a single
Opus call you could take a hundred more DeepSeek draws. *Resample or Reroute?*
([2607.08665](https://arxiv.org/abs/2607.08665)) frames the two as competing
uses of one budget and reports cascades losing on saturated tasks.

Answered here on the decisive tasks, from draws already on disk, for $0.00:

| | escalate once | best-of-9 cheap | cost of the cheap option |
|---|---|---|---|
| code (n=5) | **5/5** | 0/5 | 7% |
| math (n=10) | **9/10** | 4/10 | 7% |

**Escalation wins, and it is not close.** Nine extra cheap draws at 7% of the
cost recover none of the code tasks and fewer than half the maths ones. On the
tasks a cascade actually escalates, the cheap rung does not have the answer at
any sample count — the capability is missing, not the luck.

Their result and this one do not conflict: theirs is about saturated tasks where
both rungs already succeed, and this measures the cell where they differ, which
is the only cell a router can win.

---

## 3. What each policy got right and wrong

`llm_routing/scorecard.py` joins each policy's decision against what the two
rungs could actually do. `wide` ladder, 209 tasks:

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

`always_mid` is measured on `claude`, the only three-rung ladder: **88.5%**,
solving **12 of 22** routable tasks without ever reaching Opus. The middle rung
does real work.

---

## 4. What it cost

| | |
|---|---|
| `wide` cache | $5.5878 |
| `claude` cache | $2.8498 |
| `deepseek` cache | $0.0769 |
| **total** | **$8.5145** over 5,075 real responses |

**Cross-ladder cache reuse is worth about $1.70.** The ladder is deliberately
absent from the cache key, so `wide`'s Opus answers serve `claude`'s top rung
and `wide`'s flash answers serve `deepseek`'s bottom rung, for nothing.

**Estimates run low.** The `claude` buy came in **52% over** a $1.87 estimate;
the call count was exact (1489 vs 1491) and the per-call cost was 53% high,
because the estimate used Opus token counts as a length proxy and weaker models
write longer answers to the same question. A cheaper model is not a
proportionally cheaper call.

---

## 5. What bounds these findings

Two things matter most, and both are measured rather than asserted:

1. **The verifier that produces the signal is not the verifier that ships.** The
   code half is graded by executing the tests MBPP+ supplies; a deployed router
   does not have them. §2.8 is the experiment that prices that gap.
2. **Price ratio and capability gap are confounded** — see the note in §2.4.

[LIMITATIONS.md](LIMITATIONS.md) states these and six smaller ones once each,
with what would settle them.

---

## 6. Reproducing everything

```bash
ROUTER_MODE=replay python scripts/run_all_ladders.py   # all 3 ladders, $0.00
python -m llm_routing.stats     --results runs/results.wide.jsonl
python -m llm_routing.scorecard --results runs/results.wide.jsonl
```

The full three-ladder driver takes **about 55 minutes** and prints one line per
step. For a quick look, `--ladders wide` is a third of that.

The published figures were produced by **deleting every derived artefact** and
regenerating all three ladders from the response cache. 0 calls reached a
backend; 0 rows are simulated.

Analysis entry points, each runnable as `python -m llm_routing.<name>`:
`run_eval` (per-policy run), `frontier` (cost-quality curves and AUC), `stats`
(McNemar and paired bootstrap), `scorecard` (per-policy error attribution),
`routable` (the cross-tab), `sweep_degraded` (verifier-degradation curve),
`plot` (SVGs).

`routable` is the one exception to "the mode comes from `ROUTER_MODE`": it takes
`--real`, and without that flag it runs the mock ladder regardless of the
environment. Every other entry point reads `ROUTER_MODE`.

Paid tools, each of which prints a costed plan and refuses to spend without
`--go`: `scripts/redraw_decisive.py` (redraw or screen),
`scripts/record_missing.py` (buy the gap a replay needs).

---

## 7. Further work

1. **A verifier that needs no shipped tests** — the cheap model generating its
   own tests, or self-consistency over code. This addresses the limitation that
   currently bounds every claim on this page, and is more informative than
   another ladder.
2. **A ladder with a large price ratio and a small capability gap**, to separate
   the confound in §2.4.
3. **Redrawing `both_ok` and `inverted`**, to turn the routable fraction from a
   lower bound into a two-sided estimate.
