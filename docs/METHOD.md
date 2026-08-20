# METHOD — how the benchmark is built, and how to run it

The long-form companion to [README](../README.md) and [RESULTS](RESULTS.md).
The README says what was found and RESULTS gives every number; this file says
how, and why each choice was made the way it was.

Every figure quoted here is either measured on real models and replayable from
`cache/`, or labelled as a projection. Nothing on this page is carried over from
a superseded run.

- [The question](#the-question)
- [The task set](#the-task-set)
- [The model ladder is the main variable](#the-model-ladder-is-the-main-variable)
- [The policies](#the-policies)
- [Policies are curves, not points](#policies-are-curves-not-points)
- [Significance testing](#significance-testing)
- [The degradation sweep](#the-degradation-sweep)
- [The RouteLLM comparison](#the-routellm-comparison)
- [Why there is a response cache](#why-there-is-a-response-cache)
- [The tuneable decisions](#the-tuneable-decisions)
- [Running it](#running-it)
- [Running it for real](#running-it-for-real)
- [Serving it](#serving-it)
- [Bugs this project found in itself](#bugs-this-project-found-in-itself)
- [Where this sits in the literature](#where-this-sits-in-the-literature)

---

## The question

Two ways to spend less on LLM inference, with opposite failure modes:

- **Predictive routing.** Inspect the query, guess whether it is hard, commit to
  one model. Pays once. Misroutes silently, and never finds out.
- **Cascade routing.** Call the cheap model, verify the answer, escalate only on
  failure. Never misroutes an easy query. Double-pays on every escalation.

"Does routing work" is settled. The question here is **where the crossover is**,
and the variable this repository manipulates to find it is **verifier quality** —
the one thing a cascade depends on and a predictive router does not have at all.

## The task set

417 tasks in two domains, chosen because their **verification regimes differ**:

| domain | source | n | grading | runtime verifier |
|---|---|---|---|---|
| math | MATH-500, level 5 | 60 | exact match on the normalised answer | **proxy** — self-consistency over k samples |
| code | MBPP+ | 357 | execute the expanded evalplus suite | **free and perfect** — just run the tests |

**No LLM judge anywhere.** Every verdict is deterministic, so results are
reproducible byte for byte and there is no judge to calibrate.

### Why these two datasets

Any replacement has to satisfy two hard constraints or it breaks the experiment
rather than improving it. **The code half must ship runnable tests** — the whole
project rests on one domain having a free and perfect verifier, which is only
possible because MBPP ships assert statements that can be executed. **The maths
half must have an exact-matchable answer**, because a judge would introduce a
calibration problem the project would then have to solve, and would make the
results irreproducible.

**MBPP+ over MBPP moves exactly one variable.** MBPP+ is the *same problems* as
sanitized MBPP with roughly 35x more test cases each, so the task distribution is
unchanged and the only thing that moves is how thorough the marking is.
Demonstrated rather than asserted — on task 3 (`is_not_prime`), a solution that
forgets the `n == 1` case **passes all four original MBPP asserts and fails the
expanded suite**.

The model is shown the original thin asserts as its specification and graded
against the expanded suite; both are carried in `grader_payload`. Putting the
expanded suite in the prompt would be ten kilobytes of fuzzed input/output pairs
— an absurd prompt and most of an answer key — and would change the task rather
than the marking.

**A few tasks must be dropped, and the set is platform-specific.** MBPP+'s
assertion helper only reaches `np.allclose` when the expected value is a float or
a flat sequence of floats. For a nested tuple or a complex number it falls
through to exact `==`, so the verdict turns on whether your libm matches the
machine the expectations were generated on. Task 590 (`polar_rect`, expected
`(tuple, complex)`) passes on Linux and fails on Windows.
`scripts/provenance/fetch_mbppplus.py` therefore validates every reference
solution against its own expanded suite **on the machine that will run the
evaluation**, and drops the failures with a reason. Dropping is not tidying: the
mock emits the reference as its "correct" answer, so a task whose own reference
fails would have every mock-correct answer graded wrong.

```bash
python scripts/provenance/fetch_mbppplus.py                 # writes data/mbppplus.json
python scripts/provenance/fetch_mbppplus.py --validate-only # re-check without downloading
```

Grading the code half therefore needs **numpy**: every expanded evalplus test
program opens with `import numpy as np` and compares floats with `np.allclose`.
See `pyproject.toml`, extra `code`. Source:
<https://huggingface.co/datasets/evalplus/mbppplus>

**MATH-500 level 5 is a difficulty floor.** Level 5 alone leaves 134 candidates,
comfortably more than the 60 sampled. It was raised from levels 3–5 for one
reason: the probe showed the cheap rung solving 10 out of 10 at the old setting,
which leaves a router nothing to decide.

That had a known cost, and it cost more than expected. A single level makes
`difficulty_proxy` constant across the maths half, so the shipped `level` field
carries no signal at all — which is what killed the original hand-written
predictive policy. Its predicate `level >= 5` was true for every maths task, so
it was `always_expensive` on that half by construction while being reported as a
router. Predictive routing is now measured with `llm_router` and `routellm`,
neither of which reads a difficulty label. Recoverable with `--min-math-level 3`,
which restores two distinct levels.

**What was rejected, and why:**

| dataset | why not |
|---|---|
| **GSM8K** | current cheap models score in the low 90s, leaving a cascade nothing to route |
| **BigCodeBench** (1140 tasks) | genuinely harder, but its `test` field is a full `unittest` class rather than a list of asserts, and its tasks import real third-party libraries — the grader's subprocess would need them installed, ending the "runs on a bare interpreter" property |
| **LiveCodeBench** | contamination-controlled by release date, which is attractive, but it is competitive programming: stdin/stdout pairs rather than asserts, so the grader would need rewriting rather than adapting |
| **Omni-MATH** (4.4K olympiad) | the real upgrade if MATH-500 saturates, and it ships a `difficulty` float 4–10 that would restore a predictive router's signal. Deferred because some answers are symbolic expressions with free variables, where exact match is fragile in a way MATH-500's mostly-numeric answers are not |
| **AIME 2025** (30 problems) | integer answers make exact match trivially clean, and it is very hard — but too small to be the maths half alone |

**After any swap, re-check three things in order.** First
`python -m llm_routing.sanity_check` — the graders must still score every
reference answer at full marks; on a new maths set this is the step that catches
unmatched answer formats, and it exits non-zero so it cannot be skipped by
accident. Then the two-arm probe (`run_eval --policy always_cheap --policy
always_expensive --split all`, then `routable --real`), because a new dataset
changes the routable fraction and if there is nothing to route there is nothing
to measure. Only then, the full run. Change one thing at a time: each step moves
the failure rate, and doing two together means not knowing which one worked — the
same confound the `cascade_degraded` design exists to avoid.

### The quarantine rule

**13 tasks are quarantined as unpassable-by-specification**, each with the
disputed input recorded as evidence in `llm_routing.build_taskset.QUARANTINED`.

A `both_fail` task is either genuinely hard or broken by its own specification,
and the cross-tab cannot tell them apart. Getting it wrong is expensive both
ways: an unpassable task silently caps every policy, and deleting a
hard-but-solvable one removes the signal the experiment exists to measure. So
the bar is deliberately hard to clear — **a task may be quarantined only if every
rung's multi-draw p̂ is exactly 0.** One greedy draw cannot establish that.

The bar earns its strictness. Redrawing all 24 `both_fail` candidates before
adjudicating cost $0.25 and rescued three that a single draw had condemned —
including `codeplus-305`, whose top rung solves it half the time, which makes it
precisely a *routable* task rather than a hopeless one. `TestQuarantine` enforces
the rule against the historical data, and
`scripts/provenance/triage_both_fail.py` gathers the evidence and deliberately
refuses to make the call itself.

Half the set is held out. `llm_routing/splits.py` splits it deterministically,
stratified by domain and difficulty; thresholds and quality estimators are
fitted on the calibration half, and `run_eval` reports on the other half by
default. Every published accuracy is therefore over n=209.

### What the two-arm probe found

Before any policy is worth running, there has to be something to route. The
probe answers every task at the bottom rung and at the top rung and cross-tabs
the two. On the `wide` ladder, 417 tasks, real responses
(`runs/results.probe.jsonl`, `simulated: false`):

| | n | cheap right | top right | **routable** | both fail |
|---|---|---|---|---|---|
| code | 357 | 82.9% | 91.6% | **13.2%** | 14 |
| math | 60 | 81.7% | 96.7% | **16.7%** | 1 |
| **all** | **417** | **82.7%** | **92.3%** | **13.7%** | 15 |

*routable* — the cheap rung got it wrong and the top rung got it right — is the
only cell a router can win. 95% CI [10.7%, 17.3%]; the ceiling over
always-cheap is 17.7 points, and every policy in this repository competes over
that band.

**Run both arms, not one.** A single-arm probe measures `P(cheap fails)`, which
is `routable + both_fail` — it cannot tell a task the top rung would fix from
one it would fail too, and only the first kind is worth routing.

**And one draw per cell overstates it.** A single draw cannot distinguish "the
cheap model cannot do this" from "the cheap model usually can and missed once".
Redrawing the decisive cells three times put about a sixth of the apparent
opportunity down to one model having a bad draw (13.5% → 11.3%) — see
[RESULTS.md §2.6](RESULTS.md), which is the number to quote.

## The model ladder is the main variable

The price ratio between rungs drives the whole economics, so the ladder is
selected rather than hard-coded. Prices and API contracts were verified against
provider docs, at list rates — promotional pricing is deliberately
ignored, because an experiment whose cost axis expires is not reproducible.

| `ROUTER_LADDER` | rungs | list | effective | realized |
|---|---|---|---|---|
| `claude` (default) | Haiku 4.5 → Sonnet 5 → Opus 5 | 1x / 3x / 5x | 1x / 3.9x / 6.5x | — |
| `deepseek` | v4-flash → v4-pro | 1x / 3.1x | 1x / 3.1x | — |
| `wide` | DeepSeek v4-flash → Opus 5 | 1x / 35.7x | 1x / 46.4x | **1x / 117x** |

Three ratios, because they answer three different questions and only the last is
a measurement. **List** and **effective** are arithmetic over *input* prices, so
they cannot be wrong about the price table — but the great majority of the bill
on this workload is *output* tokens, where the `wide` rungs are far further
apart. **Realized** is `findings.realized_ratio`: the ratio the provider
actually billed, computed from the cached greedy answers over the tasks where
both rungs answered. It is *higher* than the quoted figure, and the direction
matters — understating the price gap understates the case for cascading.

Effective differs from list because Claude 4.7 and later use a newer tokenizer
emitting roughly 30% more tokens for the same text. On the `claude` ladder the
bottom rung is on the old tokenizer and everything above it on the new one, so
the rungs disagree about how many tokens the same prompt *is*. That works
against escalation, and is modelled explicitly rather than ignored.

DeepSeek is reached through its Anthropic-format endpoint, so the whole thing
needs one SDK and no provider abstraction beyond a base URL. Two properties of
that ladder are worth noting: both rungs accept a `temperature`, which Claude's
upper rungs do not, so it is the only configuration that can run a
self-consistency verifier at *every* rung; and its endpoint silently remaps
unknown model names to the cheap model instead of erroring, which is guarded at
import, because a typo would otherwise produce plausible, wrong, paid-for
numbers.

### The price ratio does not decide it

The intuition the project started from: cascading pays in proportion to the
price gap it exploits, because a cascade always pays for the cheap call *and*
for verifying it, and what those fixed costs buy is the *chance* to skip an
expensive one. Below some ratio the fixed costs swamp the saving.

The measurement, from each ladder's own frontier run (n=209, real responses):

| ladder | effective ratio | cascade vs always-best, matched accuracy | AUC over the coin flip | verdict |
|---|---|---|---|---|
| `deepseek` | 3.11x | **−4.4%** (cheaper) | +2.2 | cascade |
| `claude` | 6.5x | **+11.7%** (dearer) | +5.3 | route |
| `wide` | 46.4x | **−83.1%** (much cheaper) | +6.9 | cascade |

**That is not monotonic, and the middle row is the interesting one.** `claude`
has the *higher* price ratio of the two close ladders and is the one where
cascading costs more, because what decides the outcome is not the price gap
alone but how much verification costs on that ladder. On `claude` the cheap
rung is Haiku and the maths half draws five samples from it, and the middle rung
does not accept a temperature so it cannot be verified at all. On `deepseek`
both rungs accept one, and the top rung is barely better, so the cascade rarely
escalates and rarely pays twice.

The router therefore *computes* this rather than quoting it:
`router_agent/findings.py` reads `runs/frontier.<ladder>.jsonl`, which is
committed for every ladder, and refuses to guess a verdict for a ladder that has
none.

> **A confound this design cannot separate.** What the data distinguishes is the
> **capability gap** between rungs as much as the price gap, and here the two
> move together: the ladder with the small price ratio is also the ladder whose
> rungs are equally capable. Three ladders cannot tell them apart. The supported
> claim is *"cascading pays when the top rung is genuinely better and
> verification is cheap."* See [RESULTS.md §2.4](RESULTS.md).

## The policies

Nine policies in two families, plus the fixed rungs and two baselines.

| policy | family | what it does |
|---|---|---|
| `llm_router` | **predictive** | the cheap model classifies its own difficulty, then answers |
| `routellm` | **predictive** | RouteLLM's pretrained `bert` router at a fixed score threshold |
| `cascade` | **cascading** | answer → verify → escalate, over every rung of the ladder |
| `cascade_routing` | **cascading** | routing and cascading unified — see below |
| `cascade_degraded` | **cascading** | the cascade with a deliberately damaged verifier — **the experiment** |
| `always_<rung>` | fixed | one per rung of the loaded ladder, generated not listed |
| `random_matched` | null | coin flip at `llm_router`'s own escalation rate — **the null hypothesis** |
| `oracle` | bound | hindsight-optimal. Bounds how good any router could be |

A hand-written `predictive` policy — routing on MATH-500's shipped difficulty
level — was **deleted**. Under `MIN_MATH_LEVEL = 5` that level
is constant, so the policy sent all 60 maths tasks to the expensive rung and was
`always_expensive` on the maths half by construction while being reported as a
router. Predictive routing is half of this repository's subject and has not gone
anywhere; it is now measured with the two implementations above, neither of
which reads a difficulty label. The reasoning is preserved as a tombstone at
`DECISION #4` in `llm_routing/policies.py`.

Four of these exist for reasons worth stating outright:

- **`random_matched` is the null hypothesis.** A router that escalates the same
  fraction of tasks *at random* also gains accuracy — it just pays for it.
  Without this baseline, a gap between a router and `always_cheap` shows only
  that spending more helps. This is not a pedantic point: LLMRouterBench
  ([2601.07206](https://arxiv.org/abs/2601.07206)) finds that under unified
  evaluation many published routers, commercial ones included, fail to reliably
  beat a simple baseline — which is what happens here, on all three ladders.
  One anchored null cannot serve policies at different spending levels, so
  `run_eval` also computes an analytic null at *each* policy's own cost; this
  row is the empirical check that the analytic one is not a fiction.
- **`cascade_routing` is the literature's answer to this repo's own framing.**
  Dekoninck et al. ([2410.10347](https://arxiv.org/abs/2410.10347), ICML 2025)
  prove that routing and cascading are both special cases of one strategy
  parameterised by a single λ, and that the unified version beats either alone.
  It differs from `cascade` in two ways that matter: it need not start at the
  bottom rung, and it need not climb one rung at a time. What is implemented
  here is their **greedy** variant, not the full algorithm.

  **It does not win here, and that is the point of measuring it.** It trails the
  plain `cascade` on every ladder — 95.2% against 95.7% on `wide`, 96.2% against
  96.7% on `claude`, 84.2% against 86.6% on `deepseek`. The paper's result turns
  on a good **ex-ante** quality estimator, and this task set has none: the only
  pre-call feature available was constant (`DECISION #4`), so `_Q_EXANTE` carries
  a domain prior and an empty slot. Read these rows as the unified strategy with
  only its post-hoc half working — which is the paper's own prediction. Every
  figure above is in `runs/results.<ladder>.jsonl`.
- **`cascade_degraded` is the experiment, not a variant.** With only the two
  natural verifiers, verifier quality has two levels and they are perfectly
  confounded with task domain — perfect/code/MBPP+/asserts/free versus
  proxy/math/MATH-500/exact-match/k-extra-calls. Five things differ at once, so
  any result could equally be read as "code is different from maths". Corrupting
  `verify_code` *inside* the code domain holds everything else fixed and turns
  two points into a curve.
- **`oracle` bounds the others, and that is checked.** It enumerates the same
  action space the deployable policies have — every rung, plus majority voting
  over cheap samples, which the maths cascade uses. `run_eval` prints an
  explicit bound check, because an earlier version of this repository shipped an
  oracle the cascade could beat, which silently invalidated every routing-skill
  figure.

## Policies are curves, not points

Every policy here has a knob: an agreement threshold, a difficulty cutoff, a
score threshold, a λ. Turning any of them buys accuracy with money. So comparing
two policies at one setting each compares two arbitrary points, and the winner
can be changed by turning either knob. That is the most common way routing
results mislead — "our router beat the cascade" usually means "our router was
tuned to spend more".

`llm_routing/frontier.py` sweeps every knob across its full range and compares
the resulting curves, following RouterBench's cost-quality convex hull
([2403.12031](https://arxiv.org/abs/2403.12031)). It reports:

- each family's **achievable frontier** — the upper convex hull of its operating
  points, which is the right object because any point between two achievable
  settings is itself achievable by randomising between them;
- **AUC**, the mean accuracy across the whole budget range, so it reads in units
  of accuracy and answers "how good is this at every budget" rather than "how
  good is it at the budget somebody tuned it to";
- the same number for the random baseline, and the gap;
- **who owns each budget** on the combined frontier, and which families
  contribute no point to it at all — a far stronger statement than losing one
  table row.

## Significance testing

`llm_routing/stats.py` runs exact McNemar tests and paired bootstraps over a
short **pre-registered** list of comparisons. Paired, because every policy
answered the same tasks from the same cached responses, which is what the
response cache is for.

Exact McNemar rather than the χ² approximation, and a paired bootstrap rather
than a t-test, because the discordant-pair counts here are small — the whole
routing signal lives in a couple of dozen tasks per ladder. `scipy` is
deliberately not a dependency: an exact test on a 2x2 table is a few lines.

At n=209 the answer is **4 of 8 comparisons detectable on `wide` and on
`claude`, 1 of 8 on `deepseek`**. An earlier version of this repository reported
0 of 8 and summarised itself as *"cost differences are measurable, accuracy
differences are not"*; **that summary is retracted** — it was a statement about
a 47-task sample, and what bought the change was the code half going from 35
tasks to 357.

A comparison that does not reach significance is reported as **unresolved**,
never as a tie. That is a statement about the sample size, not about the
policies.

## The degradation sweep

`llm_routing/sweep_degraded.py` is the one thing here that is not a replication:
it degrades a *real* verifier by a controlled amount, holding the domain, the
models, the prompts and the grader fixed. It runs on the code half, on real
model responses, for $0.00 — the code verifier runs the tests and makes no model
calls, so every point replays from answers already paid for.

357 code tasks, `wide` ladder, mean over corruption draws per level. `p` is the
probability the verifier ignores the test result and guesses.

| corrupt `p` | eff. AUC | accuracy | cost/task | escalation |
|---|---|---|---|---|
| 0.00 | 1.000 | **96.1%** | **$0.000584** | 17.1% |
| 0.10 | 0.950 | 95.0% | $0.000619 | 19.0% |
| 0.25 | 0.875 | 92.2% | $0.000773 | 26.1% |
| 0.50 | 0.750 | 93.0% | $0.000937 | 32.2% |
| 0.75 | 0.625 | 89.9% | $0.001078 | 43.1% |
| 1.00 | 0.500 | 89.1% | $0.001239 | 47.3% |

**Cost rises monotonically with verifier corruption, and accuracy falls across
the range** — 96.1% to 89.1%, seven points, for a verifier degraded from perfect
to a coin flip. The fall is not strictly monotonic: p=0.50 comes in slightly
above p=0.25, which is the size of the noise on a 357-task grid at these
corruption levels and should not be read as a reversal.

The trap worth stating plainly: **a cascade with a worthless verifier still
looks good on $/correct.** At p=1.00 a coin flip sends half the traffic to the
cheap rung and saves money doing it. Cost-per-correct therefore cannot
distinguish a good verifier from no verifier, and only the accuracy-matched
comparison can. `sweep_degraded` prints both and says which is which.

The caveats, in order of size: one ladder, and the code half is the half whose
free perfect verifier **does not transfer to production** — MBPP+ ships the
tests, and a user's query does not. That is the sharpest open problem in this
repository; see [LIMITATIONS.md](LIMITATIONS.md).

## The RouteLLM comparison

`routellm` sits out unless `cache/routellm_scores.jsonl` exists. The scores are
committed, so the policy replays for anyone with no key, no torch and no GPU.
Regenerating is free and local, but does need torch:

```bash
pip install routellm==0.2.0
python -m llm_routing.routellm_router --score   # bert variant, local, no key
python -m llm_routing.routellm_router           # threshold and decisions
```

**The score distribution is the first result, and it arrives before any accuracy
is measured.** Across all 417 tasks the router's `strong_win_rate` spans
**[0.499, 0.899]** with a median of **0.790**. It judges the weak model more
likely to win on exactly one task out of 417. So the semantically natural
threshold — 0.5, "escalate when the strong model is favoured" — routes **416 of
417 tasks** to the expensive rung, and the policy degenerates into
`always_expensive`: exactly the failure the old `predictive` policy was deleted
for. The threshold used instead is a declared constant, **0.80**, which splits
the set 189/228 and is derived from nothing else in this repository. See
`DECISION #8b` in `llm_routing/routellm_router.py`.

That compression is what "out of distribution" looks like in practice.
`bert_gpt4_augmented` was trained to predict which answer a human would *prefer*
between two chat models; it is being asked about competition maths and MBPP+,
where it cannot judge, and it defaults to "the big one" every time. **Preference
is not correctness.** The measured consequence, on all three ladders: RouteLLM's
frontier AUC sits at or *below* a cost-matched coin flip everywhere.

Of RouteLLM's five routers, `bert` is the only one that is both genuinely
learned and free to serve — `mf` and `sw_ranking` call OpenAI's embedding API on
every prompt, and `causal_llm` needs a gated 16GB checkpoint.
`llm_routing/routellm_router.py` has the full table.

## Why there is a response cache

Not for speed. Every policy calls the models independently, so the same cheap
greedy call is made several times per task. In mock mode the duplicates are
identical for free; in real mode they would be hundreds of extra paid calls
returning *different* answers.

That is a validity problem rather than a cost one. Every paired statistic this
project wants assumes the policies are compared on the same model outputs.
Without the cache, `always_cheap` and `cascade` would disagree partly because of
decoding noise, and the oracle would be bounding draws nobody else received.

A cache hit still charges the policy in full, because `cost_usd` answers "what
would this cost in production", and in production there is no cross-policy
cache. The reports print both numbers: what the policies would each pay, and
what the run actually spent.

The key hashes everything that determines a response and nothing that does not —
mode, model id, prompt, temperature, sample index, max tokens, mock seed.
**Deliberately absent: the ladder.** That is what makes three ladders
affordable: `wide`'s Opus answers serve `claude`'s top rung and `wide`'s flash
answers serve `deepseek`'s bottom rung, for nothing. Cross-ladder reuse is worth
about $1.70 of the $8.51 spent.

## The tuneable decisions

Each is marked `DECISION #n` in the source next to the code it controls. Change
one, re-run, and the report shows what moved.

1. **Model ladder** (`models.py`) — the price ratio drives the whole economics
2. **Self-consistency k** (`policies.py`) — failure detection against cost, linear
3. **Agreement threshold** (`policies.py`) — when to accept the cheap answer
4. ~~**Predictive heuristic**~~ — **retracted**, tombstoned in
   place rather than renumbered. The feature was constant; see `policies.py`
5. **Verifier corruption rate** (`policies.py`) — the manipulated variable
6. **Random baseline rate** (`policies.py`) — the null, anchored to `llm_router`
7. **LLM-as-router** (`policies.py`) — the option decision 4 rejected, now measured
8. **RouteLLM variant** (`routellm_router.py`) — which learned router, and why
   `bert` — and **8b**, its operating threshold, fixed at 0.80 rather than
   calibrated
9. **Cascade routing λ** (`policies.py`) — the unified strategy's quality/cost price

The ladder itself is the tenth and largest knob, and the one that changes the
conclusion rather than the numbers.

## Running it

Mock mode needs nothing installed. It is pure standard library, offline, and
byte-deterministic — including the figures.

```bash
python -m llm_routing.build_taskset     # data/taskset.jsonl from data/
python -m llm_routing.sanity_check      # regression gate on both graders
python -m llm_routing.routable          # is there anything to decide?
python -m llm_routing.splits            # calibration / evaluation split
python -m llm_routing.run_eval          # every policy, on the held-out half
python -m llm_routing.frontier          # cost-quality curves and AUC
python -m llm_routing.stats             # paired significance tests
python -m llm_routing.sweep_degraded    # the verifier-degradation curve
python -m llm_routing.plot              # figures/*.svg, no matplotlib
```

Everything derived lands in `runs/`. Every writer takes an output override, so a
second ladder cannot silently overwrite the first — `scripts/run_all_ladders.py`
is the driver that does all three properly.

Switch model ladders with one variable. This is the main experimental knob:

```bash
ROUTER_LADDER=claude   python -m llm_routing.run_eval   # 1x / 3x / 5x (default)
ROUTER_LADDER=deepseek python -m llm_routing.run_eval   # 1x / 3.1x, one provider
ROUTER_LADDER=wide     python -m llm_routing.run_eval   # 1x / 46x, cross-provider
```

> **Shell note.** `ROUTER_LADDER=x python ...` is bash syntax and does nothing in
> PowerShell. Rather than remember three shells' worth of syntax, put the setting
> in `.env` and drop the prefix — the repository loads it, and a real environment
> variable still overrides it:
>
> ```
> ROUTER_LADDER=deepseek
> ROUTER_MODE=replay
> ```

Replay mode reruns everything above against the committed responses, with no
key and no network, for $0.00:

```bash
ROUTER_MODE=replay python scripts/run_all_ladders.py
```

## Running it for real

Real mode spends money. Do these in order — the second is the decision point,
and it costs one or two orders of magnitude less than a full run.

```bash
pip install -e ".[real]"
cp .env.example .env        # Windows: copy .env.example .env
# then open .env and paste your key(s) in
```

`.env` is gitignored, so a key cannot be committed by accident. A real
environment variable overrides the file, so a one-off run needs no edit-and-undo.
Which keys you need depends on the ladder: `claude` wants `ANTHROPIC_API_KEY`,
`deepseek` wants `DEEPSEEK_API_KEY`, `wide` spans both providers and wants both.

A Claude Pro/Max subscription does **not** include API access — separate
products, separate billing. Keys come from the Claude Console.

```bash
# 1. plumbing: keys resolve, prompts parse, nothing truncates
ROUTER_MODE=real python -m llm_routing.run_eval --limit 10

# 2. the two-arm probe: is there anything for a router to decide?
ROUTER_MODE=real python -m llm_routing.run_eval \
    --policy always_cheap --policy always_expensive --split all
ROUTER_MODE=replay python -m llm_routing.routable --real --ladders wide
```

Ten tasks cannot answer question 2 either — at n=10 the routable rate carries a
±28-point confidence interval, which spans the entire acceptable band.

There is a hard per-run spend cap in `models.call`, next to the one line that
can charge a card, overridable per run rather than by editing:

```bash
ROUTER_MAX_SPEND_USD=8 ROUTER_MODE=real python -m llm_routing.run_eval
```

Two paid tools each print a costed plan and refuse to spend without `--go`:
`scripts/provenance/redraw_decisive.py` (redraw or screen a cell) and
`scripts/provenance/record_missing.py` (buy exactly the calls a full replay is missing).

Every response from a paid run lands in `cache/raw_calls.<ladder>.jsonl`, one
file per ladder so they never mix. Afterwards everything is free forever, and
reproducible by anyone with no key at all.

**Estimates run low.** The `claude` ladder buy came in 52% over its estimate:
the call count was exact (1489 against 1491) and the per-call cost was 53% high,
because the estimate used Opus token counts as a length proxy and weaker models
write longer answers to the same question. A cheaper model is not a
proportionally cheaper call.

## Serving it

```bash
llm-router --demo                          # real cached data, no key, $0
llm-router "What is 17 * 23?"              # needs ROUTER_MODE=real + a key
llm-router --estimate "prove X"            # price every policy, no calls
llm-router --findings                      # what the benchmark measured
llm-router "..." --approve-above 0.01      # pause for approval before spending
```

As an MCP server, so any MCP client can route through it:

```json
{
  "mcpServers": {
    "llm-routing": {
      "command": "llm-router-mcp",
      "env": { "ROUTER_LADDER": "wide", "ROUTER_MODE": "replay" }
    }
  }
}
```

Four tools (`route_query`, `estimate_cost`, `compare_policies`,
`explain_routing`), four resources under `routing://`, and a prompt that walks a
client through choosing a policy. `ROUTER_MODE=replay` is the safe thing to
register: it cannot spend money. [ARCHITECTURE.md](ARCHITECTURE.md) covers what
the serving layer had to solve that the benchmark did not.

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

## Where this sits in the literature

- FrugalGPT, [2305.05176](https://arxiv.org/abs/2305.05176) — the cascade baseline
- RouteLLM, [2406.18665](https://arxiv.org/abs/2406.18665) — the learned predictive router used here
- AutoMix, [2310.12963](https://arxiv.org/abs/2310.12963) — self-verification and escalation
- RouterBench, [2403.12031](https://arxiv.org/abs/2403.12031) — the cost-quality hull and AUC
- Dekoninck et al., [2410.10347](https://arxiv.org/abs/2410.10347) — routing and cascading unified
- LLMRouterBench, [2601.07206](https://arxiv.org/abs/2601.07206) — published routers often fail to beat a simple baseline
- Agreement-Based Cascading, [2407.02348](https://arxiv.org/abs/2407.02348) — **states this repo's crossover as a threshold**
- Routing-gap decomposition, [2607.03436](https://arxiv.org/abs/2607.03436) — **and qualifies its headline**

Two of those deserve more than a line.

**The crossover has been published, from a different direction.**
*Agreement-Based Cascading* (TMLR 07/2025) uses ensemble agreement as its
deferral signal — a generalisation of the `self_consistency` verifier here — and
states the same crossover as a cost ratio: at a price ratio of 5x or less,
sequential cascading yields minimal savings and needs parallel execution to pay
at all. It also names the worst case this repository's `deepseek` ladder shows:
when nearly everything escalates, a k-model cascade can cost (k+1)x the
expensive model alone. Their 5x and this repository's ~3x are not in conflict —
their ratio is a raw token-price one, while the `effective_ratio` used here
already absorbs verification cost and the measured escalation rate. Independent
corroboration, and the more useful kind: arrived at by a different route.

**The routable fraction is a single-draw estimate, and that is a known bias.**
*How Much of the Routing Gap Is Real?* (July 2026) decomposes exactly this
measurement into reproducible specialist advantage and single-draw label noise,
and puts the noise share highest on MATH-500 — which is what the maths half of
this task set is. That prediction was tested here on independent data and it
held: redrawing the decisive cells moved the routable fraction from 13.5%
observed to 11.3% reproducible. See [RESULTS.md §2.6](RESULTS.md).

"Isn't this just FrugalGPT?" — largely yes, and deliberately: a replication with
2026 models. What is added is the manipulation. FrugalGPT and AutoMix take their
verifier as given; Dekoninck et al. identify quality-estimator accuracy as the
deciding factor but test it by injecting synthetic Gaussian noise into a quality
signal. This repository **degrades a real verifier by a controlled amount on
objectively-graded tasks**, holding everything else fixed. That is the one thing
here that is not a replication.
