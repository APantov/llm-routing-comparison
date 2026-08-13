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

Both difficulty settings were raised on 6 August 2026, because the probe showed
the cheap rung solving essentially everything at the previous ones. MBPP+ is the
same problems as sanitized MBPP with roughly 35x more test cases, so the swap
moves exactly one variable — how thorough the marking is — and the model is
still shown the original thin asserts as its specification. The easier maths
floor remains one flag away: `--min-math-level 3`. See
[DATASETS.md](DATASETS.md).

Grading the code half needs **numpy**: every expanded evalplus test program
opens with `import numpy as np` and compares floats with `np.allclose`.

**No LLM judge anywhere.** Every verdict is deterministic, so results are
reproducible byte for byte and there is no judge to calibrate.

**13 tasks are quarantined as unpassable-by-specification**, each with the
disputed input recorded as evidence in `llm_routing.build_taskset.QUARANTINED`.
The bar for quarantining is deliberately hard to clear — every rung's multi-draw
p̂ must be exactly 0 — because getting it wrong is expensive in both directions.
[ENGINEERING.md](ENGINEERING.md) has the full rule and the reversal that motivated
it.

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
Redrawing the decisive cells three times put roughly a fifth of the apparent
opportunity down to one model having a bad draw — see
[RESULTS.md §2.6](RESULTS.md), which is the number to quote.

## The model ladder is the main variable

The price ratio between rungs drives the whole economics, so the ladder is
selected rather than hard-coded. Prices and API contracts were verified against
provider docs on 2026-07-30, at list rates — promotional pricing is deliberately
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

Ten policies in two families, plus the fixed rungs and two baselines.

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
level — was **deleted on 8 August 2026**. Under `MIN_MATH_LEVEL = 5` that level
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
4. ~~**Predictive heuristic**~~ — **retracted 8 August 2026**, tombstoned in
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
`scripts/redraw_decisive.py` (redraw or screen a cell) and
`scripts/record_missing.py` (buy exactly the calls a full replay is missing).

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
