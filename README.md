# Cascade vs Predictive Routing

Measuring when each LLM routing architecture is worth its cost.

Two ways to spend less on LLM inference, with opposite failure modes:

- **Predictive routing.** Inspect the query, guess whether it is hard, commit to
  one model. Pays once. Misroutes silently, and never finds out.
- **Cascade routing.** Call the cheap model, verify the answer, escalate only on
  failure. Never misroutes an easy query. Double-pays on every escalation.

"Does routing work" is settled. The question here is **where the crossover is**,
and the variable this repo manipulates to find it is **verifier quality** — the
one thing a cascade depends on and a predictive router does not have at all.

> ### Status: no real model has been called
>
> Everything in this repo currently runs in **mock mode**, which fabricates model
> replies from answers already stored in the task set. The pipeline is complete
> and verified end to end, but **no accuracy figure produced by it means anything
> yet**, because no API key has been used and no model has been asked anything.
>
> Every simulated number is labelled as such: at the top and bottom of every run,
> above every table, and in a `"simulated": true` field on every output row.
> `results.jsonl` is deliberately not committed, so no fabricated percentage is
> published here. See [NOTES.md](NOTES.md) for what a real run would change.
>
> **Coming back to this cold? Read [STATUS.md](STATUS.md).** It says what state the
> project is in, what to run next in order, what the paid run costs, and what to
> expect to change.
>
> **Want to understand the code?** [WALKTHROUGH.md](WALKTHROUGH.md) traces one
> real task through every file. For the plain-language version of the *ideas*,
> read [EXPLAINED.md](EXPLAINED.md).

## Run it

Mock mode needs nothing installed. It is pure standard library, offline, and
byte-deterministic — including the figures.

```bash
python3 build_taskset.py     # builds taskset.jsonl from data/
python3 sanity_check.py      # regression gate: must print 40/40 and 60/60
python3 run_eval.py --policy always_cheap --split all   # difficulty probe
python3 splits.py            # the calibration / evaluation split
python3 run_eval.py          # every policy, reported on the held-out half
python3 frontier.py          # cost-quality curves and the AUC comparison
python3 stats.py             # paired significance tests over results.jsonl
python3 sweep_degraded.py    # the verifier-degradation curve
python3 plot.py              # figures/*.svg, no matplotlib
```

Switch model ladders with one variable. This is the main experimental knob:

```bash
ROUTER_LADDER=claude   python3 run_eval.py   # 1x / 3x / 5x  (default)
ROUTER_LADDER=deepseek python3 run_eval.py   # 1x / 3.1x, one provider
ROUTER_LADDER=wide     python3 run_eval.py   # 1x / 36x, cross-provider
```

Real mode. Do these in order — the second one is the decision point, and it costs
one or two orders of magnitude less than a full run:

```bash
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
# then open .env and paste your key(s) in
```

`.env` is gitignored, so a key cannot be committed by accident. A real environment
variable overrides the file, so you can still do a one-off without editing it back.
Which keys you need depends on the ladder: `claude` wants `ANTHROPIC_API_KEY`,
`deepseek` wants `DEEPSEEK_API_KEY`, `wide` spans both providers and wants both.

A Claude Pro/Max subscription does **not** include API access — separate products,
separate billing. Keys come from the Claude Console.

Then, in order — the second is the decision point and costs far less than a run:

```bash
# 1. plumbing check: keys resolve, prompts parse, nothing truncates
ROUTER_MODE=real python3 run_eval.py --limit 10

# 2. difficulty probe: does the task set discriminate at all? ~$0.01-$0.09
ROUTER_MODE=real python3 run_eval.py --policy always_cheap --split all
```

The probe answers the pilot gate on its own, because the gate depends only on the
cheap rung's failure rate. Ten tasks cannot answer it — at n=10 that rate carries a
±28-point confidence interval, which spans the entire acceptable band.

Every response from that paid run lands in `cache/raw_calls.<ladder>.jsonl`, one
file per ladder so they never mix. Afterwards everything is free forever, and
reproducible by anyone with no key at all:

```bash
ROUTER_MODE=replay python3 run_eval.py         # same numbers, no network
ROUTER_MODE=replay python3 sweep_degraded.py   # every sweep point, $0
```

A full run over all 100 tasks and every policy is under $2 per ladder; see
[STATUS.md §3](STATUS.md) for the per-ladder breakdown.

## The task set

100 tasks in two domains, chosen because their **verification regimes differ**:

| domain | source | n | grading | runtime verifier |
|---|---|---|---|---|
| math | MATH500, levels 3–5 | 60 | exact match on the normalised answer | **proxy** — self-consistency over k samples |
| code | sanitized MBPP | 40 | execute the shipped asserts | **free and perfect** — just run the tests |

No LLM judge anywhere. Every verdict is deterministic, so results are
reproducible byte for byte and there is no judge to calibrate.

Half the set is held out. `splits.py` splits it deterministically, stratified by
domain and difficulty; thresholds and quality estimators are fitted on the
calibration half and `run_eval.py` reports on the other half by default.

## The model ladder is the main variable

The price ratio between rungs drives the whole economics, so the ladder is
selected rather than hard-coded. Prices and API contracts verified against
provider docs on 2026-07-30, at list rates — promotional pricing is deliberately
ignored, because an experiment whose cost axis expires is not reproducible.

| `ROUTER_LADDER` | rungs | list ratio | effective ratio |
|---|---|---|---|
| `claude` (default) | Haiku 4.5 → Sonnet 5 → Opus 5 | 1x / 3x / 5x | 1x / 3.9x / 6.5x |
| `deepseek` | v4-flash → v4-pro | 1x / 3.1x | 1x / 3.1x |
| `wide` | DeepSeek v4-flash → Opus 5 | 1x / 36x | 1x / 46x |

Effective differs from list because Claude 4.7 and later use a newer tokenizer
emitting roughly 30% more tokens for the same text. On the `claude` ladder the
bottom rung is on the old tokenizer and everything above it on the new one, so the
rungs disagree about how many tokens the same prompt *is*. That works against
escalation and is modelled explicitly rather than ignored.

DeepSeek is reached through its Anthropic-format endpoint, so the whole thing needs
one SDK and no provider abstraction beyond a base URL. Two properties of that
ladder are worth noting: both rungs accept a `temperature`, which Claude's upper
rungs do not, so it is the only configuration that can run a self-consistency
verifier at *every* rung; and its endpoint silently remaps unknown model names to
the cheap model instead of erroring, which is guarded at import because a typo
would otherwise produce plausible, wrong, paid-for numbers.

### The finding this made possible

Matched on accuracy, against simply always paying for the best rung:

| ladder | ratio | cascade vs always-best |
|---|---|---|
| `deepseek` | 3.1x | cascade costs **33% more** |
| `claude` | 6.5x effective | cascade **12% cheaper** |
| `wide` | 46x effective | cascade **74% cheaper** |

**The sign flips.** Cascading pays in proportion to the price gap it exploits, and
below roughly 3x the wasted cheap call and its verification cost more than they
save. A cascade always pays for the cheap call and for verifying it; those are
fixed costs, and what they buy is the *chance* to skip an expensive call. When
"expensive" is only 3x "cheap", the fixed costs swamp the saving.

Averaged over the whole budget range instead of at matched accuracy, the cascade
beats a cost-matched coin flip on every ladder (+4.8% / +7.2% / +8.8% AUC). So it
is always the better *router*; it is not always cheaper than not routing at all.
Two different questions, both reported.

These are mock-mode numbers, but unlike the accuracy figures they come from the
verified price tables and the escalation logic rather than from `MOCK_SKILL`, which
is why [STATUS.md](STATUS.md) expects this one to survive a real run.

## The policies

| policy | what it does |
|---|---|
| `always_<rung>` | one per rung of the loaded ladder, generated not listed |
| `random_matched` | coin flip at predictive's own escalation rate — **the null hypothesis** |
| `predictive` | hand-written heuristic, routes once on pre-call features |
| `routellm` | RouteLLM's pretrained `bert` router, threshold-matched to `predictive` |
| `llm_router` | the cheap model classifies its own difficulty, then answers |
| `cascade` | answer → verify → escalate, over every rung of the ladder |
| `cascade_routing` | **routing and cascading unified** — see below |
| `cascade_degraded` | the cascade with a deliberately damaged verifier — **the experiment** |
| `oracle` | hindsight-optimal. Bounds how good any router could be |

Four of those exist for reasons worth stating outright:

- **`random_matched` is the null hypothesis.** A router that escalates the same
  fraction of tasks *at random* also gains accuracy — it just pays for it.
  Without this baseline, a gap between `predictive` and `always_cheap` shows only
  that spending more helps. This is not a pedantic point: LLMRouterBench
  ([arXiv:2601.07206](https://arxiv.org/abs/2601.07206)) finds that under unified
  evaluation many published routers, commercial ones included, fail to reliably
  beat a simple baseline.
- **`cascade_routing` is the literature's answer to this repo's own framing.**
  Dekoninck et al. ([arXiv:2410.10347](https://arxiv.org/abs/2410.10347), ICML
  2025) prove that routing and cascading are both special cases of one strategy
  parameterised by a single λ, and that the unified version beats either alone.
  It differs from `cascade` in two ways that matter: it need not start at the
  bottom rung, and it need not climb one rung at a time. Their headline conclusion
  is also independently this project's thesis — *quality estimation is the
  deciding factor* — which is exactly what `cascade_degraded` manipulates.
- **`cascade_degraded` is the experiment, not a variant.** With only the two
  natural verifiers, verifier quality has two levels and they are perfectly
  confounded with task domain — perfect/code/MBPP/asserts/free versus
  proxy/math/MATH500/exact-match/k-extra-calls. Five things differ at once, so
  any result could equally be read as "code is different from math". Corrupting
  `verify_code` *inside* the code domain holds everything else fixed and turns
  two points into a curve.
- **`oracle` bounds the others, and that is checked.** It enumerates the same
  action space the deployable policies have — every rung, plus majority voting over
  cheap samples, which the math cascade uses. `run_eval` prints an explicit bound
  check because an earlier version of this repo shipped an oracle that the cascade
  could beat, which silently invalidated every routing-skill figure.

## Policies are curves, not points

Every policy here has a knob: an agreement threshold, a difficulty cutoff, a score
threshold, a λ. Turning any of them buys accuracy with money. So comparing two
policies at one setting each compares two arbitrary points, and the winner can be
changed by turning either knob. That is the most common way routing results
mislead — "our router beat the cascade" usually means "our router was tuned to
spend more".

`frontier.py` sweeps every knob across its full range and compares the resulting
curves, following RouterBench's cost-quality convex hull
([arXiv:2403.12031](https://arxiv.org/abs/2403.12031)). It reports:

- each family's **achievable frontier** — the upper convex hull of its operating
  points, which is the right object because any point between two achievable
  settings is itself achievable by randomising between them;
- **AUC**, the mean accuracy across the whole budget range, so it reads in the
  units of accuracy and answers "how good is this at every budget" rather than
  "how good is it at the budget somebody tuned it to";
- the same number for the random baseline, and the gap;
- **who owns each budget** on the combined frontier, and which families contribute
  no point to it at all — a far stronger statement than losing one table row.

## Do the differences survive a significance test?

`stats.py` runs exact McNemar tests and paired bootstraps over a short
pre-registered list of comparisons. Paired, because every policy answered the same
tasks from the same cached responses, which is what the response cache is for.

The current answer, on the held-out half, is **no** — the accuracy gaps that look
decisive in the report table have confidence intervals spanning zero, while
several cost differences are comfortably significant. That is a real result about
the task set size and it is written up as item 2 of [NOTES.md](NOTES.md) rather
than quietly omitted.

## Files

| file | what it is |
|---|---|
| `build_taskset.py` | MATH500 + sanitized MBPP, stratified sample, unified schema |
| `graders.py` | deterministic grading: exact match (math), run asserts (code) |
| `models.py` | model client, mock / real / replay, price table, cost accounting |
| `response_cache.py` | one draw per distinct call, shared by every policy |
| `splits.py` | the calibration / evaluation split, stratified and deterministic |
| `policies.py` | the policies and the three verifiers |
| `run_eval.py` | batch runner, spend cap, report, oracle bound check |
| `frontier.py` | cost-quality curves, achievable frontiers, AUC |
| `stats.py` | exact McNemar and paired bootstrap over the results |
| `sweep_degraded.py` | the experiment: cascade quality against verifier quality |
| `routellm_router.py` | RouteLLM's pretrained router, cost-matched to `predictive` |
| `plot.py` | SVG figures from the standard library, no matplotlib |
| `sanity_check.py` | regression gate; exits non-zero if a grader is broken |

## The tuneable decisions

Each is marked `DECISION #n` in the source next to the code it controls. Change
one, re-run, and the report shows what moved.

1. **Model ladder** (`models.py`) — the price ratio drives the whole economics
2. **Self-consistency k** (`policies.py`) — failure detection against cost, linear
3. **Agreement threshold** (`policies.py`) — when to accept the cheap answer
4. **Predictive heuristic** (`policies.py`) — route once, blind, on pre-call features
5. **Verifier corruption rate** (`policies.py`) — the manipulated variable
6. **Random baseline seeds** (`policies.py`) — the null the others are measured against
7. **LLM-as-router** (`policies.py`) — the option decision 4 rejected, now measured
8. **RouteLLM variant and threshold** (`routellm_router.py`) — a learned router, cost-matched
9. **Cascade routing λ** (`policies.py`) — the unified strategy's quality/cost price

The ladder itself is the tenth and largest knob, and the one that changes the
conclusion rather than the numbers. See `ROUTER_LADDER` above.

## Why there is a response cache

Not for speed. Every policy calls the models independently, so the same cheap
greedy call is made several times per task. In mock mode the duplicates are
identical for free; in real mode they would be hundreds of extra paid calls
returning *different* answers.

That is a validity problem rather than a cost one. Every paired statistic this
project wants assumes the policies are compared on the same model outputs.
Without the cache, `always_cheap` and `cascade` would disagree partly because of
decoding noise, and the oracle would be bounding draws that nobody else received.

A cache hit still charges the policy in full, because `cost_usd` answers "what
would this cost in production", and in production there is no cross-policy cache.
The reports print both numbers: what the policies would each pay, and what this
run actually spent.

## The RouteLLM comparison

`routellm` sits out unless `cache/routellm_scores.jsonl` exists. Those scores are
committed, so the policy runs for everyone with no API key, no HuggingFace
access, no torch and no GPU. To regenerate them:

```bash
pip install routellm==0.2.0
python3 routellm_router.py --score     # bert variant, local, no API key
python3 routellm_router.py             # show calibration and routing decisions
```

Of RouteLLM's five routers, `bert` is the only one that is both genuinely learned
and free to serve — `mf` and `sw_ranking` call OpenAI's embedding API on every
prompt, and `causal_llm` needs a gated 16GB checkpoint. `routellm_router.py` has
the full table.

## Where this sits in the literature

- FrugalGPT, [arXiv:2305.05176](https://arxiv.org/abs/2305.05176) — the cascade baseline
- RouteLLM, [arXiv:2406.18665](https://arxiv.org/abs/2406.18665) — the learned predictive router used here
- AutoMix, [arXiv:2310.12963](https://arxiv.org/abs/2310.12963) — self-verification and escalation
- RouterBench, [arXiv:2403.12031](https://arxiv.org/abs/2403.12031) — the cost-quality hull and AUC
- Dekoninck et al., [arXiv:2410.10347](https://arxiv.org/abs/2410.10347) — routing and cascading unified
- LLMRouterBench, [arXiv:2601.07206](https://arxiv.org/abs/2601.07206) — published routers often fail to beat a simple baseline

"Isn't this just FrugalGPT?" — largely yes, and deliberately: it is a replication
with 2026 models. What is added is the manipulation. FrugalGPT and AutoMix take
their verifier as given; Dekoninck et al. identify quality-estimator accuracy as
the factor that decides whether any of this works, and test it by injecting
synthetic Gaussian noise into a quality signal. This repo instead **degrades a real
verifier by a controlled amount on objectively-graded tasks**, holding the domain,
the models, the prompts and the grader fixed. That is the one thing here that is
not a replication.

## Open issues

Tracked honestly in [NOTES.md](NOTES.md), including the ones that would weaken
the headline. The largest is at the top of this file: nothing has been measured
yet.
