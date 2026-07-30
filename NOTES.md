# Open issues and limitations

Kept in the repo rather than in a private note, because a project whose subject is
measurement honesty should be honest about its own state. Roughly ordered by how
much each one would change a conclusion.

## Blocking

### 1. Real models have been called once, and the run was degenerate

The plumbing check has been done. On 30 July 2026, `ROUTER_MODE=real
ROUTER_LADDER=wide run_eval.py --limit 10` produced **47 real model responses**
(cached in `cache/raw_calls.wide.jsonl` 45, `.claude.jsonl` 1, `.deepseek.jsonl` 1)
and **47 rows in `results.jsonl`** with `"mode": "real"`, `"simulated": false`, for
**$0.0481**. So the repo is past "nothing has ever been called".

It is not past "there is no result". All eleven policies scored **100% on all five
reported tasks**, so that run contains zero accuracy information — every pairwise
comparison is a tie by construction and the routing-skill denominator is 0/0. Every
accuracy figure the repo *reports* is still simulated. Everything below is written
on the assumption that a real run at usable scale is what is missing.

The remaining path costs little:

```bash
cp .env.example .env      # then paste your key in; .env is gitignored
# ROUTER_MODE=real python3 run_eval.py --limit 10   # plumbing check — DONE, $0.048
ROUTER_MODE=real python3 run_eval.py --policy always_cheap --split all  # the gate
ROUTER_MODE=real python3 run_eval.py              # full run
```

Then commit the ladder's cache file (`cache/raw_calls.claude.jsonl`, or whichever
ladder was run) and everything after that is free and reproducible for anyone, via
`ROUTER_MODE=replay`.

### 2. The task set is too small for the comparisons being made

This is now measured rather than suspected. `python3 stats.py` runs exact McNemar
tests and paired bootstraps over every pre-registered comparison, and on the
held-out half **none of them reaches significance**. Not one. The accuracy gaps
that look decisive in the report table — five or six points between adjacent
policies — have confidence intervals comfortably spanning zero.

What *is* significant is cost: several cost differences have intervals that
exclude zero by a wide margin. So the current honest summary of the whole project
is:

> Cost differences between routing architectures are measurable at n=100.
> Accuracy differences are not.

That is a real finding and it should be reported rather than buried, but it also
means the headline comparison needs more tasks. The response cache makes this
affordable: scale the task set up for one paid run, and every sweep afterwards
stays free. A rough power calculation for detecting a five-point paired difference
at these discordance rates needs several hundred tasks, not one hundred.

Two things partly mitigate it in the meantime:

- `frontier.py` compares whole curves rather than single points, which pools
  information across every operating point and is correspondingly less noisy than
  any one row of the table.
- The degradation sweep averages over 200 corruption draws per level, so its
  curve is an estimate with a stated spread rather than a single realisation.

## Weakens the headline

### 3. Mock mode makes majority voting far too strong

`models._wrong_answer` scatters wrong answers across distinct values, so
self-consistency recovers the truth whenever 2 of 5 samples happen to be right.
Real models cluster on the *same* wrong answer much more often, which is why
published self-consistency gains are low single digits rather than the large gain
mock mode shows.

Consequence: the maths half's cascade result is inflated in mock mode, and it will
drop — possibly a lot — on the first real run. Do not quote the mock figure.

### 4. Hyperparameters are calibrated, but the calibration is thin

Four numbers are free parameters. Two control the cascade's verifier, two control
the predictive router's guess. All four live in `policies.py` next to the code
they govern, marked `DECISION #2`, `#3` and `#4`.

| constant | value | what it controls | which way it biases |
|---|---|---|---|
| `SELF_CONSISTENCY_K` | 5 | how many times `verify_math` samples the model before judging whether it agrees with itself | higher k detects failure better and costs linearly more, so it trades the cascade's accuracy against its cost |
| `AGREEMENT_THRESHOLD` | 0.8 | what fraction of those samples must give the same answer for the cheap answer to be accepted. At k=5 this means 4 of 5 | higher escalates more often: better accuracy, more double-paying |
| `PREDICTIVE_HARD_LEVEL` | 5 | the MATH500 difficulty level at or above which `predictive` pays for the expensive model | sets how much of the maths half goes expensive, so it sets the router's whole cost position |
| `PREDICTIVE_CODE_CHARS` | 100 | the prompt-length cutoff, in characters, above which a code task is called hard | same, for the code half; chosen to match the maths half's escalation rate so the two are cost-comparable |

`splits.py` now holds out half the task set, `run_eval.py --split eval` reports on
the held-out half by default, and `cascade_routing`'s quality estimators are fitted
on the calibration half only. So the worst version of this problem is fixed.

What remains: the four constants above were *originally* chosen while looking at
all 100 tasks, and re-deriving them properly on the calibration half has not been
done — they are inherited values that the split now merely protects from getting
worse. `frontier.py` sidesteps the issue for comparison purposes by sweeping each
knob across its whole range instead of trusting any single setting.

`PREDICTIVE_CODE_CHARS` is the weakest of the four: it is pure calibration rather
than a discovered signal, so it has no independent justification at all.

Note that this is a separate problem from the mock constants in `models.py`
(`MOCK_SKILL`, `MOCK_ROUTER_SKILL`, `MOCK_TOKENS_OUT`, `MOCK_LATENCY_S`). Those are
not tuned against a result — they *are* the result, in mock mode. See items 3 and 8.

### 5. The predictive router's maths signal is flattering

It reads MATH500's shipped human-assigned difficulty `level`. That is legitimate —
it arrives with the question rather than being derived from the answer — but
production traffic does not come labelled with its difficulty. The maths half is
an optimistic upper bound on what a predictive router can do, and should be
reported that way.

The code half is the honest half, and there the available signal is close to
nothing: prompt length is the best of a weak set, and the report shows what that
is worth.

### 6. Every rung's capability is stipulated in mock mode

Each entry in `models.MODEL_SPECS` carries a `mock` block deciding how often that
model is right. Any mock-mode conclusion about whether an extra rung pays, or about
how DeepSeek compares to Claude, is a restatement of those constants. The ladders
are built and wired so a real run can answer it; mock mode cannot.

The `wide` ladder is the worst case here, because its two rungs come from different
providers, so a capability difference is confounded with a provider difference and
the tokenizer factors are unmeasured. Its COST conclusions are solid; its accuracy
conclusions are the least trustworthy in the repo.

### 7. `cascade_routing` is the greedy variant

Dekoninck et al. derive the optimal strategy by taking an expectation over every
remaining subset of the model ladder, using a per-model variance estimate. What is
implemented here compares stopping against continuing to the single best next
tier. The paper evaluates that simplification too and reports it costs 0.5% to
1.3%, mostly in low-noise settings. Implementing the full version needs a variance
estimate per model per step, which the calibration half at this n cannot support.

### 8. `llm_router` accuracy is not a measurement in mock mode

The mock router is an oracle on the mock's own latent difficulty, corrupted at
rate `1 - MOCK_ROUTER_SKILL`. Its accuracy restates that constant and nothing
else, which is why it tends to outscore every honest router in mock mode. Its
**cost and latency** overhead are real arithmetic on the price table, and those
are the only figures worth taking from it before a real run.

### 9. Latency is modelled as a constant, and it is not one

`MOCK_LATENCY_S` gives each tier a fixed latency. Real latency varies with output
length, load and time of day, and the tail matters more than the mean for anything
user-facing. A cascade's latency story is much worse than its cost story — it pays
a full extra round trip on every escalation, serially — and this repo currently
under-represents that. Latency from a real run would be worth reporting as a
distribution, not a mean.

## Resolved, recorded so they are not reintroduced

### 10. The cascade paid for rungs it had already decided to reject

On the claude ladder the mid rung (Sonnet 5) does not accept a temperature, so
`verify_math` cannot sample it and correctly refuses to accept an answer it cannot
check. The consequence went unnoticed until a task was traced by hand: on every
maths escalation the cascade bought a mid-tier answer whose rejection was certain
before it was requested, then escalated anyway. That was 25 of 25 escalations.

Measured with the guard off versus on, on otherwise identical code, the maths half:
100.0% accuracy either way, cost/task $0.004589 -> $0.003839, a **16% saving for
free**.

Fixed by `_verdict_is_predetermined`, which skips a non-final rung whose verdict is
knowable without seeing its answer. Deliberately narrow: it asks "is the verdict
knowable in advance", not "do I expect a rejection". A cascade that skipped rungs it
merely expected to fail would be a predictive router in disguise, and the whole
comparison rests on the two being different.

Worth noting as a general lesson: this was an API contract (Sonnet 5 rejects
`temperature`) propagating into economics two layers away. Aggregate tables hid it
completely; tracing one task exposed it immediately.

### 11. The oracle did not bound the cascade

`policy_oracle` chose between cheap-greedy and expensive-greedy only, while the
maths cascade also has cheap-majority-of-k available. That action was outside the
oracle's space, so the cascade could and did score above the supposed ceiling,
which silently invalidated every routing-skill figure.

Fixed: the oracle now enumerates the same action space the deployable policies
have, across every rung of `models.TIERS`, in cost order, and charges the cheapest
correct action. `run_eval` prints an explicit bound check every run.

### 12. The pilot gate mislabelled its own band

It accepted a cheap-model failure rate anywhere in 20–55% while printing
"in the 30-40% target band". Fixed: the band is now two named constants and the
gate prints the band it is actually testing against.

### 13. The two tiers use different tokenizers

Claude 4.7 and later use a newer tokenizer that produces roughly 30% more tokens
for the same text. The cheap tier here is on the old tokenizer and both upper
tiers are on the new one, so the effective input price ratios are nearer
1x / 3.9x / 6.5x than the 1x / 3x / 5x the price table suggests. This works
against the cascade, since escalation pays the inflated input cost on top of a
cheap call already made.

Fixed: modelled as `tokenizer_factor` in `models.MODELS`, applied in mock mode
only — real mode gets true counts back from the API.

### 14. The verifier resampled the wrong model

`verify_math` sampled the cheap tier unconditionally. That is correct for a
cheap-first two-tier cascade and wrong for anything else: verifying a mid-tier
answer has to resample the mid tier. Fixed by threading the tier through every
verifier. Two consequences worth knowing, both now handled explicitly:

- verification gets rapidly more expensive as a cascade climbs, because k samples
  at 3x the price is the largest single cost the middle rung adds;
- only the cheap tier accepts a temperature, so higher tiers cannot be
  self-consistency-verified at all. `_self_consistency` returns no signal there
  and the verifier REJECTS rather than accepting, so "no verifier available" can
  never be silently read as "verifier said yes".

### 15. Mock outcomes depended on RNG call order

Mock draws used to come from an unseeded global RNG, so repeat runs of the same
configuration disagreed by more than the effects being measured. Every stochastic
value now goes through `models._draw`, a pure hash of its inputs, which also makes
`--limit 10` reproduce the first ten tasks of a full run exactly — for every
policy except the two that calibrate on the task set they are given.

### 16. Artefacts disagreed about line endings

`taskset.jsonl` was written with CRLF and `results.jsonl` with LF, so the same
code produced byte-different files on different machines and no hash-based
regression gate was possible. Fixed in both places: writers pass `newline=""`, and
`.gitattributes` stops git renormalising on checkout.

## Deliberate choices that look like gaps

- **`cascade_degraded` runs on the code domain only.** That is the design. Its
  purpose is to vary verifier fidelity while holding the domain fixed; running it
  on maths, where the verifier is already a proxy, would compound two sources of
  verifier error and reintroduce the confound it exists to remove.
- **`routellm` and `cascade_routing` sit out when uncalibrated, rather than
  guessing.** A row labelled with a router's name that came from a coin flip would
  be the worst kind of number in the repo.
- **The oracle is excluded from the combined frontier.** It needs the answer in
  order to choose, so including it would collapse the frontier to one point and
  the comparison between deployable policies would say nothing. It is printed
  alongside as the ceiling it is.
- **A cache hit still charges the policy full price.** `cost_usd` answers "what
  would this cost in production", and in production there is no cross-policy
  cache. The two figures are reported separately.
- **`stats.py` compares a short pre-registered list by default.** Every pair is a
  multiple-comparisons problem; mining all 78 pairs for the significant ones is
  how false findings get published. `--all-pairs` exists and prints a warning.
- **No LLM judge anywhere.** Every verdict is deterministic. This constrains which
  datasets can be used, and that constraint is the reason the task set looks the
  way it does.

## Reading

- FrugalGPT, [arXiv:2305.05176](https://arxiv.org/abs/2305.05176) — the cascade baseline
- RouteLLM, [arXiv:2406.18665](https://arxiv.org/abs/2406.18665) — the learned predictive router used here
- AutoMix, [arXiv:2310.12963](https://arxiv.org/abs/2310.12963) — self-verification and escalation
- RouterBench, [arXiv:2403.12031](https://arxiv.org/abs/2403.12031) — where the cost-quality
  convex hull and its area-under-curve summary come from
- Dekoninck, Baader and Vechev, *A Unified Approach to Routing and Cascading for
  LLMs*, [arXiv:2410.10347](https://arxiv.org/abs/2410.10347) — proves routing and
  cascading are special cases of one strategy, and identifies quality-estimator
  accuracy as the deciding factor. This repo's `cascade_routing` policy implements
  their greedy variant, and its central claim is what `sweep_degraded.py` tests
  empirically instead of with synthetic noise.
- LLMRouterBench, [arXiv:2601.07206](https://arxiv.org/abs/2601.07206) — finds that
  under unified evaluation many published routers, including commercial ones, do
  not reliably beat a simple baseline. Directly relevant to why `random_matched`
  exists here.
