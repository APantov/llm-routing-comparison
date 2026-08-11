# Open issues and limitations

Kept in the repo rather than in a private note, because a project whose subject is
measurement honesty should be honest about its own state. Roughly ordered by how
much each one would change a conclusion.

## Blocking

### 0. Five task-set entries are unpassable, and are quarantined for good

**Standing rule, not a one-off correction: a quarantined task is never counted
again — no rerun, no ladder, no figure, nothing the agent layer serves.**

`codeplus-119`, `codeplus-305`, `codeplus-630`, `codeplus-771` and `codeplus-792`
score against whatever the MBPP reference implementation returned on inputs their
prompt never describes. `codeplus-119` is the clearest: it expects
`[1,2,3,4,5,6] -> 7`, the XOR-fold of the reference, and 7 is not in the array. A
textbook-correct solution mismatches 64 of its 110 inputs.

This mattered far more than five tasks should, because they were **all** of
`always_expensive`'s failures on the eval split. Left in, they capped every
policy in the project: `always_expensive` and `oracle` read 92% rather than 100%,
and the code half read 80% rather than 100%. **This file's own issue 18 — "the
routing signal and the usable verifier are in different halves" — was argued
partly from `both_fail` counts that were this artefact.** Code `both_fail` is now
zero and its rescue rate is 100%.

The general lesson, and the reason `llm_routing/routable.py` now prints a review queue:
**"both rungs failed" reads as "hard" and is equally consistent with "broken".**
The cross-tab cannot tell, nobody looked for four months, and the two need
different responses. Any task landing in `both_fail` is unmeasured until someone
runs a textbook solution against its tests.

**They were deleted outright on 9 August 2026, not filtered.** The first fix kept
their responses and screened them out at each point of use; that spread five
broken tasks through the codebase as permanent complexity to preserve $0.1667 of
API calls no rerun could ever use. `scripts/purge_quarantined.py` removed every
trace — 115 cache rows, 10 probe rows, 5 RouteLLM scores, 5 `runs/redraw.wide.json`
entries — and the filters went with them. Committed spend is now **982 responses,
$4.2372**. No result moved: the task set had already excluded them.

Evidence per task is in `build_taskset.QUARANTINED`, which is the only place the
rule now appears in code; the full account is STATUS.md §6 (the quarantine rule). `TestQuarantine`
is a tripwire, not a filter — it fails if a quarantined id turns up in any
artefact, because nothing downstream would catch it any more. Commit `24302ba`
is the last one holding the purged rows.

### 1. One of the ten policies still has no real accuracy data

> **RESOLVED on the `wide` ladder, 8 August 2026.** This issue was written when
> the only real run was a degenerate 5-task plumbing check. The cache now holds
> **982 real responses** for **$4.24**, and `ROUTER_MODE=replay python -m llm_routing.run_eval`
> scores nine policies on real models with `simulated: false` meaning it. The
> table is in STATUS.md §1.
>
> Nine is now *all of them* on this ladder. The count fell from eleven to ten
> when the degenerate `predictive` policy was deleted (STATUS.md §2.3), and the
> last outstanding row was filled when RouteLLM's scores were regenerated for the
> rebuilt task set — free, local, and overdue. `always_mid` is the remaining
> unmeasured policy and it exists only on the three-rung `claude` ladder, which
> has never been run for real.
>
> The last step was small and worth recording, because it is a general lesson
> about caches. A cache is only as complete as the runs that filled it, and the
> two runs that filled this one — a two-arm probe and a 21-task redraw — had no
> reason to make the calls a *cascade* makes. Every cascade verifies maths by
> drawing `SELF_CONSISTENCY_K - 1` extra samples at temperature 0.8;
> `llm_router` makes an 8-token classification call. Neither existed.
>
> `scripts/record_missing.py` enumerates that gap **from the code's own
> constants** rather than by running a policy and recording what it asks for —
> the distinction matters, because a policy's control flow depends on the
> responses it gets, so a stub-driven discovery run would request calls a real
> run never makes and the script would buy them. Two facts make the enumeration
> exact: `_self_consistency` makes zero calls on any tier that rejects a
> temperature, and it is reached only through the maths branches of `verify_math`
> and `policy_oracle`.
>
> 339 calls, **$0.0503**, and four policies went from undroppable-but-unmeasured
> to measured. That is the cheapest unit of progress in this project's history,
> and it sat unnoticed because the mock fallback (issue 19) had been quietly
> filling the hole.
>
> **Still open:** `always_mid` alone, which needs the three-rung `claude` ladder.

The original text follows, kept because its warning about degenerate runs is
still the right warning.

### 1a. Real models have been called once, and the run was degenerate

The plumbing check has been done. On 30 July 2026, `ROUTER_MODE=real
ROUTER_LADDER=wide python -m llm_routing.run_eval --limit 10` produced **47 real model responses**
(cached in `cache/raw_calls.wide.jsonl` 45, `.claude.jsonl` 1, `.deepseek.jsonl` 1)
and **47 rows in `runs/results.jsonl`** with `"mode": "real"`, `"simulated": false`, for
**$0.0481**. So the repo is past "nothing has ever been called".

It is not past "there is no result". All policies scored **100% on all five
reported tasks**, so that run contains zero accuracy information — every pairwise
comparison is a tie by construction and the routing-skill denominator is 0/0. Every
accuracy figure the repo *reports* is still simulated. Everything below is written
on the assumption that a real run at usable scale is what is missing.

The remaining path costs little:

```bash
cp .env.example .env      # then paste your key in; .env is gitignored
# ROUTER_MODE=real python -m llm_routing.run_eval --limit 10   # plumbing check — DONE, $0.048
ROUTER_MODE=real python -m llm_routing.run_eval --policy always_cheap --split all  # the gate
ROUTER_MODE=real python -m llm_routing.run_eval              # full run
```

Then commit the ladder's cache file (`cache/raw_calls.claude.jsonl`, or whichever
ladder was run) and everything after that is free and reproducible for anyone, via
`ROUTER_MODE=replay`.

### 2. The task set is too small for the comparisons being made

> **Superseded in part — read `ROUTABLE_2026-07-30.md` (deleted 9 Aug; in git history) first.**
> "Too small" is the second-order problem. Cross-tabulating cheap-tier success
> against expensive-tier success on the ten tasks with real answers from both rungs
> gives `both succeed = 10, routable = 0`: not one task where the choice of rung
> changed the outcome, so the best and worst conceivable routers score identically
> and every comparison is a tie by construction. The ceiling on the whole
> experiment is `routable + inverted`, and on real data that is 0 points with a
> 95% upper bound of 27.8%. **More tasks cannot fix a ceiling of zero.** Fix the
> task set first; the two-arm probe that decides it costs $0.44.

This is now measured rather than suspected. `python -m llm_routing.stats` runs exact McNemar
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

- `llm_routing/frontier.py` compares whole curves rather than single points, which pools
  information across every operating point and is correspondingly less noisy than
  any one row of the table.
- The degradation sweep averages over 200 corruption draws per level, so its
  curve is an estimate with a stated spread rather than a single realisation.

### 3. `routable = 15%` rests on a single draw per task per rung

The two-arm probe of 6 August is the only real accuracy measurement in this
repository, and every row in `runs/results.probe.jsonl` carries `"calls": ["cheap"]`
or `["expensive"]` — **one draw per (task, tier)**, at temperature 0.

That is the standard protocol, and it has a known bias. *How Much of the Routing
Gap Is Real?* ([arXiv:2607.03436](https://arxiv.org/abs/2607.03436), July 2026)
decomposes exactly this measurement into reproducible specialist advantage and
single-draw label noise:

| benchmark | measured gap | noise share |
|---|---|---|
| GSM8K | 3.3% | 12% |
| **MATH-500** | 10.0% | **36%** |
| GPQA | 42.8% | 13% |

The noise concentrates where few models succeed — on queries only one to three of
eleven models get right it reaches 43%. **The maths half of this task set is
MATH-500 level 5, 60 of the 100 tasks**, and level 5 is precisely that
thin-support regime. So an unknown but probably non-trivial share of the 15
routable tasks are tasks where the cheap rung would sometimes succeed and
happened not to, rather than tasks it cannot do. No single-commit router can
capture those; only resampling can.

This does not overturn the finding — McNemar's p = 0.002 is computed on the
observed discordant pairs and they were observed. It moves the *interpretation*
toward the lower end of the existing 95% CI [9.3%, 23.3%].

> **MEASURED, 7 August 2026. The answer is that about a third of it was noise.**
> `scripts/redraw_decisive.py --draws 10 --go` took ten further draws of both
> rungs on the 21 decisive tasks: 420 calls, **$2.9637**, all committed.
>
> | | observed (1 draw) | expected (10 draws) | reproducible |
> |---|---|---|---|
> | code (40) | 12.5% | 13.8% | 12.5% |
> | math (60) | 16.7% | 7.8% | 6.7% |
> | **total** | **15.0%** | **10.2%** | **9.0%** |
>
> Of the 15 tasks counted routable: **6 solid, 5 flaky, 4 phantom**. A phantom is
> a task the cheap rung solves 10 times out of 10 and simply missed on the
> probe's single draw — `math-96`, `math-147`, `math-432`, `math-481`.
>
> **Amended 9 August 2026 — `math-96` was not a phantom, it was a truncation.**
> Its cheap draw did not "miss"; it was cut off at `MAX_TOKENS` mid-derivation,
> so it never reached a `oxed{}` and the grader had nothing to mark. It is now
> left UNMEASURED rather than graded wrong, which drops the pair from the
> cross-tab entirely: **n 95 → 94, routable 15 → 14, observed 15.8% → 14.9%**.
> Three phantoms remain, all maths.
>
> `expected` barely moves, 0.1021 → 0.1026, and that is the tell: with
> p_cheap = 1.0 the task contributed (1 − p_cheap) × p_expensive = 0 anyway.
> Only the single-draw cross-tab ever thought it was routable mass.
>
> The domain split is the finding. All 5 routable code tasks are solid; all 4
> phantoms and all 5 flaky tasks are maths. **Of 10 routable maths tasks exactly
> one survives.** The maths half carried the higher headline number and almost
> none of the signal — which is what arXiv:2607.03436 predicts for MATH-500
> specifically, tested here on independent data.
>
> Two truncations surfaced (`math-422` cheap, `math-154` expensive) and graded as
> wrong regardless of content, so `both_fail` was overstated too. That is the
> artefact class in [arXiv:2605.07395](https://arxiv.org/abs/2605.07395).
>
> **CLOSED 9 August 2026, and not by raising the cap.** `MAX_TOKENS` is in the
> response-cache key, so raising it strands all 980 committed responses and
> re-charges $4.2352 — the follow-up as originally written was forbidden by the
> invariant it was written under. Truncated draws are instead **dropped from
> p̂'s numerator and denominator**: `math-154` expensive is 0.00 on **9** draws
> rather than 0.00 on 10, and `math-422` cheap rises 0.50 → 0.56. Both counts
> are recorded per task in `runs/redraw.wide.json` under `draws_used`, so a reader
> can see which figures rest on fewer draws.
>
> `models.is_truncated` is now the single definition of the rule, applied
> wherever an analysis grades straight off the cache: `redraw_decisive`,
> `routable.real_verdicts` and `resample_vs_reroute`.
>
> Per-task probabilities are in `runs/redraw.wide.json`. Everything replays for free.

The full protocol in the paper is k ≥ 20 seed-aligned draws per cell, which was
unnecessary here: the noise lives in the cells already identified as decisive, so
redrawing 21 tasks rather than 100 answered it for a fifth of the price.

```bash
ROUTER_MODE=real ROUTER_LADDER=wide python scripts/redraw_decisive.py --draws 10 --go
```

`models.call` already takes a `sample_idx` and the response cache is already
keyed on it, so distinct draws cannot collide and no plumbing was needed. Every
call is written to the ladder's cache, so the re-estimate now replays for free.

**What remains open.** `both_ok` and `inverted` were not redrawn, so a task where
the cheap rung is secretly flaky but happened to succeed still carries its
single-draw verdict. That hides routable mass rather than inventing it, so 10.2%
is a lower bound on the corrected figure. `--cells all` closes it for about
$14 — not obviously worth it, since the direction of the remaining bias is known
and it runs against the phantom effect rather than with it.

### 3b. The quoted price ratio understates the real one by 47%

`findings.price_ratios` computes `effective_ratio` from **input** prices times a
tokenizer factor. On the `wide` ladder that gives **46.4x**. Billed reality over
112 tasks with both rungs cached is **68.2x**.

The cause is not an arithmetic error, it is the wrong rate: on a reasoning
workload roughly 93% of the bill is output tokens, and these rungs are 89x apart
on output (\$25/M against \$0.28/M) versus 36x on input. Pricing a call by its
input rate prices the wrong thing.

`findings.realized_ratio` now measures it from the cache and `price_ratios`
reports it alongside, absent for any ladder never run for real. The constant is
kept because it needs no run; the measured figure is preferred when it exists —
the same discipline `ratio_verdict` already applies to frontier data.

**This runs in favour of the repo's thesis, which is why it matters.** The claim
is that cascading pays in proportion to the price gap. The gap on hard tasks is
half again as large as quoted, so the case for cascading on `wide` is understated
in every table here.

### 3c. Resampling the cheap rung does not substitute for escalating — measured

*Resample or Reroute?* ([arXiv:2607.08665](https://arxiv.org/abs/2607.08665))
argues cascades waste budget by only ever climbing, and reports them losing
22–31% on saturated tasks. At a 68x ratio one Opus call buys **69 DeepSeek
draws**, so the question is live. `scripts/resample_vs_reroute.py` answers it
from the committed redraw, for free.

| | escalate | best/majority-of-5 | -of-9 |
|---|---|---|---|
| **code** (n=10, exact verifier) | 5/10 | **0/10** | **0/10** |
| **math** (n=11, majority vote) | 10/11 | 6/11 | 5/11 |
| math, best-of-9 *oracle* | — | — | 8/11 |

**On code it buys literally nothing.** Nine draws at 12% of the escalation cost
solve zero additional tasks: the cheap rung's code failures are systematic, not
stochastic. Escalation is not one option among several, it is the only move.

**On maths majority voting gets worse past k=5** — 6/11 at k=5, 5/11 at k=7 and
k=9. More draws converge on the model's *systematic* error rather than away from
it. This is issue 4 below, confirmed on real data: wrong answers cluster.

The regime matters and this does not contradict the paper. Their result is for
saturated tasks; this task set is deliberately unsaturated (MATH500 level 5,
MBPP+). Resampling helps where failure is a coin flip, and this set was built so
failure is not.

### 3d. `AGREEMENT_THRESHOLD = 0.8` was chosen blind and appears to be right

Sorted by modal agreement at k=9 on the 11 decisive maths tasks:

| agreement | majority answer |
|---|---|
| 22–67% (n=7) | 1 correct — rejecting is the right call |
| **78%** (n=1) | **WRONG** — `math-94`, confidently wrong |
| 89% (n=4) | 4 correct — accepting is the right call |

The threshold falls exactly between 78% and 89%, so on this sample it separates
them perfectly: zero false accepts. It was set without any real data.

Do not over-read it. n=11, selected by construction, and `math-94` sits one point
below the boundary — a threshold of 0.75 would have accepted a wrong answer with
78% agreement. What the data supports is "0.8 is not obviously wrong", not "0.8
is optimal". Calibrating it properly needs the full task set redrawn.

The useful reframing: agreement is not only a correctness signal, it is a signal
for **whether more cheap draws can help at all**. Low agreement means resampling
will not converge; high agreement means the first draw was probably already
right. Neither case leaves much room for resampling to beat escalation, which is
why the table in 3c looks the way it does.

## Weakens the headline

### 4. Mock mode makes majority voting far too strong

`models._wrong_answer` scatters wrong answers across distinct values, so
self-consistency recovers the truth whenever 2 of 5 samples happen to be right.
Real models cluster on the *same* wrong answer much more often, which is why
published self-consistency gains are low single digits rather than the large gain
mock mode shows.

Consequence: the maths half's cascade result is inflated in mock mode, and it will
drop — possibly a lot — on the first real run. Do not quote the mock figure.

> **MEASURED, 7 August 2026, and the prediction was right in direction and
> gentler in size than feared.** With real self-consistency samples in the cache,
> the maths cascade scores **96.7%** on the held-out half against mock mode's
> 100%, and `always_cheap` scores 86.7%.
>
> The useful decomposition: of the 3 tasks the cascade fixes over `always_cheap`,
> **2 come from the plurality answer and 1 from escalating**, on 2 escalations in
> 30 tasks. So majority voting is doing two thirds of the work — a real effect,
> not a mock artefact, but one that has to be named rather than folded into "the
> cascade works". `verify_math`'s docstring already warned that the maths cascade
> bundles two mechanisms and is therefore not comparable to a bare cheap call.
> This is the measurement of that.
>
> Note the interaction with issue 3c, which found majority voting getting *worse*
> past k=5 on the decisive tasks. Both are true: k=5 helps, more does not, and
> the reason is that extra draws converge on the model's systematic error.

### 5. Hyperparameters are calibrated, but the calibration is thin

Three numbers are free parameters. Two control the cascade's verifier, one sets
the learned router's operating point. They live in `llm_routing/policies.py` and
`llm_routing/routellm_router.py` next to the code they govern, marked `DECISION #2`, `#3`
and `#8b`.

| constant | value | what it controls | which way it biases |
|---|---|---|---|
| `SELF_CONSISTENCY_K` | 5 | how many times `verify_math` samples the model before judging whether it agrees with itself | higher k detects failure better and costs linearly more, so it trades the cascade's accuracy against its cost |
| `AGREEMENT_THRESHOLD` | 0.8 | what fraction of those samples must give the same answer for the cheap answer to be accepted. At k=5 this means 4 of 5 | higher escalates more often: better accuracy, more double-paying |
| `FIXED_THRESHOLD` | 0.80 | the RouteLLM score at or above which `routellm` pays for the expensive model | sets the router's whole cost position. Declared rather than calibrated, so it is not tied to any other policy's spend |

**Two of these were `PREDICTIVE_HARD_LEVEL` and `PREDICTIVE_CODE_CHARS` until
8 August 2026**, and the first is the cautionary tale for this whole section. It
was set to 5 while `MIN_MATH_LEVEL` was 3, and stayed at 5 when the task set was
rebuilt at level 5 — at which point it selected everything and the policy stopped
routing. A free parameter that is never re-derived does not merely go stale; it
can silently become a constant. The policy was deleted (STATUS.md §2.3).

`llm_routing/splits.py` now holds out half the task set, `llm_routing/run_eval.py --split eval` reports on
the held-out half by default, and `cascade_routing`'s quality estimators are fitted
on the calibration half only. So the worst version of this problem is fixed.

What remains: the constants above were *originally* chosen while looking at
all 100 tasks, and re-deriving them properly on the calibration half has not been
done — they are inherited values that the split now merely protects from getting
worse. `llm_routing/frontier.py` sidesteps the issue for comparison purposes by sweeping each
knob across its whole range instead of trusting any single setting.

Note that this is a separate problem from the mock constants in `llm_routing/models.py`
(`MOCK_SKILL`, `MOCK_ROUTER_SKILL`, `MOCK_TOKENS_OUT`, `MOCK_LATENCY_S`). Those are
not tuned against a result — they *are* the result, in mock mode. See items 3 and 8.

### 6. There is no ex-ante quality estimator in this repository

This issue used to read "the predictive router's maths signal is flattering": it
read MATH500's shipped human-assigned `level`, which arrives with the question
rather than being derived from the answer, so it passed the leak test while being
unavailable to any production router.

**It was not flattering, it was absent.** Under `MIN_MATH_LEVEL = 5` the level is
constant across the maths half, so the predicate selected every task and the
policy was `always_expensive` there. It has been deleted (STATUS.md §2.3).

What is left is a genuine gap rather than a broken implementation, and it is
worth stating as one. Dekoninck et al. identify a good **ex-ante** quality
estimator — can I predict this model will do well? — as what routing needs, and
this repository does not have one. The evidence:

| candidate feature | AUC for "cheap rung fails" | AUC for "routable" |
|---|---|---|
| code `prompt_chars` | 0.688 | 0.586 |
| code `n_asserts` | 0.450 | 0.457 |
| math `level` | 0.500 (constant) | 0.500 |
| math `prompt_chars` | 0.510 | 0.460 |

Note how the one feature with any signal loses most of it when the target changes
from *hard* to *routable* — because on the code half 5 of 40 tasks defeat both
rungs, so a hardness feature partly detects tasks escalation cannot fix.

`policies._Q_EXANTE` now records the absence in its shape: it keys on
`(tier, domain, EXANTE_FEATURE)` with the feature slot empty, so `cascade_routing`
runs with a deliberately uninformative ex-ante term and only the post-hoc half
working. `EXANTE_FEATURE` is the hook a real estimator would plug into.

The **post-hoc** side is the half this repo does have, on the code domain, and it
is what `llm_routing/sweep_degraded.py` manipulates.

### 7. Every rung's capability is stipulated in mock mode

Each entry in `models.MODEL_SPECS` carries a `mock` block deciding how often that
model is right. Any mock-mode conclusion about whether an extra rung pays, or about
how DeepSeek compares to Claude, is a restatement of those constants. The ladders
are built and wired so a real run can answer it; mock mode cannot.

The `wide` ladder is the worst case here, because its two rungs come from different
providers, so a capability difference is confounded with a provider difference and
the tokenizer factors are unmeasured. Its COST conclusions are solid; its accuracy
conclusions are the least trustworthy in the repo.

### 8. `cascade_routing` is the greedy variant

Dekoninck et al. derive the optimal strategy by taking an expectation over every
remaining subset of the model ladder, using a per-model variance estimate. What is
implemented here compares stopping against continuing to the single best next
tier. The paper evaluates that simplification too and reports it costs 0.5% to
1.3%, mostly in low-noise settings. Implementing the full version needs a variance
estimate per model per step, which the calibration half at this n cannot support.

### 9. `llm_router` accuracy is not a measurement in mock mode

The mock router is an oracle on the mock's own latent difficulty, corrupted at
rate `1 - MOCK_ROUTER_SKILL`. Its accuracy restates that constant and nothing
else, which is why it tends to outscore every honest router in mock mode. Its
**cost and latency** overhead are real arithmetic on the price table, and those
are the only figures worth taking from it before a real run.

### 10. Latency is modelled as a constant, and it is not one

`MOCK_LATENCY_S` gives each tier a fixed latency. Real latency varies with output
length, load and time of day, and the tail matters more than the mean for anything
user-facing. A cascade's latency story is much worse than its cost story — it pays
a full extra round trip on every escalation, serially — and this repo currently
under-represents that. Latency from a real run would be worth reporting as a
distribution, not a mean.

### 18. The routing signal and the usable verifier are in different halves

This is the sharpest open problem in the repository, and it is the intersection
of two things already recorded separately above.

The 7 August redraw (issue 3) put the reproducible routing signal almost
entirely in the **code** half: all 5 routable code tasks are solid, while 9 of 10
routable maths tasks are flaky or phantom. So code is where a router has
something to capture.

But the code half's verifier is free and perfect **only because MBPP+ ships
tests**. That is the third consequence listed in the README's "what the product
had to solve": self-consistency transfers to a served query, running the tests
does not, because a user's query arrives without them.

Put together: **the half with the signal is the half whose verifier does not
deploy, and the half whose verifier deploys has almost no signal.** Every
real-data number in this repository is collected under a verifier the product
cannot have.

That does not sink the idea, and it is worth being precise about why:

- `llm_routing/sweep_degraded.py` is exactly the instrument for this, and now runs on real
  data. It prices the loss directly — at p=0.50 the cascade still holds
  always-expensive's accuracy 60% cheaper. A deployed verifier is a *degraded*
  verifier, not an absent one, and the sweep says how much degradation is
  survivable. The answer so far is: a lot.
- The maths half's verifier is the deployable one, and its measured signal is
  thin *on MATH500 level 5*, which is a deliberately adversarial sample. That is
  not evidence about ordinary traffic.

What would settle it is a code verifier that needs no shipped tests — asking the
cheap model to generate its own tests, or self-consistency over code — measured
on the same 40 tasks against the same graded outcomes, which the committed cache
already supports for the answer half. That is the next experiment worth running,
and it is more informative than a full ten-policy run.

## Resolved, recorded so they are not reintroduced

### 19. Replay fell back to the mock cache and labelled the result real

*Newest first in this section, because this one is the most instructive.*

The single worst bug in the project's history, because it defeated the property
the whole repository is organised around.

`models.call` in replay mode built two candidate cache keys, real and mock, and
returned whichever hit first. The fallback was deliberate and documented — it
lets the replay *path* be exercised without an API key — but nothing carried the
choice outward. `run_eval.provenance` then computed its `simulated` flag from
the run mode and whether the real cache **file existed**, which is a question
about the filesystem rather than about any response.

It had been taking the fallback. Measured 7 August 2026: of the 240
self-consistency samples the maths cascade needs across the task set, **240 came
from the mock cache and 0 from the real one** — the real temperature-0.8
responses on disk are orphaned at the old `MAX_TOKENS=2048` and never matched.
The resulting 370-row `runs/results.jsonl`, every row `simulated: false`, reported the
maths cascade at 100% accuracy. That is issue 4 above — "mock mode makes majority
voting far too strong" — restated as a measurement.

Fixed in three places, because one would not have been enough:

- **`ModelResponse.simulated`** carries per-response provenance, read from the
  cache record's own stored `mode` rather than from which key matched. Correct
  for the 758 records already on disk, and robust to the lookup order changing.
- **`run_eval.run` measures the flag per row** from `call_stats["served_mock"]`
  deltas across each policy call. One fabricated response anywhere in a row makes
  the row simulated — a cascade that verifies a real cheap answer with five mock
  samples is reporting the mock's verdict. `llm_routing/frontier.py` takes the conservative
  whole-run version, since a frontier point aggregates every task.
- **The fallback is off by default**, behind `ROUTER_REPLAY_FALLBACK=1`. A
  missing response now raises `ReplayMiss` and the policy is dropped by name.

**The lesson worth carrying: a guard downstream of a silent fallback is dead
code.** `ReplayMiss` and `_drop_uncached` were written the same week to catch
exactly this, and could never fire, because the mock cache satisfied every
lookup before the miss could happen. The fallback also hid the problem from
`llm_routing/sanity_check.py`, `check_core_unchanged.py` and all 156 tests, none of which
compare a replayed number against the cache it should have come from.

Two consequences that surfaced immediately once the fallback was off, both
correct behaviour rather than new bugs:

- `policies.fit_estimators` needs strictly more of the cache than any single
  policy, so it is the first thing to fail on a thin cache. `run_eval` and
  `llm_routing/frontier.py` now catch `ReplayMiss` there and let `cascade_routing` sit out —
  the same "sit out rather than guess" rule `routellm` already followed.
- `llm_routing/frontier.py` drops any family it cannot replay and names it, rather than
  crashing. It still refuses to run if `always_cheap` or `always_expensive` is
  missing, since those two define the cost axis.

### 11. The cascade paid for rungs it had already decided to reject

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
merely expected to fail would be a one-shot router in disguise, and the whole
comparison rests on the two being different.

Worth noting as a general lesson: this was an API contract (Sonnet 5 rejects
`temperature`) propagating into economics two layers away. Aggregate tables hid it
completely; tracing one task exposed it immediately.

### 12. The oracle did not bound the cascade

`policy_oracle` chose between cheap-greedy and expensive-greedy only, while the
maths cascade also has cheap-majority-of-k available. That action was outside the
oracle's space, so the cascade could and did score above the supposed ceiling,
which silently invalidated every routing-skill figure.

Fixed: the oracle now enumerates the same action space the deployable policies
have, across every rung of `models.TIERS`, in cost order, and charges the cheapest
correct action. `run_eval` prints an explicit bound check every run.

### 13. The pilot gate mislabelled its own band

It accepted a cheap-model failure rate anywhere in 20–55% while printing
"in the 30-40% target band". Fixed: the band is now two named constants and the
gate prints the band it is actually testing against.

### 14. The two tiers use different tokenizers

Claude 4.7 and later use a newer tokenizer that produces roughly 30% more tokens
for the same text. The cheap tier here is on the old tokenizer and both upper
tiers are on the new one, so the effective input price ratios are nearer
1x / 3.9x / 6.5x than the 1x / 3x / 5x the price table suggests. This works
against the cascade, since escalation pays the inflated input cost on top of a
cheap call already made.

Fixed: modelled as `tokenizer_factor` in `models.MODELS`, applied in mock mode
only — real mode gets true counts back from the API.

### 15. The verifier resampled the wrong model

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

### 16. Mock outcomes depended on RNG call order

Mock draws used to come from an unseeded global RNG, so repeat runs of the same
configuration disagreed by more than the effects being measured. Every stochastic
value now goes through `models._draw`, a pure hash of its inputs, which also makes
`--limit 10` reproduce the first ten tasks of a full run exactly — for every
policy except the two that calibrate on the task set they are given.

### 17. Artefacts disagreed about line endings

`data/taskset.jsonl` was written with CRLF and `runs/results.jsonl` with LF, so the same
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
- **`llm_routing/stats.py` compares a short pre-registered list by default.** Every pair is a
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
  their greedy variant, and its central claim is what `llm_routing/sweep_degraded.py` tests
  empirically instead of with synthetic noise.
- LLMRouterBench, [arXiv:2601.07206](https://arxiv.org/abs/2601.07206) — finds that
  under unified evaluation many published routers, including commercial ones, do
  not reliably beat a simple baseline. Directly relevant to why `random_matched`
  exists here.
- Kolawole, Dennis, Talwalkar and Smith, *Agreement-Based Cascading*,
  [arXiv:2407.02348](https://arxiv.org/abs/2407.02348) (TMLR 07/2025) — defers on
  ensemble disagreement, a generalisation of `_self_consistency`, and states this
  repo's crossover as a cost-ratio threshold: at γ ≥ 1/5 sequential cascading
  saves little without parallelism. Also proves the worst case the `deepseek` row
  shows: a k-model cascade can cost (k+1)x the top model when nearly everything
  escalates.
- *How Much of the Routing Gap Is Real?*,
  [arXiv:2607.03436](https://arxiv.org/abs/2607.03436) — decomposes the
  router-to-oracle gap into reproducible advantage and single-draw label noise.
  The reason issue 3 below exists.
- *Resample or Reroute?*, [arXiv:2607.08665](https://arxiv.org/abs/2607.08665) —
  treats another draw and a bigger model as competing uses of one budget, and
  reports cascades losing 22–31% on saturated tasks. Tested here in issue 3c,
  where it does not reproduce — this task set is unsaturated by construction,
  which is the condition their result depends on.
