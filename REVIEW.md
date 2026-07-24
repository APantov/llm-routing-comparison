# Repository review

Assessment of the cascade-vs-predictive routing project against its own plan and
against the current literature. 29 July 2026.

**Verdict:** the design is sound and the engineering is unusually careful for a
weekend project. There is one real bug, one dominated policy caused by a
parameter choice, and a statistical-power problem that the current n cannot
support. None of it is fatal. About 8 hours of focused work separates this from
a finished, defensible portfolio piece. Right now it is not a CV asset because
the deliverable (a public repo with a number in it) does not exist yet.

---

## 1. What the repo actually is

100 tasks (60 MATH500 levels 3-5, 40 sanitized MBPP), five routing policies,
mock and real model backends, deterministic grading, cost accounting from token
counts. The pipeline runs end to end in mock mode on real data. Nothing has
touched a live model.

Current mock numbers (`results.jsonl`, 500 rows):

| policy | acc | cost/task | escalation |
|---|---|---|---|
| always_cheap | 68.0% | $0.000489 | - |
| predictive | 84.0% | $0.001635 | - |
| cascade | 93.0% | $0.002946 | 38.0% |
| oracle | 96.0% | $0.001435 | - |
| always_expensive | 95.0% | $0.003444 | - |

Split by domain, which is where the actual result lives:

| domain | cascade cost | cascade acc | always_exp cost | always_exp acc |
|---|---|---|---|---|
| code (perfect verifier) | $0.001842 | 97.5% | $0.003541 | 97.5% |
| math (proxy verifier) | $0.003681 | 90.0% | $0.003379 | 93.3% |

Code cascade: 48% cheaper at identical accuracy. Math cascade: more expensive
**and** less accurate, so strictly dominated and off the Pareto frontier.

That contrast is the headline finding and it is a good one.

---

## 2. What is genuinely well done

**The verifier asymmetry is the real contribution.** Same cascade architecture,
two verification regimes, one task set. Most of the routing literature varies
the router and holds verification fixed. The 2026 survey identifies quality
estimation as the critical factor in cascade success, but treats it as a
component to improve rather than a variable to manipulate. Holding architecture
constant and varying verifier quality is a clean, under-covered ablation.

**No LLM judge anywhere.** This is directly validated by recent work. The May
2026 "Unsolvability Ceiling" paper finds that a substantial fraction of reported
routing headroom in existing benchmarks is evaluation artifact, from three
sources: judge bias toward verbose responses, truncation under fixed generation
budgets, and output format mismatch. This repo already defends against all
three: exact-match and execution grading (no judge), an explicit truncation
counter that flags `max_tokens` hits loudly instead of letting them grade as
wrong, and `\boxed{}` normalisation for format. Cite this. It turns three
defensive engineering choices into a literature-grounded methodology.

**Leak discipline.** Putting router-visible features in their own
`predict_features` field, so nobody can reach for `difficulty_proxy` because it
happens to be handy, is a structural fix rather than a comment. The distinction
drawn between shipped metadata (MATH500 `level`) and answer-derived leakage
(reference solution line count) is correct and stated clearly.

**Correlated failure in the mock.** `MOCK_FAILURE_CORRELATION = 0.75` shows real
understanding of what would go wrong without it. Independent failures would make
escalation look far better than any real cascade achieves.

**Operational details that show the project was actually run:** dated model IDs
rather than aliases, per-model API contract handling (temperature accepted on
Haiku, `thinking` explicitly disabled on Opus 5), a spend cap enforced in code,
brace-matched `\boxed{}` parsing, subprocess timeout on generated code.

Pricing in `models.py` is correct as of today: Haiku 4.5 at $1/$5 per MTok,
Opus 5 at $5/$25. The 5x ratio is real.

---

## 3. Problems, in priority order

> **Status, 29 July 2026.** 3.1 and 3.5 are FIXED. Fixing 3.1 exposed two new
> problems, 3.10 and 3.11, which now block any reportable math number. Mock-mode
> output is also now self-labelling (see 3.12).

### 3.1 Bug: the math cascade pays for 5 samples and uses 1 — FIXED

`verify_math` draws `SELF_CONSISTENCY_K = 5` samples and computes agreement.
`policy_cascade` then grades `cheap.text`, which is the original greedy sample,
not the plurality answer across the 5.

The entire point of self-consistency is that the majority answer is more
accurate than any individual sample. The code pays 5x the cheap-tier cost,
extracts the confidence signal, and discards the accuracy gain that comes free
with it.

**Fixed.** Verifiers now return a `Verdict` carrying `answer_text`, the response
the cascade should actually grade. `verify_code` returns what it was handed;
`verify_math` returns the plurality answer rewrapped as `\boxed{}`. Ties break
toward the greedy sample (inserted first, `most_common` is a stable sort).

Consequence to state explicitly: `cascade` now bundles two mechanisms, majority
voting for accuracy and agreement for escalation. The accepted math answer is no
longer a bare single cheap call, so it is not comparable to `always_cheap` as a
single-sample baseline. This is what AutoMix-style systems do, but it has to be
said rather than left implicit. See 3.10 and 3.11 for what it broke.

### 3.2 The math cascade is dominated, and k=5 is most of the reason

Do the arithmetic and put it in the README. At a 5x price ratio, k cheap samples
cost roughly `k/5` of one expensive call. At k=5 the verifier has already spent
the equivalent of a full expensive call *before* it decides whether to escalate.
The math cascade cannot win at that setting regardless of how good the verifier
is.

So "the math cascade loses" is currently two claims tangled together: a finding
about verifier quality, and an artifact of a parameter. Separate them by running
the k sweep (k = 1, 3, 5) and reporting the break-even. RUNBOOK decision D flags
this but does not do the algebra. Doing it converts "I chose 5" into a curve,
which is a much better answer and costs almost nothing.

### 3.3 The threshold has hidden granularity

With k=5, agreement can only take the values 0.2, 0.4, 0.6, 0.8, 1.0.
`AGREEMENT_THRESHOLD = 0.8` therefore means "4 of 5 must agree", and is
identical in behaviour to any threshold in (0.6, 0.8]. A sensitivity sweep over
0.7 / 0.75 / 0.8 would return three identical rows and look like a broken
experiment.

Sweep the integer count (3/5, 4/5, 5/5) or sweep k. Also note that `counts`
excludes unparseable answers but the denominator is `len(answers)`, so two parse
failures cap agreement at 0.6 and force escalation. That is arguably the right
behaviour, but it currently conflates "the model disagreed with itself" with
"the parser failed", and those are different events.

Minor related point: sample 1 is drawn at temperature 0.0 and samples 2-5 at
0.8. Standard self-consistency draws all k at the same temperature.

### 3.4 n = 100 cannot resolve the accuracy differences

Marginal 95% intervals on the current run:

```
always_cheap      68.0%  [58.9, 77.1]
predictive        84.0%  [76.8, 91.2]
cascade           93.0%  [88.0, 98.0]
always_expensive  95.0%  [90.7, 99.3]
oracle            96.0%  [92.2, 99.8]
```

Cascade, always_expensive and oracle overlap almost entirely. Worse, on a paired
basis there are only **four** discordant tasks between cascade and
always_expensive (b=1, c=3). No test separates those.

Two cheap fixes:

- **Use paired tests.** Every policy runs on the same tasks, so McNemar and
  paired bootstrap are available and far more powerful than comparing marginal
  accuracies. Cascade vs predictive is the one comparison with real signal
  (b=9, c=0, McNemar p ≈ 0.004).
- **Say so.** State that the accuracy comparison is underpowered at n=100 and
  that the result lives in the cost axis, which has much lower variance. Stating
  your own power limits is a strong signal. Having them pointed out to you is
  not.

### 3.5 Mock runs are not reproducible, and the noise exceeds the effect — FIXED

`_mock_call` uses `nonce = random.random()` from the unseeded global RNG whenever
temperature > 0. The docstring claims determinism; the code contradicts it.

Three identical repeats of the cascade over the same 25 math tasks:

```
correct:    20, 23, 22
escalated:   9, 11, 12
```

That is a 12 percentage point swing in accuracy and a 12 point swing in
escalation rate from run-to-run noise alone, on a project whose headline
comparison is a 2 point accuracy difference. Any mock result reported from a
single run is currently meaningless.

**Fixed.** `sample_idx` is now threaded through `models.call()`, and every mock
outcome is a pure hash of `(MOCK_SEED, task, tier, temperature, sample_idx)` via
`models._draw`. No global RNG state anywhere.

Three identical repeats now give `(25, 9)` three times. The mock is also
invariant to call order, so `--limit 10` reproduces the first 10 tasks of the
full run and running one policy alone matches running all five. `MOCK_SEED` is
an environment variable, so sweeping it gives the mock's run-to-run variance,
which is the honest way to read any small difference.

### 3.6 `predictive` is two routers averaged, one of which is a coin flip

Math routes on MATH500's shipped `level`, which is strongly predictive. Code
routes on `prompt_chars >= 100`, which the docstring itself reports at r = +0.28
against difficulty. So the aggregate 84% is the mean of a good router and a
near-random one, and it is not an interpretable number.

Report predictive per domain, always. Then frame the code half as the actual
finding it is: *this is what predictive routing looks like when no pre-call
difficulty signal exists*, which is precisely the condition under which cascade
wins. That connects the two halves of the project instead of leaving the code
router looking like a weak baseline.

### 3.7 The oracle is a single-draw union and overstates headroom

A July 2026 paper ("How Much of the Routing Gap Is Real?") decomposes exactly
this. The oracle takes one correctness label per (query, model), but under
stochastic decoding that is a Bernoulli draw, not a property of the model. They
find 12-36% of the reported router-to-oracle gap is single-draw label noise that
no single-commit router can ever capture.

In real mode this repo will have the same issue, and the math tier already runs
at temperature 0.8. Cheap mitigation: run the cheap model 3x on a subset and
report what fraction of the oracle gap is reproducible. Minimum acceptable
version: one paragraph in the limitations section saying the oracle is an
optimistic upper bound, with the citation.

### 3.8 Not a git repository, and no deliverable exists

`git status` returns "not a git repository". There is no `requirements.txt`, no
`plot.py`, no tests, and no public repo.

For a CV project the GitHub page *is* the artefact. A hiring manager clicks the
link, sees a README with a Pareto plot and a reproduction command, and forms an
opinion in about 90 seconds. There is currently nothing to click. This is the
cheapest, highest-return item in the whole review.

### 3.9 Documentation drift

README "Known state" says code tasks fail 100% and the tiers score too close.
Both are fixed. RUNBOOK line 209 already notes this. Four overlapping documents
(initial plan, plan v2, RUNBOOK, README) for a six-file project is too much
surface to keep true. Collapse to README plus RUNBOOK and delete the rest.

### 3.10 NEW: the oracle no longer bounds the cascade

Post-fix mock run: math cascade 100.0%, oracle 98.0%, always_expensive 98.0%.
The cascade beat the oracle. That is structural, not noise.

`policy_oracle` picks the cheapest correct option among {cheap-greedy,
expensive-greedy}. The cascade now has a third action available,
cheap-majority-of-5, which is not in that set. An oracle whose action space is
smaller than the policies it is meant to bound is not an upper bound, and
bounding headroom was its entire justification (plan, decision 4).

**Fix:** widen the oracle's action space to {cheap-greedy, cheap-SC-k,
expensive-greedy}, cheapest correct one wins. Until then the oracle row is
meaningless for math and should not be plotted.

### 3.11 NEW: the mock makes majority voting far too strong

Measured on the 60 math tasks after the 3.1 fix:

```
cheap greedy correct       : 44/60 = 73.3%
cheap majority-of-5 correct: 57/60 = 95.0%
self-consistency gain      : +21.7 points (13 tasks rescued)
```

Published self-consistency gains on competition math at k=5 are low single
digits. A +21.7 point gain is roughly three to five times too generous.

The cause is `models._wrong_answer`: it returns `truth + delta` for a random
delta in 1-9, so wrong samples scatter across about six distinct values while
correct samples all collapse to the identical truth string. Plurality then
recovers the truth whenever as few as 2 of 5 samples are right. Sampled 20
times, one task produced six different wrong answers and no repeats.

Real models do the opposite: when wrong, they tend to be wrong the *same way*
across samples, because the error comes from a stable mistaken belief rather
than from decoding noise. That clustering is exactly why self-consistency is a
weak confidence signal, and it is the mechanism the whole math half of this
project is supposed to be studying.

This was invisible before because the 3.1 bug meant the majority answer was
never used, so wrong-answer scatter only affected the escalation signal, where
scatter is roughly the right behaviour.

**Fix:** give each task a dominant attractor wrong answer that the mock returns
most of the time it is wrong, plus a tail of others. Correctness across samples
should also be correlated rather than independent, since `shared` is currently
redrawn per `sample_idx`. Both changes encode the same fact: the model has one
stable belief per task, and it is sometimes wrong.

### 3.12 Mock output is now self-labelling

Mock runs printed a table that looked exactly like a results table, with a
dollar figure and no indication of mode. That is how a simulated number ends up
in a README as if it had been measured.

Now: a banner at the head and foot of the run, a `### MOCK MODE - SIMULATED,
NOT MEASURED ###` tag above *each* table so a cropped screenshot still carries
it, "total MODELLED cost ... nothing was spent" instead of "total spend", and
`mode`, `mock_seed`, `k` and `agreement_threshold` stamped into every row of
`results.jsonl` so any slice of the file is self-describing and a k sweep can be
grouped rather than remembered.

`run_eval.py` also now refuses to let a mock run overwrite real results, since
real results cost money and cannot be reproduced. `--force` overrides.

---

## 4. Literature check

The reading list in the plan (FrugalGPT, RouteLLM, AutoMix) is the 2023-24
canon. It is the right foundation but it is now two years stale, and the gap
matters for one specific claim.

**The novelty framing needs softening.** The initial plan says "there is no
published answer to which one wins under what conditions". That is no longer
safe. Dekoninck et al., *A Unified Approach to Routing and Cascading for LLMs*
(arXiv 2410.10347, ICLR 2025), does exactly that unification: it selects the
best model at each step, can skip or reorder models, and shows cascade routing
consistently outperforms either approach alone. Its headline conclusion is that
quality estimation is the critical factor, which is the same variable this
project manipulates.

If you claim novelty and the interviewer knows that paper, you lose credibility
you did not need to risk.

**Safe framing that is still strong:** a controlled replication with 2026 models
in which the manipulated variable is *verifier quality* rather than router
quality. Honest, still a real contribution at portfolio scale, and it lets you
name the literature instead of being corrected by it.

**One paper to add, not three.** The 2026 survey *Dynamic Model Routing and
Cascading for Efficient LLM Inference* (arXiv 2603.04445) gives you the entire
map in one sitting: the three-stage pipeline framing (pre-router, post-generation
verifier, escalation policy), where FrugalGPT and Cascade Routing sit in it, and
the standard evaluation metrics. Reading it is the highest-leverage two hours
available and it replaces, rather than adds to, the rabbit hole the plan warns
about.

Useful specifics from it worth adopting:

- The three-stage pipeline vocabulary. Your cascade instantiates all three
  stages; saying so out loud maps your project onto the field's own framing.
- Standard metrics are quality-cost Pareto frontier plus AUC across operating
  points. You have the frontier planned. AUC over the k sweep would be a
  cheap addition and is what reviewers expect.
- The survey explicitly lists "no method pairs response-level signals with
  online adaptation" as an open gap. Not something to build here, but a good
  answer to "where would you take this next".

**Also worth knowing about, one line each:**

- *Unsolvability Ceiling in Multi-LLM Routing* (arXiv 2605.07395, May 2026) -
  judge bias, truncation and format mismatch inflate apparent routing headroom.
  Justifies your entire grading design.
- *How Much of the Routing Gap Is Real?* (arXiv 2607.03436, July 2026) -
  12-36% of the router-to-oracle gap is single-draw noise. Directly limits your
  oracle.
- *When LLMs Agree, Are They Right?* (arXiv 2607.08065, July 2026) - audits
  self-consistency as a confidence signal and finds it a positive but weak,
  regime-dependent proxy. This is your math verifier, audited by someone else.
  Strong support for the project's central claim.

---

## 5. Will it produce interesting results?

Yes, and the interesting result is already visible in the mock: cascade is 48%
cheaper than always-expensive at identical accuracy where verification is free
and perfect, and strictly dominated where verification is a guess.

That is memorable, defensible, and it corrects a mistake production teams
actually make. Teams read FrugalGPT, conclude cascading is free money, and miss
that FrugalGPT's cascade rests on a trained DistilBERT quality estimator. Take
the estimator away and the economics invert. Your project is a demonstration of
exactly that, and you can say it in one sentence.

**The caveat:** those are mock numbers, and `MOCK_SKILL` was tuned to produce
roughly the failure rate the plan wanted. The result partly reflects the tuning.
Nothing is real until the pilot runs. Do not skip the pilot gate.

**Two additions that would meaningfully raise the ceiling, both cheap:**

1. **The k sweep** (k = 1, 3, 5). Turns the math cascade's failure from an
   anecdote into a cost curve with a stated break-even point.
2. **`cascade_degraded`**, the stretch goal already in the plan. A deliberately
   weakened verifier gives a third point on the verifier-quality axis, so you can
   plot accuracy against verifier quality rather than assert a two-point
   contrast. This is the cheapest available route from "I observed X" to "I
   measured the sensitivity of X", and sensitivity is what makes a result look
   like science.

---

## 6. CV assessment

Worth putting on a CV, above the median portfolio project, but its value sits
almost entirely in work that has not been done yet.

**What it currently proves:** careful evaluation code, cost instrumentation,
awareness of leakage and correlated failure. All real, all invisible until
somebody reads the source.

**What would make it an asset:** a public repo, a Pareto plot, a real number in
the bullet, and a limitations section. The plan's draft bullet has "X%" and "Y
points" as placeholders. Until those are measured, the bullet cannot be used and
the project cannot be discussed in an interview.

### Path to done, roughly 8 hours

1. `git init`, push to GitHub, add `requirements.txt`. (30 min, unblocks
   everything)
2. ~~Fix the majority-vote bug (3.1) and the mock seeding (3.5).~~ Done. Now fix
   what that exposed: the oracle's action space (3.10) and the mock's
   wrong-answer clustering (3.11). (1h)
3. Pilot: 10 tasks, real calls, read the cheap-model failure rate. (1h, this is
   the gate)
4. Full run plus `plot.py`. (2h, this is the deliverable)
5. k sweep, paired statistics, per-domain predictive reporting. (2h, this is
   what makes it credible rather than merely present)
6. README rewrite: real numbers, limitations section citing the survey and the
   oracle-noise paper. (1.5h)

### On weekend two

LangGraph, tracing, CI and MCP are packaging. Genuinely useful for keyword
matching on a CV, and none of them add anything to the argument. None of them
matter if there is no number in the bullet. Do them strictly after step 6.

One pushback on the plan: it treats the LangGraph port as a core deliverable. It
is not. The hand-rolled loop is about thirty lines and already correct. Porting
it teaches you the framework's abstractions, which is a perfectly good reason to
do it, but frame it in interview as "I learned the framework by porting a loop I
had already written and understood", not as an architectural requirement.
Interviewers can tell the difference, and the honest version is the more
impressive one.

---

## Sources

- [FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176)
- [A Unified Approach to Routing and Cascading for LLMs (Dekoninck et al., ICLR 2025)](https://arxiv.org/pdf/2410.10347)
- [Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey (2026)](https://arxiv.org/html/2603.04445v2)
- [Unsolvability Ceiling in Multi-LLM Routing: An Empirical Study of Evaluation Artifacts](https://arxiv.org/abs/2605.07395)
- [How Much of the Routing Gap Is Real? Decomposing the Router-to-Oracle Gap](https://arxiv.org/pdf/2607.03436)
- [When LLMs Agree, Are They Right? Auditing Self-Consistency and Cross-Model Agreement as Confidence Signals](https://arxiv.org/pdf/2607.08065)
- [Claude Platform pricing](https://platform.claude.com/docs/en/about-claude/pricing)
