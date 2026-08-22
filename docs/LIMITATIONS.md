# Limitations

What bounds the claims in [RESULTS.md](RESULTS.md), ordered by how much each
one would move a conclusion. This is the complete list, and the only place each
one is stated *with what would settle it*; the two largest are also named where
they bite, in RESULTS and in METHOD.

---

## 1. The verifier that produces the signal is not the verifier that ships

The code half is graded by executing the tests MBPP+ supplies. A deployed router
does not have them. The maths half — where a self-consistency verifier *would*
deploy — carries far less routing signal.

This is the sharpest open problem here, and the repository prices it rather than
noting it: `sweep_degraded.py` degrades the perfect verifier by a controlled
amount and measures what the cascade loses. Going from shipped tests to a proxy
verifier in production is a move along that curve, not a step off it.

**What would settle it:** a verifier that needs no shipped tests — the cheap
model generating its own tests, or self-consistency over code.

## 2. Price ratio and capability gap are confounded

The project set out to test a **price-ratio** crossover. What the data
distinguishes is the **capability gap** between rungs, because the ladder with
the small price ratio (`deepseek`, 3.1x) is also the ladder whose rungs are
equally capable — its top rung is not measurably better than its bottom one.

Three ladders cannot separate the two. The supported claim is *"cascading pays
when the top rung is genuinely better"*; the price-ratio framing remains a
hypothesis.

**What would settle it:** a ladder with a large price ratio and a small
capability gap.

## 3. The task set is 86% code

357 of 417 tasks. Every aggregate is therefore close to a code number. Per-domain
figures are reported throughout and should be preferred to the "all" row.

## 4. Verification is not available at every rung

Claude's upper rungs do not accept a `temperature`, so they cannot be resampled
and self-consistency is simply unavailable on them. On the `claude` ladder the
middle rung cannot be verified at all; `deepseek` is the only ladder that can
verify everywhere.

The verifier reports this as *unverifiable* rather than as agreement. A verifier
that always accepts is worse than no verifier, because it is invisible.

## 5. Routing opportunity is a lower bound, not a two-sided estimate

Redrawing the decisive cells three times moved the routable fraction from 13.5%
observed to 11.3% reproducible. Only the decisive cells were redrawn, so
flakiness hidden in `both_ok` and `inverted` is still uncounted — the correction
is a floor on itself.

**What would settle it:** redrawing the other two cells, costed at roughly $14.

## 6. Maths cannot discriminate one-shot routers

`llm_router` sends every maths task to the expensive rung, so on that half it is
`always_expensive` by behaviour rather than by design. This is declared rather
than measured away: the fix was costed and cut.

## 7. `cascade_routing` is the greedy variant

Dekoninck et al.'s unified strategy is implemented in its greedy form — best
quality-per-dollar at each step — not the full algorithm. Its ex-ante estimator
is also uninformative on this task set, so it runs with only its post-hoc half
working; see [METHOD.md](METHOD.md#the-policies).

## 8. Latency is modelled, not measured

Costs are real throughout. Latency is a constant per rung, which is enough to
keep the accounting honest and not enough to say anything about tail behaviour.

## 9. The code grader is load-sensitive, to about one task in 417

`graders.grade_run_asserts` executes candidate solutions in a subprocess and
treats `subprocess.TimeoutExpired` as a **failure** rather than as an unmeasured
result. A task whose expanded MBPP+ suite runs near the 30s ceiling can
therefore grade wrong on a loaded machine and right on an idle one.

Observed, not hypothesised: across three regenerations the `wide` top rung has
come back at 91.8%, 92.1% and 92.3%, one task moving between `both_ok` and
`inverted` each time, with the loaded-machine run lowest. `claude` and
`deepseek` reproduced exactly. The counts the argument rests on — `routable`,
`both_fail` — and the McNemar p have never moved.

The same asymmetry is already handled correctly elsewhere: a response that hit
`max_tokens` is dropped from the cross-tab as unmeasured rather than scored
wrong. A timeout deserves the same treatment and does not get it.

**What would settle it:** distinguish timeout from failure in the grader's
return, drop timed-out pairs from the cross-tab the way truncated ones are, and
record how often it fires. Until then, treat per-rung accuracies in
[the cross-tab](RESULTS.md#24-the-third-ladder-has-almost-nothing-to-route) as ±0.3 points and the cell counts as firm.

---

The bugs this project found in itself — each of which changed a published number,
and each of which now has a permanent test behind it — are in
[METHOD.md](METHOD.md#bugs-this-project-found-in-itself).
