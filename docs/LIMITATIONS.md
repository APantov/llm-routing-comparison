# Limitations

What bounds the claims in [RESULTS.md](RESULTS.md), ordered by how much each
one would move a conclusion. Everything here is stated once and not repeated
elsewhere.

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
quality-per-dollar at each step — not the full algorithm.

## 8. Latency is modelled, not measured

Costs are real throughout. Latency is a constant per rung, which is enough to
keep the accounting honest and not enough to say anything about tail behaviour.

---

The bugs this project found in itself — each of which changed a published number,
and each of which now has a permanent test behind it — are in
[ENGINEERING.md](ENGINEERING.md).
