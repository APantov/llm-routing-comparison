# STATUS — read this first

Written 30 July 2026, **updated 6 August 2026 after the two-arm probe**. This is
the "you just woke up and forgot everything" file. It answers four questions in
order: where the project is, what to do next, what it will cost, and what to
expect when you do it.

If you only read one section, read [§2 Do this next](#2-do-this-next).

---

## 1. Where the project is

**The machine is finished. The task set has been hardened. The two-arm probe has
been run against real models, and it says the task set now discriminates — but
only just, and n=100 cannot close the question.**

Everything works end to end: 100 tasks, 11 policies, three model ladders, a
calibration split, a cost-quality frontier, paired significance tests, a
degradation sweep, and figures. All of it reproduces byte-for-byte on a bare
Python install with no API key and no network.

### The probe result, 6 August 2026 — the first real accuracy data in this repo

`python run_eval.py --policy always_cheap --policy always_expensive --split all`
on the `wide` ladder (DeepSeek v4-flash → Opus 5), all 100 tasks, both arms.
200 real calls, **$0.9234**, written to `results.probe.jsonl` and cached in
`cache/raw_calls.wide.jsonl`.

| cell | n=100 | code (40) | math (60) | meaning |
|---|---|---|---|---|
| `both_ok` | 77 | 29 | 48 | tie — nothing to route |
| **`routable`** | **15** | **5** | **10** | **cheap wrong, expensive right — the only cell where escalating pays** |
| `both_fail` | 6 | 5 | 1 | tie |
| `inverted` | 2 | 1 | 1 | escalating *loses* |

- **routable = 15.0%**, 95% CI [9.3%, 23.3%]. Ceiling (the most any router can add
  over `always_cheap`) = **17.0%**. McNemar **p = 0.002**.
- `always_cheap` **79.0%**, `always_expensive` **92.0%**.
- **Rescue rate 71.4%** — of the cheap rung's failures, 71% are ones escalating
  actually fixes. That is the number a cascade lives on.
- Per domain: code 12.5% routable, maths 16.7%.

> These are the figures **after the grader fix of 6 August**. The probe as first
> reported read routable 13.0%, cheap 77.0%, expensive 87.0%, p=0.021 — seven of
> its twenty wrong maths answers turned out to be correct answers the normaliser
> could not match. See "the grader was lying" below.

**This is a real change and it is the point of the 31 July / 6 August work.** On
30 July the cross-tab over the ten tasks that had both arms cached was
`both_ok=10, routable=0`: nothing to route, at p≈0.02. Hardening both halves —
MBPP+ on code, MATH500 level 5 on maths — moved the routable fraction to 15%.

**Both hardenings worked, and they worked independently**, because they apply to
disjoint task subsets. But they did not work *equally*, and the difference is the
most useful thing in this table:

| | cheap | expensive | both_fail | routable | rescue |
|---|---|---|---|---|---|
| code (MBPP+) | 75.0% | 85.0% | **5 of 40** | 12.5% | 50% |
| maths (level 5) | 81.7% | **96.7%** | **1 of 60** | 16.7% | 91% |

**Opus solves 58 of 60 MATH500 level-5 problems.** The maths half is close to
saturated at the top rung, so its routing signal is almost entirely "DeepSeek
fails on something Opus finds easy" — a clean cascade signal, 91% rescue.

**Code is now the harder domain in absolute terms**, which reverses this file's
long-standing assumption. MBPP+ produced genuine `both_fail` content: 5 of 40
tasks neither rung can solve, against 1 of 60 on maths. Half of the cheap rung's
code failures are unfixable by escalating.

### But read the verdict carefully

`routable.py` still returns **UNRESOLVED**: the CI [9.3%, 23.3%] straddles the
15% floor of its [15%, 45%] band. The point estimate now sits exactly *on* the
floor rather than below it, so this is "plausibly adequate, not demonstrated"
rather than "probably too easy".

The gate `run_eval.py` prints is friendlier — cheap-model failure rate 21.0%
[13%, 29%] against a 20–55% target. **Prefer the routable figure.** `P(cheap
fails)` is `routable + both_fail`, and 6 of those 21 failures are tasks the
expensive rung cannot fix either. That is the exact confusion
[ROUTABLE_2026-07-30.md](ROUTABLE_2026-07-30.md) was written to warn about,
though the gap is much smaller now that the grader is not manufacturing failures.

So the honest reading: **there is a real routing signal, it is comfortably
significant at n=100 (p=0.002), and it is thin.** A perfect oracle beats
`always_cheap` by 17 points and `always_expensive` beats it by 13. Every policy in
the repo is competing over that 17-point band.

### The grader was lying, and the regression gate could not see it

Seven of the twenty wrong maths answers in the probe were **correct answers the
normaliser rejected** — five of them on Opus, which is why its accuracy moved
87% → 92%. The classes, each now covered by a test:

| the model wrote | the truth said | the bug |
|---|---|---|
| `1+\sqrt{19}, 1-\sqrt{19}` | `1 \pm \sqrt{19}` | no `\pm` handling at all |
| `\{-2, 1+\sqrt5, 1-\sqrt5\}` | `\{1\pm\sqrt{5},-2\}` | set ordering, and `\sqrt5` vs `\sqrt{5}` |
| `\frac{270}{7}` | `\frac{270}7\text{ degrees}` | trailing unit, and `\frac{270}7` unbraced |
| `18.90` | `\$18.90` | `.replace("$","")` left the escaping backslash |
| `3<\lambda\le4` | `(3,4]` | interval vs inequality |
| `AF^2+BF^2+CF^2 = 3R^2` | `3R^2` | answer restated with its left-hand side |

**`sanity_check.py` printed 60/60 throughout.** It feeds the ground-truth string
back into `\boxed{}`, so it only ever tested `grade(GT, GT)` — which passes by
construction however broken the normaliser is. It now also checks a table of
known-equivalent pairs *and* a table of near-miss wrong answers, because a grader
that accepts everything would pass the first table perfectly.

### The models are not deterministic, and it is measured

Abandoning the 2048-cap run and re-running at 4096 left **73 pairs of independent
draws** — same model, same prompt, same temperature, two separate calls, both on
disk because `max_tokens` is in the cache key. Grading both:

| domain | verdict flips |
|---|---|
| code | 0 of 27 (**0%**) |
| maths | 4 of 46 (**8.7%**) |

`math-94` is the clean example: `\boxed{80}` on one draw, `\boxed{130}` on the
other. **8.7% per-task noise against a 15% routable signal.** The response cache
guarantees every *policy* sees the same draw, which is what the paired statistics
need — but a full re-run of the experiment would move individual cells around,
and the routable fraction carries this on top of its sampling CI. Treat one draw
per task as the floor of what the design needs, not a comfortable margin.

### What is still not measured

The probe ran **two** policies. The other nine — every cascade, every predictive
router, the degradation sweep, the frontier — have still never been run against a
real model. Their accuracy figures remain mock-mode fabrications and are labelled
as such everywhere.

What is *already* real, even in mock mode:

- the price tables, verified against provider docs on 2026-07-30
- all the cost arithmetic built on them
- the graders, the caching, the escalation logic, the statistics

So the cost conclusions below are trustworthy. The accuracy conclusions are not,
except for the two arms in the table above, and everything is labelled.

### What the probe cost, and why it was 2x the estimate

`ROUTABLE_2026-07-30.md` priced this probe at **$0.44**. It came in at **$0.9234**,
and the reason is worth carrying forward because it repriced the whole project:

| domain | mean output tokens | cost per call |
|---|---|---|
| code | 55 | $0.0012 |
| maths (level 5) | 650 | $0.0078 |

**Maths costs 6.5x what code costs**, because level-5 problems produce long
derivations and Opus 5 charges $25/MTok on output. The $0.44 estimate was built
from the old, easier task set, where the modelled reply was 80–120 tokens. The
prediction in §5 that real runs would come in "2–3x higher" is confirmed — and
every cost figure in §3 that was extrapolated from the mock is low for the same
reason.

**`MAX_TOKENS` was raised from 2048 to 4096 on 6 August 2026**, and this is the
second thing the probe paid to discover. At 2048, two of the first 118 real
responses truncated — both coherent level-5 derivations that simply ran long, one
of them a few hundred tokens short of its `\boxed{}`. 1.7% sounds ignorable and is
not: truncations land on the hardest tasks, which are exactly the ones that decide
the routable fraction, and a truncated cheap answer reads as "the cheap rung
failed" and inflates it. Note the consequence before you touch that constant
again — `max_tokens` is in the response-cache key, so changing it **invalidates
every cached response and re-charges the whole run**. That is what the first
$0.39 of the 6 August spend bought.

One call still truncated at 4096 (`math-96`, cheap rung). It was inspected rather
than assumed: DeepSeek was doing hand bisection on a cubic at "Step 15" and not
converging. That is a genuine capability failure, correctly graded wrong, and it
sits in `both_fail` — so it does not touch the routable count either way.

### The finding that made the ladder work worth doing

The repo can now switch model ladders with an environment variable, and doing so
**flips the project's headline conclusion**. Matched on accuracy, against simply
always paying for the best model:

| ladder | rungs | price ratio | cascade vs always-best, same accuracy |
|---|---|---|---|
| `deepseek` | v4-flash → v4-pro | 3.1x | cascade costs **33% more** |
| `claude` | Haiku 4.5 → Sonnet 5 → Opus 5 | 5x list, 6.5x effective | cascade **12% cheaper** |
| `wide` | DeepSeek v4-flash → Opus 5 | 36x list, 46x effective | cascade **74% cheaper** |

That is the answer to "when is each architecture worth it", and it is a sharper
answer than a single ladder could ever have given:

> **Cascading pays in proportion to the price gap it is exploiting.** Below roughly
> a 3x ratio the wasted cheap call and its verification cost more than they save,
> and you should just route. The wider the gap, the more cascading wins.

The mechanism is not subtle once stated: a cascade always pays for the cheap call,
and pays for verification on top. Those are fixed costs. What it buys is the chance
to skip an expensive call. When "expensive" is only 3x "cheap", the fixed costs
swamp the saving.

Averaged across the whole budget range (the AUC column in `frontier.py`), the
cascade beats a cost-matched coin flip on **every** ladder: +4.8% claude, +7.2%
deepseek, +8.8% wide. So cascading is always a better *router* than chance — it
just is not always cheaper than not routing at all. Those are two different
questions and the repo now reports both.

A second, more uncomfortable finding: **RouteLLM's pretrained router scores below a
coin flip on all three ladders** (−2.8%, −1.9%, −2.3% AUC). It was trained on human
preference between chat models; this asks it about objectively-graded maths and
code. Out of distribution, and it shows.

### The thing that will bite you

`python3 stats.py` runs exact McNemar tests over the held-out half. **Zero of eight
comparisons reach significance.** The six-point gaps in the table have confidence
intervals spanning zero. Cost differences *are* significant; accuracy differences
are not.

So the current honest summary is: *cost differences are measurable at n=100,
accuracy differences are not.* Do not quote an accuracy gap as a finding until
§2 step 5 is done.

The probe is the one exception, and it is instructive: the cheap-vs-expensive gap
**is** significant at n=100 (McNemar p=0.002) because it is the largest gap in the
project. Every policy comparison is a contest over the 17-point band inside it,
and those are the ones n=100 cannot resolve.

---

## 2. Do this next

In order. Steps 1–3 are free and take about five minutes total.

> **Where you actually are, 6 August 2026.** Steps 1–4b are done. The task set has
> been hardened (MBPP+ code half, MATH500 level 5 maths half — both now the
> defaults in `build_taskset.py`) and the two-arm probe has been run and read.
> **Go to step 5**, but read [§4](#4-do-you-need-more-tasks) first: at
> routable=15% the case for enlarging the task set before paying for a full run is
> stronger than it was, not weaker.

> **Shell note.** `ROUTER_LADDER=x python3 ...` is bash syntax and does nothing on
> Windows PowerShell. Rather than remember three shells' worth of syntax, put the
> setting in `.env` and drop the prefix entirely — the repo loads it, and a real
> environment variable still overrides it:
>
> ```
> ROUTER_LADDER=deepseek
> ROUTER_MODE=real
> ```
>
> Every command below then works unchanged in bash, PowerShell and cmd.


### Step 1 — confirm nothing rotted (2 min, free)

```bash
python3 build_taskset.py && python3 sanity_check.py
```

Expect `code source: mbppplus   math levels: >= 5`, then `40/40` and `60/60`, then
`both graders score every reference answer correctly`. If either count is short,
**stop** — every number downstream is invalid until it is fixed, and
`sanity_check.py` exits non-zero to make that hard to miss.

The code half needs **numpy** to grade, because every MBPP+ test program compares
floats with `np.allclose`. The easier originals are still one flag away:
`python3 build_taskset.py --code mbpp --min-math-level 3`.

### Step 2 — look at the three ladders (2 min, free)

```bash
ROUTER_LADDER=claude   python3 run_eval.py
ROUTER_LADDER=deepseek python3 run_eval.py
ROUTER_LADDER=wide     python3 run_eval.py
```

Expect the table from §1. The `oracle bound check` must say `ok` on all three rows
of each run; if it ever says `VIOLATED`, a policy has gained an action the oracle
cannot reach and every routing-skill figure is void until fixed.

### Step 3 — see the curves and the tests (1 min, free)

```bash
python3 frontier.py && python3 stats.py && python3 plot.py
```

`figures/frontier.svg` and `figures/degradation.svg` are written. Open them.

### Step 4a — plumbing check (2 minutes, about $0.02) — **DONE, 30 July 2026**

Ten tasks, everything wired up, just to prove the keys work and nothing errors.
This has been run on the `wide` ladder: it cost **$0.0481**, wrote 47 real
responses to `cache/raw_calls.{wide,claude,deepseek}.jsonl` and 47 `simulated:
false` rows to `results.jsonl`, and passed on every check below (no truncation,
graders ran, cache replays field-for-field). Every policy scored 100% on all five
reported tasks, which is exactly the non-result a plumbing check is allowed to
produce. Re-run it only if the ladder or the prompts change.

```bash
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
# open .env, paste ANTHROPIC_API_KEY and DEEPSEEK_API_KEY, save

ROUTER_LADDER=wide ROUTER_MODE=real python3 run_eval.py --limit 10
```

`.env` is gitignored so a key cannot be committed by accident, and a real
environment variable beats the file if you want a one-off override. Note that a
Claude Pro/Max subscription does **not** include API access — separate products,
separate billing; the key comes from the Claude Console.

What this is for: API keys resolve, prompts send, responses parse, the graders
run, the cache writes. Check for `!! TRUNCATED at max_tokens` — any at all means
raise `models.MAX_TOKENS` and re-run, because a truncated answer grades as *wrong*
and would read as a capability result.

**Do NOT read the pilot gate off this run.** At n=10 the failure rate has a 95%
confidence interval of roughly ±28 points, so it cannot distinguish "too easy"
from "too hard" from "fine". It spans the entire band. That is what step 4b is for.

### Step 4b — the two-arm probe (5 minutes, $0.92) — **DONE, 6 August 2026**

**This was the actual decision point**, and the numbers are in §1. Summary:
routable **15.0%** [9.3%, 23.3%], ceiling 17.0%, McNemar p=0.002, cheap 79.0% /
expensive 92.0%, code 12.5% and maths 16.7%.

Re-run it only if the ladder, the prompts, the task set or `MAX_TOKENS` change —
each of those invalidates the cached responses and re-charges the run.

```bash
# both arms, all 100 tasks, writes results.probe.jsonl (never results.jsonl)
ROUTER_LADDER=wide ROUTER_MODE=real python3 run_eval.py \
    --policy always_cheap --policy always_expensive --split all

# then the cross-tab, free, off the cache
ROUTER_MODE=replay python3 routable.py --real --ladders wide
```

> **Do not use the one-arm version, and do not read the gate `run_eval.py`
> prints.** See [ROUTABLE_2026-07-30.md](ROUTABLE_2026-07-30.md). `P(cheap fails)`
> is `routable + both_fail`, and the one-arm gate cannot see the split. This run
> is the concrete demonstration: it reports a 21.0% cheap failure rate against a
> 20–55% target, while the quantity that actually matters is 15.0% and sits on
> the floor of its band. Six of those 21 failures are tasks the expensive rung
> cannot fix either — and on the code half it is 5 of 10.

How to read the routable fraction, against `routable.py`'s [15%, 45%] band:

- **inside the band** → the task set discriminates. Go to step 5.
- **below 15%** → too easy. Not "no experiment", but a thin one: every policy is
  competing over a narrow ceiling and n must rise to see anything. **This is where
  we are, at 13%, with the CI straddling the floor.**
- **above 45%** → too hard; everything escalates and every policy collapses onto
  `always_expensive`.

Run the probe on **each ladder you care about**, because the answer is a property
of the cheap rung, not of the task set alone: DeepSeek v4-flash and Haiku 4.5 will
not fail on the same fraction. Only `wide` has been probed.

### Step 5 — the full paid run

**Do [§4](#4-do-you-need-more-tasks) first.** At routable=15% with a 17-point
ceiling, a full run at n=100 would be measuring nine policies against each other
inside a band that n=100 already cannot resolve. Enlarging the task set costs the
same order of magnitude and is the difference between a result and another
"nothing is significant".

Before any full run, note two things the probe changed:

- **RouteLLM's cached scores no longer cover the task set.** The rebuild changed
  every code id to `codeplus-*` and resampled the maths half, so `routellm` now
  sits out with `SKIPPED - no cached scores`. Regenerate them first — it is free,
  local, and needs no API key: `python3 routellm_router.py --score`.
- **Budget from the measured per-call costs in §1, not from the table in §3.**
  Maths is $0.0078/call and code $0.0012/call on `wide`. Self-consistency
  verification samples the cheap rung k times per task, so the cascade policies
  multiply the maths figure.

```bash
ROUTER_LADDER=wide ROUTER_MODE=real python3 run_eval.py
git add -f cache/raw_calls.wide.jsonl results.jsonl
git commit -m "raw responses, wide ladder"
```

Committing the cache is the important half. After that, **everything is free
forever** and reproducible by anyone with no key at all:

```bash
ROUTER_MODE=replay ROUTER_LADDER=wide python3 run_eval.py
ROUTER_MODE=replay ROUTER_LADDER=wide python3 frontier.py
ROUTER_MODE=replay ROUTER_LADDER=wide python3 sweep_degraded.py
ROUTER_MODE=replay ROUTER_LADDER=wide python3 stats.py
```

Then repeat step 5 for `claude` and `deepseek`. Each writes its own cache file, so
they do not interfere.

### Step 6 — decide about more tasks

See [§4](#4-do-you-need-more-tasks). Short answer: yes, and you will know exactly
how many once step 5 gives you real discordance counts.

---

## 3. What the runs cost

> **These are MODELLED figures and the 6 August probe showed them to be low by
> roughly 2.5x on the hardened task set.** The measured numbers are in §1: maths
> **$0.0078/call**, code **$0.0012/call** on `wide`, against a modelled reply
> length of 80–120 tokens that turned out to be 650 for level-5 maths. The
> two-arm probe was priced here at $0.008 and cost **$0.9234**. Scale everything
> below accordingly, and prefer a measured number wherever one exists.

Modelled from the verified price tables, for all 100 tasks and every policy. The
response cache deduplicates identical calls, so the number that costs money is
*distinct* calls, not policy-attributed calls.

| ladder | probe (cheap rung, 100 calls) | full run (all policies) |
|---|---|---|
| `deepseek` | 100 calls, **$0.004** | 772 calls, **$0.06** |
| `wide` | 100 calls, **$0.004** | 540 calls, **$0.37** |
| `claude` | 100 calls, **$0.049** | 640 calls, **$0.72** |

**Actually spent to date: $1.36**, all on `wide`, all in
`cache/raw_calls.wide.jsonl`. $0.0481 on the 30 July plumbing run, $0.39 on a
two-arm probe at `MAX_TOKENS=2048` that was abandoned when it truncated, and
$0.9234 on the probe that replaced it.

The `wide` probe and the `deepseek` probe hit the SAME cheap rung
(deepseek-v4-flash), so running both is redundant. There are only two distinct
bottom rungs across the three ladders, and therefore only two probes worth paying
for: one on DeepSeek v4-flash, one on Haiku 4.5.

These are BACKEND costs - what actually leaves your account, after the response
cache deduplicates identical calls across policies. The larger "total attributed
cost" the report also prints is the sum over policies of what each would pay to
serve alone, which is the right number for a production comparison and the wrong
one for your invoice.

All three ladders, full runs, come to about **$1.15 total.** The spend cap in
`run_eval.MAX_SPEND_USD` is $20 and enforced in code, so a bug cannot run away with
your money.

Two caveats, both in your favour:

- These are *modelled* output lengths (80–120 tokens). Real replies with thinking
  disabled are usually longer, so budget maybe 2–3x. Still under $10 for everything.
- The sweeps and the frontier cost **nothing extra** after the first run. That is
  the entire reason the response cache was built before any money was spent.

---

## 4. Do you need more tasks?

**Yes.** This is the clearest actionable finding in the repo.

n=100, halved by the calibration split, leaves 51 tasks to report on. One task is
two accuracy points. The gaps you care about are five or six points, which means
they hinge on three or four tasks going one way rather than the other. `stats.py`
confirms it: **nothing is significant.**

### How many more — now answerable from real discordance counts

The 6 August probe supplies what step 5 was supposed to: **17 discordant pairs per
100 tasks** on the widest comparison in the project (15 routable + 2 inverted).

That comparison is comfortably significant (p=0.002). The ones that are not are
the policy-vs-policy contests *inside* the 17-point ceiling, and they are strictly
harder — a cascade and a predictive router disagree on far fewer than 17 tasks per
100, because both spend most of the set agreeing with `always_cheap` on the 77
`both_ok` tasks.

Concretely, on the routable fraction itself (p̂ = 15.0%):

| n | 95% CI on routable | resolves the [15%, 45%] band? |
|---|---|---|
| 100 | [9.3%, 23.3%] | no — sits on the floor |
| 504 (the whole pool) | ≈ [12.0%, 18.5%] | no, but much tighter |
| ~1900 | ≈ [13.4%, 16.7%] | still not, because the estimate *is* the boundary |

An estimate sitting exactly on a band edge cannot be resolved by more tasks — the
CI shrinks around 15% and never clears it. So the honest framing is that the task
set is **marginal by this criterion and no amount of n will change that verdict**;
what n buys is power for the policy comparisons, which is the thing actually
short. Do not read "UNRESOLVED" as "one more probe will settle it".

Note also the noise floor: 8.7% of maths verdicts flip between independent draws
of the same prompt. Part of the width above is decoding noise, not task sampling,
and adding tasks does not reduce it — sampling each task more than once would.

**The pool is smaller than it used to be, because level 5 costs maths tasks.**
Verified on 6 August rather than quoted:

| source | available | currently used |
|---|---|---|
| MATH500 level ≥ 5 | **134** | 60 |
| MATH500 level ≥ 4 | 262 | — |
| MATH500 level ≥ 3 | 367 | — |
| MBPP+ | **370** | 40 |

Taking everything at the current settings gives **504 tasks**, not the 787 this
file previously promised — `MIN_MATH_LEVEL = 5` cut the maths pool from 367 to
134. Dropping to level 4 would buy back 128 maths tasks and restore the
predictive router's difficulty signal (see the comment on `MIN_MATH_LEVEL`), at
the cost of an easier maths half.

### Why this is cheap to fix

Cost scales linearly with tasks. From the **measured** per-call figures in §1, a
two-arm probe over all 504 costs about **$3.60**; a full eleven-policy run is the
number to be careful with, because self-consistency verification samples the
cheap rung k times per task and maths is the expensive domain. Budget in the
$15–25 range for `wide` at n=504 rather than the $0.37 in §3, and re-read the
spend cap in `run_eval.MAX_SPEND_USD` (currently $20) before starting — **it will
bind.**

Every analysis afterwards is free via replay. This is still the highest-value
spend available to the project.

### The risk that the code half was too easy — resolved, and the answer is no

This file used to warn that MBPP is saturated and the code cascade might have
nothing to route. **The probe settled it: code routable = 12.5%, maths = 16.7%.**
Both halves discriminate, and the verifier-quality experiment — which lives on the
code half because that is the half with the perfect verifier — has material to
work with.

The surprise is the direction. **Code is now the harder half**: 5 of its 40 tasks
defeat both rungs, against 1 of 60 on maths, and Opus scores 85% on code against
96.7% on maths. MBPP+ did not merely restore the code signal, it overshot maths.

Credit where due: that took the MBPP+ swap. Plain MBPP's thin asserts are what
made the code half look saturated, and the expanded evalplus suites recovered the
signal without changing a single problem — the swap moves exactly one variable,
how thorough the marking is. See [DATASETS.md](DATASETS.md).

The remaining escalation path, if 12.5% is judged too thin:

1. **Drop `MIN_MATH_LEVEL` to 4** and rebalance toward maths — buys 128 maths
   tasks and restores the predictive router's difficulty signal, at the cost of an
   easier maths half.
2. **BigCodeBench** for the code half. 1140 tasks, genuinely hard, but the `test`
   field is a `unittest` class rather than an assert list so `graders.py` needs
   adapting, and its tasks import real third-party libraries.
3. **Omni-MATH filtered to `difficulty >= 7`** for the maths half. 4,428 olympiad
   problems with a shipped difficulty float, but some answers are symbolic
   expressions that exact match handles badly — `sanity_check.py` is the tool for
   finding out how many.

### Where to get more tasks

Change `N_MATH` and `N_CODE` in `build_taskset.py` and rebuild. No new data, no
new code. Verify the pool sizes yourself rather than trusting this file:

```bash
python3 -c "import build_taskset as b; print(len(b.load_math500(5)), len(b.load_mbppplus()))"
```

> **Recommendation: set `N_MATH = 134` and `N_CODE = 370`** for the full 504 at
> the current difficulty settings, then re-probe before committing to a full run.
> The probe over 504 costs about $3.60 from the measured per-call figures.
>
> Two knock-on effects to expect. `stratified_sample` samples across four
> difficulty buckets and taking the whole pool makes it a no-op — fine, but the
> maths/code balance inverts from 60/40 to **27/73**, so the code half would
> dominate every aggregate. Report per-domain figures, as the tables already do,
> or cap `N_CODE` to hold the ratio. Second, the maths half is already at its
> difficulty ceiling for this dataset: 134 is *all* of MATH500 level 5, so maths
> cannot grow further without a new source.

### What NOT to do

Do not add a third domain yet. The two-domain design is load-bearing: it is what
creates the perfect-verifier / proxy-verifier contrast the whole experiment rests
on. A third domain adds tasks but also adds a confound, and you already have a
cheaper way to add tasks.

---

## 5. What to expect to change when the numbers become real

Written down in advance so the predictions could be scored later. Two of them now
can be.

| what | mock says | expect on a real run | outcome |
|---|---|---|---|
| **cost per task** | modelled | **2–3x higher** | ✅ **CORRECT, and if anything understated.** Measured 6 August: level-5 maths runs 650 output tokens against a modelled 80–120, and the two-arm probe cost $0.92 against a $0.44 estimate. |
| **routable fraction** | 29%, "comfortably in band" | §4 expects it low | ✅ **CORRECT.** Real answer 15.0%. The mock's 29% was a restatement of `MOCK_SKILL` — `difficulty_pct` is a rank within the sampled set, so the mock is structurally blind to absolute difficulty. |
| **grader correctness** | assumed | not predicted at all | ❌ **MISSED.** Nobody wrote down that the grader itself might be wrong. It was, on 7 of 20 wrong maths answers, and `sanity_check.py` could not see it. Worth remembering next time a measurement looks like a capability result. |
| maths cascade accuracy | very high | **notably lower** | untested — needs the cascade policies run for real |
| `llm_router` accuracy | competitive | **unknown, probably worse** | untested |
| the ratio finding | sign flips across ladders | **should hold** | untested; only `wide` has real data |
| RouteLLM below random | −2 to −3% AUC | **direction should hold** | untested, and currently unrunnable — the cached scores no longer cover the rebuilt task set |
| three-rung ladder | middle rung barely used | **genuinely unknown** | untested; needs the `claude` ladder |

The two that resolved both resolved *against* the mock, in the direction of "the
mock flatters the project". Weight the remaining five accordingly.

---

## 6. Where things live

| file | read it when |
|---|---|
| `STATUS.md` | now |
| `WALKTHROUGH.md` | you want to understand the code: file by file, with a real trace |
| `README.md` | you want the technical overview and the citations |
| `EXPLAINED.md` | you want the plain-language version of any concept |
| `NOTES.md` | you want the honest list of what is wrong and what is unresolved |
| `models.py` | changing models, prices or ladders — `DECISION #1` at the top. `MAX_TOKENS` carries the truncation measurement |
| `policies.py` | changing what a policy does — `DECISION #2`–`#9` |
| `build_taskset.py` | changing how many tasks, or which. Both difficulty defaults were raised on 6 August |
| `DATASETS.md` | choosing a different benchmark, and why MBPP+ was the one taken |
| `ROUTABLE_2026-07-30.md` | why the routable fraction is the quantity that matters |

Everything *fabricated* is gitignored: `frontier.jsonl`, `sweep_degraded.jsonl`,
`figures/`, and the mock caches (`raw_calls.*.mock.jsonl`). That is deliberate — a
plausible percentage sitting in a repo is how a simulated number ends up quoted as
a measurement.

The real artefacts are the exception and are **force-added to git** despite the
ignore rule:

- `cache/raw_calls.{wide,claude,deepseek}.jsonl` — **318 real responses, $1.36 of
  spend.** The irreplaceable asset. `ROUTER_MODE=replay` reproduces every paid run
  from them field-for-field.
- `results.jsonl` — 47 rows from the 30 July plumbing run, all `simulated: false`.
  Historical: it predates the 6 August rebuild, so its `code-*` task ids no longer
  exist in `taskset.jsonl`. Do not run `frontier.py` or `stats.py` against it.
- `results.probe.jsonl` — 200 rows, both arms, all `simulated: false`. Normally
  gitignored as fabricated; force-added because this copy is real.

**`results.jsonl` was clobbered by a mock run once, on 31 July, and recovered from
git.** The clobber guard in `run_eval.guard_clobber` exists to prevent exactly
that; `--force` defeats it. If it happens again, `git checkout HEAD -- results.jsonl`
is the recovery — note `HEAD`, because the file may be staged.

---

## 7. One-paragraph summary

The pipeline is done, the task set has been hardened on both halves, and the
two-arm probe has been run for real: **routable = 15.0%** [9.3%, 23.3%] on the
`wide` ladder, cheap 79% against expensive 92%, McNemar p=0.002, rescue rate 71%.
That is up from an effective zero on 30 July, so the MBPP+ and level-5 hardening
did what it was supposed to. Two measurement bugs were found and fixed on the way
and both mattered: the maths grader was rejecting 7 of 20 correct-but-differently-
formatted answers (worth 5 points of Opus accuracy, and invisible to a regression
gate that only tested `grade(GT, GT)`), and the models turn out to disagree with
themselves on 8.7% of maths tasks between independent draws of the same prompt.
The signal is real, significant, and thin: every policy competes over a 17-point
ceiling, and 15% sits exactly on the floor of the band `routable.py` asks for —
which more tasks cannot resolve, since the CI just shrinks around the boundary.
The next move is to enlarge the task set to the full 504 available at the current
settings for statistical power, then pay for a full eleven-policy run — budgeting
from the measured $0.0078/call on maths rather than the modelled figures in §3,
which the probe showed to be low by about 2.5x. Nine of the eleven policies still
have no real accuracy data at all.
