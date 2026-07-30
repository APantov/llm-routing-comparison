# STATUS — read this first

Written 30 July 2026. This is the "you just woke up and forgot everything" file.
It answers four questions in order: where the project is, what to do next, what it
will cost, and what to expect when you do it.

If you only read one section, read [§2 Do this next](#2-do-this-next).

---

## 1. Where the project is

**The machine is finished. The plumbing has been checked against real models. The
experiment has not been run.**

Everything works end to end: 100 tasks, 11 policies, three model ladders, a
calibration split, a cost-quality frontier, paired significance tests, a
degradation sweep, and figures. All of it reproduces byte-for-byte on a bare
Python install with no API key and no network.

**Step 4a below has been done.** `ROUTER_MODE=real ROUTER_LADDER=wide python3
run_eval.py --limit 10` was run on 30 July 2026. It produced 47 real model
responses across `cache/raw_calls.{wide,claude,deepseek}.jsonl` (45 / 1 / 1) and
47 rows in `results.jsonl` carrying `"mode": "real"`, `"simulated": false`, at a
total cost of **$0.0481**. Five tasks are reported (`code-475`, `code-86`,
`math-105`, `math-331`, `math-92`); the cache covers ten because `fit_estimators`
also calls on the calibration half.

**That run carries no accuracy information.** Every policy scored 100% on all five
tasks, so the routing-skill denominator is 0/0 and every comparison is a tie by
construction. What it did establish, which is the point of a plumbing check: keys
resolve, both providers answer, prompts parse, the graders score real output, the
cache writes and replays field-for-field, nothing truncated (max 621 output tokens
against `MAX_TOKENS = 2048`), and **real latency is ~2.7s mean against the 0.4–0.9s
the mock stipulates — 3–7x the modelled figure.**

**Every accuracy number this repo reports is still fabricated.** Mock mode invents
model replies from constants in `models.py`, and the one real run is too small and
too degenerate to replace any of them. That is the one thing standing between this
repo and a result.

What is *already* real, even in mock mode:

- the price tables, verified against provider docs on 2026-07-30
- all the cost arithmetic built on them
- the graders, the caching, the escalation logic, the statistics

So the cost conclusions below are trustworthy. The accuracy conclusions are not,
and are labelled as such everywhere.

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

---

## 2. Do this next

In order. Steps 1–3 are free and take about five minutes total.

### Step 1 — confirm nothing rotted (2 min, free)

```bash
python3 build_taskset.py && python3 sanity_check.py
```

Expect `40/40` and `60/60`, then `both graders score every reference answer
correctly`. If either count is short, **stop** — every number downstream is invalid
until it is fixed, and `sanity_check.py` exits non-zero to make that hard to miss.

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

### Step 4b — the difficulty probe (5 minutes, about $0.01–$0.09)

**This is the actual decision point**, and it is the cheapest meaningful thing in
the whole project. The pilot gate depends only on the *cheap rung's* failure rate,
so there is no need to run eleven policies to find it — run one, over all 100
tasks:

```bash
ROUTER_LADDER=wide ROUTER_MODE=real python3 run_eval.py \
    --policy always_cheap --split all
```

100 calls on the bottom rung. About **$0.01** on a DeepSeek cheap rung, **$0.09**
on Haiku. It writes to `results.probe.jsonl`, never touching `results.jsonl`, so it
cannot corrupt a full run's rows.

At n=100 the failure rate is resolved to about ±9 points, which is enough to place
it in the band:

- **20–55%** → the task set discriminates. Go to step 5.
- **below 20%** → the cheap model is too good here. Almost nothing to route, and
  every policy collapses onto `always_cheap`. Fix by raising `MIN_MATH_LEVEL` to 4
  or 5 in `build_taskset.py`, and by replacing the code half — see §4.
- **above 55%** → too hard; everything escalates and every policy collapses onto
  `always_expensive`. Lower `MIN_MATH_LEVEL` to 3.

Run the probe on **each ladder you care about**, because the answer is a property
of the cheap rung, not of the task set alone: DeepSeek v4-flash and Haiku 4.5 will
not fail on the same fraction.

In mock mode this reads 29%, comfortably in band — but that is a restatement of
`MOCK_SKILL`, not evidence. The probe is the first time the number means anything,
and §4 explains why there is real reason to expect it to come in low.

### Step 5 — the full paid run (about 30 minutes, under $1)

```bash
ROUTER_LADDER=wide ROUTER_MODE=real python3 run_eval.py
git add cache/raw_calls.wide.jsonl && git commit -m "raw responses, wide ladder"
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

Modelled from the verified price tables, for all 100 tasks and every policy. The
response cache deduplicates identical calls, so the number that costs money is
*distinct* calls, not policy-attributed calls.

| ladder | probe (cheap rung, 100 calls) | full run (all policies) |
|---|---|---|
| `deepseek` | 100 calls, **$0.004** | 772 calls, **$0.06** |
| `wide` | 100 calls, **$0.004** | 540 calls, **$0.37** |
| `claude` | 100 calls, **$0.049** | 640 calls, **$0.72** |

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

### How many more

To detect a five-point paired difference at the discordance rates seen here, you
need roughly **400–600 tasks**, so 4–6x the current set. Get the real discordance
counts from step 5 and the number becomes exact rather than a rule of thumb.

### Why this is cheap to fix

Cost scales linearly with tasks, and from §3 a full run is about $1.50. So **600
tasks is roughly $10**, once. Every analysis afterwards is free via replay. This is
the highest-value spend available to the project by a wide margin.

### First, a real risk: the code half may be too easy in absolute terms

Not a dilution problem from expanding — a problem with the dataset choice, and it
applies to the current 100 tasks just as much.

MBPP is a saturated benchmark. Frontier models score in the mid-90s pass@1 on it,
and the general assessment is that by 2026 it is only useful for separating models
in roughly the 7B–30B range. Both cheap rungs here (Haiku 4.5, DeepSeek v4-flash)
are well above that class. MATH500 has held up better and still separates frontier
models, particularly at levels 4–5.

So the plausible failure mode is: **the probe comes back below 20% on the code half
and the code cascade has nothing to route.** That would not invalidate the project —
it would relocate it, because the *verifier-quality* experiment lives on the code
half, which is the half with the perfect verifier.

If that happens, the fix in order of effort:

1. **Raise `MIN_MATH_LEVEL` to 4** and rebalance toward maths. Cheapest, no new
   data, but it shrinks the perfect-verifier half.
2. **Replace MBPP with a harder executable-test benchmark.** LiveCodeBench is the
   named successor and is contamination-controlled by release date. This preserves
   the design — the whole experiment needs code tasks that ship runnable tests, so
   whatever replaces MBPP must too.
3. **Keep MBPP but filter it** to items whose reference solution exceeds some line
   count. Free, but a small and biased subset.

Do not do any of these until the probe says so. It costs $0.09 to find out, and
guessing wrong in either direction wastes far more.

### Where to get more tasks

- **Maths**: MATH500 at levels 3–5 leaves **367** candidates and you are using 60.
  Take all 367. Free, already downloaded, one constant change.
- **Code**: sanitized MBPP leaves **420** usable items after dropping the ten
  canonical few-shot examples, and you are using 40. Take all 420. Also free and
  already on disk.

That gets you to **787 tasks** with no new data and no new code — change `N_MATH`
and `N_CODE` in `build_taskset.py` and rebuild. Verify the pool sizes yourself
rather than trusting this file:

```bash
python3 -c "import build_taskset as b; print(len(b.load_math500()), len(b.load_mbpp()))"
```

> **Recommendation: do this first.** Set `N_MATH = 367` and `N_CODE = 420`, run
> `build_taskset.py`, then pilot. There is no reason to pay for a run at n=100 when
> n=787 costs the same order of magnitude and is the difference between "no
> detectable difference" and an actual result.
>
> One knock-on effect to expect: `stratified_sample` currently samples across four
> difficulty buckets, and taking the whole pool makes it a no-op. That is fine, but
> the maths/code balance shifts from 60/40 to 47/53, so the two domains carry
> slightly different weight in any aggregate number. Report per-domain figures, as
> the tables already do.

### What NOT to do

Do not add a third domain yet. The two-domain design is load-bearing: it is what
creates the perfect-verifier / proxy-verifier contrast the whole experiment rests
on. A third domain adds tasks but also adds a confound, and you already have a
cheaper way to add tasks.

---

## 5. What to expect to change when the numbers become real

Written down now so you can check your predictions later. This is the honest part.

| what | mock says | expect on a real run | why |
|---|---|---|---|
| maths cascade accuracy | very high | **notably lower** | `_wrong_answer` scatters wrong answers, so majority voting recovers truth far too easily. Real models cluster on the *same* wrong answer. This is the biggest single overstatement in the repo. |
| `llm_router` accuracy | competitive | **unknown, probably worse** | the mock router is an oracle on the mock's own difficulty, corrupted by a constant. Its accuracy measures nothing. Its cost overhead is real. |
| the ratio finding | sign flips across ladders | **should hold** | it comes from the price tables and the escalation logic, not from `MOCK_SKILL`. This is the most robust conclusion here. |
| RouteLLM below random | −2 to −3% AUC | **direction should hold** | out-of-distribution transfer, and the scores are real bert forward passes already. Magnitude may move. |
| three-rung ladder | middle rung barely used | **genuinely unknown** | `MOCK_SKILL["claude-sonnet-5"]` decides this and it is a guess. This question cannot be answered in mock mode at all. |
| cost per task | modelled | **2–3x higher** | modelled replies are 80–120 tokens, shorter than real ones. Ratios between policies should survive. |

---

## 6. Where things live

| file | read it when |
|---|---|
| `STATUS.md` | now |
| `WALKTHROUGH.md` | you want to understand the code: file by file, with a real trace |
| `README.md` | you want the technical overview and the citations |
| `EXPLAINED.md` | you want the plain-language version of any concept |
| `NOTES.md` | you want the honest list of what is wrong and what is unresolved |
| `models.py` | changing models, prices or ladders — `DECISION #1` at the top |
| `policies.py` | changing what a policy does — `DECISION #2`–`#9` |
| `build_taskset.py` | changing how many tasks, or which |

Everything *fabricated* is gitignored: `frontier.jsonl`, `sweep_degraded.jsonl`,
`results.probe.jsonl`, `figures/`, and the mock caches (`raw_calls.*.mock.jsonl`).
That is deliberate — a plausible percentage sitting in a repo is how a simulated
number ends up quoted as a measurement.

The real artefacts are the exception and are **force-added to git** despite the
ignore rule: `results.jsonl` (47 rows, all `simulated: false`) and
`cache/raw_calls.{wide,claude,deepseek}.jsonl` (47 real responses, $0.0481 of
spend). They cost money, they cannot be regenerated for free, and `ROUTER_MODE=replay`
reproduces the paid run from them field-for-field. Note the consequence: a mock run
with `--force` would overwrite `results.jsonl`, and the clobber guard is what stops
that — if it ever gets bypassed, `git checkout results.jsonl` is the recovery.

---

## 7. One-paragraph summary

The pipeline is done and verified, and one 5-task plumbing run against real models
has confirmed it end to end for $0.05 — but every policy tied at 100% on it, so no
accuracy number means anything yet. Before spending more, raise the task count to the full 794
available in the data you already have, because the current n=100 cannot resolve
any of the comparisons and the bigger run costs roughly the same. Then pilot on 10
tasks (~$0.14), read the failure-rate gate, and do the full run (~$1.50 per
ladder). The strongest result already visible is that cascading's advantage scales
with the price ratio and reverses below about 3x — and that one comes from the price
tables rather than the mock, so it should survive contact with reality.
