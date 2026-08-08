# SHIP PLAN — read this before STATUS §2

> **Written 8 August 2026, after an independent audit of the working tree.**
>
> **This file supersedes [STATUS.md](STATUS.md) §2 "Do this next".** That section
> tells you to go to step 5 and pay for a full eleven-policy run at n=100. Do not
> do that. The audit found three defects that make the current headline numbers
> mean something other than what they say, and all three are free to fix. Fixing
> them first changes what is worth buying.
>
> **Budget: about $6 of new spend buys a materially better experiment.** Nothing
> below requires more. $4.40 has already been spent; none of it is wasted, and
> two invariants in §1 are what keep it that way.

---

## 0. What the audit found

Three defects, in descending order of how much they change the story. Every one
is reproducible from the committed cache with no API key.

### 0.1 Four of the five `both_fail` code tasks are broken, not hard

On the eval split, `always_expensive` fails exactly four tasks, and all four are
unpassable from their prompt:

| task | the defect |
|---|---|
| `codeplus-119` | "find the element that appears only once in a sorted array". MBPP+ expects `[1,2,3,4,5,6]` → **7**. Seven is not in the array — the expectations are the XOR-fold of the reference implementation. A textbook-correct solution mismatches **64 of 110** inputs. |
| `codeplus-792` | "count the number of lists in a given number of lists" is ambiguous English. All three shipped asserts are consistent with both `len(x)` and "count sublists"; only the hidden inputs disambiguate. Opus wrote `return len(x)`. |
| `codeplus-771` | balanced-parentheses. Expects `""` → `False`. An empty expression is balanced. A textbook solution mismatches exactly 1 of 106 inputs — that one. |
| `codeplus-305` | the reference produces mutually inconsistent expectations for identically-shaped inputs. |

Correcting for them moves every row of the headline table:

| policy | as reported | excluding the 4 broken | code only |
|---|---|---|---|
| `always_expensive` | 92.0% | **100.0%** | 80% → **100%** |
| `oracle` | 92.0% | 100.0% | 80% → 100% |
| `cascade` | 90.0% | 97.8% | 80% → 100% |
| `always_cheap` | 78.0% | 84.8% | 65% → 81.2% |

So STATUS's "**Code is now the harder domain in absolute terms** — MBPP+ produced
genuine `both_fail` content" is wrong, and so is the derived "half of the cheap
rung's code failures are unfixable by escalating". They are unfixable because
they are broken. The 17-point ceiling is set by four defective tasks.

**The routing signal itself survives untouched** — see §0.4.

### 0.2 The one-shot router comparison is degenerate

`MIN_MATH_LEVEL = 5` makes `predict_features.level` **constant at 5 across all 60
math tasks**. `predict_is_hard` returns `level >= 5`, so `predictive` escalates
60/60 math. `random_matched` calibrates to predictive's realised rate, so it does
too. `llm_router` routed 30/30 math expensive as well.

On 60% of the task set, three "routers" are literally `always_expensive`.
`run_eval` already prints routing skill `n/a` on math because there is nothing
left to compare. So "every one-shot router costs 90–93% of always-expensive and
gets less accuracy — LLMRouterBench reproduced on real data" is an arithmetic
consequence of the level filter, not a measurement.

[docs/NOTES.md](docs/NOTES.md) issue 6 calls the math signal "flattering". It is
not flattering, it is absent.

### 0.3 Smaller measurement gaps

- **The grader bug class fixed on 6 August is still half-present.** `\sqrt5` vs
  `\sqrt{5}` was fixed; redundant braces on exponents were not. `x^{2}` ≠ `x^2`,
  `a_{1}` ≠ `a_1`, `2^{10}` ≠ `2^10` all grade False. Three math ground truths are
  exposed (`120^\circ`, `3R^2`, `8n^2 + 4n + 1`) and `sanity_check.EQUIVALENT`
  does not cover the case.
- **Three live truncations at 4096, not one.** STATUS documents `math-96`. Two
  more entered in the 7 August redraw and were never inspected: `math-422` idx=10
  and `math-154` opus idx=2. `math-154` is scored `both_fail` at expensive 0/10 —
  one of those ten "failures" is a truncation.
- **$0.13 of the cache is unreachable.** Two orphaned 2048-cap rows ($0.053, correctly
  unreachable since `max_tokens` is in the key) and 58 rows on 16 task ids that no
  longer exist in the task set ($0.073).

### 0.4 What the halves actually contain

Per-task success probability across every greedy draw on disk, both rungs:

**Math, 60 tasks** — mean p: cheap 0.891, expensive 0.967

| cheap × expensive | tasks |
|---|---|
| always / always | **47** |
| flaky / always | 10 |
| never / never | 1 |
| **never / always** | **1** ← the only clean routing signal |
| always / never | 1 (inverted) |

**Code, 40 tasks** — mean p: cheap 0.750, expensive 0.861

| cheap × expensive | tasks | broken |
|---|---|---|
| always / always | 29 | 0 |
| **never / always** | **5** | **0** ← clean routing signal |
| never / never | 4 | **4** |
| never / flaky | 1 | **1** |
| always / never | 1 (`codeplus-800`, inverted) | 0 |

Two things follow.

**Delete the 5 broken code tasks and the code half is a textbook cascade
setting**: 29 ties, 5 routable (14.3%), 1 inverted, **zero `both_fail`**, rescue
rate 100%, and deterministic on both rungs. That structure already exists, in the
half that was written off as the weaker one.

**The math half bought one task.**

| | tasks | spend | clean routable | $ per unit of signal |
|---|---|---|---|---|
| code | 40 | $0.397 | 5 | **$0.079** |
| math | 60 | $3.932 | 1 | **$3.93** |

89.3% of all spend went to the half that produced one reproducibly routable task.

**Opus is perfectly deterministic** — 0 flaky across 28 redrawn math tasks, 39 of
40 on code. DeepSeek is flaky on 10 of 29. All the decoding noise is on the cheap
rung, where draws cost $0.000036–0.000245.

---

## 1. Two invariants

**Do not touch `MAX_TOKENS` or the prompt templates.** Both are in the cache key.
Changing either invalidates all 1,095 cached responses and re-charges $4.40. That
has already happened once — the abandoned 2048 run cost $0.39. The three live
truncations must be handled by *flagging*, not by raising the cap.

**The ladder is not in the cache key, and prompts are tier-independent.**
Verified: 112 of 112 tasks have identical `prompt_sha256` across rungs, and
`response_cache.make_key` takes `(mode, model, prompt, temperature, sample_idx,
max_tokens, mock_seed)` — no ladder. So Opus rows recorded under `wide` are
directly reusable by the `claude` ladder, and the DeepSeek-flash rows by the
`deepseek` ladder. Only `READ_PATHS` keeps them apart. Wiring that up saves
**$1.91** and is the best free optimisation available.

---

## 2. Phase 0 — free, and blocking

Do all of these before spending anything. They change what is worth buying.

1. **Commit the working tree.** 1,864 uncommitted lines including 759 new cache
   lines. The cache is the irreplaceable asset and it is currently unversioned.
   This is the most urgent item in the repository.
2. **Quarantine the 5 broken code tasks** with the evidence from §0.1, and add the
   systematic version: during screening, any task where *both* rungs fail goes to
   a manual review queue rather than straight into the set. The screen data in
   Phase 1 gives you that queue for free.
3. **Fix the grader's brace normalisation** (`x^{2}`≡`x^2`, `a_{1}`≡`a_1`) and add
   rows to `sanity_check.EQUIVALENT`.
4. **Make truncation an error, not a wrong answer.** No `\boxed{}` *and* at
   `max_tokens` should raise, not grade False.
5. **Lower `run_eval.MAX_SPEND_USD` from $20 to $3.** At $20 a bug eats the whole
   budget before the cap binds.
6. **Regenerate RouteLLM scores** — `python routellm_router.py --score`. Free,
   local, no API key. Recovers one of the two never-run policies.
7. **Add the cross-ladder cache read path** described in §1.

---

## 3. Phase 1 — the buys

Costs use **measured** per-call figures where they exist (DeepSeek-flash and Opus,
from 1,095 real responses) and the verified price table elsewhere. Modelled
figures use pooled token counts and run ~10–35% high, so treat the total as an
upper bound.

Measured: code cheap $0.000029 · code expensive $0.002562 · math cheap $0.000245 ·
math expensive $0.019648. Measured tokens: code 141 in / 59 out, math 112 in / 811 out.

| | what | cost |
|---|---|---|
| **B** | **Rebuild the code half.** Screen all 370 MBPP+ ×3 cheap draws, then Opus on all 370 | **$0.98** |
| **C** | **Rebuild the math half by screening.** 367 MATH500 (level ≥3) ×5 cheap draws, Opus on the ~90 that fail reproducibly | **$2.22** |
| **D** | Full 9-policy run on the new set — only self-consistency samples and router calls are new | **$0.05** |
| **E** | `claude` ladder, reusing Opus rows from `wide` (modelled; likely ~$1.60 real) | **$2.19** |
| **F** | `deepseek` ladder — cheap rung already cached, only v4-pro is new | **$0.06** |
| **G** | Noise floor: 10 extra **cheap** draws on all tasks, 3 extra Opus draws only where Opus failed | **$0.51** |
| | **total** | **~$6.00** |

**B is the best purchase in the project.** Under $1 buys complete two-arm data
over the *entire* MBPP+ pool — 3.7× the current sample, on the half that carries
reproducible signal, at $0.079 per unit of signal against math's $3.93.

**C changes the selection criterion, which is the actual fix.**
`build_taskset.py` selects on MATH500's shipped `level` — absolute difficulty
against a 2021-era notion of hard. Difficulty and routability are different axes
and at the top of the scale they decouple. Screening on *measured, reproducible
cheap-rung failure* targets the quantity the experiment needs, and restores level
variance, which is what un-breaks the predictive router from §0.2.

**G is deliberately not what the 7 August redraw did.** That cost $2.96 and
established that Opus is deterministic. Redrawing the expensive rung again would
cost $6.13 and tell you nothing. Draw the cheap rung heavily; the expensive rung
barely.

> **Screening enriches the sample, and that has to be designed in.** A screened
> set cannot answer "what fraction of natural traffic is routable". Keep a small
> random-sample arm for the prevalence estimate and use the enriched arm for the
> policy comparison with weights — ordinary case-control. Report the two numbers
> separately and never let the second stand in for the first.

---

## 4. Phase 2 — free, off the new cache

Re-run `run_eval` / `frontier` / `stats` / `sweep_degraded` / `plot` on all three
ladders. With B–F committed this costs nothing and is reproducible by anyone
without a key — the property that makes the repository shippable at all.

Promote **p̂ per task** to the primary unit of analysis instead of a binary cell.
`scripts/redraw_decisive.py` already produces it. The 10 flaky math tasks are not
worthless — if cheap is right 40% of the time and Opus is certain, escalating
genuinely helps in expectation. They are badly *measured* by a one-draw
cross-tab, which is why the labels flipped on redraw.

---

## 5. Phase 3 — the write-up

Three retractions are required. They are the difference between a defensible
repository and an overclaiming one.

| claim | where | status |
|---|---|---|
| "Code is now the harder domain in absolute terms" | STATUS §1 | **retract** — all 4 `both_fail` code tasks are spec-broken (§0.1) |
| "`predictive` contributes no frontier point — LLMRouterBench reproduced on real data" | STATUS §1, README | **retract** — degenerate feature (§0.2) |
| "The cascade is within one task of always-expensive — the project's thesis, confirmed" | STATUS §1, README | **qualify** — rests on b/c = 0/1, one discordant task |

Two findings that *are* real and currently buried should be promoted:

- **`scripts/resample_vs_reroute.py`** — majority-of-k does not substitute for
  escalating, and self-consistency is *confidently wrong*: `math-94` shows 78%
  modal agreement with 2 of 9 draws correct, one point below
  `AGREEMENT_THRESHOLD`.
- **The determinism asymmetry** — Opus deterministic, DeepSeek not. It reframes
  the whole noise discussion and it is what makes G cheap.

Also correct: spend to date is **$4.4020** across **1,095** committed responses
(README and STATUS say 1,097), of which $0.13 is unreachable (§0.3).

---

## 6. Budget

**Spend ~$6 (≈€5.50) of a €10–15 budget. Hold the rest as contingency rather
than allocating it** — the most likely real cost is a re-run after a bug, and
this project has already lost $0.39 to an abandoned run plus $0.13 stranded.
Do not pre-commit the remainder to more tasks.

**Minimum viable shippable result: B + D + F = $1.09.** That is a 370-task code
experiment with real statistical power plus the `deepseek` ladder on real data.
If budget tightens, drop **C** then **E**, in that order.

---

## 7. What to skip

- **Harder maths.** Omni-MATH ≥7, AIME and friends push *both* rungs down. You
  would trade "everything ties at the top" for "everything ties at the bottom".
- **A third domain.** The two-domain design is load-bearing; a third adds a
  confound.
- **More Opus redraws.** It is deterministic. §0.4.
- **Raising `N_MATH` at level 5.** The math half's job is the deployable-verifier
  contrast, not routing signal. Keep it at n≈60 and stop spending on it.

---

## 8. Reproducing the audit

Everything in §0 comes from the committed cache and needs no API key. The
decompositions in §0.4 are per-task success rates over
`cache/raw_calls.wide.jsonl` filtered to `kind == "answer"`, `temperature == 0.0`,
excluding the two orphaned rows at `tokens_out == 2048`, graded through
`graders.grade`. §0.1 is a textbook solution run against
`grader_payload.test_program`'s `inputs`/`results` arrays. The full 9-policy
replay reproduces byte-for-byte: 1,173 calls served from cache, 0 fabricated,
0 backend hits, and `pytest` is 163/163.
