# NEXT STEP — the execution sequence

> **Written 9 August 2026, after an audit of the working tree against
> [SHIP_PLAN.md](SHIP_PLAN.md).**
>
> This file is the **order of operations**. SHIP_PLAN.md says *what* is worth
> buying and why; this says *in what sequence*, *what code is missing first*,
> and *what has changed since it was written*. Where the two disagree, the
> disagreements are called out explicitly below — SHIP_PLAN is not wrong, it is
> costed against a task set that Stage 3 replaces.
>
> **Read [SHIP_PLAN.md](SHIP_PLAN.md) §0 and §1 before acting.** Everything here
> assumes its two invariants hold.

---

## 0. The one rule that outranks the sequence

`response_cache.make_key` hashes `(mode, model, prompt, temperature,
sample_idx, max_tokens, mock_seed)`.

**Prompt templates and `MAX_TOKENS` are in the cache key. Do not touch them —
not to tidy them, not to refactor around them, not in any stage of this file.**
Changing either strands every committed response and re-charges it. That has
already happened once; the abandoned 2048-token run cost $0.39.

This is not a sequencing question. It holds before the buys, after the buys, and
during the lean-out in Stage 5. If prompt-building code is ugly, **leave it
ugly** and put a comment on it. Same for model ids in `MODEL_SPECS` and anything
that feeds `grader_payload` into a prompt.

The second invariant from SHIP_PLAN §1 still holds and is now load-bearing for
Stage 4: the ladder is *not* in the cache key, and prompts are tier-independent.
Cross-ladder reuse is wired and verified — `_sibling_real_paths` at
[response_cache.py:131](response_cache.py#L131).

---

## 1. Where the tree actually stands (verified 9 August 2026)

Facts a cold reader should not have to re-derive:

| | |
|---|---|
| tests | **168 passing**, ~3s (README still says 164) |
| working tree | clean, on `master`, tag `pre-predictive-removal` |
| **git remote** | **none — the only copy of $4.24 of paid data is this disk** |
| task set | 95 tasks = **60 math + 35 code** |
| real cache | `raw_calls.wide.jsonl` — **980 rows, $4.2352**, all `mode: real` |
| other ladders | `raw_calls.claude.jsonl` **1 row**, `raw_calls.deepseek.jsonl` **1 row** |
| results | `results.jsonl` — 393 rows, 9 policies, **every row `ladder: wide`** |
| figures | `frontier.svg`, `degradation.svg` regenerated 9 Aug against the 95-task set |
| candidate pools | MBPP+ **370** · MATH500 level≥3 **367** · level≥4 **262** · level≥5 **134** |
| Phase 0 | complete, with one deviation — see below |

**Phase 0 deviation.** SHIP_PLAN §2 item 4 asked for truncation to be *an error*.
It is instead **flagged**: counted in `models.call_stats["truncated"]`, stamped
`truncated: true` on the row, and warned loudly at
[run_eval.py:728](run_eval.py#L728) — but it still grades False. Decide
deliberately whether that is enough before Stage 3; it is defensible, since the
alternative interacts with the `MAX_TOKENS` invariant, but it should be a choice
rather than an oversight.

**What already exists that SHIP_PLAN implies is missing:**

- `routable.review_queue` at [routable.py:218](routable.py#L218) — the
  `both_fail` manual-review queue from SHIP_PLAN §2 item 2 is **built**.
- Cross-ladder cache reuse — **built and verified**.
- `build_taskset.py` already accepts `--n-code`, `--n-math`, `--min-math-level`.

---

## 2. What changed against SHIP_PLAN: buy C is cut

**SHIP_PLAN §3-C ("rebuild the maths half by screening", $2.22) is dropped from
this sequence.** Four reasons, in order of weight:

1. **It needs three components that do not exist.** No multi-draw cheap-only
   screener; no path from screening results back into task selection
   (`build_taskset` samples by `SEED`, there is no `--from-ids`); and no
   case-control weighting anywhere — `weight|prevalence|enrich` returns **zero
   hits** across [stats.py](stats.py), [run_eval.py](run_eval.py),
   [splits.py](splits.py), [frontier.py](frontier.py). SHIP_PLAN's own blockquote
   requires that weighting for a screened set.
2. **It costs ~$3.2, not $2.22.** C grows the maths half 60→90, and maths
   self-consistency at the cheap rung is the single most expensive line in the
   plan. That adds ~$1.00 to buy E on top of C's own $2.22. Over half the budget.
3. **It targets the wrong half.** SHIP_PLAN §0.4 measures maths at **$3.93 per
   unit of routing signal** against code's **$0.079**.
4. **SHIP_PLAN §7 already says not to.** "The maths half's job is the
   deployable-verifier contrast, not routing signal. Keep it at n≈60 and stop
   spending on it."

**What is lost, and must be stated in the write-up rather than measured:**
`llm_router` sends 30 of 30 maths tasks to the expensive rung, so the maths half
cannot discriminate one-shot routers at all. That is a real limitation. It costs
$0 to declare and $3.2 to remove, and the one-shot-router claim is already
carried on real data by `routellm` (AUC 85.4%, −0.9% against a cost-matched coin
flip). **Declare it.**

If a later decision reinstates C, Stage 2's screener is the same script it would
need, and the two missing pieces are then `build_taskset --from-ids` plus a
weighting design.

---

## 3. The sequence

Stages are ordered by dependency. **Do not reorder Stage 5 earlier** — the
reasoning is in §4.

### Stage 1 — free, and blocking

1. **Create a git remote and push, including the tag.** Everything else in this
   file is recoverable; `cache/raw_calls.wide.jsonl` is not. Do this first.
2. **Lean out the paid path — and only the paid path.** The code that decides
   *what gets called*, *what is served from cache*, and *what is stamped real vs
   simulated*: `models.call`, [response_cache.py](response_cache.py), the replay
   guards, the spend cap. Nothing else.

   The justification is this project's own history: a silent mock-cache fallback
   fabricated all 240 self-consistency samples and produced a `simulated: false`
   result set reporting 100% maths-cascade accuracy. The recorded lesson — *"a
   guard downstream of a silent fallback is dead code"* — is precisely what
   layered abstraction on a spend path produces. Stage 3 is the largest paid
   batch in the project's history. That path should be short enough to read in
   one sitting before it spends money.

   Gate: `pytest` still 168/168, `scripts/check_core_unchanged.py` still passes,
   and a full replay still produces `results.jsonl` byte-identical to the
   committed one. If any of those move, the refactor is wrong.
3. **Fix the spend guards** (both are real hazards for Stage 3):
   - `MAX_SPEND_USD = 3.0` at [run_eval.py:41](run_eval.py#L41) is checked at
     [run_eval.py:269](run_eval.py#L269) and **stops silently** — it prints
     "stopping early" and continues, which would produce a half-measured task
     set that looks complete. Make it fail loudly, or make the partial state
     unmistakable in the output.
   - [scripts/redraw_decisive.py](scripts/redraw_decisive.py) has **no spend cap
     at all**. Give it one.
4. **Remove dead weight that no finding can resurrect** — this is not the
   Stage 5 lean-out, just deletion of superseded material:
   - `docs/ROUTABLE_2026-07-30.md` (35KB) and `docs/SURVEY_2026-07-30.md` (33KB)
     — superseded by SHIP_PLAN and STATUS.
   - `load_mbpp` / `CODE_SOURCES["mbpp"]` and `data/sanitized-mbpp.json`, kept
     only to reproduce the thin-asserts marking that nothing now reports.
   - the `*.mock.jsonl` caches — derived, regenerate free.

   **Do NOT delete `archive/`.** Leave it exactly as it is.

Commit at the end of Stage 1. Everything above is free and reversible.

### Stage 2 — build the one missing tool

**A cheap-rung-only multi-draw screener.** Neither existing entry point can do
this:

- `run_eval.py` has no `--draws`. Its whole CLI is `--limit --domain --split
  --policy --force`.
- [scripts/redraw_decisive.py](scripts/redraw_decisive.py) always redraws **both
  rungs** — [line 213](scripts/redraw_decisive.py#L213) hard-codes
  `len(targets) * 2 * args.draws` — and it needs an existing probe to compute
  cells from, so it cannot run over a raw candidate pool.

Simplest honest options, in order of preference:

1. Add `--tier {cheap,expensive,both}` and a candidate-pool source to
   `redraw_decisive.py`. It already has the costing, the `--go` gate, the
   incremental cache write, and the mode check that refuses to "redraw" in
   replay mode. Reuse all of that.
2. Failing that, a new ~100-line `scripts/screen_pool.py` following the same
   pattern: **print a costed plan and exit unless `--go`**, write every response
   to the cache as it arrives, refuse to run unless `ROUTER_MODE=real`.

Note: **B does not strictly require this.** The three cheap draws in B are for
per-task p̂ (noise), not for selection — B is a census of the whole pool, so a
single-draw version is a valid experiment costing $0.03 less. If the screener
slips, B can run without it and the draws can be backfilled. Do not let this
block Stage 3.

### Stage 3 — the buys: B → D → F → E

Costs below are recomputed against the **post-B** task set, using the measured
per-call figures where they exist (code 141 in / 59 out, maths 112 in / 811 out)
and the shipped `MODEL_SPECS` price table. **SHIP_PLAN's figures for E and F were
priced against the old ~100-task set and are superseded by these.**

Run each buy as its own commit. Never start the next one with a dirty tree.

---

#### B — rebuild the code half · **~$0.97**

```bash
python build_taskset.py --n-code 370          # 370 pool − 5 quarantined = 365
ROUTER_MODE=real ROUTER_LADDER=wide python run_eval.py \
    --policy always_cheap --policy always_expensive --split all
# then the 3 cheap draws via the Stage 2 screener
```

Resulting task set: **365 code + 60 maths = 425 tasks**.

**This is the best purchase in the project, and inspection strengthened the
case: it is a complete enumeration of MBPP+, not a screen.** No enrichment, so
no weighting problem, so none of C's missing infrastructure applies.

Breakdown: 365 Opus code calls at measured $0.002562 = $0.935; 365 × 3 cheap
draws at $0.000036 = $0.039.

> **⚠ THE REAL COST OF B IS HUMAN, NOT FINANCIAL.**
>
> SHIP_PLAN §0.1 found **5 spec-broken tasks in 40** code tasks — 12.5%. At 365
> tasks, expect roughly **40–45 tasks that neither rung can pass**, each needing
> the same manual adjudication the original five got. Budget real time for this,
> and do not skip it: those five broken tasks were *all* of `always_expensive`'s
> failures and capped every policy in the project at 92% instead of 100%.
>
> `routable.review_queue` ([routable.py:218](routable.py#L218)) produces the
> queue. The standing rule in SHIP_PLAN §0.5 applies unchanged: evidence goes in
> `build_taskset.QUARANTINED` with the specific input that breaks the task, the
> responses are **deleted** rather than filtered (`scripts/purge_quarantined.py`),
> and `TestQuarantine` is the tripwire that catches reintroduction.
>
> Adjudicate **before** running D. A policy comparison over a set with 40
> unpassable tasks in it measures the same artefact all over again.

After adjudication, regenerate RouteLLM scores for the new task set —
`cache/routellm_scores.jsonl` currently holds 95 and needs one per task. The
`bert` variant is local and free.

#### D — full 9-policy run on the new set · **~$0.05–0.10**

Cheap because almost everything is already bought: code verification is running
asserts ($0), the maths half is unchanged so its 240 self-consistency samples
(`SELF_CONSISTENCY_K = 5`) are still cached, `routellm` scores locally, and
`random_matched` and `cascade_degraded` replay. Genuinely new: `llm_router`'s
routing call per new code task, ~365 × $0.000036 ≈ $0.013.

**Cutting C is what keeps D this cheap** — a new maths half would re-buy every
self-consistency sample.

#### F — the `deepseek` ladder · **≤$0.27**

The cheap rung is already on disk (`deepseek-v4-flash` is `wide`'s cheap rung),
so cross-ladder reuse serves it and only `v4-pro` is new.

SHIP_PLAN says $0.06. That looks like answers-only: the deepseek ladder can
**verify at every rung** ([models.py:157](models.py#L157)), so escalated maths
tasks buy self-consistency at the pro rung too. $0.27 is the upper bound with
every maths task escalating; the real figure will be lower.

#### E — the `claude` ladder · **~$2.64**

Run **last**. Its cross-ladder reuse only pays once B's Opus rows are on disk;
run it before B and you buy Opus twice.

| | E (claude) | F (deepseek) |
|---|---|---|
| old 35c/60m set (SHIP_PLAN's basis) | $2.06 | ≤$0.23 |
| **post-B 365c/60m set** | **$2.64** | **≤$0.27** |
| post-B+C 365c/90m (not being done) | $3.64 | ≤$0.38 |

Opus is reused free; haiku and sonnet are entirely new. Of E's $2.64, **$1.00 is
haiku-rung self-consistency on the 60 maths tasks** — the most expensive single
line in the whole sequence, and the reason the third row of that table is why C
was cut.

---

**Stage 3 total: ~$4.0**, against a €10–15 budget. That leaves the contingency
SHIP_PLAN §6 asks you to hold rather than allocate.

### Stage 4 — freeze the numbers · free

Re-run `run_eval` / `frontier` / `stats` / `sweep_degraded` / `plot` across all
three ladders. Costs nothing and needs no API key — the property that makes the
repository shippable.

Promote **p̂ per task** to the primary unit of analysis instead of a binary cell,
per SHIP_PLAN §4. `scripts/redraw_decisive.py` already produces it.

**New reporting requirement created by B:** the task set is now 86% code by
count, so **every aggregate number is code-dominated**. Report per-domain
throughout and never let an aggregate stand in for either half. The old
`routable = 15%` framing over a roughly balanced set does not carry over.

Commit the frozen outputs. **This is the checkpoint the lean-out is verified
against.**

### Stage 5 — the real lean-out · free

Now remove the higher-level abstraction.

**Why here and not earlier.** Replay is $0.00 and byte-for-byte reproducible, so
post-data refactoring is: rip out an abstraction → re-run the whole pipeline for
nothing → diff against Stage 4's frozen outputs. That is close to an ideal
refactoring environment, and it only exists at full strength *once the data is
bought*. Several thousand real responses across three ladders is a far stronger
regression corpus than today's 980 on one.

**And because you cannot know what is load-bearing until Stage 3 reports.** The
ladder machinery — `LADDERS`, `_build_ladder`, positional tier naming, the
2-vs-3-rung handling in `_TIER_NAMES` — is the largest abstraction in the repo,
and whether it earns its keep is exactly what E and F measure. If the three-rung
`claude` ladder adds nothing over two rungs, that machinery collapses. Delete it
in Stage 1 and you may well rewrite it.

Candidates to assess once the findings are in, none of them pre-judged here:

- the ladder abstraction, if E and F say a rung is redundant
- `cascade_degraded` and the degradation sweep, if the verifier-quality curve
  reduces to a sentence
- the policy registry, if fewer policies survive as interesting
- the agent layer's indirection over `models.call` — but keep the property that
  makes it honest: it shares the price table, cache and replay
- `scripts/` — several exist to answer a question that is now answered

**Gate for every deletion:** `pytest` green, `scripts/check_core_unchanged.py`
passes, and the full replay reproduces Stage 4's outputs byte-for-byte. If the
numbers move, the abstraction was load-bearing — put it back.

**Still do not touch** prompt templates, `MAX_TOKENS`, `MODEL_SPECS` ids, or
`archive/`.

### Stage 6 — the write-up · free

Docs come last because Stage 3 rewrites the headline numbers and Stage 5
rewrites what the code looks like. Doing this earlier means doing it twice.

The documentation problem is structural, not factual: **corrections have been
added as warning banners on top of documents whose bodies still assert the old
thing.** A reader hits contradictions within one screen. Fix the bodies and
delete the banners.

Known outstanding items:

| item | where | action |
|---|---|---|
| "Code is now the harder domain in absolute terms" | [STATUS.md:196](STATUS.md#L196) | **retract in the body** — the banner at STATUS.md:17 already does, the section does not |
| stale n=100 / 92% / 40-code tables | STATUS §1 | rewrite against Stage 4 |
| "cascade is within one task of always-expensive" | [README.md:222](README.md#L222), [STATUS.md:288](STATUS.md#L288) | **qualify** — it rested on b/c = 0/1, one discordant task; restate at the new n |
| "100 tasks", "10 policies", "164 tests" | README header table | now 425, 9, 168 |
| spend to date | README, STATUS | **$4.2352 across 980 responses**, plus Stage 3 |
| maths cannot discriminate one-shot routers | new | **state it** — see §2 above |
| `predictive` retraction | — | already done, leave it |

Two real findings SHIP_PLAN §5 says are buried and should be promoted:

- **`scripts/resample_vs_reroute.py`** — majority-of-k does not substitute for
  escalating, and self-consistency is *confidently wrong*: `math-94` shows 78%
  modal agreement with 2 of 9 draws correct, one point below
  `AGREEMENT_THRESHOLD`.
- **The determinism asymmetry** — Opus deterministic, DeepSeek not. It reframes
  the noise discussion and it is what made a cheap noise floor possible.

Finally: **fold SHIP_PLAN.md and this file into STATUS.md and delete both.**
They are scaffolding for a transition. Once Stage 6 is done the repository should
have one status document that is true on its own, with no banner telling the
reader that the text below it is wrong.

---

## 4. Summary

| stage | what | cost | blocks |
|---|---|---|---|
| 1 | remote, paid-path simplification, spend guards, dead weight | $0 | everything |
| 2 | cheap-only multi-draw screener | $0 | nothing hard — B can run without it |
| 3 | **B** → adjudicate → **D** → **F** → **E** | **~$4.0** | Stages 4–6 |
| 4 | freeze all numbers across three ladders | $0 | Stage 5's regression gate |
| 5 | the real lean-out | $0 | Stage 6 |
| 6 | write-up, retractions, collapse the docs | $0 | — |

**Cut:** SHIP_PLAN §3-C. **Do not delete:** `archive/`. **Never touch:** prompt
templates and `MAX_TOKENS`.
