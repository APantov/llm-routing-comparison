# LLM Routing: a measured benchmark, and the router it argues for

**A cost-aware LLM routing service (LangGraph + MCP), and the 100-task
benchmark that decides its policy.**

Answer at the cheapest model that can be *verified* to have got it right;
escalate only when verification fails. Whether that beats simply paying for
the best model is not a matter of opinion — it depends on the price ratio
between your models, and this repository measures where the crossover is.

|  |  |
|---|---|
| **What runs** | A LangGraph state machine — `classify → answer → verify → escalate ⟲` — with human-in-the-loop approval *inside* the escalation loop and checkpointed resume, exposed over an MCP server with four tools and four resources. |
| **What decides its policy** | 100 tasks (MBPP+ code, MATH500 level 5), 10 policies, 3 price ladders, cost–quality frontiers, exact McNemar and a paired bootstrap. |
| **Built with** | Python 3.10–3.13 · LangGraph · MCP · Anthropic + DeepSeek APIs · pytest (164 tests) · GitHub Actions. The research core is **pure standard library** — no dependency can change a benchmark number. |
| **Measured against real models** | **~10%** of tasks are ones the cheap model reliably gets wrong and the expensive one reliably gets right (n=100, wide ladder). A single draw per task said 15%; ten draws on the 21 decisive tasks showed **four of them were coin-flips the cheap model wins every time**. The routing signal is real, thinner than it first looked, and lives almost entirely in the code half. 1,097 real responses committed, so anyone can replay all of it for $0.00. Total spend to date: **$4.40**. |
| **Measured, not simulated** | **Every policy the `wide` ladder defines — 9 of 10**, on real models, on the held-out half. `cascade` reaches **90.0% against always-expensive's 92.0% at 23% of the cost**, and the cost gap is significant while the accuracy gap is not. See [§ Status](#status-every-policy-on-this-ladder-now-has-real-numbers). |
| **Not yet measured** | `always_mid` only, because it exists only on the three-rung `claude` ladder. Any policy a replay cannot serve is **dropped by name**, never scored on a subset. |

> **⚠ [SHIP_PLAN.md](SHIP_PLAN.md) — 8 August 2026.** An independent audit found
> that the four `both_fail` code tasks are *broken, not hard* (corrected,
> `always_expensive` and `oracle` score 100%, not 92%). **Read it before quoting
> any accuracy figure from this page.**
>
> Its second finding has been acted on. The `predictive` policy routed on a
> difficulty label that `MIN_MATH_LEVEL = 5` made constant, so it was
> `always_expensive` on 60% of the task set while being reported as a router. It
> has been **deleted**, and predictive routing is now measured with the two real
> implementations it always had: `llm_router` and `routellm`.

New here? [SHIP_PLAN.md](SHIP_PLAN.md) says what is wrong and what to do about
it; [STATUS.md](STATUS.md) says where the project stands;
[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) traces one task through every file;
[docs/EXPLAINED.md](docs/EXPLAINED.md) is the plain-language version of the ideas.

```bash
pip install -e ".[agent]"
python build_taskset.py
python -m router_agent.cli --demo      # real model output, no API key, $0.00
```

```
  classify  domain=math, start=cheap, verifier=self_consistency
  answer    cheap (deepseek-v4-flash) answered
  verify    self_consistency -> REJECT
  escalate  cheap -> expensive
  answer    expensive (claude-opus-5) answered
  finalize  done: exhausted_ladder

  verdict   UNVERIFIED
  answered  expensive - claude-opus-5
  cost      $0.059111  (2 calls, 1 escalation)
```

That demo runs against **1,097 real model responses committed to this
repository**, so it needs no API key, no account and no money — and it is real
output, not a simulation.

## The finding the router implements

Matched on accuracy, against simply always paying for the best model:

| ladder | effective price ratio | cascade vs always-best | verdict |
|---|---|---|---|
| `deepseek` v4-flash → v4-pro | 3.11x | **costs more** | just route |
| `claude` Haiku 4.5 → Sonnet 5 → Opus 5 | 6.5x | cheaper | cascade |
| `wide` DeepSeek v4-flash → Opus 5 | 46.4x | **much cheaper** | cascade |

**The sign flips, and the sign is the finding.** Cascading pays in proportion
to the price gap it exploits. A cascade always pays for the cheap call *and*
for verifying it — fixed costs — and what they buy is the *chance* to skip an
expensive call. Below roughly 3x, the fixed costs swamp the saving.

The magnitudes are deliberately not in that table, because they depend on the
task set and this one was rebuilt on 6 August. On the 30 July set the figures
were +33% / −12% / −74%; a mock run over the current set gives roughly
+10% / −46% / −66%; and the `wide` ladder, now measured against real models,
comes in at **−60.9%**. Every sign has held across all three; not one magnitude
has.

So the router computes them rather than quoting them. Price ratios come from
the price table (exact, no run needed); the cost and AUC figures are derived
from `frontier.jsonl` when a run for that ladder is on disk, and fall back to
the 30 July constants **labelled with their date** when it is not. On `wide`
that fallback is no longer needed:

```bash
$ llm-router --findings | jq '.ratio | {verdict, economics_source, economics_simulated}'
{ "verdict": "cascade", "economics_source": "frontier.jsonl", "economics_simulated": false }

$ ROUTER_LADDER=deepseek llm-router --findings | jq '.ratio.economics_source'
"historical"            # never run for real; the date-labelled constant is used
```

The router exposes this rather than hiding it: ask it and it will tell you not
to cascade.

```bash
$ llm-router --findings | jq .ratio.verdict     # on the deepseek ladder
"route"
```

## Two halves

| | |
|---|---|
| **The experiment** (repo root) | 100 tasks, 10 policies, 3 ladders, cost-quality frontiers, paired significance tests. Pure standard library in mock mode. |
| **The product** (`router_agent/`) | A LangGraph cascade and an MCP server, built on the same substrate, implementing what the experiment found. |

They share one model client, one price table and one response cache — which is
what makes a dollar figure from the router mean the same thing as a dollar
figure in the tables below. See [ARCHITECTURE.md](ARCHITECTURE.md).

### What the product had to solve that the benchmark did not

A served query has no ground truth, no difficulty label, and usually no tests.
Three consequences, each documented where it bites:

1. **No correctness, only verification.** `RouteOutcome` has no `correct`
   field. It reports `verified` — a verifier's opinion — plus a
   `verified_meaning` string stating what was and was not measured.
2. **A difficulty label is not available, and leaning on one was worse than
   unfair.** The benchmark's `predictive` policy routed on MATH500's shipped
   `level`, which no user query carries. That was documented as an optimistic
   upper bound — until the level-5 filter made the field constant, at which
   point the policy stopped routing altogether. It was deleted on 8 August 2026.
   Both surviving predictive routers, `llm_router` and `routellm`, read only
   what a served query actually carries.
3. **The perfect verifier usually isn't available.** "Run the tests" is free
   and exact in the benchmark only because MBPP+ ships them. Self-consistency
   transfers unchanged; running tests does not. `sweep_degraded.py` is the
   experiment that prices that loss.

## Using it

```bash
llm-router --demo                          # real cached data, no key, $0
llm-router "What is 17 * 23?"              # needs ROUTER_MODE=real + a key
llm-router --estimate "prove X"            # price every policy, no calls
llm-router --findings                      # what the benchmark measured
llm-router "..." --approve-above 0.01      # pause for approval before spending
```

As an MCP server, so any MCP client can route through it:

```json
{
  "mcpServers": {
    "llm-routing": {
      "command": "llm-router-mcp",
      "env": { "ROUTER_LADDER": "wide", "ROUTER_MODE": "replay" }
    }
  }
}
```

It offers four tools (`route_query`, `estimate_cost`, `compare_policies`,
`explain_routing`), four resources under `routing://`, and a prompt that walks
a client through choosing a policy. `ROUTER_MODE=replay` is the safe thing to
register: it cannot spend money.

---

# The experiment

Two ways to spend less on LLM inference, with opposite failure modes:

- **Predictive routing.** Inspect the query, guess whether it is hard, commit to
  one model. Pays once. Misroutes silently, and never finds out.
- **Cascade routing.** Call the cheap model, verify the answer, escalate only on
  failure. Never misroutes an easy query. Double-pays on every escalation.

"Does routing work" is settled. The question here is **where the crossover is**,
and the variable this repo manipulates to find it is **verifier quality** — the
one thing a cascade depends on and a predictive router does not have at all.

> ### Status: every policy on this ladder now has real numbers
>
> **Updated 8 August 2026.** Every accuracy figure in this section was measured
> against real models. `cache/raw_calls.wide.jsonl` holds **1,095** real
> responses, 1,097 across all three ladder files, for **$4.40** total.
>
> It got there in four steps: the 30 July plumbing run ($0.05), the 6 August
> two-arm probe ($0.92), the 7 August redraw of the 21 decisive tasks ($2.96),
> and 339 calls recording the self-consistency samples and routing calls every
> cascade needs and no earlier run had reason to make ($0.05).
>
> That last $0.05 is what made the rest of this table possible. **97.3% of all
> spend went to the expensive rung**; the cheap rung's 739 calls cost $0.12
> in total.
>
> Held-out half, n=50, `wide` ladder (DeepSeek v4-flash → Opus 5):
>
> | policy | family | acc | cost/task | vs `always_expensive` |
> |---|---|---|---|---|
> | `always_cheap` | fixed | 78.0% | $0.000130 | −14 pts at 1.5% of cost |
> | `random_matched` | null | 86.0% | $0.008008 | −6 pts at 92% of cost |
> | `routellm` | **predictive** | 82.0% | $0.004232 | −10 pts at 49% of cost |
> | `llm_router` | **predictive** | 88.0% | $0.008115 | −4 pts at 93% of cost |
> | `cascade_routing` | **cascading** | 88.0% | $0.000915 | −4 pts at **10%** of cost |
> | **`cascade`** | **cascading** | **90.0%** | **$0.002035** | **−2 pts at 23% of cost** |
> | `oracle` | bound | 92.0% | $0.001053 | match at 12% of cost, **not deployable** |
> | `always_expensive` | fixed | 92.0% | $0.008719 | — |
>
> **The cascade is within one task of always-expensive at 23% of the price.**
> That is the headline this repository was built to test, and it survives contact
> with real models on the ladder where the price gap is wide.
>
> **Neither predictive router is cheap, and neither beats the null at its own
> spend.** `llm_router` lands at 93% of the top rung's cost with less accuracy,
> because on the maths half it routes 30 of 30 tasks expensive. `routellm` is
> cheaper but pays for it in accuracy, scoring 82.0% — below `random_matched`.
> Only the cascades are cheap, and the reason is structural: a cascade finds out
> by trying, so it pays the top rung only on the tasks that need it.
>
> A previous version of this table carried a `predictive` row and reported that
> *"every one-shot router costs about as much as always-expensive"* as a finding.
> That policy routed on a difficulty label the level-5 filter made constant, so
> on the maths half it was `always_expensive` by construction rather than by
> measurement. It has been deleted; see [§ The policies](#the-policies).
>
> `always_mid` is the only policy in the repository without real numbers, and it
> exists only on the three-rung `claude` ladder — so on this ladder the table is
> complete.
>
> The `oracle` bound check passes on all three rows — `all`, `math` and `code` —
> so no policy has gained an action the ceiling cannot reach.
>
> #### Routing skill, measured against a null at each policy's own spend
>
> Raw accuracy conflates skill with willingness to spend, so `run_eval` reports
> what fraction of the available headroom each policy captured — against the
> accuracy a *signal-free* policy would reach at that same budget, obtained by
> interpolating the `always_cheap` → `always_expensive` chord.
>
> | policy | cost/task | null | actual | oracle | skill |
> |---|---|---|---|---|---|
> | `routellm` | $0.004232 | 84.7% | 82.0% | 92.0% | **−36.7%** |
> | `llm_router` | $0.008115 | 91.0% | 88.0% | 92.0% | n/a |
> | **`cascade`** | $0.002035 | 81.1% | 90.0% | 92.0% | **+81.6%** |
> | `cascade_routing` | $0.000915 | 79.3% | 88.0% | 92.0% | **+68.6%** |
>
> **The two cascades capture 69–82% of the headroom available at their price. The
> two predictive routers capture none of it.** `llm_router` reads `n/a` rather
> than a number because it spends so close to `always_expensive` that the oracle
> sits less than one task above its null — at that budget there is essentially
> nothing left for any router to win, which is itself the finding.
>
> #### What survives a significance test, and what does not
>
> `stats.py`, exact McNemar and paired bootstrap on the held-out half:
>
> - **0 of 8 accuracy comparisons reach significance.** The cascade's 2-point
>   deficit against `always_expensive` is not distinguishable from zero at n=50.
> - **The cost differences are significant and large.** `cascade` against
>   `always_expensive` is **−$0.006684/task, 95% CI [−0.008563, −0.004954]** — an
>   interval nowhere near zero.
>
> So the repository's standing summary now holds on real data with the cascade in
> it: *cost differences are measurable at this n, accuracy differences are not.*
> Read as a decision, that is a good result rather than a null one — it says the
> saving is solid and the accuracy price is too small to detect.
>
> On the cost-quality frontier (`frontier.py`, whole curves rather than single
> points), `cascade` scores **AUC 91.2%, +4.9% over a cost-matched coin flip**,
> and owns every budget from $0.000915 upward. **`routellm` scores 85.4% — −0.9%
> against random, and it contributes no point to the frontier above the floor.**
> That is LLMRouterBench's finding, reproduced on real data with an actual
> published learned router. The repository previously made the same claim on the
> strength of a hand-written heuristic whose feature was constant; that version
> is retracted and this one replaces it.
>
> #### Per domain: the two halves win differently
>
> | | `always_cheap` | `cascade` | `always_expensive` | cascade escalated |
> |---|---|---|---|---|
> | code (20) | 65.0% · $0.000027 | **80.0% · $0.000844** | 80.0% · $0.002243 | 35.0% |
> | math (30) | 86.7% · $0.000198 | **96.7% · $0.002829** | 100% · $0.013036 | 6.7% |
>
> **On code the cascade equals always-expensive exactly, at 38% of the cost.** The
> verifier there is free and perfect, so escalation happens precisely when it
> should. This is the half the 7 August redraw showed carries the reproducible
> routing signal, and it behaves like it.
>
> **On maths, most of the gain is not escalation at all.** The cascade fixes 3 of
> `always_cheap`'s 4 maths failures — but **2 of those 3 come from majority
> voting, and only 1 from escalating**. It escalated on 2 tasks out of 30.
> `verify_math` bundles two mechanisms on purpose (agreement decides escalation,
> the plurality answer replaces the greedy one), and its docstring says so; this
> is the first time the split has been measured. Any reading of "the maths cascade
> works" has to attribute two thirds of it to self-consistency voting rather than
> to routing.
>
> That also settles [NOTES.md](docs/NOTES.md) issue 4 in the good direction: mock
> mode was expected to inflate majority voting badly, and the real gain is
> smaller but genuinely there.
>
> | | n=100 | code (40) | math (60) |
> |---|---|---|---|
> | `always_cheap` | 79.0% | 75.0% | 81.7% |
> | `always_expensive` | 92.0% | 85.0% | 96.7% |
> | **routable** (cheap wrong, expensive right) | **15.0%** | 12.5% | 16.7% |
> | `both_fail` (neither rung) | 6.0% | 12.5% | 1.7% |
> | ceiling over `always_cheap` | 17.0% | 15.0% | 18.3% |
>
> 95% CI on routable [9.3%, 23.3%], McNemar **p = 0.002**, rescue rate 71%. This
> is what the task set hardening bought: on 30 July the same cross-tab read
> `both_ok=10, routable=0`. Swapping the code half to **MBPP+** and the maths half
> to **MATH500 level 5** moved it to 15%, and because those changes hit disjoint
> task subsets, each is visible working on its own half.
>
> Two things in that table are worth more than the headline. **Opus solves 58 of
> 60 MATH500 level-5 problems**, so the maths half is near-saturated at the top
> rung and its routing signal is almost purely "the cheap model fails at something
> the expensive one finds easy" — 91% rescue. And **code is now the harder
> domain**: 5 of its 40 tasks defeat both rungs, against 1 of 60 on maths.
>
> Read it soberly: 15% sits exactly *on* the 15% floor `routable.py` asks for. The
> routing signal is real, significant, and thin — every policy here competes over
> a 17-point band.
>
> ### And 15.0% turned out to be a third noise — 7 August 2026
>
> That whole table rests on **one draw per task per rung**. A single draw cannot
> tell "the cheap model cannot do this" apart from "the cheap model usually can
> and missed once". `scripts/redraw_decisive.py` took **10 further draws** of both
> rungs on the 21 tasks that decide the number — the 15 `routable` and 6
> `both_fail` — for 420 calls and **$2.96**:
>
> | | observed (1 draw) | expected (10 draws) | reproducible |
> |---|---|---|---|
> | code (40) | 12.5% | **13.8%** | 12.5% |
> | math (60) | 16.7% | **7.8%** | 6.7% |
> | **total** | **15.0%** | **10.2%** | **9.0%** |
>
> *expected* is the routable fraction a fresh probe would find on average;
> *reproducible* is the part a router could actually capture, being the tasks
> where the cheap rung reliably fails and the expensive rung reliably succeeds.
>
> **The half that looked stronger was the fake one.** Of the 15 tasks counted
> routable, 6 are solid, 5 are flaky, and **4 are phantoms — the cheap model
> solves them 10 times out of 10** and simply missed on the probe's single draw.
> Every one of the 5 code tasks is solid. Every one of the 4 phantoms is maths,
> as are all 5 flaky ones; of 10 routable maths tasks exactly **one** survives
> intact. Maths carried the higher headline number and almost none of the signal.
>
> This is not a surprise so much as a confirmation.
> [arXiv:2607.03436](https://arxiv.org/abs/2607.03436) predicts precisely this and
> puts the single-draw noise share highest on MATH-500 — which is what the maths
> half is. The prediction was tested here on independent data and it held.
>
> The honest headline is therefore **routable ≈ 10%, of which ~9% is
> capturable**, and the code half is where the routing signal actually lives.
> `redraw.wide.json` holds the per-task probabilities; every response is committed,
> so the re-estimate replays for free.
>
> Two truncations surfaced during the redraw (`math-422` cheap, `math-154`
> expensive), which grade as wrong regardless of content — the evaluation-artifact
> class from [arXiv:2605.07395](https://arxiv.org/abs/2605.07395). `math-154` is
> counted `both_fail` and at least partly should not be.
>
> **These numbers are post-fix.** As first run the probe read routable 13.0%,
> cheap 77.0%, expensive 87.0%: seven of its twenty wrong maths answers were
> correct answers the normaliser could not match (`1+\sqrt{19}, 1-\sqrt{19}`
> against a ground truth of `1 \pm \sqrt{19}`, and five more classes like it).
> `sanity_check.py` printed 60/60 throughout, because feeding the ground truth
> back into `\boxed{}` only ever tests `grade(GT, GT)`. It now checks equivalent
> formattings and near-miss wrong answers too.
>
> Every *other* accuracy figure below still comes from **mock mode**, which
> fabricates model replies from answers already stored in the task set. Those
> figures mean nothing about any model. Every simulated number is labelled as such:
> at the top and bottom of every run, above every table, and in a
> `"simulated": true` field on every output row. See [NOTES.md](docs/NOTES.md) for what
> a full real run would change.
>
> **Coming back to this cold? Read [STATUS.md](STATUS.md).** It says what state the
> project is in, what to run next in order, what the paid run costs, and what to
> expect to change.
>
> **Want to understand the code?** [WALKTHROUGH.md](docs/WALKTHROUGH.md) traces one
> real task through every file. For the plain-language version of the *ideas*,
> read [EXPLAINED.md](docs/EXPLAINED.md).

## The degradation sweep, on real data

`sweep_degraded.py` is the one thing here that is not a replication: it degrades
a *real* verifier by a controlled amount, holding the domain, the models, the
prompts and the grader fixed. As of 7 August 2026 it runs on **real model
responses**, on the code half, for $0 — the code verifier runs the tests and
makes no model calls, so every point in the sweep replays from the greedy
answers the probe already paid for.

40 code tasks, `wide` ladder, mean of 200 corruption draws per level. `p` is the
probability the verifier ignores the test result and guesses.

| corrupt `p` | eff. AUC | accuracy | cost/task | escalation |
|---|---|---|---|---|
| 0.00 | 1.000 | **87.5%** | **$0.000682** | 25.0% |
| 0.10 | 0.950 | 86.8% | $0.000743 | 27.8% |
| 0.25 | 0.875 | 85.6% | $0.000812 | 31.0% |
| 0.50 | 0.750 | 83.7% | $0.000964 | 38.5% |
| 0.75 | 0.625 | 81.8% | $0.001075 | 43.2% |
| 1.00 | 0.500 | 79.8% | $0.001236 | 50.1% |
| | | | | |
| `always_cheap` | — | 75.0% | $0.000027 | — |
| `always_expensive` | — | 85.0% | $0.002394 | — |

**With a perfect verifier the cascade matches always-expensive at 28% of the
cost.** 87.5% against 85.0% is one task at n=40, so read it as "matches", not as
"beats" — but it matches while spending $0.000682 against $0.002394, and that
gap is not one task wide.

**It degrades gracefully, which is the actual finding.** Accuracy falls
monotonically and cost rises monotonically across all six levels, and the
cascade holds always-expensive's accuracy up to **p = 0.50** — a verifier that
ignores the evidence half the time — where it is still 60% cheaper. Even at
p = 1.00, a pure coin flip, it beats `always_cheap` by 4.8 points at half the
escalation rate of a router with no signal at all.

That last row is worth stating plainly, because it is the trap: **a cascade with
a worthless verifier still looks good against always-expensive on $/correct.**
At the **68.2x** ratio this ladder actually billed, a coin flip sends half the
traffic to the cheap rung and saves money doing it. Cost-per-correct therefore cannot distinguish a
good verifier from no verifier, and only the accuracy-matched comparison can.
`sweep_degraded.py` prints both and says which is which.

The caveats, in order of size: n=40, one ladder, and the code half is the half
whose free perfect verifier **does not transfer to production** — MBPP+ ships
the tests, and a user's query does not. That last point is the sharpest open
problem in this repository; see [NOTES.md](docs/NOTES.md) issue 18.

## Run it

Mock mode needs nothing installed. It is pure standard library, offline, and
byte-deterministic — including the figures.

```bash
python3 build_taskset.py     # builds taskset.jsonl from data/
python3 sanity_check.py      # regression gate: must print 40/40 and 60/60
python3 routable.py          # is there anything left for a router to decide?
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

Real mode. Do these in order — the second one is the decision point, and it costs
one or two orders of magnitude less than a full run:

```bash
pip install -e ".[real]"
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

# 2. the two-arm probe: is there anything for a router to decide? ~$0.92
ROUTER_MODE=real python3 run_eval.py \
    --policy always_cheap --policy always_expensive --split all
ROUTER_MODE=replay python3 routable.py --real --ladders wide
```

**Run both arms, not one.** The single-arm version measures `P(cheap fails)`,
which is `routable + both_fail` — it cannot tell a task the expensive rung would
fix from one it would fail too, and only the first kind is worth routing. This
repo has a concrete demonstration of the difference: the 6 August probe reports a
21.0% cheap failure rate that reads as comfortably in band, while the routable
fraction is 15.0% and sits on its floor. Six of those 21 failures are tasks
neither rung can solve — and on the code half it is 5 of 10, so half the code
cascade's escalations buy nothing. See
[ROUTABLE_2026-07-30.md](docs/ROUTABLE_2026-07-30.md).

Ten tasks cannot answer this either — at n=10 the rate carries a ±28-point
confidence interval, which spans the entire acceptable band.

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
| math | MATH500, level 5 | 60 | exact match on the normalised answer | **proxy** — self-consistency over k samples |
| code | MBPP+ | 40 | execute the expanded evalplus suite | **free and perfect** — just run the tests |

Both difficulty settings were raised on 6 August 2026, because the probe showed
the cheap rung solving essentially everything at the previous ones. MBPP+ is the
same 378 problems as sanitized MBPP with roughly 35x more test cases, so the swap
moves exactly one variable — how thorough the marking is — and the model is still
shown the original thin asserts as its specification. The easier originals remain
one flag away: `--code mbpp --min-math-level 3`. See [DATASETS.md](docs/DATASETS.md).
Grading the code half now needs **numpy**, because the expanded suites compare
floats with `np.allclose`.

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

| `ROUTER_LADDER` | rungs | list ratio | effective ratio | realized ratio |
|---|---|---|---|---|
| `claude` (default) | Haiku 4.5 → Sonnet 5 → Opus 5 | 1x / 3x / 5x | 1x / 3.9x / 6.5x | never run |
| `deepseek` | v4-flash → v4-pro | 1x / 3.1x | 1x / 3.1x | never run |
| `wide` | DeepSeek v4-flash → Opus 5 | 1x / 36x | 1x / 46x | **1x / 68.2x** |

Three ratios, because they answer three different questions and only the last one
is a measurement. **List** and **effective** are arithmetic over *input* prices,
so they cannot be wrong about the price table — but roughly 93% of the bill on
this workload is *output* tokens, where the `wide` rungs are 89x apart rather
than 36x. **Realized** is `findings.realized_ratio`: the ratio the provider
actually billed, from the cached greedy answers, over the 112 tasks where both
rungs answered. It is *higher* than the quoted figure, and the direction matters
— understating the price gap understates the case for cascading.

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

| ladder | ratio | 30 July set (mock) | current set (mock) | **current set (real)** |
|---|---|---|---|---|
| `deepseek` | 3.11x | **+33%** (costs more) | **+10%** (costs more) | not run |
| `claude` | 6.5x effective | −12% | −46% | not run |
| `wide` | 46.4x effective | −74% | −66% | **−60.9%** |

The `wide` row is now measured. `frontier.py` on real responses puts the
cascade **60.9% below always-expensive at matched accuracy**, against a mock
estimate of −66% and a 30 July estimate of −74%. The mock overstated the saving
by about five points; the sign and the order of magnitude held.

That is the third time these numbers have moved and the third time no sign
flipped, which is why the router derives them rather than quoting them —
`router_agent/findings.py` reads `frontier.jsonl` when a run for the ladder
exists and falls back to the 30 July constants *labelled with their date* when
it does not. On `wide` it now reports `economics_source: "frontier.jsonl"` and
`economics_simulated: false`.

The two mock columns remain because the other two ladders have never been run
for real, and they are labelled at every point they appear.

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

Ten policies in two families, plus the fixed rungs and two baselines. The
families are the comparison this repository exists to make.

| policy | family | what it does |
|---|---|---|
| `llm_router` | **predictive** | the cheap model classifies its own difficulty, then answers |
| `routellm` | **predictive** | RouteLLM's pretrained `bert` router at a fixed score threshold |
| `cascade` | **cascading** | answer → verify → escalate, over every rung of the ladder |
| `cascade_routing` | **cascading** | **routing and cascading unified** — see below |
| `cascade_degraded` | **cascading** | the cascade with a deliberately damaged verifier — **the experiment** |
| `always_<rung>` | fixed | one per rung of the loaded ladder, generated not listed |
| `random_matched` | null | coin flip at `llm_router`'s own escalation rate — **the null hypothesis** |
| `oracle` | bound | hindsight-optimal. Bounds how good any router could be |

A `predictive` policy — a hand-written heuristic routing on MATH500's shipped
difficulty level — was **deleted on 8 August 2026**. Under `MIN_MATH_LEVEL = 5`
that level is constant, so the policy sent all 60 maths tasks to the expensive
rung and was `always_expensive` on 60% of the set while being reported as a
router. It scored 86.0% against the coin flip's 88.0%. Predictive routing is
half of this repository's subject and has not gone anywhere; it is now measured
with the two implementations above, neither of which reads a difficulty label.
The reasoning is preserved as a tombstone at `DECISION #4` in `policies.py`.

Four of these exist for reasons worth stating outright:

- **`random_matched` is the null hypothesis.** A router that escalates the same
  fraction of tasks *at random* also gains accuracy — it just pays for it.
  Without this baseline, a gap between a router and `always_cheap` shows only
  that spending more helps. This is not a pedantic point: LLMRouterBench
  ([arXiv:2601.07206](https://arxiv.org/abs/2601.07206)) finds that under unified
  evaluation many published routers, commercial ones included, fail to reliably
  beat a simple baseline — which is what `routellm` does here, scoring 82.0%
  against this baseline's 86.0%.
  One anchored null cannot serve policies at different spending levels, so
  `run_eval` also computes an analytic null at *each* policy's own cost. This
  row is the empirical check that the analytic one is not a fiction.
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
the task set size and it is written up as item 2 of [NOTES.md](docs/NOTES.md) rather
than quietly omitted.

## Files

```
.                       README.md  STATUS.md  ARCHITECTURE.md  LICENSE
├── *.py                the experiment: 15 flat, independently runnable scripts
├── router_agent/       the product: LangGraph cascade + MCP server
├── scripts/            demo and CI guard scripts
├── tests/              pytest, agent layer only
├── docs/               reference and dated analyses  (docs/README.md indexes them)
├── data/               MATH500 and MBPP+ source data
├── cache/              1,097 real model responses - what makes replay free
└── archive/            real data superseded by the 6 Aug rebuild, kept not deleted
```

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
| `routellm_router.py` | RouteLLM's pretrained router at a fixed score threshold |
| `plot.py` | SVG figures from the standard library, no matplotlib |
| `sanity_check.py` | regression gate: reference answers, equivalent formattings, and near-miss wrong answers. Exits non-zero if a grader is broken *or* too lax |

And the product layer, which depends on the above but is never depended on by
it — see [ARCHITECTURE.md](ARCHITECTURE.md):

| file | what it is |
|---|---|
| `router_agent/live.py` | query → task dict; where the missing difficulty label is documented |
| `router_agent/verifiers.py` | verification without ground truth |
| `router_agent/pricing.py` | cost projection, calibrated on the probe's measured token counts |
| `router_agent/findings.py` | the benchmark's results, recomputed from committed data rather than transcribed |
| `router_agent/state.py` | graph state; reducers that keep cost accounting correct across the cascade loop |
| `router_agent/nodes.py` | one function per node, testable without LangGraph |
| `router_agent/graph.py` | the cyclic graph: answer → verify → escalate → answer |
| `router_agent/engine.py` | the façade the CLI and MCP server share |
| `router_agent/cli.py` | `llm-router` |
| `router_agent/mcp_server.py` | four tools, four resources, one prompt |
| `scripts/check_core_unchanged.py` | proves the agent layer's one edit to `models.py` cannot move a benchmark number |
| `scripts/record_missing.py` | finds the calls a full replay needs that no paid run recorded, prices them, and buys them. Plan-only until `--go` |
| `tests/` | pytest, agent layer only — the research core keeps `sanity_check.py` |

## The tuneable decisions

Each is marked `DECISION #n` in the source next to the code it controls. Change
one, re-run, and the report shows what moved.

1. **Model ladder** (`models.py`) — the price ratio drives the whole economics
2. **Self-consistency k** (`policies.py`) — failure detection against cost, linear
3. **Agreement threshold** (`policies.py`) — when to accept the cheap answer
4. ~~**Predictive heuristic**~~ — **retracted 8 August 2026**, tombstoned in place
   rather than renumbered. The feature was constant; see `policies.py`
5. **Verifier corruption rate** (`policies.py`) — the manipulated variable
6. **Random baseline rate** (`policies.py`) — the null, anchored to `llm_router`
7. **LLM-as-router** (`policies.py`) — the option decision 4 rejected, now measured
8. **RouteLLM variant** (`routellm_router.py`) — which learned router, and why `bert`
   — and **8b**, its operating threshold, fixed at 0.80 rather than calibrated
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

`routellm` sits out unless `cache/routellm_scores.jsonl` exists. **It ran for the
first time on 8 August 2026**, after the scores were regenerated for the rebuilt
task set — the 6 August rebuild had left only 10 of 100 matching, and
`routellm_router.CALIBRATED` correctly refused to run a router calibrated on a
tenth of the set. The scores are committed, so the policy now replays for anyone
with no key, no torch and no GPU.

Regenerating is free, local, and needs no API key — but it does need torch:

```bash
pip install routellm==0.2.0
python routellm_router.py --score     # bert variant, local, no API key
python routellm_router.py             # show the threshold and routing decisions
```

**The score distribution is the first result, and it arrives before any accuracy
is measured.** Across all 100 tasks the router's `strong_win_rate` spans
**[0.509, 0.898]** with a median of 0.783. It never once judges the weak model
more likely to win. So the semantically natural threshold — 0.5, "escalate when
the strong model is favoured" — routes **everything** to the expensive rung, and
the policy degenerates into `always_expensive`: exactly the failure the
`predictive` policy was deleted for. The threshold used instead is a declared
constant, **0.80**, which splits the set 42/58 and is derived from nothing else
in this repository. See `DECISION #8b` in `routellm_router.py`.

That compression is what "out of distribution" looks like in practice.
`bert_gpt4_augmented` was trained to predict which answer a human would *prefer*
between two chat models; it is being asked about competition maths and MBPP+,
where it cannot judge, and it defaults to "the big one" every time. Preference is
not correctness. The measured consequence: **AUC 85.4%, −0.9% against a
cost-matched coin flip, and no point on the combined frontier above the floor.**

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
- Agreement-Based Cascading, [arXiv:2407.02348](https://arxiv.org/abs/2407.02348) — **states this repo's crossover as a threshold**
- Routing-gap decomposition, [arXiv:2607.03436](https://arxiv.org/abs/2607.03436) — **and qualifies its headline**

Two of those deserve more than a line, because they bear directly on what this
repository claims.

**The crossover has been published, from a different direction.**
*Agreement-Based Cascading* (TMLR 07/2025) uses ensemble agreement as its
deferral signal — a generalisation of the `self_consistency` verifier here — and
states the same crossover as a cost ratio: at γ ≥ 1/5, i.e. a price ratio of 5x
or less, sequential cascading yields minimal savings and needs parallel execution
to pay at all. It also names the worst case this repo's `cascade` row shows on
the `deepseek` ladder: when nearly everything escalates, a k-model cascade can
cost (k+1)x the expensive model alone. Their 5x and this repo's ~3x are not in
conflict — their γ is a raw token-price ratio, while the `effective_ratio` used
here already absorbs verification cost and the measured escalation rate.
Independent corroboration, and the more useful kind: arrived at by a different
route.

**The routable fraction is a single-draw estimate, and that is now a known
bias.** *How Much of the Routing Gap Is Real?* (July 2026) decomposes exactly
this measurement into reproducible specialist advantage and single-draw label
noise, and puts the noise share at **36% on MATH-500** — rising to 43% on
queries only a few models get right. The maths half of this task set *is*
MATH-500 level 5, 60 of the 100 tasks, and `results.probe.jsonl` holds exactly
one draw per (task, tier). So some share of the 15 routable tasks are
coin-flips rather than capability differences. See
[NOTES.md](docs/NOTES.md) for the experiment that would settle it, and what it
costs.

"Isn't this just FrugalGPT?" — largely yes, and deliberately: it is a replication
with 2026 models. What is added is the manipulation. FrugalGPT and AutoMix take
their verifier as given; Dekoninck et al. identify quality-estimator accuracy as
the factor that decides whether any of this works, and test it by injecting
synthetic Gaussian noise into a quality signal. This repo instead **degrades a real
verifier by a controlled amount on objectively-graded tasks**, holding the domain,
the models, the prompts and the grader fixed. That is the one thing here that is
not a replication.

## Open issues

Tracked honestly in [NOTES.md](docs/NOTES.md), including the ones that would weaken
the headline. The two largest:

- **One of the ten policies still has no real accuracy data.** `always_mid` exists
  only on the three-rung `claude` ladder, which has never been run for real. Any
  policy a replay cannot serve is dropped by name with the reason printed. That
  list was six policies long until `scripts/record_missing.py` bought the missing
  self-consistency samples for $0.05, and two until the RouteLLM scores were
  regenerated; on the `wide` ladder it is now empty.
- **The routing signal and the good verifier are in the same half, and only one
  of them transfers.** The 7 August redraw put nearly all the reproducible
  routing signal in the code half; the code half's verifier is free and perfect
  only because MBPP+ ships tests. A deployed router has neither the tests nor,
  on this evidence, much signal in the half where it can still verify. Issue 18.
