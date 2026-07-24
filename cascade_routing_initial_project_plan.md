# Cascade Routing: When Is a Cheap Model Good Enough?

**A measurement project on LLM cost-quality routing**

Owner: Asen Pantov
Version: 1.0, 28 July 2026
Status: concept locked, not built
Budget: 18–22 hours

> This document is self-contained. It assumes no prior conversation and can be handed to a fresh working context. Everything needed to start is in sections 4 through 7.

---

## 1. The problem, in one paragraph

Every company running LLMs in production sends far too many queries to models that are far too expensive for them. A question like "what is 2+2" and a question requiring eight steps of chained arithmetic cost the same to serve if you route both to the biggest model, but only one of them needs it. The obvious fix is routing: decide, per query, which model to use. The non-obvious part is that there are two fundamentally different ways to do this, they have opposite failure modes, and there is no published answer to which one wins under what conditions.

This project answers that question empirically, on a task set where every answer can be graded objectively.

---

## 2. The two routing strategies

### Predictive routing (route once, up front)

Look at the query. Predict whether it is hard. Send it to one model. Done.

- **Cost:** one classification, then one model call.
- **Failure mode:** misrouting. You send a hard query to the cheap model and get a wrong answer, or you send an easy query to the expensive model and overpay. You never find out which.
- This is what most published routers do (the RouteLLM family, and most vendor "smart routing" features).

### Cascade routing (try cheap, verify, escalate)

Send the query to the cheap model. Check whether the answer is good. If not, escalate to the expensive model.

- **Cost:** cheap call, plus verification, plus (on escalation) the expensive call as well.
- **Failure mode:** double payment. Every escalated query costs cheap + verify + expensive, which is *more* than routing to the expensive model directly would have.
- **Advantage:** it can never misroute an easy query. If the cheap model got it right, you stop and you paid almost nothing.

### The question

**Where is the crossover?** Intuition says cascade wins when the cheap model succeeds often, because you rarely double-pay. Predictive wins when the cheap model succeeds rarely, because then cascade double-pays on most queries. The crossover point depends on three things you can measure: the cheap model's success rate, the price ratio between the models, and the accuracy of the verifier.

That last one is where this project earns its keep.

---

## 3. The centrepiece: verifier quality is the hidden variable

In cascade routing you must decide "is this answer good enough?" **without knowing the correct answer.** That is the entire difficulty, and it is glossed over in almost every treatment of the topic.

The task set is deliberately built from two domains with opposite verification properties:

| Domain | Source | Verifier available at runtime | Quality |
|---|---|---|---|
| **Code** | MBPP | Execute the assert statements shipped with each task | Free and near-perfect |
| **Math** | GSM8K | No ground truth available; must use a proxy | Imperfect, and the proxy design is a research decision |

This gives you a controlled comparison that most projects cannot construct: **the same cascade architecture, run under a perfect verifier and under a proxy verifier.**

And it enables the experiment that makes this project genuinely uncommon:

> **Deliberately degrade the perfect verifier and measure what happens.**
>
> On the code domain, instead of running all three asserts, run only two. Then one. Then replace it with a proxy verifier. At each level, measure verifier precision and recall, and plot cascade accuracy and cost against them.
>
> The output is a curve: **how good does your verifier have to be before cascade routing stops being worth it?**

Nobody asking you about this in an interview will have seen that number. It is the difference between "I built a router" and "I found the condition under which routers work."

### Proxy verifier options for the math domain

These are yours to choose between, and the choice is a load-bearing design decision:

1. **Self-reported confidence.** Ask the cheap model whether it is confident. Cheap and obvious. Known to be poorly calibrated. Worth including as a baseline precisely *because* it is the naive approach — showing it fails is a result.
2. **Self-consistency.** Sample the cheap model k times at non-zero temperature and check whether the answers agree. Strong signal for math. Costs k cheap calls, which may still be far below one expensive call. Recommended as the primary.
3. **Separate verifier model.** A second cheap call that critiques the first answer. Different failure mode from the generator, which is the argument for it.
4. **Logprob-based uncertainty.** Clean in theory, awkward through most APIs. Skip unless it falls out easily.

---

## 4. Task set (already built — spec below)

**100 tasks: 60 math, 40 code.** Stratified across difficulty, not sampled uniformly.

Stratification matters. A router evaluated only on mid-difficulty tasks tells you nothing: you need genuinely easy items where escalating is pure waste, and genuinely hard items where staying cheap is a failure. The build script buckets each source by a difficulty proxy and samples across all four buckets.

**Sources, both fetchable from GitHub raw (no Hugging Face needed):**

- GSM8K test split: `raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl` — 1,319 problems. Final answer follows `#### `. Difficulty proxy: count of `<<...>>` computation annotations (observed range 1–8).
- MBPP: `raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl` — 974 problems, each with a `test_list` of assert statements. Difficulty proxy: line count of the reference solution (observed range 2–18).

**Unified task schema:**

```json
{
  "id": "math-42",
  "domain": "math",
  "prompt": "...",
  "grader": "exact_match_int",
  "grader_payload": {"answer": "18"},
  "difficulty_proxy": 3
}
```

Graders are `exact_match_int` (last number in the response, permissive on format, strict on position) and `run_asserts` (subprocess with a 10-second timeout so generated infinite loops cannot hang the run).

**No LLM-as-judge anywhere in this project.** Every task is graded deterministically. That removes the judge calibration problem entirely, makes runs reproducible, and is the reason these two domains were chosen over anything more interesting-sounding.

---

## 5. Policies to compare

Six, all cheap to implement, and the comparison is the deliverable.

| Policy | Description | Role in the comparison |
|---|---|---|
| `always_cheap` | Every query to the small model | Cost floor, accuracy floor |
| `always_expensive` | Every query to the large model | Accuracy ceiling, cost ceiling |
| `predictive` | Classify difficulty, route once | The published-state-of-the-art comparator |
| `cascade` | Cheap → verify → escalate on failure | The one under test |
| `cascade_degraded` | Same, with a deliberately weakened verifier | Produces the verifier-quality curve |
| `oracle` | Hindsight-optimal choice per query | Ceiling on what *any* router could achieve |

The oracle matters more than it looks. It tells you how much headroom routing has at all. If the oracle only beats `always_cheap` by three points, the entire routing question is uninteresting on this task set and you should say so.

---

## 6. Metrics

**Per policy:**

- Task accuracy (fraction graded correct)
- Mean cost per task, priced from actual token counts
- Mean and p95 latency per task
- Escalation rate (cascade family)
- Misroute rate (predictive): routed cheap and failed, or routed expensive when cheap would have sufficed

**Verifier diagnostics (cascade family), and these are the interesting ones:**

- Verifier precision: of the answers it flagged as bad, how many were actually bad
- Verifier recall: of the actually-bad answers, how many it caught
- **False-escalation cost:** money spent escalating queries the cheap model had already answered correctly
- **Missed-escalation accuracy loss:** points of accuracy lost to bad answers the verifier let through

**Headline artifact: a cost-quality Pareto plot.** Accuracy on the y-axis, mean cost per task on the x-axis, one point per policy. The question the plot answers in one glance: which policies sit on the frontier, and which are strictly dominated?

**Second artifact: the verifier-quality curve.** Verifier accuracy on the x-axis, cascade net benefit on the y-axis, with the break-even point marked.

---

## 7. Build plan

Total 18–22 hours. Ordered so that nothing depends on a component that has not been verified.

### Block 0 — Task set and graders (3h) *(largely complete)*
Download both sources, build the stratified 100-task set, implement both graders.

**Exit condition, and do not skip this:** grade the *reference* solutions from both datasets. A grader that does not score known-good answers at or near 100% is broken, and you will otherwise discover this after spending money on model calls.

### Block 1 — Model client and cost accounting (3h)
A thin wrapper over the provider API with two configured models. Every call returns the response plus input tokens, output tokens, latency, and computed cost. Include a **mock mode** that returns canned responses so the whole harness is testable end to end before spending anything.

**Exit condition:** the full pipeline runs in mock mode and produces a results file.

### Block 2 — Policies (4h)
Implement all six. The cascade is a loop with a hard escalation cap; the predictive router is a single classification call with a documented prompt. Keep policies behind one interface so the runner does not care which is executing.

**Exit condition:** each policy runs over 10 real tasks and produces sane traces.

### Block 3 — Pilot run and calibration (2h)
Run 20 tasks against the real models. Check the cheap model's success rate.

**This is a decision gate.** If the cheap model solves 95% or 15% of the task set, the interesting region has collapsed and routing has nothing to decide. Adjust the model pair or the difficulty stratification before running the full set. Discovering this after the full run is an avoidable waste.

### Block 4 — Full eval (4h)
All policies over all 100 tasks, with checkpointing so a crash at task 80 does not cost the run. Results to a per-run CSV plus a summary JSON.

### Block 5 — Verifier degradation experiment (3h)
The centrepiece. On the code domain, run cascade at four verifier strengths: all asserts, two, one, proxy-only. Compute precision and recall at each level. Produce the verifier-quality curve.

### Block 6 — Framework port and packaging (4h)
Port the cascade loop to LangGraph (it is a genuine state machine: generate → verify → escalate → done, with a cap). Wrap the router as an MCP server. README with problem, design, results, limitations. Regression subset wired into GitHub Actions.

The port order is deliberate: hand-roll first so you understand the loop, then express it in the framework. In interview that produces a better answer than either alone — you can say what the framework bought you and what it hid.

---

## 8. Stack

- **Python 3.12.** Provider SDK for model calls.
- **Models:** one cheap, one expensive, from the same provider so the comparison is not confounded by provider differences. A large price ratio makes the economics legible. Cross-provider routing is a noted variation, not the main experiment.
- **LangChain** as substrate, **LangGraph** for the cascade state machine. Both are the most-hired agent frameworks in the current market and the pairing is the most common in listings, which is the reason for choosing them over alternatives that might be marginally cleaner.
- **MCP** wrapper. It is the one framework-agnostic companion skill in the current listing data and it costs about four hours on top of finished work.
- **Tracing:** Langfuse or Arize Phoenix, self-hosted, free. Wire it in from the first commit rather than retrofitting.
- **matplotlib** for the two plots. No dashboard, no UI.

---

## 9. Cost estimate

Roughly 100 tasks × 6 policies, with the cascade and self-consistency policies making multiple calls each: on the order of 1,500–2,500 model calls, most of them to the cheap model. Short prompts and short completions throughout.

Budget €20 including reruns and the pilot. If you are above that, something is looping or self-consistency k is set too high. Set a hard spend cap in the runner and have it abort rather than trusting yourself to watch it.

---

## 10. Decisions you must own cold

These are what an interviewer probes. If you cannot answer one of these from memory, the project has not done its job regardless of code quality.

**1. How does the verifier know an answer is wrong without knowing the right answer?**
It does not, and that is the point. It estimates. Self-consistency works because independent samples from the same model tend to agree when the model has a stable internal answer and diverge when it is guessing. Be ready to explain why self-reported confidence is worse: models are trained to sound helpful, not to be calibrated, and confidence expressed in text carries almost no information about correctness.

**2. What is your escalation threshold and why that value?**
Yours to set and defend. Whatever you choose, be ready for "what happens just below it." The honest answer is that any hard threshold has an arbitrary boundary, that you chose a hard one for auditability and reproducibility over a fuzzy one, and that in production it would be a governed parameter with a review cycle rather than a constant.

**3. When does cascade lose to predictive?**
When the cheap model's success rate is low, because then you double-pay on most queries. Give the number your own run produced, not a general argument.

**4. Why no LLM judge?**
Because both domains grade objectively, and introducing a judge would add a calibration problem — needing hundreds of human-labelled cases to trust the judge — for zero gain. Choosing task domains to avoid needing a judge is a design decision, not a limitation.

**5. Why is the oracle in the comparison?**
It bounds the headroom. Without it you cannot tell whether a small gap between policies means your router is bad or means routing cannot help much on this task set.

**6. What does this cost a real company, and what would it save?**
Have an order-of-magnitude answer ready. This is the question that turns a portfolio project into a business conversation, and it is the one most candidates cannot answer.

---

## 11. Why this is worth your time

**It is a real production problem, not a toy.** Inference cost is a live line item at every company running LLMs, and it is one of the few AI engineering problems with a directly measurable financial payoff.

**The cascade is a genuine control loop.** The escalation decision depends on what came back at runtime and cannot be hard-coded. That is a defensible answer to "is this actually agentic," which a plain classifier-router is not.

**Objective evaluation throughout.** Eval maturity is the most consistently cited separator between AI engineering candidates, and most portfolio projects have none. Yours has a Pareto frontier and a verifier-quality curve.

**It speaks the language of the target listings.** Cost-per-request and latency tracking appear verbatim in senior AI engineering listings at Dutch fintechs. Evaluation frameworks, tracing, and observability appear across the board.

**The verifier-degradation experiment is genuinely uncommon.** Router projects exist. Router projects that quantify how good the verifier has to be do not.

**Framework keywords come without compromising defensibility**, because the hand-roll-then-port sequence means you own the loop before the framework abstracts it.

---

## 12. Scope killers

Watch for these by name.

- **Adding models.** Two models, one comparison. A third turns the experiment into a matrix and doubles the runtime for no additional finding.
- **Adding domains.** Two domains were chosen for a specific structural reason (verifier asymmetry). A third adds cost without adding contrast.
- **Tuning prompts to improve accuracy.** You are measuring routing, not prompt engineering. Fix reasonable prompts early and leave them alone; changing them mid-experiment invalidates the comparison.
- **Building a UI.** Two matplotlib figures in a README is correct.
- **Chasing the perfect verifier.** The imperfection *is* the experiment.

---

## 13. Risks

**The interesting region collapses.** If the cheap model solves nearly all or nearly none of the tasks, routing has nothing to decide. Mitigated by the Block 3 pilot gate. Do not skip it.

**The result is "predictive wins everywhere."** Fine. That is a finding, it is publishable in a README, and it makes a better interview answer than a manufactured win. The project was framed as a measurement for exactly this reason.

**Cost overrun.** Mitigated by a hard spend cap enforced in code.

**Scope inflation into a general routing library.** The deliverable is an experiment with a result, not a product. If you find yourself designing a plugin architecture, stop.

---

## 14. How it appears on the CV

Calibrated to what a weekend of work actually supports. Fill in the real numbers once you have them.

> **Cascade routing for LLM cost optimisation — evaluation study**
> Built a cascade router (cheap model → verifier → escalation) and benchmarked it against predictive routing, static baselines, and a hindsight oracle on 100 objectively-graded math and code tasks. Measured accuracy, cost per task, latency, and verifier precision/recall; quantified how cascade benefit degrades as verifier accuracy falls. Implemented as a hand-rolled loop, ported to LangGraph, exposed via MCP.

Note what that does: it names a measured comparison and a quantified relationship. It does not claim the router won.

**Skills this legitimately unlocks:** agent control loops, LLM evaluation and benchmarking, cost and latency instrumentation, LangGraph, MCP, tracing and observability.

**It does not unlock:** production deployment, scale, multi-agent orchestration, cloud infrastructure. Do not add those.

---

## 15. Open decisions before Block 1

1. **Model pair.** Which cheap and which expensive, from one provider. The price ratio drives the whole economics.
2. **Primary math verifier.** Self-consistency recommended; if so, choose k.
3. **Escalation threshold**, with the reasoning you would give in an interview.
4. **Tracing tool:** Langfuse or Phoenix.
5. **Hard spend cap** in euros, enforced in the runner.

---

## Appendix — assets already built

- `build_taskset.py` — downloads both sources, stratified sampling, unified schema. Verified: 100 tasks, math difficulty 1–8, code difficulty 2–18.
- `graders.py` — `exact_match_int` and `run_asserts` (subprocess, 10s timeout).
- `data/gsm8k_test.jsonl` (1,319 rows), `data/mbpp.jsonl` (974 rows).

Both scripts are standalone and depend only on the standard library.
