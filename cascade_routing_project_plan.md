# Cascade vs Predictive Routing — Build Plan

**Two weekends. 20 hours. One repo, one result, one talking point.**

Version 2.0, 29 July 2026. Replaces all earlier plans.

---

## What it is

Two ways to save money on LLM inference, with opposite failure modes:

- **Predictive routing.** Look at the query, guess if it's hard, send it to one model. Pays once. Sometimes guesses wrong and you never find out.
- **Cascade routing.** Send it to the cheap model, check the answer, escalate only if it failed. Never misroutes an easy query. Double-pays every time it escalates.

You build both, plus three reference points, and measure which wins where.

**The question:** *when is each architecture the right choice?* Not "does routing work" — that's settled. The trade-off between these two is what production teams actually argue about.

---

## Five policies

| Policy | What it does | Why it's in |
|---|---|---|
| `always_cheap` | Small model, every query | Cost floor, accuracy floor |
| `always_expensive` | Big model, every query | Accuracy ceiling, cost ceiling |
| `predictive` | Heuristic difficulty classifier, route once | The comparator |
| `cascade` | Cheap → verify → escalate on failure | The other comparator |
| `oracle` | Best model per query, chosen with hindsight | Bounds how good *any* router could be |

Stretch goal if weekend two runs short: `cascade_degraded`, same cascade with a deliberately weakened verifier, to show how sensitive the whole thing is to verifier quality.

**Measure per policy:** accuracy, cost per task from real token counts, latency, escalation rate (cascade), misroute rate (predictive).

**One headline plot:** accuracy on y, cost on x, one point per policy. Which sit on the frontier, which are dominated.

---

## Task set

Roughly 100 tasks, split between code and math.

**Code from MBPP.** Each task ships assert statements. Run them. The verifier is free and perfect.

**Math from a set the cheap model actually struggles with.** Not GSM8K — modern cheap models score in the low 90s there and your cascade will have nothing to route. Use MATH500 or similar competition-level problems.

**The rule that matters more than the benchmark name: you want the cheap model failing 30–40% of the time.** Too easy and every policy converges. Too hard and the cascade escalates everything.

**Pilot gate, do not skip.** Run 10 tasks against both models before anything else. Check the failure rate. If the cheap model aces them, go harder. Discovering this after the full run wastes the weekend.

---

## The verifier contrast

This is the part that's interesting to talk about, and it costs almost nothing.

**Code self-grades.** Run the tests. You know with certainty whether the answer was right, at runtime, for free.

**Math doesn't.** No ground truth available when you need it. So you sample the cheap model a few times and check whether the answers agree. Agreement is a *guess* at correctness.

Same cascade architecture, two verification regimes. The math cascade should perform noticeably worse, and being able to explain why is the best 90 seconds in your interview: a cascade is only as good as its ability to detect its own failures, and most real problems look like the math case, not the code case.

---

## Build

### Weekend 1 — hand-rolled, get a number (11h)

- **Setup and task set (3h).** Download both sources, build the task file, write both graders. Sanity check: grade the *reference* solutions. A grader that doesn't score known-good answers near 100% is broken.
- **Model client (2h).** Thin wrapper, two models, returns response plus tokens plus latency plus cost. Include a mock mode so you can test the pipeline without spending anything.
- **Pilot (1h).** Ten tasks, real calls, check the failure rate. Adjust the task set if needed.
- **Five policies (4h).** The cascade is a loop with a hard escalation cap. The predictive router is a heuristic on features you already have — task length, difficulty proxy, keywords. Not a trained model, not an LLM call.
- **Full run and plot (1h).** All policies, all tasks, checkpointed. Pareto plot.

**End of weekend 1 you have a result.** Everything after this is packaging and skills.

### Weekend 2 — LangGraph, tracing, README (9h)

- **LangGraph port (4h).** Same cascade as a `StateGraph`: node for the cheap call, node for the verifier, `add_conditional_edges` on the verifier result, node for the expensive call. Add a checkpointer.

  Port order is the whole point. You wrote the loop by hand, so when the framework turns it into nodes and edges you know exactly what it did and what it hid. That's the difference between listing LangGraph and understanding it.

- **Tracing (2h).** Langfuse or Phoenix, both free and self-hostable. Trace every node. Screenshot for the README.
- **CI eval gate (1h).** Pin a 20-task subset, GitHub Action that fails if accuracy drops. Cheapest MLOps evidence you can buy.
- **README and framing (2h).** Problem, design, results, limitations, reproduction command.

**Optional if time remains:** wrap the router as an MCP server. About an hour with FastMCP. Keyword value, low learning value. Cut this first.

---

## Decisions you must own cold

An interviewer will find these. If you can't answer from memory, the project didn't do its job.

**1. Your predictive router is a heuristic; RouteLLM trained theirs on preference data. Isn't that an unfair comparison?**

Yes, in absolute terms. You're comparing architectures at a fixed, documented router quality. The oracle bounds how good any predictive router could get, so you can say where yours sits between random and optimal and reason about what a trained one would change. Say this before they say it.

**2. How does the verifier know an answer is wrong without knowing the right answer?**

It doesn't. It estimates. Self-consistency works because independent samples converge when the model has a stable answer and diverge when it's guessing. Be ready to say why asking the model "are you confident?" is worse: models are trained to sound helpful, not calibrated.

**3. What's your escalation threshold and why that value?**

Yours to set and defend. Expect "what happens just below it." Honest answer: any hard threshold has an arbitrary boundary, you chose a hard one for reproducibility, and in production it'd be a governed parameter, not a constant.

**4. Why is the oracle in there?**

It bounds headroom. Without it you can't tell whether a small gap between policies means your routers are bad or means routing can't help much on this task set.

**5. When does cascade lose to predictive?**

When the cheap model fails often, because then you double-pay on most queries. Give the number your run produced, not the general argument.

---

## Literature: two hours, three papers, one paragraph

Read **FrugalGPT** (arxiv.org/abs/2305.05176) properly. Skim **RouteLLM** (arxiv.org/abs/2406.18665) and **AutoMix** (arxiv.org/abs/2310.12963).

Their only job is this exchange: *"Isn't this just FrugalGPT?"* → *"Yes, it's a replication of FrugalGPT-style cascading with 2026 models, and here's what changed."* Ten seconds, question closed. You are not writing a paper. Do not read more than three.

---

## Scope killers

- **Adding models.** Two. A third turns the experiment into a matrix.
- **Tuning prompts to improve accuracy.** You're measuring routing, not prompt engineering. Fix reasonable prompts early and leave them.
- **Training the predictive router.** A heuristic is the design. Training one is a different project.
- **Reading more papers.** Three. The research rabbit hole nearly ate this project once already.
- **Building a UI.** Two matplotlib figures in a README.

---

## On AI assistance

If it's a decision, you make it. If it's typing, delegate it.

Hand-write the escalation predicate, the verifier, and the metric definitions, even if you write them badly, then have AI review. Delegate the batch runner, the plotting, the CI YAML.

Then volunteer it in interview before you're asked: *"Boilerplate was AI-assisted. The parts I designed and can defend are the escalation predicate, the verifier, and the evaluation design. Ask me about any of those."*

---

## CV bullet

Fill in the real numbers once you have them.

> **Cascade vs predictive LLM routing — cost-quality benchmark**
> Implemented and benchmarked five routing policies (cascade, predictive, two static baselines, hindsight oracle) on 100 objectively-graded code and math tasks. Measured accuracy, cost per task, latency, and escalation/misroute rates; found cascade routing saves X% of cost for Y points of accuracy, with the advantage concentrated where answers are cheaply verifiable. Hand-rolled control loop ported to LangGraph, traced end to end, with an evaluation gate in CI.

**Skills this honestly unlocks:** LangGraph, agent control loops, LLM evaluation and benchmarking, cost and latency instrumentation, tracing, eval in CI.

**It does not unlock:** production deployment, scale, multi-agent orchestration, cloud.

---

## Open decisions before you start

1. **Model pair.** One provider, big price ratio. The ratio drives the economics.
2. **Math task source**, given the 30–40% cheap-model failure rule.
3. **Escalation threshold**, with the reasoning you'd give out loud.
4. **Langfuse or Phoenix.** Five minutes, then move.
5. **Spend cap in euros**, enforced in the runner, not by willpower. Budget €20.
