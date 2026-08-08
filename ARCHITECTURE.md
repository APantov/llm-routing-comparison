# Architecture

Two halves that share one substrate.

```
┌─────────────────────────── the product ───────────────────────────┐
│                                                                    │
│   llm-router CLI            MCP server (stdio)                     │
│        │                         │                                 │
│        └────────────┬────────────┘                                 │
│                     ▼                                              │
│              router_agent.engine        the façade                 │
│                     │                                              │
│                     ▼                                              │
│              router_agent.graph         LangGraph state machine    │
│              ┌──────────────────────────────────────┐              │
│              │ classify → answer → verify → escalate│              │
│              │              ▲__________________|    │              │
│              └──────────────────────────────────────┘              │
│                     │                                              │
│         ┌───────────┼───────────┐                                  │
│         ▼           ▼           ▼                                  │
│      live.py   verifiers.py  pricing.py                            │
│    (no ground   (no ground   (projects cost                        │
│     truth)       truth)       before spending)                     │
└─────────────────────┬──────────────────────────────────────────────┘
                      │  models.call() — the ONLY door to a provider
┌─────────────────────┼──────────────────────────────────────────────┐
│                     ▼                                              │
│   models.py ── response_cache.py ── graders.py                     │
│   price table    one draw per        deterministic                 │
│   mock/real/     distinct call       marking                       │
│   replay                                                           │
│                                                                    │
│   policies.py → run_eval.py → frontier.py / stats.py               │
│   10 policies    100 tasks     curves, McNemar, bootstrap          │
│                                                                    │
└─────────────────────── the experiment ─────────────────────────────┘
```

The product does not open its own HTTP connections, keep its own price table,
or define its own notion of a model tier. It calls `models.call`, exactly as
`policies.py` does. Three properties follow, and none of them are obtainable
by a serving layer that reimplements its own client:

| property | why it follows |
|---|---|
| costs are comparable | one verified price table, one piece of arithmetic |
| the demo is free | `ROUTER_MODE=replay` serves from 318 committed real responses |
| spending is auditable | one function reaches a provider, not a package |

---

## The graph, and why it is a graph

```
START → classify → answer → verify ─┬─ accept ──→ finalize → END
                     ▲              ├─ stop ────→ finalize
                     │              └─ escalate → escalate ─┬→ answer
                     └────────────────────────────────────  └→ finalize
```

The edge from `escalate` back to `answer` is a cycle, and it is what makes
LangGraph load-bearing rather than decorative:

- **Checkpointing across the cycle.** A run that pauses mid-cascade resumes on
  the rung it stopped at, instead of restarting and re-paying for the cheap
  call.
- **The interrupt is inside the loop.** Human approval is needed *before an
  escalation* — a point in the middle of an iteration. A `while` loop would
  have to hand-roll suspend and resume; `interrupt()` suspends the graph and
  `Command(resume=...)` continues it.
- **The trace is structural.** Reducer-backed channels accumulate across laps,
  so "why did this query cost 40 cents" is answered by reading state rather
  than by correlating log lines.

### State: two kinds of channel

The characteristic bug in a loop with accumulated state is that the second lap
overwrites the first lap's spend instead of adding to it — producing a router
that under-reports its own cost. On a project whose subject *is* cost, that is
the bug most worth designing against.

| channel | reducer | meaning |
|---|---|---|
| `rung_index`, `answer`, `verified` | none — last write wins | current position |
| `cost_usd`, `latency_s`, `calls`, `events` | `operator.add` | cumulative over the run |

`tests/test_graph.py::TestAccounting` asserts `cost_usd == sum(call costs)`
after a run that actually escalated, which is the regression test for exactly
this.

### The three guards on `escalate`

Checked cheapest-first, so a human is never interrupted to approve a call the
budget was going to refuse anyway:

1. **Ladder** — is there a rung above this one.
2. **Budget** — would the next call breach the ceiling, using a *projected*
   cost. A check afterwards is a receipt, not a budget.
3. **Human** — if `require_approval_above_usd` is set and the projection
   exceeds it, suspend.

---

## What does not survive the move from benchmark to product

This is the substantive engineering content of the serving layer, and the
part the benchmark alone cannot tell you.

### 1. Ground truth is gone

`graders.grade` needs the answer. A served query has none. So the serving
layer reports **`verified`** — a verifier's opinion — and never `correct`.
`RouteOutcome` has no `correct` field, and a test asserts it never gains one.

Every result carries `verified_meaning`, which states in words what was and
was not measured, and distinguishes three cases that are easy to conflate:

- the model agreed with itself → weak positive evidence
- the model disagreed with itself → evidence of unreliability, not proof of error
- agreement **could not be measured** → unverified, which is not the same as either

### 2. The difficulty label is gone — and leaning on it cost the benchmark a policy

The benchmark's `predictive` policy routed maths on
`predict_features["level"] >= 5`. That level is MATH500's own annotation,
shipped with the dataset, written by someone who had already solved the
problem. **A user's query has no such field and never will.**

That was documented here as an *upper bound* on what a deployed predictive
router could score. It was not one. `MIN_MATH_LEVEL = 5` makes the field
constant, so the predicate was true for every maths task and the policy was
`always_expensive` on 60% of the set — scoring below the coin flip it existed
to beat. **It was deleted on 8 August 2026.** The benchmark now measures
predictive routing with `llm_router` and `routellm`, neither of which reads a
label, and `live.predict_is_hard_live` no longer has the branch that consulted
the benchmark predicate: it reads query text, unconditionally.

The structural point stands and is why the deletion costs the argument
nothing. It cuts in the cascade's favour, and it is not a thumb on the scale —
a cascade needs no difficulty label because it finds out by trying.

### 3. The perfect verifier is usually gone

| benchmark verifier | quality | transfers? | why |
|---|---|---|---|
| `verify_math` — self-consistency | proxy | **yes** | agreement is computed over the model's own draws; it never consults an answer key |
| `verify_code` — run the asserts | perfect | **no** | free and exact only because MBPP+ *ships* the tests; a served query carries none |

This is the largest gap between the experiment and the product, and the
repository already contains the experiment that prices it: `sweep_degraded.py`
corrupts `verify_code` by a controlled amount *inside the code domain* —
holding models, prompts, domain and grader fixed — and measures how cascade
quality falls as verifier quality does. Losing the real tests in production is
a move along that curve, not a step off it.

Consequence, stated plainly:

> **the cascade's production economics are governed by which verifier you can
> actually obtain, and for most workloads that is the proxy, not the perfect
> one.**

### 4. Verification is not available at every rung

`temperature` was removed on Opus 4.7+ and Sonnet 5, so those rungs cannot be
resampled and self-consistency is simply unavailable on them. On the `claude`
and `wide` ladders that applies to **every rung above the bottom**, which means
a cascade that fails at the bottom climbs to the top without being able to
check the intermediate rung. `deepseek` is the only ladder that can verify
everywhere — both rungs accept a temperature.

The verifier reports this as *unverifiable* rather than as unanimous
agreement. A verifier that always accepts is worse than no verifier, because
it is invisible.

---

## The one change the agent layer made to the research core

`models.py` gained a `general` domain prompt, a `code_untested` prompt, and a
branch in `_mock_call` for live queries with no ground truth to perturb.

The claim is that none of it is reachable from `build_taskset.py` output.
`scripts/check_core_unchanged.py` **proves** it rather than asserting it: it
fingerprints every mock response the task set can produce — 100 tasks × 3
ladders × 4 sample indices × 2 temperatures, plus both prompt kinds — and
compares against `models.py` at a git revision.

```
  claude    identical  6a463523b63ce57a
  deepseek  identical  2ffdab4328c47b17
  wide      identical  4f2133356a22d25e

OK: byte-identical to HEAD.
```

It runs in CI. If a future edit to the serving path leaks into the experiment,
the build fails.

---

## Modes

| mode | what it does | costs | can serve a live query? |
|---|---|---|---|
| `mock` | fabricates responses | nothing | yes, but the text is a labelled placeholder |
| `replay` | serves from `cache/raw_calls.*.jsonl` | nothing | only prompts that were actually paid for |
| `real` | calls the provider | money | yes |

The interesting one is **replay**. The response cache is keyed on the *prompt
text*, not the task id, so feeding a benchmark question in as a live query
hits the committed real responses:

```bash
llm-router --demo     # real Opus 5 and DeepSeek output, $0.00, no API key
```

Self-consistency verification asks for draws at `sample_idx >= 1`, which the
two-arm probe never bought. Replay therefore reports *unverifiable* on those
and the cascade escalates — which is the honest outcome, and visible in the
trace.

---

## Layout

| file | what it is |
|---|---|
| `router_agent/config.py` | settings; every default cites the measurement behind it |
| `router_agent/live.py` | query → task dict; where the missing difficulty label is documented |
| `router_agent/verifiers.py` | verification without ground truth |
| `router_agent/pricing.py` | cost projection, calibrated on measured token counts |
| `router_agent/findings.py` | the benchmark's results, recomputed from committed data |
| `router_agent/state.py` | the graph state and its reducers |
| `router_agent/nodes.py` | one function per node, testable without LangGraph |
| `router_agent/graph.py` | pure wiring |
| `router_agent/engine.py` | the façade |
| `router_agent/cli.py` | `llm-router` |
| `router_agent/mcp_server.py` | tools, resources, prompts |
| `scripts/check_core_unchanged.py` | proves the research core is untouched |
| `scripts/check_mcp_server.py` | MCP registration smoke test |

For the experiment's own files, see the table in [README.md](README.md).

---

## Design decisions worth arguing with

**Why not LangChain model wrappers?** They would replace `models.call`, and
with it the price table, the response cache, replay, and cost comparability
with the benchmark. The wrapper would buy provider portability this project
does not need — both providers already speak the Anthropic wire format.

**Why is domain inference a regex and not a model call?** A learned classifier
would be a second routing decision hidden inside the first: it costs money,
needs its own evaluation, and its errors get attributed to the router. The
repository already has a policy (`llm_router`) measuring what happens when you
ask a model to classify before answering. The cost of being wrong here is
bounded — the domain picks a prompt template and a default verifier, not a
model.

**Why is executing tests off by default?** `verify_tests` runs
model-generated code through `graders.grade_run_asserts`, which shells out to
`[sys.executable, path]`. No container, no seccomp, no network policy. In the
benchmark that risk is bounded — the code is written against MBPP+ problems on
a machine the author controls. In serving it is arbitrary code execution as a
service. It is gated behind `allow_code_execution`, and the fallback is the
proxy verifier rather than a refusal to answer.
