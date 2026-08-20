# Architecture

Two halves that share one substrate.

```
┌─────────────────────────── the product ────────────────────────────┐
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
│   9 policies     417 tasks     curves, McNemar, bootstrap          │
│                                                                    │
└────────────────── the experiment: llm_routing/ ────────────────────┘
```

The product does not open its own HTTP connections, keep its own price table,
or define its own notion of a model tier. It calls `models.call`, exactly as
`llm_routing/policies.py` does. Three properties follow, and none of them are obtainable
by a serving layer that reimplements its own client:

| property | why it follows |
|---|---|
| costs are comparable | one verified price table, one piece of arithmetic |
| the demo is free | `ROUTER_MODE=replay` serves from 5,075 committed real responses |
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
to beat. **It was deleted.** The benchmark now measures
predictive routing with `llm_router` and `routellm`, neither of which reads a
label, and `live.predict_is_hard_live` no longer has the branch that consulted
the benchmark predicate: it reads query text, unconditionally.

The structural point stands and is why the deletion costs the argument
nothing. It cuts in the cascade's favour, and it is not a thumb on the scale —
a cascade needs no difficulty label because it finds out by trying.

> **One name, two things.** `predictive` still exists *here*, in the serving
> layer: `llm-router --policy predictive` routes once from the query text, and
> `--estimate` prices it as the one-shot bracket. What was deleted is the
> **benchmark** policy of that name, which routed on MATH500's shipped
> difficulty label. They are not the same object, and only the benchmark one
> could read a field a user's query does not have.

### 3. The perfect verifier is usually gone

| benchmark verifier | quality | transfers? | why |
|---|---|---|---|
| `verify_math` — self-consistency | proxy | **yes** | agreement is computed over the model's own draws; it never consults an answer key |
| `verify_code` — run the asserts | perfect | **no** | free and exact only because MBPP+ *ships* the tests; a served query carries none |

This is the largest gap between the experiment and the product, and the
repository already contains the experiment that prices it: `llm_routing/sweep_degraded.py`
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

## The one-way arrow, and the one change that crossed it

`router_agent` imports `llm_routing`, never the reverse, and CI has a job whose
only purpose is to keep it that way.

Exactly one change went the other way. `llm_routing/models.py` gained a `general` domain prompt, a `code_untested` prompt, and a
branch in `_mock_call` for live queries with no ground truth to perturb.

The claim is that none of it is reachable from `llm_routing/build_taskset.py` output.
`scripts/check_core_unchanged.py` **proves** it rather than asserting it: it
fingerprints every mock response the task set can produce — 417 tasks × 3
ladders × 4 sample indices × 2 temperatures, plus both prompt kinds — and
compares against `llm_routing/models.py` at a git revision.

```
  claude    identical  c467911f3b282a8d
  deepseek  identical  ecda0b686b9cb668
  wide      identical  3d19dda9a9df1565

OK: 417 tasks x 3 ladders x 4 samples x 2 temperatures - byte-identical to HEAD.
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

### The experiment, in the order data flows through it

Every module below lives in `llm_routing/` and is runnable on its own with
`python -m llm_routing.<name>`.

```
data/                    raw downloaded datasets (MATH-500, MBPP+)
    |
    v
build_taskset.py         pick 417, normalise into one schema
    |
    v
data/taskset.jsonl       <-- the 417 questions, the input to everything
    |
    v
splits.py                cut in half: calibration / evaluation
    |
    v
run_eval.py              THE CONDUCTOR. for each task, for each policy:
    |                        |
    |                        v
    |                    policies.py        decides WHICH model(s) to call
    |                        |
    |                        v
    |                    models.py          makes the call (or fakes it)
    |                        |
    |                        v
    |                    response_cache.py  has this exact call been made before?
    |                        |
    |                        v
    |                    graders.py         was the answer right?
    v
runs/results.<ladder>.jsonl   <-- one row per (task, policy)
    |
    +---> stats.py            are the differences real, or noise?
    +---> frontier.py         sweep every knob, compare curves not points
    +---> sweep_degraded.py   the actual experiment
    +---> scorecard.py        what each policy got right and wrong, and why
    +---> plot.py             figures/*.svg
```

| file | owns | does NOT own |
|---|---|---|
| `llm_routing/build_taskset.py` | picking and normalising questions | anything about models or policies |
| `llm_routing/graders.py` | "is this answer correct?" | anything about cost |
| `llm_routing/models.py` | prices, model IDs, API contracts, the mock | what a policy decides |
| `llm_routing/response_cache.py` | "have I seen this exact call?" | what a call costs a policy |
| `llm_routing/policies.py` | every routing strategy and verifier | how models are called |
| `llm_routing/splits.py` | which tasks are for tuning vs reporting | anything else |
| `llm_routing/run_eval.py` | running everything, the report | any decision a policy makes |
| `llm_routing/stats.py` | significance testing | producing results |
| `llm_routing/frontier.py` | sweeping knobs, cost-quality curves | single operating points |
| `llm_routing/sweep_degraded.py` | the verifier-degradation experiment | anything not about verifiers |
| `llm_routing/plot.py` | SVG figures | computing anything |
| `llm_routing/sanity_check.py` | proving the graders work | everything else |

The rule the whole layout follows: **a policy never knows what mode it is in.**
`llm_routing/policies.py` calls `models.call("cheap", task)` and gets a reply.
Whether that reply came from a real API, a hash function, or a file on disk is
entirely `llm_routing/models.py`'s business. That is why the same policy code
produces the mock run and the paid run with no branching.

---

## The four data structures

Everything in the repo is one of these four shapes.

**A task** — one row of `data/taskset.jsonl`:

```json
{
  "id": "codeplus-592",
  "domain": "code",
  "prompt": "Write a python function to find the sum of the product of consecutive binomial co-efficients.",
  "grader": "test_program",
  "grader_payload": {
    "tests": ["assert sum_Of_product(3) == 15", "..."],
    "test_program": "import numpy as np\n..."
  },
  "difficulty_proxy": 10,
  "predict_features": {"prompt_chars": 93, "n_asserts": 3},
  "difficulty_pct": 0.9173
}
```

`tests` is what the model is *shown* as its specification — the original thin
MBPP asserts. `test_program` is the expanded evalplus suite it is *graded*
against. Keeping the specification fixed while making the marking stricter is
what makes MBPP+ a one-variable change over MBPP.

Three fields carry the leak discipline, which is where cheating would happen:

- **`predict_features`** is the *only* thing a router may read before calling.
  It holds what is derivable from the question alone.
- **`difficulty_proxy`** for code is the reference solution's line count — **not
  knowable before answering**, so a router reading it would be cheating. It
  exists for sampling and for driving the mock, and keeping it out of
  `predict_features` is what stops the leak.
- **`_ref_code`**, where present, is the known-good answer. Used only by the mock
  and by `llm_routing/sanity_check.py`; a real model never sees it.

**A model response** (`models.ModelResponse`) — `text, tier, tokens_in,
tokens_out, latency_s, cost_usd`. Cost travels *with* the response, because
bolting cost accounting on afterwards is how a project ends up unable to answer
"what did that policy spend".

**A verdict** (`policies.Verdict`) — `accepted, answer_text, cost_usd,
latency_s`. `answer_text` is not always the answer that went in: the maths
verifier buys 5 samples, so it has a better answer available (the majority vote)
than the one it was handed, and returning it is free.

**A policy result** (`policies.PolicyResult`) — `task_id, policy, correct,
cost_usd, latency_s, escalated, calls`. `calls` is the honest receipt: every rung
actually paid for, in order. Read it rather than `escalated` when asking "did the
expensive model get involved", because a one-shot router never *escalates* no
matter which model it picked.

---

## Standing invariants

- **Never touch prompt templates, `models.MAX_TOKENS`, or `MODEL_SPECS` ids.**
  All are in the cache key. Changing one strands 5,075 responses and re-charges
  $8.51. It has happened once, for $0.39.
- **Never delete `archive/`.** It holds superseded real data that cost money.
- **A quarantined task is never counted again**, in any rerun, ladder, or figure.
  Responses are deleted, not filtered; `TestQuarantine` is the tripwire.
- **CI can never spend.** `ROUTER_MODE: mock` is hard-set and no keys are
  configured.

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
