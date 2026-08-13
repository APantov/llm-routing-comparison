# WALKTHROUGH — how the code actually works

[EXPLAINED.md](EXPLAINED.md) covers the *ideas*. This file covers the *code*: what
each file owns, how data moves between them, and a trace of one real task through
the whole system with real numbers.

Read this before spending money. If you understand the trace in §4, you understand
the repo.

---

## 1. The one-sentence version

Load 417 questions → hand each one to 10 routing policies → each policy decides
which model(s) to pay for → grade every answer → compare the policies on
accuracy and cost.

Everything else is machinery for making that comparison *fair*.

---

## 2. The files, in the order data flows through them

Every module named below lives in `llm_routing/`, and is runnable on its own
with `python -m llm_routing.<name>`.

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

### What each file owns, and nothing else

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
`llm_routing/policies.py` calls `models.call("cheap", task)` and gets a reply. Whether that
reply came from a real API, a hash function, or a file on disk is entirely
`llm_routing/models.py`'s business. That is why the same policy code produces the mock run and
the paid run with no branching.

---

## 3. The four data structures

Everything in the repo is one of these four shapes.

### A task (a row of `data/taskset.jsonl`)

Real example, exactly as stored:

```json
{
  "id": "codeplus-592",
  "domain": "code",
  "prompt": "Write a python function to find the sum of the product of consecutive binomial co-efficients.",
  "grader": "test_program",
  "grader_payload": {
    "tests": ["assert sum_Of_product(3) == 15", "assert sum_Of_product(4) == 56", "..."],
    "test_program": "import numpy as np\n..."
  },
  "difficulty_proxy": 10,
  "predict_features": {"prompt_chars": 93, "n_asserts": 3},
  "difficulty_pct": 0.9173
}
```

`tests` is what the model is *shown* as its specification — the original thin
MBPP asserts. `test_program` is the expanded evalplus suite it is *graded*
against, roughly 35x more cases. Keeping the specification fixed while making
the marking stricter is what makes MBPP+ a one-variable change over MBPP.

Three fields deserve attention because they are where cheating would happen:

- **`predict_features`** is the *only* thing a router may read before calling. It
  contains things derivable from the question alone.
- **`difficulty_proxy`** for code is the reference solution's line count. That is
  **not knowable before answering**, so a router reading it would be cheating. It
  exists for sampling and for driving the mock. Keeping it out of
  `predict_features` is what stops that leak.
- **`_ref_code`**, on the rows that carry it, is the known-good answer. Used
  only by the mock and by `llm_routing/sanity_check.py`. A real model never sees
  it.

### A model response (`models.ModelResponse`)

```
text, tier, tokens_in, tokens_out, latency_s, cost_usd
```

Cost travels *with* the response. That is deliberate — bolting cost accounting on
afterwards is how projects end up unable to answer "what did that policy spend".

### A verdict (`policies.Verdict`)

What a verifier hands back: `accepted`, `answer_text`, `cost_usd`, `latency_s`.

`answer_text` is subtle and matters. It is not always the answer that went in. The
maths verifier buys 5 samples, so it has a *better* answer available (the majority
vote) than the single one it was handed. Returning it is free, so it does.

### A policy result (`policies.PolicyResult`)

```
task_id, policy, correct, cost_usd, latency_s, escalated, calls
```

`calls` is the honest receipt — every rung actually paid for, in order. Read that
rather than `escalated` when asking "did the expensive model get involved", because
a one-shot router never "escalates" no matter which model it picked.

---

## 4. One real task, traced end to end

Task `math-284`, MATH-500 level 5. Truth: `3 \pm 2 \sqrt{2}`.

> Find all solutions to
> `sin(tan⁻¹(x) + cot⁻¹(1/x)) = 1/3`.

These are **real numbers from real model calls** on the `wide` ladder
(DeepSeek v4-flash → Opus 5), replayed from `cache/raw_calls.wide.jsonl`.

### What `always_cheap` does

```
1. models.call("cheap", task)
     -> build_prompt() makes the text
     -> response_cache checks: seen this exact (model, prompt, temp, sample)?
        HIT -> returns the response the paid run received
     DeepSeek v4-flash answers
2. graders.grade(task, reply)
     extract_answer() finds the last \boxed{}
     normalise, compare to truth  ->  WRONG
```

Result: `correct=False`, `cost=$0.000287`.

### What `cascade` does on the same task

```
1. models.call("cheap", task)   -- IDENTICAL call, so it is a CACHE HIT.
   Same reply, same $0.000287. This is the whole point of the cache:
   both policies are judged on the same model output.

2. verify_math(task, reply, "cheap")
     samples the cheap model 4 more times at temperature 0.8
     the 5 answers disagree, and agreement lands under the 0.8 threshold
     -> REJECTED, escalate

3. models.call("expensive", task)   -- Opus 5
     reply is correct

Result: correct=True, cost=$0.030078, calls=[cheap x5, expensive]
```

**105 times the cost of `always_cheap`, and right instead of wrong.** That single
task is the entire trade-off the project measures, and every table in the
repository is that comparison aggregated over 209 held-out tasks.

It also shows why the maths half is the expensive half: five cheap calls bought
a verdict, not an answer. On the code half the same verdict costs nothing,
because the tests are executed instead of sampled.

### The bug that trace exposed

Step 3 did not always exist. Before it, the cascade paid for the mid rung, had the
answer rejected for a reason knowable in advance, and escalated anyway — on **25
out of 25** maths escalations. Measured with the guard off versus on, on identical
code:

| | accuracy | cost/task |
|---|---|---|
| guard off | 100.0% | $0.004589 |
| guard on | 100.0% | $0.003839 |

Same answers, **16% less money**. The cause is an API contract (Sonnet 5 rejects
`temperature`) propagating into economics two layers away. This is a good example
of the kind of thing only a trace finds.

---

## 5. The five ideas that make the comparison fair

Everything confusing in this repo is one of these.

### 5.1 The response cache is not a speed optimisation

Ten policies all need "the cheap model's answer to task 7". Without a cache
that is ten API calls returning **ten different answers**, because models are
not deterministic. Then when two policies disagree you cannot tell whether their
*strategies* differ or whether they got different dice rolls.

So the first answer is stored and everyone gets that one. Every policy is judged on
identical model output. That is what makes the paired statistics in `llm_routing/stats.py`
valid at all.

**But a cache hit still charges the policy full price.** In production you run one
policy and there is nothing to share with, so `cost_usd` must answer "what would
this cost to deploy". The cache saves *the experiment* money, not the policy. The
report prints both numbers separately.

### 5.2 The mock is a pure function, not random

`models._draw` hashes its inputs into a seed. So the mock's answer to
(task, model, temperature, sample index) is always the same. Consequences:

- a run reproduces byte for byte
- `--limit 10` gives exactly the first 10 tasks of a full run
- running one policy alone gives the same answers as running all of them

This used to be an unseeded global RNG, and repeat runs disagreed by more than the
effects being measured.

### 5.3 The oracle must be able to do everything the policies can

The oracle cheats — it tries everything and reports the cheapest option that
worked. It exists as a ceiling: if the best real router scores 90% and the oracle
scores 91%, routing is nearly maxed out.

A ceiling that is not a ceiling is worse than no ceiling. An earlier version let
the oracle choose only cheap-vs-expensive, but the maths cascade also has
majority-vote-over-5-samples available — so the cascade **scored above the
supposed maximum**. `run_eval` now prints an explicit bound check every run.

### 5.4 Half the tasks are hidden from the tuning

Pick a threshold by trying several on all 417 tasks, then report the best score,
and the number partly measures your own choice. `llm_routing/splits.py` holds out
half; thresholds and quality estimates are fitted on the other half. That is why
runs report `n=209` and not `n=417`.

### 5.5 Policies are curves, not points

Every policy has a knob that trades accuracy for money. Comparing two policies at
one setting each lets whoever set the knobs pick the winner. `llm_routing/frontier.py` sweeps
every knob across its full range and compares the resulting curves.

---

## 6. Where every decision lives

Each is marked `DECISION #n` in the source, next to the code it controls.

| # | what | file |
|---|---|---|
| 1 | the model ladder and its prices | `llm_routing/models.py` |
| 2 | self-consistency sample count (k=5) | `llm_routing/policies.py` |
| 3 | agreement threshold (0.8) | `llm_routing/policies.py` |
| 4 | ~~the predictive heuristic~~ retracted, tombstoned | `llm_routing/policies.py` |
| 5 | verifier corruption rate — **the manipulated variable** | `llm_routing/policies.py` |
| 6 | random baseline seed | `llm_routing/policies.py` |
| 7 | LLM-as-router | `llm_routing/policies.py` |
| 8 | RouteLLM variant and threshold | `llm_routing/routellm_router.py` |
| 9 | cascade routing λ | `llm_routing/policies.py` |

---

## 7. Reading order, if you want to read the source

1. **`llm_routing/build_taskset.py`** — simplest. Shows you what a task is.
2. **`llm_routing/graders.py`** — self-contained. "Is this answer right?"
3. **`llm_routing/models.py`**, just `MODEL_SPECS` and `LADDERS` at the top — the price tables.
4. **`llm_routing/policies.py`**, just `policy_always` then `_cascade` — the two ends of the
   spectrum. Skip `cascade_routing` on a first pass; it is the most involved.
5. **`llm_routing/run_eval.py`**, just `run()` and `report()` — the loop and the table.

Skip on a first pass: `llm_routing/frontier.py`, `llm_routing/stats.py`, `llm_routing/plot.py`, `llm_routing/routellm_router.py`.
They are analysis and reporting, and none of them changes what a policy does.

---

## 8. The three modes, mechanically

| mode | what `models.call()` does | costs money |
|---|---|---|
| `mock` | hashes the inputs, fabricates a reply | no |
| `real` | calls the API, writes the reply to cache | **yes** |
| `replay` | reads from cache, **errors** on a miss | no |

Replay is the reproducibility guarantee. After one paid run, anyone can re-run the
entire experiment — every sweep, hundreds of times over — and get identical numbers
with no API key. That is why the cache had to exist before any money was spent.

The cache files are per ladder (`cache/raw_calls.wide.jsonl`), and real and mock
live in separate files so a fabricated response can never be one careless `git add`
away from being published as a measurement.

---

## 9. What to be suspicious of

Every published number comes from replay over real responses. These are the
places to push anyway:

- **Anything printed in mock mode.** Mock fabricates replies from a formula, so
  its accuracies restate a constant in `models.py`. It is labelled at every
  point of output, and nothing fabricated is committed — but it is what you get
  if you forget `ROUTER_MODE=replay`.
- **Aggregates over both domains.** The task set is 86% code, so any "all"
  figure is close to a code figure. Per-domain numbers are reported throughout
  and should be preferred.
- **The `wide` ladder's capability gap.** Its two rungs are different providers,
  so capability and provider are confounded.
- **Any single-draw claim about routing opportunity.** Greedy decoding is not
  deterministic; see [LIMITATIONS.md](LIMITATIONS.md).
