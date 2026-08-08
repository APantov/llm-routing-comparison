# WALKTHROUGH — how the code actually works

[EXPLAINED.md](EXPLAINED.md) covers the *ideas*. This file covers the *code*: what
each file owns, how data moves between them, and a trace of one real task through
the whole system with real numbers.

Read this before spending money. If you understand the trace in §4, you understand
the repo.

---

## 1. The one-sentence version

Load 100 questions → hand each one to 11 different routing strategies → each
strategy decides which model(s) to pay for → grade every answer → compare the
strategies on accuracy and cost.

Everything else is machinery for making that comparison *fair*.

---

## 2. The files, in the order data flows through them

```
data/                    raw downloaded datasets (MATH500, MBPP)
    |
    v
build_taskset.py         pick 100, normalise into one schema
    |
    v
taskset.jsonl            <-- the 100 questions, the input to everything
    |
    v
splits.py                cut in half: calibration / evaluation
    |
    v
run_eval.py              THE CONDUCTOR. for each task, for each policy:
    |                        |
    |                        v
    |                    policies.py      decides WHICH model(s) to call
    |                        |
    |                        v
    |                    models.py        makes the call (or fakes it)
    |                        |
    |                        v
    |                    response_cache.py  has this exact call been made before?
    |                        |
    |                        v
    |                    graders.py       was the answer right?
    v
results.jsonl            <-- one row per (task, policy)
    |
    +---> stats.py       are the differences real, or noise?
    +---> frontier.py    sweep every knob, compare curves not points
    +---> sweep_degraded.py   the actual experiment
    +---> plot.py        figures/*.svg
```

### What each file owns, and nothing else

| file | owns | does NOT own |
|---|---|---|
| `build_taskset.py` | picking and normalising questions | anything about models or policies |
| `graders.py` | "is this answer correct?" | anything about cost |
| `models.py` | prices, model IDs, API contracts, the mock | what a policy decides |
| `response_cache.py` | "have I seen this exact call?" | what a call costs a policy |
| `policies.py` | every routing strategy and verifier | how models are called |
| `splits.py` | which tasks are for tuning vs reporting | anything else |
| `run_eval.py` | running everything, the report | any decision a policy makes |
| `stats.py` | significance testing | producing results |
| `frontier.py` | sweeping knobs, cost-quality curves | single operating points |
| `sweep_degraded.py` | the verifier-degradation experiment | anything not about verifiers |
| `plot.py` | SVG figures | computing anything |
| `sanity_check.py` | proving the graders work | everything else |

The rule the whole layout follows: **a policy never knows what mode it is in.**
`policies.py` calls `models.call("cheap", task)` and gets a reply. Whether that
reply came from a real API, a hash function, or a file on disk is entirely
`models.py`'s business. That is why the same policy code produces the mock run and
the paid run with no branching.

---

## 3. The four data structures

Everything in the repo is one of these four shapes.

### A task (a row of `taskset.jsonl`)

Real example, exactly as stored:

```json
{
  "id": "code-86",
  "domain": "code",
  "prompt": "Write a function to find nth centered hexagonal number.",
  "grader": "run_asserts",
  "grader_payload": {"tests": ["assert centered_hexagonal_number(10) == 271", ...]},
  "difficulty_proxy": 2,
  "predict_features": {"prompt_chars": 55, "n_asserts": 3},
  "_ref_code": "def centered_hexagonal_number(n):\n  return 3 * n * (n - 1) + 1",
  "difficulty_pct": 0.0513
}
```

Three fields deserve attention because they are where cheating would happen:

- **`predict_features`** is the *only* thing a router may read before calling. It
  contains things derivable from the question alone.
- **`difficulty_proxy`** for code is the reference solution's line count. That is
  **not knowable before answering**, so a router reading it would be cheating. It
  exists for sampling and for driving the mock. Keeping it out of
  `predict_features` is what stops that leak.
- **`_ref_code`** is the known-good answer. Used only by the mock (it needs
  something that genuinely passes the tests) and by `sanity_check.py`. A real
  model never sees it.

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

Task `math-331`, level 5. Truth: `\frac{13}{6}`. These are actual numbers from a
mock run on the `claude` ladder.

### What `always_cheap` does

```
1. models.call("cheap", task)
     -> build_prompt() makes the text
     -> response_cache checks: seen this exact (model, prompt, temp, sample) before?
        MISS -> mock generates a reply, writes it to cache
     reply: "Reasoning... the answer is $\boxed{\frac{13}{6}+7}$"
     55 tokens in, 80 out  ->  $0.000455
2. graders.grade(task, reply)
     extract_answer() finds the last \boxed{} -> "\frac{13}{6}+7"
     normalise, compare to truth "\frac{13}{6}"  ->  WRONG
```

Result: `correct=False`, `cost=$0.000455`.

### What `cascade` does on the same task

```
1. models.call("cheap", task)   -- IDENTICAL call, so it is a CACHE HIT.
   Same reply, same $0.000455. This is the whole point of the cache:
   both policies are judged on the same model output.

2. verify_math(task, reply, "cheap")
     samples the cheap model 4 more times at temperature 0.8
     +$0.001820
     the 5 answers disagree: plurality is "\frac{13}{6}+7" with 40% agreement
     threshold is 80%  ->  REJECTED, escalate

3. next rung is "mid" (Sonnet 5).
     _verdict_is_predetermined() asks: could verify_math ever accept this rung?
     Sonnet 5 does not accept a temperature -> cannot be sampled -> always rejected
     -> SKIP IT. Do not pay for an answer already certain to be discarded.

4. models.call("expensive", task)   -- Opus 5
     reply is correct

Result: correct=True, cost=$0.007351, calls=[cheap x5, expensive]
```

Sixteen times the cost of `always_cheap`, and right instead of wrong. That single
task is the entire trade-off the project measures, and every table in the repo is
that comparison aggregated over 100 tasks.

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
identical model output. That is what makes the paired statistics in `stats.py`
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

Pick a threshold by trying several on all 100 tasks, then report the best score,
and the number partly measures your own choice. `splits.py` holds out half;
thresholds and quality estimates are fitted on the other half. That is why runs
report `n=51` and not `n=100`.

### 5.5 Policies are curves, not points

Every policy has a knob that trades accuracy for money. Comparing two policies at
one setting each lets whoever set the knobs pick the winner. `frontier.py` sweeps
every knob across its full range and compares the resulting curves.

---

## 6. Where every decision lives

Each is marked `DECISION #n` in the source, next to the code it controls.

| # | what | file |
|---|---|---|
| 1 | the model ladder and its prices | `models.py` |
| 2 | self-consistency sample count (k=5) | `policies.py` |
| 3 | agreement threshold (0.8) | `policies.py` |
| 4 | ~~the predictive heuristic~~ retracted, tombstoned | `policies.py` |
| 5 | verifier corruption rate — **the manipulated variable** | `policies.py` |
| 6 | random baseline seed | `policies.py` |
| 7 | LLM-as-router | `policies.py` |
| 8 | RouteLLM variant and threshold | `routellm_router.py` |
| 9 | cascade routing λ | `policies.py` |

---

## 7. Reading order, if you want to read the source

1. **`build_taskset.py`** (206 lines) — simplest. Shows you what a task is.
2. **`graders.py`** (200) — self-contained. "Is this answer right?"
3. **`models.py`**, just `MODEL_SPECS` and `LADDERS` at the top — the price tables.
4. **`policies.py`**, just `policy_always` then `_cascade` — the two ends of the
   spectrum. Skip `cascade_routing` on a first pass; it is the most involved.
5. **`run_eval.py`**, just `run()` and `report()` — the loop and the table.

Skip on a first pass: `frontier.py`, `stats.py`, `plot.py`, `routellm_router.py`.
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

- **Any accuracy number, right now.** Mock mode fabricates replies from constants.
  Cost numbers are real; accuracy numbers are not. See [NOTES.md](NOTES.md).
- **`llm_router`'s accuracy specifically.** In mock mode its router is an oracle on
  the mock's own difficulty. It measures a constant.
- **The `wide` ladder's accuracy.** Two providers, so capability and provider are
  confounded, and the tokenizer factors are unmeasured.
- **Anything from the maths half involving self-consistency.** The mock scatters
  wrong answers, so majority voting works far better than it does on real models.
