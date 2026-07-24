# Runbook

Ordered steps to get from the current state to a reportable result.

Current state: pipeline runs end to end in mock mode on **synthetic** data.
Nothing here has touched a real model yet.

---

## Phase 0 — Replace the fake data (blocking everything)

**Step 1. Download MBPP.**
`mbpp.jsonl` from github.com/google-research/google-research/tree/master/mbpp
→ save as `data/mbpp.jsonl`. Has to be done from your machine; the sandbox
can't reach github raw.

**Step 2. Pick and download the math set.** See DECISION A below. If you pick
anything other than GSM8K you also have to do step 3.

**Step 3. If not GSM8K, rewrite two things in `build_taskset.py`:**

- the answer parser. GSM8K puts the answer after `#### `. MATH500 puts it in
  `\boxed{}` and the contents may be a fraction, a radical or an expression,
  not an integer. `graders.extract_final_int` cannot handle those — you'd need
  `grade_exact_match_str` with light normalisation.
- the difficulty proxy. GSM8K's `<<...>>` step count disappears. MATH500 ships
  a `level` field (1–5), which is a better proxy anyway.

**Step 4. Delete `make_fixture_data.py`** and rebuild:

```bash
python3 build_taskset.py
```

**Step 5. Sanity-check the graders.** A grader that can't score known-good
answers is broken, and you will not notice until the numbers are meaningless.

```bash
python3 -c "
import json
from graders import grade
tasks=[json.loads(l) for l in open('taskset.jsonl')]
code=[t for t in tasks if t['domain']=='code']
print('code refs:', sum(grade(t,'\`\`\`python\n'+t['_ref_code']+'\n\`\`\`') for t in code), '/', len(code))
math=[t for t in tasks if t['domain']=='math']
print('math truth:', sum(grade(t,'the answer is '+t['grader_payload']['answer']) for t in math), '/', len(math))
"
```

Both must be at or very near 100%. Stop and fix if not.

---

## Phase 1 — Make the predictive router honest

**Step 6. Fix the leak in `policies.predict_is_hard`.** It currently reads
`difficulty_pct`, which for code tasks derives from the line count of MBPP's
reference solution — information you do not have before answering. See
DECISION B. Until this is fixed, no `predictive` number is reportable.

**Step 7. Re-run mock** to confirm nothing broke:

```bash
python3 run_eval.py
```

---

## Phase 2 — Pilot against real models (the gate)

**Step 8. Set the model pair and real prices** in `models.py`. See DECISION C.
Check the prices against the current pricing page rather than trusting the
values sitting in the file.

**Step 9. Install and authenticate:**

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
```

**Step 10. Run the 10-task pilot. Do not skip this.**

```bash
ROUTER_MODE=real python3 run_eval.py --limit 10
```

**Step 11. Read the `cheap-model failure rate` line at the bottom.**

| Reading | Meaning | Action |
|---|---|---|
| under 20% | every policy will converge, nothing to route | harder tasks, go back to step 2 |
| 30–40% | the band where routing decisions matter | proceed |
| over 55% | cascade escalates nearly everything | easier tasks, go back to step 2 |

Iterating here is cheap. Discovering it after the full run costs you a weekend.

---

## Phase 3 — Full run and the headline plot

**Step 12. Set the two cascade parameters.** DECISIONS D and E.

**Step 13. Confirm the spend cap** in `run_eval.py` (`MAX_SPEND_USD = 20.0`)
matches your actual budget in the currency you care about.

**Step 14. Full run:**

```bash
ROUTER_MODE=real python3 run_eval.py
```

**Step 15. Write `plot.py`.** It does not exist yet and the plan calls for it.
Accuracy on y, cost per task on x, one point per policy, from `results.jsonl`.
Mark which points are on the Pareto frontier and which are dominated. A second
figure splitting the cascade by domain is the one that shows the verifier
contrast. Delegate this — it's typing, not a decision.

**Step 16. Sensitivity sweep, if time.** Re-run with 2–3 values of the
agreement threshold and plot the resulting curve. Turns "I picked 0.8" into
"here is what the threshold buys you", which is a much better answer.

---

## Phase 4 — Weekend 2, packaging

17. Port the cascade to LangGraph as a `StateGraph` with a checkpointer.
18. Add tracing (Langfuse or Phoenix), screenshot for the README.
19. Pin a 20-task subset, add a GitHub Action that fails on accuracy regression.
20. Rewrite the README: problem, design, results, limitations, repro command.
21. Add `requirements.txt` — it doesn't exist.
22. Fill the real numbers into the CV bullet in the plan.

---

# Decisions

Seven. The first five you must be able to defend from memory.

### A. Math task source

The plan says explicitly "not GSM8K" because modern cheap models score in the
low 90s there and the cascade has nothing to route. `build_taskset.py`
currently loads GSM8K anyway. Resolve this.

| Option | Trade-off |
|---|---|
| **MATH500 levels 3–5** | Best default. Ships a difficulty level, so the proxy is free and honest. Costs you a `\boxed{}` parser. |
| **GSM8K-Hard / GSM-Symbolic** | Smallest change: keeps the existing parser and step-count proxy, just drops accuracy. Weakest story. |
| **AIME** | Integer answers so grading stays trivial, but ~30 items and near-0% cheap accuracy. Hard tail only. |
| **GPQA-Diamond** | Non-math option, exact-match MCQ grading. But multiple choice breaks self-consistency — 4 options means agreement by chance. |

Also consider whether MBPP is too easy for the code half. HumanEval+ or
LiveCodeBench are harder, but MBPP's shipped asserts are what make the free
verifier work, so don't swap it without checking the replacement has tests.

### B. The predictive router's features

It has to run on information available *before* any model call. Candidates:
prompt character or token length, question-mark and clause counts, keyword hits
("prove", "minimum", "probability", "recursive"), number of asserts for code,
MATH500's `level` field if you use it.

The `level` field is legitimate metadata, not a leak — it's shipped with the
question, not derived from the answer. Reference-solution line count is a leak.
Be able to state which of your features is which and why.

Set `PREDICTIVE_HARD_PCT` and be ready for "what happens just below it".

### C. Model pair

One provider, big price ratio — the ratio drives the entire economics. Currently
Haiku 4.5 vs Sonnet 5 in `models.py`. Verify the prices are current. A larger
ratio makes cascade look better; be honest that the result is ratio-dependent.

### D. Self-consistency k (`SELF_CONSISTENCY_K`, currently 5)

Linear cost, diminishing returns on failure detection. k=5 means the math
cascade pays 5× the cheap model before it even considers escalating, which is
most of why math loses. Try k=3 and see whether detection degrades enough to
matter.

### E. Agreement threshold (`AGREEMENT_THRESHOLD`, currently 0.8)

Fraction of samples that must agree to accept the cheap answer. Higher =
escalate more = more accuracy, more cost. Expect "what happens just below your
threshold?" Honest answer: every hard threshold has an arbitrary boundary; you
chose a hard one for reproducibility; in production it'd be a governed
parameter, not a constant. The step 16 sweep is the strong version of this.

### F. Spend cap

€20 in the plan. Enforced in `run_eval.py`, not by willpower. Note it's
currently in USD.

### G. Langfuse or Phoenix

Five minutes of reading, then commit. Both free and self-hostable. Do not
spend an evening on this.

---

# Known gaps

- `data/` is synthetic (`make_fixture_data.py`). Delete before real mode.
- `predict_is_hard` leaks reference-solution length. Blocks reporting.
- No `plot.py`. The plan's single headline deliverable does not exist.
- No `requirements.txt`.
- README "Known state" section is stale — both bugs it lists are fixed.
- `cascade_degraded` (weakened verifier) is still an unbuilt stretch goal.
