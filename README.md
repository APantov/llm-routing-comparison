# Cascade vs Predictive Routing

Measuring when each LLM routing architecture is worth it.

## Run it

Mock mode needs nothing installed. It is pure standard library, offline, and
byte-deterministic.

```bash
python3 build_taskset.py     # builds taskset.jsonl from data/
python3 sanity_check.py      # graders must print 40/40 and 60/60
python3 run_eval.py          # mock mode, no API key, no spend
python3 sweep_degraded.py    # the verifier-degradation curve
python3 random_baseline.py   # the null hypothesis, over 200 seeds
```

Real mode, when the pipeline works and you've done the pilot:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...
ROUTER_MODE=real python3 run_eval.py --limit 10   # 10-task pilot FIRST
```

Every response from that paid run lands in `cache/raw_calls.jsonl`. After it,
everything is free forever:

```bash
ROUTER_MODE=replay python3 run_eval.py         # same numbers, no network, no key
ROUTER_MODE=replay python3 sweep_degraded.py   # every sweep point, $0
```

## Files

| File | What it is | Whose |
|---|---|---|
| `build_taskset.py` | MATH500 (levels 3–5) + sanitized MBPP, stratified sample, unified schema | plumbing |
| `graders.py` | Deterministic grading: exact match (math), run asserts (code) | plumbing |
| `models.py` | Model client, mock / real / replay, cost accounting | plumbing |
| `response_cache.py` | One draw per distinct call, shared by every policy | plumbing |
| `run_eval.py` | Batch runner, spend cap, report | plumbing |
| `policies.py` | The nine policies + the three verifiers | **yours** |
| `sweep_degraded.py` | The experiment: cascade quality vs verifier quality | **yours** |
| `random_baseline.py` | The null hypothesis, over many seeds | **yours** |

## The decisions you own

1. **Model pair** (`models.py`) — price ratio drives the economics
2. **Self-consistency k** (`policies.py`) — failure detection vs cost, linear
3. **Agreement threshold** (`policies.py`) — when to accept the cheap answer
4. **Predictive heuristic** (`policies.py`) — route once, blind, on pre-call features only
5. **Verifier corruption rate** (`policies.py`) — the manipulated variable
6. **Random baseline seeds** (`policies.py`) — the null the others are measured against
7. **LLM-as-router** (`policies.py`) — the router decision #4 rejected, now measured

Change one, re-run, watch the numbers move. That is how you learn what they do.

## Known state

Pipeline runs end to end in mock mode, and every number in the repo is still
**simulated** — no API call has been made. Verified on the current tree:
`taskset.jsonl` rebuilds byte-identically, both graders score reference answers
40/40 and 60/60, and `run_eval.py` reproduces `results.jsonl` exactly.

Live issues, in rough priority order:

- **No real run yet.** `plot.py` does not exist either, so there is no figure.
- **The mock makes majority voting far too strong.** `_wrong_answer` scatters
  wrong answers across distinct values, so self-consistency recovers the truth
  whenever 2 of 5 samples are right — worth +21.7 points here against low single
  digits in the literature. The math half's headline is inflated by this.
- **The oracle no longer bounds the cascade.** `policy_oracle` chooses between
  cheap-greedy and expensive-greedy; the cascade also has cheap-majority-of-5,
  which is outside the oracle's action space.
- **The pilot gate mislabels itself.** It accepts 20–55% but prints
  "GOOD - in the 30-40% target band", currently at 26.0%.
- **Hyperparameters were chosen on the full evaluation set.** There is no
  calibration/evaluation split yet.

`STRATEGY_2026-07-29.md` has the full list and an ordering.

## Reading list — two hours, three papers, stop

- FrugalGPT, arxiv.org/abs/2305.05176 (read properly)
- RouteLLM, arxiv.org/abs/2406.18665 (skim)
- AutoMix, arxiv.org/abs/2310.12963 (skim)

Their only job: answering "isn't this just FrugalGPT?" with "yes, it's a
replication with 2026 models, and here's what changed."
