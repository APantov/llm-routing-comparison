# Cascade vs Predictive Routing

Measuring when each LLM routing architecture is worth it.

## Run it

```bash
python3 build_taskset.py     # builds taskset.jsonl from data/
python3 run_eval.py          # mock mode, no API key, no spend
```

Real mode, when the pipeline works and you've done the pilot:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
ROUTER_MODE=real python3 run_eval.py --limit 10   # 10-task pilot FIRST
```

## Files

| File | What it is | Whose |
|---|---|---|
| `build_taskset.py` | Downloads GSM8K + MBPP, stratified sample, unified schema | plumbing |
| `graders.py` | Deterministic grading: exact match (math), run asserts (code) | plumbing |
| `models.py` | Model client, mock + real, cost accounting | plumbing |
| `run_eval.py` | Batch runner, spend cap, report | plumbing |
| `policies.py` | The five policies + the two verifiers | **yours** |

## The four decisions you own

1. **Model pair** (`models.py`) — price ratio drives the economics
2. **Self-consistency k** (`policies.py`) — failure detection vs cost, linear
3. **Agreement threshold** (`policies.py`) — when to accept the cheap answer
4. **Predictive heuristic** (`policies.py`) — route once, blind, on pre-call features only

Change one, re-run, watch the numbers move. That is how you learn what they do.

## Known state

Pipeline runs end to end in mock mode. Numbers are currently wrong:
- code tasks fail 100% of the time (bug, findable, start at `grade()`)
- cheap and expensive tiers score too close together (mock tuning)

Fix both before switching to real mode.

## Reading list — two hours, three papers, stop

- FrugalGPT, arxiv.org/abs/2305.05176 (read properly)
- RouteLLM, arxiv.org/abs/2406.18665 (skim)
- AutoMix, arxiv.org/abs/2310.12963 (skim)

Their only job: answering "isn't this just FrugalGPT?" with "yes, it's a
replication with 2026 models, and here's what changed."
