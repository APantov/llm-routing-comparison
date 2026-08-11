<!--
  Add a CI badge once this repository has a GitHub remote — replace OWNER:
  [![CI](https://github.com/OWNER/llm-routing-comparison/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/llm-routing-comparison/actions/workflows/ci.yml)
-->

# LLM Routing: a measured benchmark, and the router it argues for

**A cost-aware LLM routing service (LangGraph + MCP), and the 417-task benchmark
that decides its policy.**

Answer at the cheapest model that can be *verified* to have got it right, and
escalate only when verification fails. Whether that beats simply paying for the
best model is not a matter of opinion — it depends on the models you are
choosing between, and this repository measures it on three real ladders.

| | |
|---|---|
| **What runs** | A LangGraph state machine — `classify → answer → verify → escalate ⟲` — with human-in-the-loop approval *inside* the escalation loop and checkpointed resume, served over MCP with four tools and four resources. |
| **What decides its policy** | 417 tasks (MBPP+ code, MATH-500 level 5), 10 policies, 3 price ladders, **all measured on real models**: cost–quality frontiers, exact McNemar, paired bootstrap. |
| **Built with** | Python 3.10–3.13 · LangGraph · MCP · Anthropic + DeepSeek APIs · pytest (205 tests) · GitHub Actions. The research core is **pure standard library** — no dependency can change a benchmark number. |
| **Evidence** | **5,075 real model responses**, committed. **$8.51** spent. Every figure and table regenerates offline, with no API key, for **$0.00**. |

**Start with [STATUS.md](STATUS.md)** — it is the one file kept current, and it
states the findings, the costs and the limitations in that order.

---

## Quickstart

```bash
pip install -e ".[agent]"
python -m llm_routing.build_taskset
python -m router_agent.cli --demo      # real model output, no API key, $0.00
```

No account, no key, no money: the responses were bought once and committed, so
the router replays genuine model output rather than simulating it.

```
2. The cascade's cost - it disagreed with itself, so it paid twice
--------------------------------------------------------------------------
  query: Let N be the number of ordered triples (a,b,c) of positive integers
         with a*b*c = 2310 and a<b<c. Compute N.

    classify  domain=math, start=cheap, verifier=self_consistency
    answer    cheap (deepseek-v4-flash) answered
    verify    self_consistency -> REJECT, confidence=0.75
    escalate  cheap -> expensive
    answer    expensive (claude-opus-5) answered
    verify    self_consistency -> REJECT
    finalize  done: exhausted_ladder

    answered by  claude-opus-5
    verified     False  (self_consistency)
    cost         $0.013450   backend $0.000000
```

That is the expensive case on purpose. The cheap model agreed with itself only
75% of the time, so the cascade escalated — and **paid for both rungs**. Whether
that is a good trade is exactly what the price ratio decides, and what the rest
of this repository measures. `python scripts/demo.py` prints all three canonical
traces.

## The finding

Pre-registered comparison against simply always paying for the best model.
Exact McNemar over paired outcomes, n=209 held-out tasks per ladder.

| ladder | rungs | cascade | always-expensive | Δ acc | p | Δ cost/task |
|---|---|---|---|---|---|---|
| `wide` | v4-flash → Opus 5 | **95.7%** | 92.3% | +3.3% | **0.039** | **−$0.00307** |
| `claude` | Haiku 4.5 → Sonnet 5 → Opus 5 | **96.7%** | 92.3% | +4.3% | **0.012** | +$0.00097 |
| `deepseek` | v4-flash → v4-pro | 86.6% | 83.7% | +2.9% | 0.070 | −$0.00000 |

**On `wide` the cascade is more accurate *and* four times cheaper. On `claude`
it buys the accuracy at a premium** — verification is not free when the cheap
rung is Haiku and the maths half draws five samples from it. Do not quote
"cheaper and better" without naming the ladder.

![Cost–quality frontier on the wide ladder](figures/frontier.wide.svg)

**Predictive routing does not beat a coin flip — six comparisons out of six.**
`random_matched` flips a coin at the learned router's own escalation rate, which
holds spend roughly fixed and isolates skill:

| comparison | `wide` | `claude` | `deepseek` |
|---|---|---|---|
| `llm_router` vs `random_matched` | p=0.167 | p=0.549 | p=1.000 |
| `routellm` vs `random_matched` | p=0.454 | p=1.000 | p=0.727 |
| `cascade` vs `llm_router` | **p=0.003** | **p=0.006** | **p=0.012** |

Neither an LLM-as-router nor RouteLLM's pretrained BERT beats a cost-matched
coin flip on any ladder, while the cascade beats both on every ladder. **The
distinction is when the decision is made**: a predictive router commits before
seeing an attempt, a cascade decides after verifying one.

The full results — the per-policy error attribution, what the routing signal is
actually worth once decoding noise is priced out, and the confound three ladders
cannot separate — are in [STATUS.md](STATUS.md).

## Layout

```
.
├── llm_routing/      the experiment — 16 modules, standard library only
├── router_agent/     the product — LangGraph cascade + MCP server
├── scripts/          operator tools: paid buys, guards, the demo, the driver
├── tests/            pytest, 205 tests
├── docs/             method, architecture, walkthrough, datasets, open issues
├── data/             MATH-500 and MBPP+ sources, and the built task set
├── cache/            5,075 real model responses — what makes replay free
├── runs/             every derived artefact: results, frontiers, scorecards
├── figures/          SVGs, written from the standard library
└── archive/          real data superseded by the 6 Aug rebuild, kept not deleted
```

The two halves share one model client, one price table and one response cache,
which is what makes a dollar figure from the router mean the same thing as a
dollar figure in the tables. The arrow runs one way — `router_agent` imports
`llm_routing`, never the reverse — and CI has a job whose only purpose is to
keep it that way. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Reproducing everything

Mock mode needs nothing installed: pure standard library, offline, and
byte-deterministic down to the figures.

```bash
python -m llm_routing.build_taskset    # data/taskset.jsonl, 417 tasks
python -m llm_routing.sanity_check     # regression gate on both graders
python -m llm_routing.splits           # the calibration / evaluation split
```

Replay mode reruns the published analysis against the committed responses. No
key, no network, $0.00:

```bash
ROUTER_MODE=replay python scripts/run_all_ladders.py    # all 3 ladders
python -m llm_routing.stats     --results runs/results.wide.jsonl
python -m llm_routing.scorecard --results runs/results.wide.jsonl
```

The published figures were produced by deleting every derived artefact and
regenerating all three ladders from the cache. 0 calls reached a backend; 0 rows
are simulated.

Real mode needs a key and spends money. `pip install -e ".[real]"`, copy
`.env.example` to `.env`, and read [docs/METHOD.md](docs/METHOD.md#running-it-for-real)
first — it gives the order to buy things in, starting with the two-arm probe
that costs a hundredth of a full run and decides whether the rest is worth
doing.

## Documentation

| file | read it when |
|---|---|
| [STATUS.md](STATUS.md) | you want the current findings, costs and limitations. **Start here.** |
| [docs/METHOD.md](docs/METHOD.md) | you want the method: task set, ladders, policies, verifiers, the degradation experiment, and how to run it for real |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | you want to know how the benchmark and the serving layer fit together |
| [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) | you want to understand the code, one real task traced through every module |
| [docs/EXPLAINED.md](docs/EXPLAINED.md) | you want the plain-language version, no familiarity with routing assumed |
| [docs/NOTES.md](docs/NOTES.md) | you want the honest list of what is wrong, unresolved, or would weaken the headline |
| [docs/DATASETS.md](docs/DATASETS.md) | you are choosing a benchmark, or want to know why MBPP+ and MATH-500 level 5 |

## What this is not

The sharpest limitation, stated here rather than buried: **the verifier that
produces the signal is not the verifier that ships.** The code half is graded by
executing the tests MBPP+ supplies, which a deployed router does not have. The
maths half — where a self-consistency verifier *would* deploy — carries far less
routing signal. Every number here is collected under a verifier the product
cannot have, and [docs/NOTES.md](docs/NOTES.md) tracks that alongside seven
others in the same spirit.

## Where this sits in the literature

FrugalGPT ([2305.05176](https://arxiv.org/abs/2305.05176)) is the cascade
baseline and this is partly a replication of it with 2026 models. What is added
is the manipulation: FrugalGPT and AutoMix take their verifier as given, and
Dekoninck et al. ([2410.10347](https://arxiv.org/abs/2410.10347)) identify
quality-estimator accuracy as the factor that decides whether any of this works
but test it by injecting synthetic noise. This repository instead **degrades a
real verifier by a controlled amount on objectively-graded tasks**, holding the
domain, the models, the prompts and the grader fixed. Full bibliography and what
each paper contributes: [docs/METHOD.md](docs/METHOD.md#where-this-sits-in-the-literature).

## License

MIT — see [LICENSE](LICENSE).
