# LLM Routing: a measured benchmark, and the router it argues for

[![CI](https://github.com/APantov/llm-routing-comparison/actions/workflows/ci.yml/badge.svg)](https://github.com/APantov/llm-routing-comparison/actions/workflows/ci.yml)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A cost-aware LLM routing service (LangGraph + MCP), and the 417-task benchmark
that decides its policy.**

Answer at the cheapest model that can be *verified* to have got it right, and
escalate only when verification fails. Whether that beats simply paying for the
best model is not a matter of opinion — it depends on the models you are
choosing between, and this repository measures it on three real ladders.

> **The finding, in one sentence: cascade when the top rung is genuinely better
> and verification is cheap — the price ratio alone gets two of three ladders
> backwards.** The shipped router computes that verdict per ladder from the
> committed measurements, and refuses to answer for a ladder it has no data on.

| | |
|---|---|
| **What runs** | A LangGraph state machine — `classify → answer → verify → escalate ⟲` — with human-in-the-loop approval *inside* the escalation loop and checkpointed resume, served over MCP with four tools and four resources. |
| **What decides its policy** | 417 tasks (MBPP+ code, MATH-500 level 5), 9 policies, 3 price ladders, **all measured on real models**: cost–quality frontiers, exact McNemar, paired bootstrap. |
| **Built with** | Python 3.10–3.13 · LangGraph · MCP · Anthropic + DeepSeek APIs · pytest (210 tests) · GitHub Actions. The research core is **pure standard library** — no dependency can change a benchmark number. |
| **Evidence** | **5,075 real model responses**, committed. **$8.51** spent. Every figure and table regenerates offline, with no API key, for **$0.00**. |

---

## Quickstart

```bash
pip install -e ".[agent]"
python -m llm_routing.build_taskset
python -m router_agent.cli --demo      # real model output, no API key, $0.00
```

No account, no key, no money: the responses were bought once and committed, so
the router replays genuine model output rather than simulating it.

`python scripts/demo.py` prints the three canonical traces — the cascade winning
at the cheap rung, the cascade paying twice, and the code case where
verification is exact and free. The first of them:

```
1. The cascade's win - verified at the cheap rung
--------------------------------------------------------------------------
  query: Let f(x) = x^3 - 3x + 1. Find the sum of the squares of all real roots.

    classify  domain=math, start=cheap, verifier=self_consistency
    answer    cheap (deepseek-v4-flash) answered
    verify    self_consistency -> ACCEPT, confidence=1.00
    finalize  done: verified

    answered by  deepseek-v4-flash
    verified     True  (self_consistency)
    cost         $0.000315   backend $0.000000
```

Three independent draws from DeepSeek all gave the right answer, so the cascade
accepted and never called Opus 5 — roughly **27x cheaper** than routing straight
to the top rung. When verification *fails* the cascade escalates and pays for
both rungs. Whether that trade is worth making is what the rest of this
repository measures.

## The finding

Pre-registered comparison against simply always paying for the best model.
Exact McNemar over paired outcomes, n=209 held-out tasks per ladder.

| ladder | rungs | cascade | always-expensive | Δ acc | p | Δ cost/task |
|---|---|---|---|---|---|---|
| `wide` | v4-flash → Opus 5 | **95.7%** | 92.3% | +3.3% | **0.039** | **−$0.00307** |
| `claude` | Haiku 4.5 → Sonnet 5 → Opus 5 | **96.7%** | 92.3% | +4.3% | **0.012** | +$0.00097 |
| `deepseek` | v4-flash → v4-pro | 86.6% | 83.7% | +2.9% | 0.070 | −$0.00000 |

![Cascade against always-expensive on all three ladders](figures/ladders.svg)

**On `wide` the cascade is more accurate *and* four times cheaper. On `claude`
it buys the accuracy at a premium** — verification is not free when the cheap
rung is Haiku and the maths half draws five samples from it. The ladder decides
the sign, which is why the router below reads it rather than assuming it.

Three further results, each with its numbers and its caveats in
[docs/RESULTS.md](docs/RESULTS.md):

- **Predictive routing does not beat a coin flip — six comparisons out of six.**
  Neither an LLM-as-router nor RouteLLM's pretrained BERT beats a cost-matched
  random null on any ladder, while the cascade beats both on every one. The
  distinction is *when* the decision is made: a predictive router commits before
  seeing an attempt, a cascade decides after verifying one.
  → [§2.3](docs/RESULTS.md#23-predictive-routing-does-not-beat-a-coin-flip--six-of-six)
- **Accuracy hides what a router actually did.** Two policies can reach the same
  accuracy by escalating the right ten tasks or by escalating everything.
  `always_expensive` escalates 201 tasks to buy 27 rescues, burning **$0.71** on
  escalations that could not improve the answer; `cascade` gets 24 of those
  rescues and wastes **$0.084**.
  → [§3](docs/RESULTS.md#3-what-each-policy-got-right-and-wrong)
- **Every policy is a curve, not a point.** Each router here has a knob trading
  accuracy for money, so comparing two at one setting each lets whoever set the
  knobs pick the winner. `frontier.py` sweeps each knob across its whole range
  and compares the resulting curves.
  → [§2.5](docs/RESULTS.md#25-the-price-ratio-does-not-decide-whether-to-cascade)

## The benchmark ships its own conclusion

A benchmark that ends in a table leaves the reader to apply it. This one ends in
a function. `findings.ratio_verdict(ladder)` reads that ladder's committed
frontier and returns the verdict for it. Same query, two ladders, opposite
answers:

```bash
$ llm-router --estimate "prove that sqrt(2) is irrational" --ladder wide
  recommended policy   cascade        (measured on the wide ladder)
  cascade vs always-best, at matched accuracy   -83.1%

$ llm-router --estimate "prove that sqrt(2) is irrational" --ladder claude
  recommended policy   route        (measured on the claude ladder)
  cascade vs always-best, at matched accuracy   +11.7%
```

That flip is the finding, and the router reads it rather than assuming it — and
**declines for a ladder it has no data on**. The CLI, the MCP `explain_routing`
tool and the `RouterConfig` defaults all call the same function, so changing what
the benchmark measured changes what the router recommends. There is no constant
to drift out of date — there used to be one, and two of its three verdicts were
backwards.

## Layout

```
llm_routing/    the experiment — 16 modules, standard library only
router_agent/   the product — LangGraph cascade + MCP server
cache/          5,075 real model responses — what makes replay free
runs/           every derived artefact: results, frontiers, scorecards
data/  docs/  figures/  scripts/  tests/  archive/
```

The two halves share one model client, one price table and one response cache,
which is what makes a dollar figure from the router mean the same thing as a
dollar figure in the tables. The arrow runs one way — `router_agent` imports
`llm_routing`, never the reverse — and CI has a job whose only purpose is to
keep it that way. Module by module:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Reproducing everything

Replay mode reruns the published analysis against the committed responses — no
key, no network, $0.00:

```bash
ROUTER_MODE=replay python scripts/run_all_ladders.py --ladders wide   # ~18 min
```

Drop `--ladders wide` for all three, about 55 minutes. The published figures
were produced exactly that way after deleting every derived artefact: 0 calls
reached a backend, 0 rows are simulated, and every regenerated file came back
byte-identical to the committed one.

Mock mode needs nothing installed at all — pure standard library, offline,
byte-deterministic down to the figures. Real mode needs a key and spends money.
Both, plus every analysis entry point and the order to buy data in, are in
[docs/METHOD.md](docs/METHOD.md#running-it).

## Documentation

| file | read it when |
|---|---|
| [docs/EXPLAINED.md](docs/EXPLAINED.md) | you want the plain-language version, no familiarity with routing assumed — **start here** |
| [docs/RESULTS.md](docs/RESULTS.md) | you want every finding, with the numbers and what they cost |
| [docs/METHOD.md](docs/METHOD.md) | you want the method: task set, why these datasets, ladders, policies, verifiers, the degradation experiment, how to run it for real, and the bugs this project found in itself |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | you want to know how the benchmark and the serving layer fit together, module by module |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | you want what bounds the claims |

## What bounds the claims, and what is new here

Stated on the front page rather than buried: **the verifier that produces the
signal is not the verifier that ships.** The code half is graded by executing
the tests MBPP+ supplies, and a deployed router does not have them.

That gap is priced rather than noted, and pricing it is what this repository
adds to the literature. FrugalGPT ([2305.05176](https://arxiv.org/abs/2305.05176))
is the cascade baseline, and it and AutoMix take their verifier as given;
Dekoninck et al. ([2410.10347](https://arxiv.org/abs/2410.10347)) identify
quality-estimator accuracy as the factor deciding whether any of this works, but
test it by injecting synthetic noise. Here `sweep_degraded.py` instead
**degrades a real verifier by a controlled amount on objectively-graded tasks**,
holding domain, models, prompts and grader fixed — so shipping a proxy verifier
is a move along a measured curve rather than a step into the unknown.

Every other bound is stated once in
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) with what would settle it, and the
full bibliography is in
[docs/METHOD.md](docs/METHOD.md#where-this-sits-in-the-literature).

## License

MIT — see [LICENSE](LICENSE).
