"""RouteLLM's pretrained routers, as a policy in this experiment.

STRATEGY_2026-07-29.md 2.2. Replaces the project's weakest link - a hand-tuned
heuristic with r=+0.28 on the code half - with a published learned router,
trained on Chatbot Arena preference data, at zero training cost.

The interesting outcome is the unflattering one. RouteLLM's routers were trained
on human preference between GPT-4-class and Mixtral-class models on open-ended
chat. This applies them to Haiku 4.5 vs Opus 5 on objectively-graded competition
maths and MBPP. That is squarely out of distribution, and if the learned router
loses to `level >= 5`, that is a real finding: preference-trained routers
transfer poorly to objectively-graded reasoning, because preference is not
correctness.


WHICH VARIANT, AND WHY - read this before changing it
-----------------------------------------------------
RouteLLM 0.2.0 ships five routers. Checked by reading the shipped wheel rather
than the README, because the difference matters a lot here:

  router        per-query third-party API call?          weights
  ------------  ---------------------------------------  ------------------------
  bert          NO - local torch forward pass            HF bert_gpt4_augmented
  causal_llm    NO - local, but needs Meta-Llama-3-8B     ~16GB, gated repo
  mf            YES - OpenAI text-embedding-3-small,      HF mf_gpt4_augmented
                on EVERY prompt (matrix_factorization/
                model.py, MFModel.forward)
  sw_ranking    YES - same, plus two arena datasets       HF datasets
  random        NO                                       none

So `bert` is the one to use: a genuine learned router, one checkpoint download,
then entirely local inference. No API key, no per-query spend, no third-party
dependency at serving time. `mf` is the documented-best performer but it cannot
be run without paying OpenAI per prompt, so it is the fallback, not the default.

THE IMPORT TRAP. routellm/routers/routers.py imports
routellm.routers.similarity_weighted.utils, which does `OPENAI_CLIENT = OpenAI()`
at MODULE level. The openai SDK raises OpenAIError when constructed with no
credentials. So importing RouteLLM's router classes AT ALL fails without an
OPENAI_API_KEY, including for the bert router that never uses one. This module
sets a placeholder key before importing on the bert path. No call is ever made
with it - BERTRouter.calculate_strong_win_rate only tokenises and runs a local
forward pass. If you are tempted to remove that, you will spend an evening on a
credentials error from a code path you are not using.


THE SCORE CACHE
---------------
A router score is a pure function of (variant, checkpoint, prompt), so it is
cached exactly like a model response, in cache/routellm_scores.jsonl. This is
what makes the whole thing a one-time cost:

  - on the mf path, each prompt is embedded by OpenAI ONCE, ever;
  - the scores file is small (one float per task) and is COMMITTED, so anyone
    can reproduce the routing decisions with no API key, no HuggingFace access,
    no torch and no GPU. That is a stronger reproducibility property than the
    router itself has.

If every task is already scored, this module never imports routellm, never
touches the network, and never loads a model.


    python3 routellm_router.py --score      # populate the cache (needs the deps)
    python3 routellm_router.py              # show calibration + routing decisions
    python3 run_eval.py                     # picks the policy up automatically
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import models

HERE = Path(__file__).parent
SCORES_PATH = Path(os.environ.get("ROUTELLM_SCORES_PATH", HERE / "cache" / "routellm_scores.jsonl"))

# Preference order. bert first because it needs no key and no per-query spend.
VARIANTS = {
    "bert": {"checkpoint": "routellm/bert_gpt4_augmented", "needs_openai": False},
    "mf": {"checkpoint": "routellm/mf_gpt4_augmented", "needs_openai": True},
}

_scores = None


# ---------------------------------------------------------------------------
# Score cache
# ---------------------------------------------------------------------------

def score_key(variant, checkpoint, prompt):
    blob = json.dumps({"v": variant, "c": checkpoint, "p": prompt},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load():
    global _scores
    if _scores is not None:
        return _scores
    _scores = {}
    if SCORES_PATH.exists():
        with SCORES_PATH.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "key" in rec:
                    _scores[rec["key"]] = rec
    return _scores


def _put(rec):
    _load()[rec["key"]] = rec
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCORES_PATH.open("a", encoding="utf-8", newline="") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cached_variant():
    """Which variant the cached scores came from, or None if there are none.

    If two variants are present the newest wins, but say so - silently mixing
    scores from two routers into one policy would be a fabricated router.
    """
    recs = list(_load().values())
    if not recs:
        return None
    variants = {r["variant"] for r in recs}
    if len(variants) > 1:
        print(f"  !! routellm_router: scores present from MORE THAN ONE variant "
              f"{sorted(variants)}. Delete cache/routellm_scores.jsonl and rescore "
              f"with one. Using the last one seen.", file=sys.stderr)
    return recs[-1]["variant"]


def available(tasks=None):
    """True if every task already has a cached score."""
    variant = cached_variant()
    if variant is None:
        return False
    if tasks is None:
        return True
    ck = VARIANTS[variant]["checkpoint"]
    return all(
        score_key(variant, ck, models.build_prompt(t)) in _load() for t in tasks
    )


def score_of(task, variant=None):
    variant = variant or cached_variant()
    if variant is None:
        raise RuntimeError(
            "routellm_router: no cached scores.\n"
            "  Populate them once with:  python3 routellm_router.py --score\n"
            "  This module never invents a score - a router that falls back to a\n"
            "  coin flip when its model is missing is a coin flip, not a router."
        )
    key = score_key(variant, VARIANTS[variant]["checkpoint"], models.build_prompt(task))
    rec = _load().get(key)
    if rec is None:
        raise KeyError(
            f"routellm_router: {task['id']} has no cached score for variant "
            f"{variant!r}. Re-run: python3 routellm_router.py --score"
        )
    return rec["strong_win_rate"]


# ---------------------------------------------------------------------------
# Scoring (the only part that needs the dependency)
# ---------------------------------------------------------------------------

def _load_router(variant):
    """Instantiate one RouteLLM router. Raises with a usable message if it can't."""
    cfg = VARIANTS[variant]

    if cfg["needs_openai"]:
        # Checked BEFORE anything heavy loads, so a missing key costs a second
        # rather than a model download followed by a credentials error.
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                f"\nSTOPPING: the {variant!r} router embeds every prompt with OpenAI's\n"
                f"  text-embedding-3-small and OPENAI_API_KEY is not set.\n\n"
                f"  Either set it:      export OPENAI_API_KEY=...\n"
                f"  or use the self-contained router, which needs no key at all:\n"
                f"                      python3 routellm_router.py --score --variant bert\n"
            )
    else:
        # See THE IMPORT TRAP above. routellm's package-level import chain
        # constructs an OpenAI client even for routers that never use one.
        os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder-bert-path-makes-no-openai-calls")

    try:
        from routellm.routers.routers import ROUTER_CLS
    except ImportError as e:
        raise SystemExit(
            f"\nSTOPPING: could not import routellm ({e}).\n"
            f"  pip install routellm\n"
            f"  It is deliberately absent from requirements.txt - it pulls torch,\n"
            f"  transformers, datasets and litellm, none of which the experiment needs.\n"
        )

    return ROUTER_CLS[variant](checkpoint_path=cfg["checkpoint"])


def score_tasks(tasks, variant="bert", force=False):
    """Score every task, loading the router only for cache misses."""
    cfg = VARIANTS[variant]
    todo = []
    for t in tasks:
        prompt = models.build_prompt(t)
        key = score_key(variant, cfg["checkpoint"], prompt)
        if force or key not in _load():
            todo.append((t, prompt, key))

    if not todo:
        print(f"all {len(tasks)} tasks already scored by {variant!r}; "
              f"nothing to do, no model loaded, no network call made")
        return

    print(f"scoring {len(todo)}/{len(tasks)} tasks with routellm {variant!r} "
          f"({cfg['checkpoint']})", file=sys.stderr)
    if cfg["needs_openai"]:
        print(f"  this will make {len(todo)} OpenAI text-embedding-3-small calls "
              f"(~$0.00001 each, one time - they are cached)", file=sys.stderr)

    router = _load_router(variant)
    for i, (t, prompt, key) in enumerate(todo, 1):
        rate = float(router.calculate_strong_win_rate(prompt))
        _put({
            "key": key, "task_id": t["id"], "domain": t["domain"],
            "variant": variant, "checkpoint": cfg["checkpoint"],
            "strong_win_rate": rate,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        })
        if i % 20 == 0:
            print(f"  {i}/{len(todo)}", file=sys.stderr)
    print(f"wrote {len(todo)} scores -> {SCORES_PATH}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

# Filled by calibrate(). Per domain, like RANDOM_MATCHED_RATES, so the
# comparison against predictive AND random_matched is cost-matched on both
# halves rather than only in aggregate.
THRESHOLDS = {}


def calibrate(tasks, target_rates):
    """Pick the score threshold that reproduces predictive's expensive-call rate.

    RouteLLM's own threshold is a free parameter, so comparing its default
    against a heuristic tuned to a particular escalation rate would compare
    two different spending levels and call the difference quality. Calibrating
    to a matched rate is what makes the comparison like-for-like, and it is what
    STRATEGY 2.2 specifies.

    Per domain rather than global, because predictive's rate differs by domain
    (math 40%, code 35%) and a single global threshold would let the router
    spend unevenly across the two halves while the heuristic could not.

    Concretely: take the (1 - rate) quantile of the scores in that domain, so
    exactly `rate` of tasks score at or above it. Ties can make the realised
    rate differ slightly from the target; the report prints the realised rate,
    which is the one that matters.
    """
    global THRESHOLDS
    variant = cached_variant()
    out = {}
    for domain, rate in target_rates.items():
        sub = sorted((score_of(t, variant) for t in tasks if t["domain"] == domain),
                     reverse=True)
        if not sub:
            continue
        k = int(round(rate * len(sub)))
        # k = how many tasks should go expensive. The threshold is the score of
        # the k-th highest, so exactly those k are at or above it.
        out[domain] = sub[k - 1] if k > 0 else (sub[0] + 1.0)
    THRESHOLDS = out
    return out


def routes_expensive(task):
    th = THRESHOLDS.get(task["domain"])
    if th is None:
        raise RuntimeError("routellm_router: calibrate() has not been called")
    return score_of(task) >= th


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true", help="populate the score cache")
    ap.add_argument("--variant", default="bert", choices=list(VARIANTS),
                    help="bert (default): local, no key. mf: needs OPENAI_API_KEY.")
    ap.add_argument("--force", action="store_true", help="rescore even if cached")
    args = ap.parse_args()

    import run_eval
    import policies

    tasks = run_eval.load_tasks()

    if args.score:
        score_tasks(tasks, args.variant, args.force)

    variant = cached_variant()
    if variant is None:
        print("\nno cached scores yet. Populate them with:")
        print("  python3 routellm_router.py --score            # bert, no API key")
        print("  python3 routellm_router.py --score --variant mf   # needs OPENAI_API_KEY")
        return

    rates = policies.calibrate_random_rates(tasks)
    th = calibrate(tasks, rates)
    print(f"\nrouter: routellm {variant!r} ({VARIANTS[variant]['checkpoint']})")
    print(f"scored tasks: {sum(1 for t in tasks if available([t]))}/{len(tasks)}")
    print(f"\n{'domain':<8} {'target rate':>12} {'threshold':>11} {'realised':>10} "
          f"{'score min':>10} {'median':>9} {'max':>9}")
    print("-" * 74)
    for domain in sorted(th):
        sub = [t for t in tasks if t["domain"] == domain]
        sc = sorted(score_of(t) for t in sub)
        realised = sum(routes_expensive(t) for t in sub) / len(sub)
        print(f"{domain:<8} {rates[domain]:>12.1%} {th[domain]:>11.4f} "
              f"{realised:>10.1%} {sc[0]:>10.4f} {sc[len(sc) // 2]:>9.4f} {sc[-1]:>9.4f}")

    # How much the learned router and the heuristic actually disagree. If they
    # route the same tasks, any accuracy difference is noise.
    agree = sum(1 for t in tasks if routes_expensive(t) == policies.predict_is_hard(t))
    print(f"\nagreement with predict_is_hard: {agree}/{len(tasks)} = {agree / len(tasks):.1%}")
    print("  (both send the same fraction of traffic expensive by construction,")
    print("   so this is purely whether they pick the SAME tasks)")


if __name__ == "__main__":
    main()
