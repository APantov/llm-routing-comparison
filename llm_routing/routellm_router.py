"""RouteLLM's pretrained routers, as a policy in this experiment.

One of the two predictive routers this repository compares against cascading -
the other being `llm_router`, which asks the cheap model. This one is a published
learned router, trained on Chatbot Arena preference data, at zero training cost
and with no per-query API call on the bert path.

It replaced a hand-written heuristic that was deleted on 8 August 2026 for being
a constant on 60% of the task set (policies.py DECISION #4).

The interesting outcome is the unflattering one. RouteLLM's routers were trained
on human preference between GPT-4-class and Mixtral-class models on open-ended
chat. This applies them to DeepSeek v4-flash vs Opus 5 on objectively-graded
competition maths and MBPP+. That is squarely out of distribution, and the
prediction has now been checked: across all 417 tasks the router's scores span
[0.499, 0.899] with a median of 0.790, and exactly ONE task falls below 0.5. So
on 416 of 417 it judges the strong model favoured, and at the semantically
natural threshold it degenerates into always_expensive. Preference-trained routers transfer poorly to objectively
graded reasoning, because preference is not correctness - and the failure is
visible in the score distribution before any accuracy is measured. See
DECISION #8b for what threshold is used instead, and why.


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
forward pass. Removing that line produces a credentials error from a code path
that makes no API calls, which is a confusing hour to spend.


THE SCORE CACHE
---------------
A router score is a pure function of (variant, checkpoint, prompt), so it is
cached exactly like a model response, in cache/routellm_scores.jsonl. This is
what makes the whole thing a one-time cost:

  - on the mf path, each prompt is embedded by OpenAI ONCE, ever;
  - the scores file is small, one float per task, and it is committed to the
    repo, so the routing decisions reproduce with no API key, no HuggingFace
    access, no torch and no GPU. That is a stronger reproducibility property
    than the router itself has.

If every task is already scored, this module never imports routellm, never
touches the network, and never loads a model.


    python -m llm_routing.routellm_router --score      # populate the cache (needs the deps)
    python -m llm_routing.routellm_router              # show calibration + routing decisions
    python -m llm_routing.run_eval                     # picks the policy up automatically
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from llm_routing import models
from llm_routing import paths

SCORES_PATH = Path(os.environ.get("ROUTELLM_SCORES_PATH", paths.CACHE / "routellm_scores.jsonl"))

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
            "  Populate them once with:  python -m llm_routing.routellm_router --score\n"
            "  This module never invents a score - a router that falls back to a\n"
            "  coin flip when its model is missing is a coin flip, not a router."
        )
    key = score_key(variant, VARIANTS[variant]["checkpoint"], models.build_prompt(task))
    rec = _load().get(key)
    if rec is None:
        raise KeyError(
            f"routellm_router: {task['id']} has no cached score for variant "
            f"{variant!r}. Re-run: python -m llm_routing.routellm_router --score"
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
                f"                      python -m llm_routing.routellm_router --score --variant bert\n"
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
            f"  It is deliberately an optional extra, not a default - it pulls torch,\n"
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

# Filled by use_fixed_threshold() for the headline run, or by calibrate() for
# frontier.py's rate sweep. Per domain, because a single global threshold would
# let the router spend unevenly across the two halves.
THRESHOLDS = {}

# ---------------------------------------------------------------------------
# DECISION #8b: the operating threshold. FIXED, not calibrated. 8 August 2026.
#
# This used to be calibrated to reproduce `predictive`'s expensive-call rate, so
# the two routers were cost-matched by construction. `predictive` was deleted
# (policies.py DECISION #4), and re-anchoring to another policy would just move
# the dependency: a router whose spending level is defined by a different
# router's spending level cannot be read on its own.
#
# So the threshold is now a declared constant. It is ARBITRARY in the sense that
# it was not tuned for accuracy, and it is not derived from anything else in this
# repository.
#
# WHY NOT 0.5. `calculate_strong_win_rate` returns P(the strong model wins), so
# 0.5 is the semantically natural cut: escalate when the strong model is more
# likely than not to win. On this task set it routes all but one task expensive.
# Measured across all 417 tasks the range is [0.499, 0.899], median 0.790, and
# exactly ONE task scores below 0.5 (at 0.499). The router effectively never
# says the weak model is favoured, so at 0.5 this is always_expensive with an
# extra step.
#
# The figures above were [0.509, 0.898] over 100 tasks before the 6 August 2026
# rebuild grew the set to 417. Re-measured rather than carried over: the old
# range supported the strictly stronger claim that no task ever falls below
# 0.5, and that claim is now false by one task.
#
# That is a finding rather than an inconvenience, and it is the out-of-
# distribution behaviour this module's docstring predicted: bert_gpt4_augmented
# was trained on human preference between chat models, and it is being asked
# about competition maths and MBPP+. Preference is not correctness, and a model
# asked "which answer would a human prefer" on a task it cannot judge says
# "the big one" every time.
#
# 0.80 is a round number inside the observed range. It splits the set 42/58
# (math 50%, code 30%) rather than degenerating - which is the whole point, since
# a policy that routes 100% one way is `always_expensive` wearing a router's
# name, and that is exactly what `predictive` was deleted for.
#
# The consequence to keep in view: routellm is no longer cost-matched to any
# other policy, so an accuracy difference against one conflates routing skill
# with spending level. run_eval.routing_skill handles this by computing a null at
# each policy's OWN cost, and frontier.py sweeps the rate across its whole range,
# which is the comparison that never depended on a matched operating point.
# ---------------------------------------------------------------------------
FIXED_THRESHOLD = 0.80

# Whether calibrate() has run against the current task set.
#
# This exists because two different questions were being asked with one answer.
# run_eval calls available(ALL tasks) to decide whether to calibrate, and
# available([one task]) to decide whether the policy runs on that task. Those
# disagree the moment SOME tasks have cached scores and others do not - which is
# exactly what happens after a task-set swap, since scores are keyed on the prompt.
# The result was routellm running uncalibrated on the tasks it did have scores
# for, and raising from routes_expensive mid-run.
CALIBRATED = False


def use_fixed_threshold(tasks=None, value=None):
    """Set the operating threshold to the declared constant. See DECISION #8b.

    This is what the headline run uses. `tasks` is optional and only narrows
    which domains get an entry; without it every domain this repo knows about is
    covered, because routes_expensive raises on a domain it has no threshold for.

    IT MUST SET `CALIBRATED`. run_eval gates the policy on that flag, and it was
    previously assigned only inside calibrate(). Setting THRESHOLDS from outside
    without it makes routellm sit out the entire run in silence - no error, no
    dropped-policy line, just a missing row. That is why this is a function
    rather than an assignment at the call site.
    """
    global THRESHOLDS, CALIBRATED
    th = FIXED_THRESHOLD if value is None else value
    domains = ({t["domain"] for t in tasks} if tasks else {"math", "code"})
    THRESHOLDS = {d: th for d in domains}
    CALIBRATED = bool(THRESHOLDS)
    return THRESHOLDS


def calibrate(tasks, target_rates):
    """Pick the score threshold that reproduces a target expensive-call rate.

    NOT used by the headline run any more - see use_fixed_threshold() and
    DECISION #8b. This is the rate-parameterised form frontier.py needs: it
    sweeps `target_rates` from 0 to 1 to trace the router's whole cost-quality
    curve, which is the comparison that never depended on matching some other
    policy's operating point.

    Per domain rather than global, so the router cannot spend unevenly across
    the two halves at a given point on the sweep.

    Concretely: take the (1 - rate) quantile of the scores in that domain, so
    exactly `rate` of tasks score at or above it. Ties can make the realised
    rate differ slightly from the target; the report prints the realised rate,
    which is the one that matters.
    """
    global THRESHOLDS, CALIBRATED
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
    CALIBRATED = bool(out)
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

    from llm_routing import run_eval

    tasks = run_eval.load_tasks()

    if args.score:
        score_tasks(tasks, args.variant, args.force)

    variant = cached_variant()
    if variant is None:
        print("\nno cached scores yet. Populate them with:")
        print("  python -m llm_routing.routellm_router --score            # bert, no API key")
        print("  python -m llm_routing.routellm_router --score --variant mf   # needs OPENAI_API_KEY")
        return

    th = use_fixed_threshold(tasks)
    print(f"\nrouter: routellm {variant!r} ({VARIANTS[variant]['checkpoint']})")
    print(f"scored tasks: {sum(1 for t in tasks if available([t]))}/{len(tasks)}")
    print(f"threshold: {FIXED_THRESHOLD} (fixed, not calibrated - DECISION #8b)")
    print(f"\n{'domain':<8} {'threshold':>11} {'realised':>10} "
          f"{'score min':>10} {'median':>9} {'max':>9}")
    print("-" * 62)
    for domain in sorted(th):
        sub = [t for t in tasks if t["domain"] == domain]
        sc = sorted(score_of(t) for t in sub)
        realised = sum(routes_expensive(t) for t in sub) / len(sub)
        print(f"{domain:<8} {th[domain]:>11.4f} "
              f"{realised:>10.1%} {sc[0]:>10.4f} {sc[len(sc) // 2]:>9.4f} {sc[-1]:>9.4f}")

    # The distribution matters more than the threshold, because it is what makes
    # the natural cut unusable. A preference-trained router that never scores
    # below 0.5 cannot be thresholded at 0.5.
    allsc = sorted(score_of(t) for t in tasks)
    below = sum(s < 0.5 for s in allsc)
    print(f"\nscores below 0.5 (the weak model favoured): {below}/{len(allsc)}")
    if below == 0:
        print("  none. At the semantically natural threshold this router is")
        print("  always_expensive. See DECISION #8b - preference is not correctness.")

    # What the threshold is sitting on. A fixed cut is only meaningful if it
    # lands somewhere the scores actually are, so print how far it is from the
    # nearest score and how fast the rate moves around it. A threshold in a gap
    # is brittle: a rescore that shifts scores by 0.01 would move the rate a lot.
    print(f"\nsensitivity of the {FIXED_THRESHOLD} cut:")
    for delta in (-0.05, -0.02, 0.0, 0.02, 0.05):
        th = FIXED_THRESHOLD + delta
        rate = sum(score_of(t) >= th for t in tasks) / len(tasks)
        mark = "  <- in use" if delta == 0.0 else ""
        print(f"  threshold {th:.2f}  ->  {rate:>5.1%} expensive{mark}")


if __name__ == "__main__":
    main()
