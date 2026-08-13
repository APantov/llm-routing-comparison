"""
Batch runner. Runs every policy over every task, writes results, prints a report.

Usage:
    python -m llm_routing.run_eval                                # mock mode, no spend
    python -m llm_routing.run_eval --limit 10                      # first 10 tasks only
    python -m llm_routing.run_eval --domain math                   # one domain
    ROUTER_MODE=real   python -m llm_routing.run_eval --limit 10   # 10-task pilot, real calls
    ROUTER_MODE=replay python -m llm_routing.run_eval              # replay a paid run, free

Start in mock mode. Get the whole pipeline working and the report printing, then
switch to real. Debugging a broken pipeline while paying per call is miserable.

Output is results.jsonl, one row per (task, policy), with enough provenance on
each row to interpret it without an external note.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from llm_routing import models
from llm_routing import paths
from llm_routing import policies
from llm_routing import response_cache
from llm_routing import splits
from llm_routing.policies import POLICIES, POLICY_DOMAINS

# One file per ladder, always. There is no unsuffixed `results.jsonl`:
# it used to be a copy of whichever ladder ran last, which is a second
# source of truth that can silently disagree with the first.
RESULTS = paths.RUNS / f"results.{models.LADDER}.jsonl"

# Hard spend cap, enforced in code rather than by willpower.
#
# It MOVED on 9 August 2026, from a check in this module's policy loop to
# `models.call`, next to the one line that can charge a card. The old placement
# only covered the loop, so everything spent before it - the llm_router routing
# pre-pass, estimator fitting on the calibration half - was outside the cap.
#
# Re-exported under the old name because this is where a reader looks for it,
# and because the README and docs/RESULTS.md point at `run_eval.MAX_SPEND_USD`.
#
# PER RUN, not per project: a runaway guard, sitting above the largest planned
# run and far below the funded card, so a bug costs one run rather than the
# budget. $20 -> $3 on 8 August, $3 -> $5 on 9 August. The $5 is set against
# docs/RESULTS.md section 4 (what it cost), whose largest single buy is E at ~$2.64 and whose whole
# sequence is ~$4.0; $3 would have bound partway through E, which is the worst
# place for a cap to bind - not a bug, just a plan the guard had not been told
# about.
#
# Override per run rather than editing anything - a cap that is routinely
# edited is not a cap:
#
#     ROUTER_MAX_SPEND_USD=8 ROUTER_MODE=real python -m llm_routing.run_eval
MAX_SPEND_USD = models.MAX_SPEND_USD

# The EXCEPTION is deliberately not aliased here. `models` is reloaded by the
# test suite's `use_ladder` fixture, which rebuilds the class object, and an
# alias captured at import would then name a class that nothing raises any more
# - so the handler at the bottom of this file would silently stop catching. It
# reaches through `models.` at except time instead. The float above is safe to
# alias because it is a value, and a reload reads the same environment variable.

# ---------------------------------------------------------------------------
# The pilot gate: the band the cheap-model failure rate has to land in for the
# task set to be worth running at all.
#
# Below the floor, the cheap model is nearly always right and a cascade has
# almost nothing to route, so every policy collapses towards always_cheap. Above
# the ceiling, the cheap model is nearly always wrong, the cascade escalates
# everything, and every policy collapses towards always_expensive. Routing
# decisions only carry information in between.
# ---------------------------------------------------------------------------
FAILURE_RATE_FLOOR = 0.20
FAILURE_RATE_CEILING = 0.55


def write_jsonl(path, rows):
    """Write JSONL with LF endings on every platform.

    Not cosmetic. Without newline="", Python translates \\n to \\r\\n on Windows,
    so the same code produces byte-different artefacts on different machines and
    no hash-based regression gate is possible.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


MOCK_BANNER = """\
================================================================
  MOCK MODE - these numbers are SIMULATED, not measured.

  Model replies are FABRICATED from answers already stored in the
  taskset. Accuracy below restates models.MOCK_SKILL; it does not
  measure any model. Costs are modelled from synthetic token
  counts - nothing was spent and no network call was made.

  For a real result:  ROUTER_MODE=real python -m llm_routing.run_eval
================================================================"""

REAL_BANNER = """\
================================================================
  REAL MODE - live API calls, real money, real numbers.
  Every response is written to cache/raw_calls.<ladder>.jsonl
  as it arrives, so this run replays free forever after.
================================================================"""

REPLAY_REAL_BANNER = """\
================================================================
  REPLAY MODE - served from this ladder's real response cache.
  No network call. No spend. These are the SAME responses the
  paid run received, so the numbers are the paid run's numbers.
================================================================"""

REPLAY_MOCK_BANNER = """\
================================================================
  REPLAY MODE, BUT THE CACHE IS SIMULATED.

  This ladder's real cache holds no responses, so the replay
  is being served from the MOCK cache. The numbers below
  are FABRICATED, exactly as in mock mode. This is useful for
  testing the replay path; it is not a result.
================================================================"""


def banner():
    if models.MODE != "replay":
        return MOCK_BANNER if models.MODE == "mock" else REAL_BANNER
    # Say which it is. A replay banner that claims real responses while serving
    # simulated ones is the single most dangerous label in this repo.
    return REPLAY_REAL_BANNER if response_cache.REAL_PATH.exists() else REPLAY_MOCK_BANNER


def tag():
    """Short mode marker, printed above EVERY table rather than once at the top.

    A screenshot is usually a crop of one table, and a crop that loses the banner
    is exactly how a simulated number ends up quoted as if it were measured.
    """
    if models.MODE == "replay":
        real, mock = models.call_stats["served_real"], models.call_stats["served_mock"]
        if mock and real:
            # The dangerous case, and the one that has actually happened. Neither
            # of the two clean labels is true, so print neither.
            return (f"### REPLAY, {mock} OF {real + mock} RESPONSES FABRICATED "
                    f"- CONTAMINATED, NOT A RESULT ###")
        if mock:
            return "### REPLAY OF A MOCK CACHE - SIMULATED, NOT MEASURED ###"
        if real:
            return "### REPLAY MODE - cached responses from a real run ###"
        # Nothing served yet: tag() was called before the run. Fall back to the
        # file-existence guess, which is all that is knowable at that point.
        return ("### REPLAY MODE - cached responses from a real run ###"
                if response_cache.REAL_PATH.exists()
                else "### REPLAY OF A MOCK CACHE - SIMULATED, NOT MEASURED ###")
    return ("### MOCK MODE - SIMULATED, NOT MEASURED ###" if models.MODE == "mock"
            else "### REAL MODE - live API calls ###")


def credentials_line():
    """One line saying where the keys came from, and whether they are there.

    Printed in the header of every run rather than discovered at the first API
    call, because the failure it catches is silent until then: a missing or
    deleted `.env` looks exactly like a working setup right up to the moment
    money would have been spent.

    Key NAMES only, never values, and never a prefix of a value. A log line is
    the easiest place in a project to leak a secret.
    """
    from pathlib import Path as _P

    env_file = _P(models.__file__).parent / ".env"
    where = f"{env_file.name} loaded" if env_file.exists() else f"no {env_file.name}"
    if models.DOTENV_LOADED:
        where += f" ({', '.join(models.DOTENV_LOADED)})"

    needed = sorted({models.PROVIDERS[models.MODELS[t]["provider"]]["key_env"]
                     for t in models.TIERS})
    have = [k for k in needed if os.environ.get(k)]
    missing = [k for k in needed if not os.environ.get(k)]

    status = f"keys: {where}; "
    status += f"set: {', '.join(have) or 'none'}"
    if missing:
        status += f"; MISSING: {', '.join(missing)}"
        if models.MODE == "real":
            status += "  <- real mode will fail on the first call"
        else:
            status += "  (fine for mock/replay)"
    return status


def provenance(simulated=None):
    """Everything needed to interpret a row, carried by the row itself.

    Stamped per row rather than in a sidecar header, so that a slice of
    results.jsonl - one policy, one domain, one k - is still self-describing. The
    parameters are here specifically so a sweep over k or the agreement threshold
    produces a file that can be grouped by, rather than a pile of runs whose
    settings have to be remembered.

    Pass `simulated` when the caller knows what actually served the row - see
    run(), which measures it per row from models.call_stats. The fallback below
    is a guess from the run mode, and on 7 August 2026 it guessed wrong for 370
    rows: it asks whether the real cache FILE exists, which says nothing about
    whether any given response came out of it.
    """
    return {
        "mode": models.MODE,
        # The single most important bit on the row. `mode` alone is ambiguous: a
        # replay is real if the cache was populated by a paid run and fabricated
        # if it was populated by the mock, and those two look identical in every
        # other field. A reader should never have to infer which one this is.
        "simulated": simulated if simulated is not None else (
            models.MODE == "mock" or (
                models.MODE == "replay" and not response_cache.REAL_PATH.exists()
            )
        ),
        "mock_seed": models.MOCK_SEED if models.MODE == "mock" else None,
        "k": policies.SELF_CONSISTENCY_K,
        "agreement_threshold": policies.AGREEMENT_THRESHOLD,
        # The manipulated variable, stamped on every row so the degradation sweep
        # produces one file that can be grouped by corruption level.
        "verifier_corruption": policies.VERIFIER_CORRUPTION,
        # Which draw the random baselines took. Meaningless for every other
        # policy, and recorded anyway, because a row that needs an external note
        # to interpret is a row that will eventually be misread.
        "random_seed": policies.RANDOM_SEED,
        # Which half of the split produced this row, and which lambda
        # cascade_routing was operating at. Both change what the row means.
        "split_seed": splits.SPLIT_SEED,
        "cascade_routing_lambda": policies.CASCADE_ROUTING_LAMBDA,
        # Which model ladder produced this row. Without it, two results files from
        # two ladders look identical in every field and are not comparable at all.
        "ladder": models.LADDER,
    }


def load_tasks(limit=None, domain=None):
    with open(paths.TASKSET, encoding="utf-8") as f:
        tasks = [json.loads(l) for l in f]
    if domain:
        tasks = [t for t in tasks if t["domain"] == domain]
    tasks = tasks[:limit] if limit else tasks
    # Tell the cache which ids are live, so a conflicting key on a stranded task
    # stays a note and one on a task this run will score becomes a warning.
    response_cache.LIVE_TASK_IDS = {t["id"] for t in tasks}
    return tasks


def applicable(name, task):
    """Does this policy run on this task?

    cascade_degraded is defined on the code domain only, and that is the point of
    it rather than a limitation: it varies verifier fidelity WITHIN a domain,
    holding tasks, models, grader and prompts fixed, so verifier quality stops
    being confounded with math-vs-code. Running it on math would put the confound
    straight back.

    routellm runs only when real cached scores exist for the task. It is skipped
    rather than approximated - see policies.py DECISION #8.
    """
    if name == "routellm":
        from llm_routing import routellm_router
        # BOTH conditions. A cached score for this task is not enough: the
        # threshold is set from the whole task set, so a partially-scored set must
        # keep the policy out entirely rather than let it run uncalibrated on the
        # subset that happens to have scores.
        return routellm_router.CALIBRATED and routellm_router.available([task])
    if name in policies.NEEDS_ESTIMATORS and not policies.ESTIMATORS_FITTED:
        return False
    allowed = POLICY_DOMAINS.get(name)
    return allowed is None or task["domain"] in allowed


def run(tasks):
    rows = []
    attributed = 0.0   # what the policies would each pay in production
    total = sum(1 for t in tasks for n in POLICIES if applicable(n, t))
    done = 0

    # policy name -> the task whose response was missing from the replay cache.
    # A cache recorded by one set of policies has no reason to contain the calls
    # another set would make, so this is a normal state for replay, not a fault.
    uncached = {}

    for task in tasks:
        for name, fn in POLICIES.items():
            if name in uncached or not applicable(name, task):
                continue
            # Measured per row, not inferred from the mode. A policy that made
            # even one fabricated call produced a fabricated row, whatever the
            # other calls were: a cascade that verifies a real cheap answer with
            # five mock samples is reporting the mock's verdict.
            mock_before = models.call_stats["served_mock"]
            # Same argument as mock_before, for the same reason: a row whose
            # answer was cut off at max_tokens is a MISSING measurement, not a
            # wrong one, and downstream analysis has to be able to tell.
            trunc_before = models.call_stats["truncated"]
            try:
                res = fn(task)
            except models.ReplayMiss:
                # Drop the policy ENTIRELY, including rows already collected for
                # it, rather than scoring it on whichever tasks happened to be
                # cached. A partially-replayed policy is measured on a biased
                # subset - the tasks some earlier run chose to record - and would
                # be reported next to fully-scored policies as if comparable.
                # `applicable` already keeps routellm out on exactly this
                # reasoning; this is the same rule applied at call time, where a
                # missing recording is the thing that reveals the problem.
                uncached[name] = task["id"]
                continue
            attributed += res.cost_usd
            rows.append(
                {
                    "task_id": res.task_id,
                    "domain": task["domain"],
                    "difficulty": task.get("difficulty_proxy"),
                    "policy": res.policy,
                    "correct": res.correct,
                    "cost_usd": res.cost_usd,
                    "latency_s": res.latency_s,
                    "escalated": res.escalated,
                    "calls": res.calls,
                    "truncated": models.call_stats["truncated"] > trunc_before,
                    **provenance(
                        simulated=models.call_stats["served_mock"] > mock_before
                    ),
                }
            )
            done += 1
            if done % 50 == 0:
                # Both, labelled. One line showing a single "spend" figure is how
                # a replay gets read as though it cost money, and how a real run
                # gets read as though it cost several times what it did.
                print(f"  {done}/{total}  spent=${models.backend_spend_usd:.4f}"
                      f"  attributed=${attributed:.4f}", file=sys.stderr)
    return _drop_uncached(rows, uncached)


def _drop_uncached(rows, uncached):
    """Remove every row belonging to a policy that hit a replay cache miss.

    Says so on stderr rather than silently, because "this policy is absent" and
    "this policy scored zero" look identical in a results file, and the second
    reading is the dangerous one.
    """
    if not uncached:
        return rows
    kept = [r for r in rows if r["policy"] not in uncached]
    dropped = len(rows) - len(kept)
    print(
        f"\n!! replay cache is incomplete for {len(uncached)} "
        f"{'policy' if len(uncached) == 1 else 'policies'}; "
        f"dropped from this run along with {dropped} already-scored "
        f"{'row' if dropped == 1 else 'rows'}:",
        file=sys.stderr,
    )
    for name, task_id in sorted(uncached.items()):
        print(f"     {name:<18} first missing at {task_id}", file=sys.stderr)
    print(
        "   These policies make calls the cache was never populated with. That is\n"
        "   expected when the cache came from a run of DIFFERENT policies - the\n"
        "   two-arm probe recorded always_cheap and always_expensive only.\n"
        "   Record them with:  ROUTER_MODE=real python -m llm_routing.run_eval"
        + "".join(f" --policy {n}" for n in sorted(uncached))
        + "\n   Or run everything free and offline with ROUTER_MODE=mock.",
        file=sys.stderr,
    )
    return kept


def _used_expensive(row):
    return "expensive" in (row.get("calls") or [])


def routing_skill(by_policy):
    """How much of the achievable routing gain each router actually captured.

        skill = (acc_router - null_at_router's_cost) / (acc_oracle - null)

    THE NULL IS COMPUTED AT EACH POLICY'S OWN SPEND, changed 8 August 2026.

    It used to be `random_matched`'s accuracy for everybody. That was only ever
    valid for one policy - the one `random_matched` was rate-matched to - and it
    quietly compared every other policy against a null at somebody else's
    spending level. Both cascades were being scored against a null matched to the
    predictive heuristic's rate, which was never the intent. Deleting that
    heuristic and putting `routellm` on a fixed threshold removed the last reason
    to pretend one null fits all.

    The null for a one-shot router that spends C is closed-form: randomising
    between the two rungs at probability p costs C_cheap + p*(C_exp - C_cheap)
    and scores A_cheap + p*(A_exp - A_cheap). Invert the first for p, substitute
    into the second, and you have the chord of the cost-accuracy line - the
    accuracy a policy with NO signal would reach at exactly this budget.

    p is taken from cost rather than from the realised expensive-call rate, so
    one rule covers cascades too: a cascade that spends more than always_expensive
    clamps to p=1, which correctly says that at that budget randomising cannot
    beat simply always paying for the best rung.

    `random_matched` stays in the report as the empirical check that the chord is
    not a fiction: it is a real policy that actually flips a coin, and its
    measured accuracy should land near its own chord value.

    Reported PER DOMAIN as well as overall, because the aggregate averages two
    halves whose routing signal differs by an order of magnitude. A negative
    value means the router did worse than chance at its own spend. That is a
    result and is printed as one rather than clipped to zero.
    """
    def stat(name, domain=None):
        rs = by_policy.get(name, [])
        if domain:
            rs = [r for r in rs if r["domain"] == domain]
        if not rs:
            return None, None, 0
        return (sum(r["correct"] for r in rs) / len(rs),
                sum(r["cost_usd"] for r in rs) / len(rs),
                len(rs))

    def null_at(cost, domain):
        """Accuracy of a signal-free policy at this budget. None if unavailable."""
        a_lo, c_lo, _ = stat("always_cheap", domain)
        a_hi, c_hi, _ = stat("always_expensive", domain)
        if None in (a_lo, a_hi) or c_hi - c_lo < 1e-12:
            return None
        p = (cost - c_lo) / (c_hi - c_lo)
        p = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)
        return a_lo + p * (a_hi - a_lo)

    ranked = ("routellm", "llm_router", "random_matched", "cascade",
              "cascade_routing")
    routers = [n for n in ranked if by_policy.get(n)]
    if not routers or not by_policy.get("oracle"):
        return
    if not (by_policy.get("always_cheap") and by_policy.get("always_expensive")):
        return

    print()
    print(tag())
    print("routing skill = (router - null) / (oracle - null)")
    print("  null = the accuracy a policy with NO signal reaches at the SAME")
    print("  cost, by randomising between the rungs. Each policy gets its own.")
    print(f"{'router':<16} {'domain':<8} {'cost/task':>10} {'null':>8} "
          f"{'router':>8} {'oracle':>8} {'skill':>9}")
    print("-" * 74)
    any_na = False
    for name in routers:
        for domain in (None, "math", "code"):
            mid, cost, n = stat(name, domain)
            hi, _, _ = stat("oracle", domain)
            if None in (mid, cost, hi):
                continue
            lo = null_at(cost, domain)
            if lo is None:
                continue
            label = domain or "all"
            # The ratio needs headroom to be meaningful, and "meaningful" here
            # has a natural unit: one task, worth 1/n of accuracy. A policy that
            # spends near always_expensive gets a null that sits within a task or
            # two of the oracle, and then a single task flipping moves the ratio
            # by more than the entire headroom - which is how a coin flip prints
            # -417% and looks like a finding. Report n/a instead. This is a
            # property of the policy's SPENDING LEVEL, not of its skill: at that
            # budget there is almost nothing left for any router to win.
            headroom = hi - lo
            if headroom < 1.0 / max(n, 1) + 1e-12:
                skill = "     n/a"
                any_na = True
            else:
                skill = f"{(mid - lo) / headroom:>8.1%}"
            print(f"{name:<16} {label:<8} {cost:>10.6f} {lo:>8.1%} "
                  f"{mid:>8.1%} {hi:>8.1%} {skill:>9}")
    if any_na:
        print("  n/a = the oracle is less than one task above the null at that")
        print("        spend, so no ratio computed from n tasks can be estimated.")


def oracle_bound_check(by_policy):
    """Assert the invariant the oracle claims: nothing beats it on accuracy.

    Printed rather than merely believed. The oracle is only a ceiling if it can
    reach every action the other policies can, and this repo has already shipped
    a version where it could not - the math cascade accepted a self-consistency
    answer that was outside the oracle's action space, and duly scored above it.
    A silent ceiling that is not a ceiling invalidates every "fraction of
    headroom captured" figure in the report, so it gets a check.
    """
    oracle = by_policy.get("oracle")
    if not oracle:
        return
    print()
    print("oracle bound check (nothing should exceed the oracle's accuracy):")
    for domain in (None, "math", "code"):
        def acc(name):
            rs = by_policy.get(name, [])
            if domain:
                rs = [r for r in rs if r["domain"] == domain]
            return (sum(r["correct"] for r in rs) / len(rs)) if rs else None

        hi = acc("oracle")
        if hi is None:
            continue
        # One task of slack, because cascade_degraded runs on a subset and a
        # rounding difference is not a violation.
        over = []
        for name in by_policy:
            if name == "oracle":
                continue
            a = acc(name)
            if a is not None and a > hi + 1e-9:
                over.append(f"{name} {a:.1%}")
        label = domain or "all"
        if over:
            print(f"  {label:<6} VIOLATED - oracle {hi:.1%} beaten by: {', '.join(over)}")
            print("         The oracle's action space is missing something these")
            print("         policies can do. See policies.policy_oracle.")
        else:
            print(f"  {label:<6} ok - oracle {hi:.1%} is the ceiling")


def report(rows):
    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r["policy"]].append(r)

    # Roughly a cost ladder, so the table reads bottom-to-top on spend. Built
    # from models.TIERS rather than listed, so it adapts to the loaded ladder.
    #
    # random_matched sits immediately before the predictive family - routellm and
    # llm_router - because it is what they have to beat before their accuracy
    # means anything. The cascades follow, so the two architectures are adjacent
    # blocks rather than interleaved.
    order = (
        ["always_cheap", "random_matched", "routellm", "llm_router",
         "cascade_degraded", "cascade_routing"]
        + [f"always_{t}" for t in models.TIERS[1:-1]]
        + ["cascade", "oracle", f"always_{models.TIERS[-1]}"]
    )

    print()
    print(banner())
    print()
    print(tag())
    print(f"{'policy':<20} {'acc':>7} {'cost/task':>12} {'lat/task':>10} {'exp%':>7} {'escal':>7}")
    print("-" * 70)
    for name in order:
        rs = by_policy.get(name, [])
        if not rs:
            continue
        acc = sum(r["correct"] for r in rs) / len(rs)
        cost = sum(r["cost_usd"] for r in rs) / len(rs)
        lat = sum(r["latency_s"] for r in rs) / len(rs)
        # Two different things, and the distinction is the reason both columns
        # exist. exp% is how often the expensive model was used at all, which is
        # what drives cost and is defined for every policy. escal is how often a
        # cascade escalated AFTER seeing a cheap answer, which is 0 by definition
        # for a one-shot router however it routed.
        exp = sum(_used_expensive(r) for r in rs) / len(rs)
        esc = sum(r["escalated"] for r in rs) / len(rs)
        # cascade_degraded runs on code only, so its row covers fewer tasks than
        # the others. Print n rather than let the reader assume.
        note = f"  (n={len(rs)}, code only)" if name == "cascade_degraded" else ""
        print(f"{name:<20} {acc:>6.1%} {cost:>12.6f} {lat:>9.2f}s "
              f"{exp:>6.1%} {esc:>6.1%}{note}")
    print("  exp%  = fraction of tasks that used the expensive model at all")
    print("  escal = fraction where a cascade escalated after verifying; a")
    print("          one-shot router never escalates, so its escal is always 0")

    routing_skill(by_policy)
    oracle_bound_check(by_policy)

    # The verifier contrast: the cascade should do noticeably worse where the
    # verifier is a guess (math) than where it is perfect (code).
    print()
    print(tag())
    print("cascade by domain (the verifier contrast):")
    print(f"{'domain':<10} {'acc':>7} {'cost/task':>12} {'escal':>8}")
    print("-" * 40)
    for domain in ("code", "math"):
        rs = [r for r in by_policy.get("cascade", []) if r["domain"] == domain]
        if not rs:
            continue
        acc = sum(r["correct"] for r in rs) / len(rs)
        cost = sum(r["cost_usd"] for r in rs) / len(rs)
        esc = sum(r["escalated"] for r in rs) / len(rs)
        verifier = "perfect" if domain == "code" else "proxy"
        print(f"{domain:<10} {acc:>6.1%} {cost:>12.6f} {esc:>7.1%}   <- {verifier} verifier")

    # The claim under test in DECISION #7: that an LLM routing call "would add a
    # full round trip and defeat the purpose". Half of that is a cost claim, so
    # print the cost rather than argue about it.
    if by_policy.get("llm_router") and by_policy.get("always_cheap"):
        n = len(by_policy["llm_router"])
        router_cost = sum(policies.ROUTER_CALL_COST) / max(1, len(policies.ROUTER_CALL_COST))
        router_lat = sum(policies.ROUTER_CALL_LATENCY) / max(1, len(policies.ROUTER_CALL_LATENCY))
        cheap = sum(r["cost_usd"] for r in by_policy["always_cheap"]) / n
        exp = sum(r["cost_usd"] for r in by_policy.get("always_expensive", [])) / n \
            if by_policy.get("always_expensive") else None
        print()
        print(tag())
        print("LLM-as-router overhead (the claim recorded at policies.py DECISION #4):")
        print(f"  routing call        ${router_cost:.6f}/task, +{router_lat:.2f}s/task")
        print(f"    as % of a cheap answer call     {100 * router_cost / cheap:>5.1f}%")
        if exp:
            print(f"    as % of an expensive answer call{100 * router_cost / exp:>6.1f}%")
        print("  This is also what random_matched does NOT pay: it is calibrated to")
        print("  llm_router's rate but flips a coin, so it is cheaper by this much.")
        print("  Cost and latency above are arithmetic on the price table and are")
        print("  the only part of this policy that mock mode can measure. Its")
        print("  ACCURACY restates models.MOCK_ROUTER_SKILL and measures nothing.")
        if models.MODE == "mock":
            print("  The PERCENTAGES are softer than the dollar figure: mock answer calls")
            print("  emit a fixed 80/120 output tokens, well under a real reply, so the")
            print("  router's share of a real answer call will be smaller than shown.")
            print("  Quote the absolute cost from a real run, not these ratios.")

    # The pilot gate. This is the number that decides whether the task set works
    # at all, so it prints the band it is testing against rather than asserting a
    # verdict the reader cannot check.
    cheap = by_policy.get("always_cheap", [])
    if cheap:
        n = len(cheap)
        fail = 1 - sum(r["correct"] for r in cheap) / n
        # A rate without its precision is how a 10-task pilot gets read as a
        # verdict. At n=10 the interval is about +-28 points, which spans the
        # entire band, so the gate reports the interval and refuses to rule when
        # the interval does not fit inside a region.
        se = (fail * (1 - fail) / n) ** 0.5 if n else 0.0
        half = 1.96 * se
        lo, hi = max(0.0, fail - half), min(1.0, fail + half)
        # At exactly 0 or 1 the normal interval has zero width, which would let a
        # single lucky task read as a confident verdict. The rule of three gives
        # the correct bound for "no events observed in n trials": the true rate
        # could still be as high as about 3/n.
        if fail == 0.0:
            hi = min(1.0, 3.0 / n)
        elif fail == 1.0:
            lo = max(0.0, 1.0 - 3.0 / n)
        print()
        print(f"pilot gate - cheap-model failure rate: {fail:.1%}  "
              f"95% CI [{lo:.0%}, {hi:.0%}]  (n={n})")
        print(f"  target band: {FAILURE_RATE_FLOOR:.0%} to {FAILURE_RATE_CEILING:.0%}")
        if hi < FAILURE_RATE_FLOOR:
            print("  TOO EASY - the cascade has almost nothing to route. Use harder tasks.")
        elif lo > FAILURE_RATE_CEILING:
            print("  TOO HARD - the cascade escalates nearly everything. Use easier tasks.")
        elif FAILURE_RATE_FLOOR <= lo and hi <= FAILURE_RATE_CEILING:
            print("  IN BAND - routing decisions carry information on this task set.")
        else:
            print("  UNRESOLVED - the interval straddles a boundary, so this n cannot")
            print("  decide it. The point estimate leans "
                  + ("too easy" if fail < FAILURE_RATE_FLOOR else
                     "too hard" if fail > FAILURE_RATE_CEILING else "in band")
                  + ". Run more tasks:")
            print("    python -m llm_routing.run_eval --policy always_cheap --split all")

        # Per domain as well as overall, because the two halves can fail this gate
        # in opposite directions and the aggregate would hide it. The code half is
        # the one to watch: it carries the perfect verifier, so the whole
        # verifier-quality experiment lives there, and MBPP is a much older and
        # more saturated benchmark than MATH500.
        by_domain = defaultdict(list)
        for r in cheap:
            by_domain[r["domain"]].append(r)
        if len(by_domain) > 1:
            print("  per domain (the aggregate can hide a split verdict):")
            for domain in sorted(by_domain):
                rs = by_domain[domain]
                dn = len(rs)
                df = 1 - sum(r["correct"] for r in rs) / dn
                dse = (df * (1 - df) / dn) ** 0.5
                dlo, dhi = max(0.0, df - 1.96 * dse), min(1.0, df + 1.96 * dse)
                if df == 0.0:
                    dhi = min(1.0, 3.0 / dn)
                elif df == 1.0:
                    dlo = max(0.0, 1.0 - 3.0 / dn)
                if dhi < FAILURE_RATE_FLOOR:
                    verdict = "TOO EASY"
                elif dlo > FAILURE_RATE_CEILING:
                    verdict = "TOO HARD"
                elif FAILURE_RATE_FLOOR <= dlo and dhi <= FAILURE_RATE_CEILING:
                    verdict = "in band"
                else:
                    verdict = "unresolved"
                note = "  <- the verifier experiment lives here" if domain == "code" else ""
                print(f"    {domain:<6} {df:>6.1%}  CI [{dlo:.0%}, {dhi:.0%}]  "
                      f"n={dn:<4} {verdict}{note}")

        # Failure rate by difficulty band, which is the number that says HOW to fix
        # a failing gate rather than merely that it failed. If the cheap model only
        # starts failing at the top of the range, the fix is to keep that band and
        # drop the rest; if it never fails anywhere, the dataset itself has to go.
        # This is the whole reason the probe is worth paying for once the direction
        # is already obvious.
        print("  by difficulty band - where the cheap model starts to struggle:")
        for domain in sorted(by_domain):
            rs = by_domain[domain]
            bands = defaultdict(list)
            for r in rs:
                bands[r.get("difficulty")].append(r)
            cells = []
            for band in sorted(bands, key=lambda b: (b is None, b)):
                sub = bands[band]
                bf = 1 - sum(x["correct"] for x in sub) / len(sub)
                cells.append(f"{band}:{bf:.0%}(n={len(sub)})")
            print(f"    {domain:<6} " + "  ".join(cells))
        print("    math bands are MATH500 levels; code bands are reference-solution")
        print("    line counts. Keep the bands that fail, drop the ones that do not.")

    # Two different numbers, and conflating them is the mistake the cache makes
    # easy. `attributed` is what the policies cost: every call charged to every
    # policy that made it, which is what a production deployment of one policy
    # would pay. `backend` is what THIS RUN actually spent, after deduplicating
    # identical calls across policies. Deduplication changes the second and must
    # never change the first.
    attributed = sum(r["cost_usd"] for r in rows)
    st = models.call_stats
    print()
    label = "total MODELLED cost" if models.MODE == "mock" else "total attributed cost"
    print(f"{label}: ${attributed:.4f}   (sum over policies - what they would each pay)")
    print(
        f"model calls: {st['requested']} requested, {st['from_cache']} served from cache, "
        f"{st['backend']} reached a backend"
    )
    if st["requested"]:
        saved = 100 * st["from_cache"] / st["requested"]
        print(f"  cache deduplicated {saved:.1f}% of calls  (cache now holds {response_cache.size()})")
    if models.MODE == "mock":
        print("  (simulated - nothing was spent and no network call was made)")

    # Where the TEXT came from, which is a different question from where the
    # lookup went and is the one that decides whether any of this is a result.
    # Printed unconditionally in replay, including when the answer is the good
    # one, so that its absence can never be read as reassurance.
    if models.MODE == "replay":
        real, mock = st["served_real"], st["served_mock"]
        print(f"  provenance: {real} real response(s), {mock} fabricated")
        if mock:
            sim_rows = sum(1 for r in rows if r["simulated"])
            print(
                f"  !! {mock} response(s) came from the MOCK cache, contaminating "
                f"{sim_rows} of {len(rows)} rows, each stamped simulated: true.\n"
                f"     Unset ROUTER_REPLAY_FALLBACK to make these a hard error instead."
            )

    # Truncation is invisible in the accuracy table, since a cut-off answer just
    # scores as wrong, so it gets its own section. Named, not counted: the point
    # is to be able to go and look at the task.
    if models.truncated_ids:
        rows_hit = sum(1 for r in rows if r.get("truncated"))
        print()
        print(
            f"!! {len(models.truncated_ids)} response(s) hit max_tokens and grade as "
            f"WRONG, affecting {rows_hit} of {len(rows)} scored rows "
            f"(stamped truncated: true)."
        )
        for task_id, tier, kind, sample_idx in sorted(models.truncated_ids):
            print(f"     {task_id:<14} {tier:<10} {kind} sample={sample_idx}")
        if not rows_hit:
            print(
                "   0 scored rows: these were served while fitting on the calibration\n"
                "   half, so they touch the estimators rather than the reported table."
            )
        print(
            "   These are MISSING measurements, not capability failures. Raising\n"
            "   models.MAX_TOKENS re-charges every cached response (see\n"
            "   docs/ENGINEERING.md), so exclude the task or accept the row as\n"
            "   unmeasured -\n"
            "   do not read it as the model getting the answer wrong."
        )


def guard_clobber(force: bool):
    """Refuse to overwrite REAL results with a MOCK run.

    A real run costs money and cannot be reproduced, because sampling at
    temperature > 0 is stochastic. A mock run is free and takes seconds. Losing
    the former to the latter by typing `python -m llm_routing.run_eval` out of habit is a
    mistake worth making impossible rather than merely unlikely.
    """
    if models.MODE != "mock" or force or not RESULTS.exists():
        return
    try:
        with RESULTS.open(encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    except (json.JSONDecodeError, OSError):
        return

    # Guard on `simulated`, not on `mode`. A replay of a real cache is money; a
    # replay of a mock cache is not, and blocking the second would train the habit
    # of reaching for --force, which defeats the guard on the first.
    def is_real(r):
        if "simulated" in r:
            return not r["simulated"]
        return r.get("mode") == "real"

    if any(is_real(r) for r in rows):
        sys.exit(
            f"\nREFUSING TO RUN.\n"
            f"  {RESULTS.name} holds REAL results, which cost money and cannot be\n"
            f"  reproduced. This is a MOCK run and would overwrite them.\n\n"
            f"  Back them up:   cp {RESULTS.name} results.real.jsonl\n"
            f"  Or override:    python -m llm_routing.run_eval --force\n"
        )


def guard_regression(rows, force: bool):
    """Refuse to replace a results file with a strictly poorer one.

    guard_clobber runs before the run and can only see the ARGUMENTS. This runs
    after and sees what was actually produced, which is the only place the real
    hazard is visible: policies are dropped mid-run by ReplayMiss, so a command
    that asks for all nine can legitimately finish with one.

    That is not hypothetical. On 8 August 2026 `ROUTER_LADDER=deepseek python
    run_eval.py` - with ROUTER_MODE=replay set in .env - found cached data for
    always_cheap only (the deepseek and wide ladders share a bottom rung, so the
    cross-ladder cache served it) and dropped the other eight. It overwrote a
    complete nine-policy wide run with 47 rows of one policy. guard_clobber
    passed it, correctly by its own terms: both files were real, and it only
    refuses mock-over-real.

    A results file is not an output to be regenerated at will. Recomputing it
    needs the cache to still cover every policy, and that is exactly what fails
    when a ladder changes. So: refuse a write that covers a different ladder, or
    fewer policies, than the file already on disk.
    """
    if force or not rows or not RESULTS.exists():
        return
    try:
        with RESULTS.open(encoding="utf-8") as f:
            old = [json.loads(l) for l in f if l.strip()]
    except (json.JSONDecodeError, OSError):
        return
    if not old:
        return

    old_pol, new_pol = {r["policy"] for r in old}, {r["policy"] for r in rows}
    old_lad = {r.get("ladder") for r in old}
    new_lad = {r.get("ladder") for r in rows}
    lost = sorted(old_pol - new_pol)
    changed_ladder = old_lad != new_lad

    if not lost and not changed_ladder:
        return

    why = []
    if changed_ladder:
        why.append(
            f"  ladder:   on disk {sorted(x for x in old_lad if x)} "
            f"-> this run {sorted(x for x in new_lad if x)}"
        )
    if lost:
        why.append(
            f"  policies: {len(old_pol)} on disk -> {len(new_pol)} here; "
            f"would lose {', '.join(lost)}"
        )
    sys.exit(
        "\nREFUSING TO WRITE.\n"
        f"  {RESULTS.name} holds a broader run than this one produced.\n"
        + "\n".join(why)
        + "\n\n"
        "  Most likely this ladder's cache cannot serve every policy, and they\n"
        "  were dropped mid-run - scroll up for the SKIPPED lines. Regenerating\n"
        "  the file you are about to destroy needs a cache that still covers it.\n\n"
        "  Keep both:      python -m llm_routing.run_eval --out results.<ladder>.jsonl\n"
        "                  (the right answer for a ladder change - one file per\n"
        "                   ladder is what scripts/run_all_ladders.py does)\n"
        "  Write anyway:   python -m llm_routing.run_eval --force\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="run only the first N tasks (use 10 for a pilot)")
    ap.add_argument("--domain", choices=["math", "code"])
    ap.add_argument(
        "--split", choices=["eval", "all"], default="eval",
        help="eval (default): fit on the calibration half, report on the held-out "
             "half. all: report on every task, which is optimistic because the "
             "thresholds were tuned while looking at them.",
    )
    ap.add_argument(
        "--policy", action="append", default=None, metavar="NAME[,NAME...]",
        help="restrict the run to these policies. Repeatable, and also accepts a "
             "comma-separated list, because a bare comma is awkward in some "
             "shells: --policy a --policy b and --policy a,b are equivalent. "
             "Writes to results.probe.jsonl rather than results.jsonl. The main "
             "use is the TWO-ARM PROBE: --policy always_cheap --policy "
             "always_expensive --split all, which measures the routable fraction "
             "for a fraction of a full run's cost.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="allow a mock run to overwrite existing real results",
    )
    ap.add_argument(
        "--out", metavar="PATH", default=None,
        help="write results here instead of results.jsonl. THE POINT IS LADDERS: "
             "one ladder per file, so results.wide.jsonl and results.claude.jsonl "
             "can sit side by side and be compared. Without this there is exactly "
             "one results.jsonl, and guard_regression correctly refuses to let a "
             "second ladder overwrite the first - see its docstring for the run "
             "that made that guard necessary.",
    )
    args = ap.parse_args()

    # A filtered run is a probe, not a result. It writes elsewhere so it can never
    # overwrite a full run's rows with a partial set, which would silently break
    # every paired comparison in stats.py.
    global RESULTS
    selected = None
    if args.policy:
        # Flatten: each --policy may itself hold a comma list.
        selected = [p.strip() for chunk in args.policy
                    for p in chunk.split(",") if p.strip()]
        unknown = [p for p in selected if p not in POLICIES]
        if unknown:
            sys.exit(
                f"unknown policy: {', '.join(unknown)}\n"
                f"  available on ladder {models.LADDER!r}: {', '.join(sorted(POLICIES))}"
            )
        RESULTS = paths.RUNS / "results.probe.jsonl"
        for name in list(POLICIES):
            if name not in selected:
                del POLICIES[name]

    # --out wins over the probe redirect: it is an explicit instruction, where the
    # redirect is a default applied on the caller's behalf. Said out loud, because
    # silently ignoring either one writes a file somewhere the caller is not
    # looking, and a results file in the wrong place is how a partial run gets
    # read as a complete one.
    if args.out:
        if selected:
            print(
                f"--policy would redirect to results.probe.jsonl; --out overrides "
                f"that.\n  This is a PARTIAL run ({len(selected)} of the available "
                f"policies) going to a\n  path of your choosing. Do not let it "
                f"stand in for a full one.",
                file=sys.stderr,
            )
        RESULTS = Path(args.out)
        if not RESULTS.parent.exists():
            sys.exit(f"--out directory does not exist: {RESULTS.parent}")

    guard_clobber(args.force)

    all_tasks = load_tasks(args.limit, args.domain)
    calibration, evaluation = splits.split(all_tasks)
    report_tasks = evaluation if args.split == "eval" else all_tasks

    print(banner(), file=sys.stderr)
    print(f"mode={models.MODE}  policies={len(POLICIES)}", file=sys.stderr)
    for line in models.ladder_summary():
        print(line, file=sys.stderr)
    print(credentials_line(), file=sys.stderr)
    for line in splits.describe(calibration, evaluation):
        print(line, file=sys.stderr)
    if args.split == "eval":
        print(f"reporting on the HELD-OUT half: n={len(report_tasks)}", file=sys.stderr)
    else:
        print(
            f"reporting on ALL {len(report_tasks)} tasks. These numbers are "
            f"OPTIMISTIC:\n  the thresholds were chosen while looking at this same "
            f"set. Use --split eval\n  for the honest figure.",
            file=sys.stderr,
        )

    # Fit cascade_routing's quality estimators on the CALIBRATION half only. This
    # uses ground truth, which is what a calibration split is for; doing it on the
    # reporting tasks would make the policy's numbers a measure of its hindsight.
    #
    # ONLY WHEN A POLICY ACTUALLY NEEDS THEM. Fitting is not free: it calls EVERY
    # rung on every calibration task, top rung included. Doing that unconditionally
    # made `--policy always_cheap` - the difficulty probe, whose whole point is to
    # be the cheapest possible real call - quietly spend most of its money on the
    # expensive tier it was never meant to touch. Measured on the shipped set:
    # 318 backend calls instead of 100, of which 49 were to the top rung.
    needs_fit = any(n in POLICIES for n in policies.NEEDS_ESTIMATORS)
    fit_failed = None
    if calibration and needs_fit:
        try:
            tables = policies.fit_estimators(calibration)
        except models.ReplayMiss as exc:
            # Fitting calls every rung and every verifier on the calibration
            # half, so it needs strictly more of the cache than any single policy
            # does - it is the first place a thin cache shows up. Leave
            # ESTIMATORS_FITTED False and let `applicable` drop the policies that
            # need it, which is the same "sit out rather than guess" rule
            # routellm already follows. Crashing the run instead would take eight
            # fully-replayable policies down with the one that has no data.
            fit_failed = str(exc).splitlines()[0]
            tables = None
    if fit_failed is not None:
        print(
            f"cascade_routing: SKIPPED - the estimators cannot be fitted from "
            f"this cache.\n  {fit_failed}", file=sys.stderr,
        )
    elif calibration and needs_fit:
        print(
            "cascade_routing estimators fitted on the calibration half:",
            file=sys.stderr,
        )
        for tier in models.TIERS:
            row = "  ".join(
                f"hard={h}:{tables['exante'].get((tier, h), float('nan')):.0%}"
                for h in (False, True)
            )
            print(f"  ex-ante  {tier:<10} {row}", file=sys.stderr)
        for domain in ("math", "code"):
            acc = tables["posthoc"].get((domain, True))
            rej = tables["posthoc"].get((domain, False))
            if acc is None or rej is None:
                continue
            # This gap IS the verifier's quality, and it is the number the whole
            # repo manipulates. A wide gap means the verdict is informative.
            print(
                f"  post-hoc {domain:<10} P(correct|accept)={acc:.0%}  "
                f"P(correct|reject)={rej:.0%}  gap={acc - rej:+.0%}",
                file=sys.stderr,
            )
    elif not needs_fit:
        print("cascade_routing: not selected, so its estimators are not fitted "
              "(saves calls to every rung)", file=sys.stderr)
    else:
        print("cascade_routing: SKIPPED - calibration half is empty", file=sys.stderr)

    # Match the random baseline's spend to llm_router's on the REPORTING tasks,
    # before anything runs. Doing it afterwards would compare against a rate
    # measured on a different task set. See policies.py DECISION #6.
    #
    # The pre-pass reads the same cached routing calls policy_llm_router will
    # make, so it costs nothing. In replay a missing one raises ReplayMiss, and
    # the null then falls back to its declared rates - loudly, because a null
    # that is silently at the wrong spend is worse than no null at all.
    decisions = None
    try:
        decisions = policies.llm_router_decisions(report_tasks)
    except models.ReplayMiss as e:
        print(
            f"random_matched: llm_router's routing calls are not in the cache "
            f"({e}).\n  Falling back to the DECLARED rates - the null is NOT "
            f"cost-matched to any router in this run.",
            file=sys.stderr,
        )
    rates = policies.calibrate_random_rates(report_tasks, decisions)
    anchor = "llm_router" if decisions is not None else "the declared defaults"
    print(
        f"random_matched calibrated to {anchor}: "
        + ", ".join(f"{d}={r:.0%}" for d, r in sorted(rates.items())),
        file=sys.stderr,
    )
    tasks = report_tasks

    # RouteLLM runs at a FIXED threshold rather than one calibrated to another
    # policy's spend - see routellm_router.py DECISION #8b. With no scores it
    # sits out; it is never approximated.
    from llm_routing import routellm_router
    if routellm_router.available(tasks):
        th = routellm_router.use_fixed_threshold(tasks)
        realised = {}
        for d in sorted(th):
            sub = [t for t in tasks if t["domain"] == d]
            realised[d] = sum(routellm_router.routes_expensive(t) for t in sub) / len(sub)
        print(
            f"routellm ({routellm_router.cached_variant()}) at fixed "
            f"score>={routellm_router.FIXED_THRESHOLD}: "
            + ", ".join(f"{d}={r:.0%} expensive" for d, r in sorted(realised.items()))
            + "\n  Not cost-matched to any other policy by construction; read the"
              " routing-skill table, which nulls each policy at its own spend.",
            file=sys.stderr,
        )
    else:
        print(
            "routellm: SKIPPED - no cached scores. Populate them once with\n"
            "  python -m llm_routing.routellm_router --score      (bert variant, no API key)",
            file=sys.stderr,
        )

    rows = run(tasks)
    guard_regression(rows, args.force)
    paths.ensure_runs()
    write_jsonl(RESULTS, rows)
    report(rows)
    print(f"\nwrote {len(rows)} rows -> {RESULTS}")
    # Last line of output as well as the first. Terminal scrollback usually shows
    # the end of a run rather than the beginning.
    print()
    print(banner())


if __name__ == "__main__":
    try:
        main()
    except models.SpendCapExceeded as exc:
        # Caught only to print it as a block rather than a traceback. The exit
        # status is still non-zero, so a script that chains runs stops here.
        print(
            "\n" + "=" * 62
            + f"\n  SPEND CAP - RUN ABORTED, {RESULTS.name} NOT WRITTEN\n"
            + "=" * 62 + f"\n{exc}\n"
            + "  A partial run must not look like a complete one, so the\n"
            + f"  previous {RESULTS.name} is left exactly as it was.",
            file=sys.stderr,
        )
        raise SystemExit(2)
