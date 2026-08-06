"""
Batch runner. Runs every policy over every task, writes results, prints a report.

Usage:
    python3 run_eval.py                                # mock mode, no spend
    python3 run_eval.py --limit 10                      # first 10 tasks only
    python3 run_eval.py --domain math                   # one domain
    ROUTER_MODE=real   python3 run_eval.py --limit 10   # 10-task pilot, real calls
    ROUTER_MODE=replay python3 run_eval.py              # replay a paid run, free

Start in mock mode. Get the whole pipeline working and the report printing, then
switch to real. Debugging a broken pipeline while paying per call is miserable.

Output is results.jsonl, one row per (task, policy), with enough provenance on
each row to interpret it without an external note.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import models
import policies
import response_cache
import splits
from policies import POLICIES, POLICY_DOMAINS

HERE = Path(__file__).parent
RESULTS = HERE / "results.jsonl"

# Hard spend cap, enforced in code rather than by willpower.
MAX_SPEND_USD = 20.0

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

  For a real result:  ROUTER_MODE=real python3 run_eval.py
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
    import os
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


def provenance():
    """Everything needed to interpret a row, carried by the row itself.

    Stamped per row rather than in a sidecar header, so that a slice of
    results.jsonl - one policy, one domain, one k - is still self-describing. The
    parameters are here specifically so a sweep over k or the agreement threshold
    produces a file that can be grouped by, rather than a pile of runs whose
    settings have to be remembered.
    """
    return {
        "mode": models.MODE,
        # The single most important bit on the row. `mode` alone is ambiguous: a
        # replay is real if the cache was populated by a paid run and fabricated
        # if it was populated by the mock, and those two look identical in every
        # other field. A reader should never have to infer which one this is.
        "simulated": models.MODE == "mock" or (
            models.MODE == "replay" and not response_cache.REAL_PATH.exists()
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
    with open(HERE / "taskset.jsonl", encoding="utf-8") as f:
        tasks = [json.loads(l) for l in f]
    if domain:
        tasks = [t for t in tasks if t["domain"] == domain]
    return tasks[:limit] if limit else tasks


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
        import routellm_router
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
    spend = 0.0
    total = sum(1 for t in tasks for n in POLICIES if applicable(n, t))
    done = 0

    for task in tasks:
        for name, fn in POLICIES.items():
            if not applicable(name, task):
                continue
            if spend > MAX_SPEND_USD:
                print(f"\n!! spend cap ${MAX_SPEND_USD} hit, stopping early", file=sys.stderr)
                return rows
            res = fn(task)
            spend += res.cost_usd
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
                    **provenance(),
                }
            )
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{total}  spend=${spend:.4f}", file=sys.stderr)
    return rows


def _used_expensive(row):
    return "expensive" in (row.get("calls") or [])


def routing_skill(by_policy):
    """How much of the achievable routing gain each router actually captured.

        skill = (acc_router - acc_random_matched) / (acc_oracle - acc_random_matched)

    Random is the floor and the oracle is the ceiling, so this reads as "what
    fraction of the available headroom did the router find". It is the number that
    makes the routing arm interpretable at all: raw accuracy conflates routing
    skill with willingness to spend, and a cost-matched random baseline holds the
    spending fixed so only skill is left.

    Reported PER DOMAIN as well as overall, because the aggregate is the mean of a
    router with a real signal (math, where MATH500 ships a difficulty level) and
    one with almost none (code, where nothing in an MBPP prompt predicts
    difficulty). Averaging those two into one number hides the finding.

    A negative value means the router did worse than a coin flip at the same
    spend. That is a result and is printed as one rather than clipped to zero.
    """
    def acc(name, domain=None):
        rs = by_policy.get(name, [])
        if domain:
            rs = [r for r in rs if r["domain"] == domain]
        return (sum(r["correct"] for r in rs) / len(rs)) if rs else None

    routers = [n for n in ("predictive", "routellm", "llm_router", "cascade",
                           "cascade_routing") if by_policy.get(n)]
    if not routers or not by_policy.get("random_matched") or not by_policy.get("oracle"):
        return

    print()
    print(tag())
    print("routing skill = (router - random_matched) / (oracle - random_matched)")
    print(f"{'router':<16} {'domain':<8} {'random':>8} {'router':>8} {'oracle':>8} {'skill':>9}")
    print("-" * 62)
    for name in routers:
        for domain in (None, "math", "code"):
            lo, mid, hi = acc("random_matched", domain), acc(name, domain), acc("oracle", domain)
            if None in (lo, mid, hi):
                continue
            label = domain or "all"
            if abs(hi - lo) < 1e-9:
                # No headroom: random already matches the oracle, so the ratio is
                # 0/0 and any number printed here would be invented.
                skill = "     n/a"
            else:
                skill = f"{(mid - lo) / (hi - lo):>8.1%}"
            print(f"{name:<16} {label:<8} {lo:>8.1%} {mid:>8.1%} {hi:>8.1%} {skill:>9}")


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
            print(f"         The oracle's action space is missing something these")
            print(f"         policies can do. See policies.policy_oracle.")
        else:
            print(f"  {label:<6} ok - oracle {hi:.1%} is the ceiling")


def report(rows):
    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r["policy"]].append(r)

    # Ordered roughly by spend, so the table reads as a cost ladder. The two
    # random policies sit next to predictive on purpose: they are what predictive
    # has to beat before its accuracy means anything.
    # Roughly a cost ladder, so the table reads bottom-to-top on spend. Built
    # from models.TIERS rather than listed, so it adapts to the loaded ladder.
    order = (
        ["always_cheap", "random_matched", "predictive", "routellm", "llm_router",
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
        print("LLM-as-router overhead (the claim in policies.py DECISION #4):")
        print(f"  routing call        ${router_cost:.6f}/task, +{router_lat:.2f}s/task")
        print(f"    as % of a cheap answer call     {100 * router_cost / cheap:>5.1f}%")
        if exp:
            print(f"    as % of an expensive answer call{100 * router_cost / exp:>6.1f}%")
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
            print("    python3 run_eval.py --policy always_cheap --split all")

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

    # Truncation is invisible in the accuracy table, since a cut-off answer just
    # scores as wrong, so it gets its own line.
    if models.truncated_calls:
        print()
        print(
            f"!! {models.truncated_calls} call(s) hit max_tokens and were graded as "
            f"WRONG. Raise models.MAX_TOKENS and re-run; these numbers are not valid."
        )


def guard_clobber(force: bool):
    """Refuse to overwrite REAL results with a MOCK run.

    A real run costs money and cannot be reproduced, because sampling at
    temperature > 0 is stochastic. A mock run is free and takes seconds. Losing
    the former to the latter by typing `python3 run_eval.py` out of habit is a
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
            f"  Or override:    python3 run_eval.py --force\n"
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
        RESULTS = HERE / "results.probe.jsonl"
        for name in list(POLICIES):
            if name not in selected:
                del POLICIES[name]

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
    if calibration and needs_fit:
        tables = policies.fit_estimators(calibration)
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

    # Match the random baseline's spend to predictive's on the REPORTING tasks,
    # before anything runs. Doing it afterwards would compare against a rate
    # measured on a different task set.
    rates = policies.calibrate_random_rates(report_tasks)
    print(
        "random_matched calibrated to predictive: "
        + ", ".join(f"{d}={r:.0%}" for d, r in sorted(rates.items())),
        file=sys.stderr,
    )
    tasks = report_tasks

    # RouteLLM shares the same calibration target, so its threshold is set from
    # the same rates rather than from its own default. With no scores it sits out;
    # it is never approximated.
    import routellm_router
    if routellm_router.available(tasks):
        th = routellm_router.calibrate(tasks, rates)
        print(
            f"routellm ({routellm_router.cached_variant()}) calibrated to the same rates: "
            + ", ".join(f"{d}=score>={v:.4f}" for d, v in sorted(th.items())),
            file=sys.stderr,
        )
    else:
        print(
            "routellm: SKIPPED - no cached scores. Populate them once with\n"
            "  python3 routellm_router.py --score      (bert variant, no API key)",
            file=sys.stderr,
        )

    rows = run(tasks)
    write_jsonl(RESULTS, rows)
    report(rows)
    print(f"\nwrote {len(rows)} rows -> {RESULTS}")
    # Last line of output as well as the first. Terminal scrollback usually shows
    # the end of a run rather than the beginning.
    print()
    print(banner())


if __name__ == "__main__":
    main()
