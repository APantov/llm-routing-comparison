#!/usr/bin/env python3
"""Run the whole pipeline once per ladder, writing one set of files per ladder.

WHY THIS EXISTS
---------------
Every analysis script in this repo wrote to a single fixed path -
`results.jsonl`, `frontier.jsonl`, `sweep_degraded.jsonl` - which was correct
while there was one measured ladder and wrong the moment there were three. The
project's whole thesis is that the sign of the cascading-vs-routing result
depends on the PRICE RATIO between rungs, and a price ratio is a property of a
ladder. Testing that claim means holding three ladders' numbers side by side, so
they cannot share a filename.

`run_eval.guard_regression` was already refusing the alternative, and correctly:
on 8 August 2026 a `deepseek` run found cached data for one policy, dropped the
other eight, and overwrote a complete nine-policy `wide` run with 47 rows. The
fix is not to weaken that guard. It is to give each ladder its own file, so the
guard compares like with like and there is nothing to clobber.

WHAT IT DOES

    for each ladder:
        run_eval.py       --out results.<ladder>.jsonl
        frontier.py       --out frontier.<ladder>.jsonl
        sweep_degraded.py --out sweep_degraded.<ladder>.jsonl
        plot.py           --frontier ... --sweep ... --suffix .<ladder>

then copies the canonical ladder's results to the unsuffixed `results.jsonl` and
`frontier.jsonl`, because README, STATUS, the test suite and `router_agent`
all read those names.

EACH LADDER RUNS IN ITS OWN SUBPROCESS. `models.LADDER` is read from the
environment at import time and baked into module-level constants (`TIERS`,
`MODEL_SPECS` lookups, the policy registry built from them), so switching ladder
inside one process means reloading half the repo. A subprocess per ladder is the
honest version of that, and it also means one ladder failing cannot leave global
state behind for the next.

    python scripts/run_all_ladders.py                     # all three
    python scripts/run_all_ladders.py --ladders wide      # just one
    python scripts/run_all_ladders.py --keep-going        # don't stop on failure

MODE IS NOT SET HERE, DELIBERATELY. It comes from the environment or `.env`,
exactly as it does for a direct run. This script has no --go gate and no spend
cap of its own because it adds neither: it shells out to the same entry points,
which carry `models.MAX_SPEND_USD` and the replay guards. Running it with
ROUTER_MODE=real spends money, and the individual scripts are what stop it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The ladder whose files are also copied to the unsuffixed names. `wide` is the
# canonical one: it is the ladder every committed number was measured on, and
# the one the 46x price-ratio finding belongs to.
CANONICAL = "wide"


def ladders_available():
    """Read the ladder names from models rather than hard-coding them.

    Imported lazily and in a subprocess-free way: this only touches LADDERS,
    which does not depend on which ladder is selected.
    """
    import models
    return list(models.LADDERS)


def run(cmd, env, log):
    """Run one step, streaming nothing, capturing everything.

    Output is captured rather than streamed because three ladders x four steps
    is a lot of scrollback, and the interesting part - which policies were
    dropped, what was spent - is a handful of lines. The full text goes to the
    log file so nothing is actually lost.
    """
    started = time.time()
    proc = subprocess.run(
        [sys.executable, *cmd],
        cwd=REPO, env=env,
        capture_output=True, text=True,
    )
    elapsed = time.time() - started
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 70}\n$ {' '.join(cmd)}\n{'=' * 70}\n")
        f.write(proc.stdout)
        f.write(proc.stderr)
    return proc, elapsed


def interesting(proc):
    """The lines worth putting on screen from a captured run.

    SKIPPED lines are the ones that matter most: a policy dropped for a replay
    miss is the difference between a nine-policy comparison and a one-policy
    file that looks like one.
    """
    out = []
    for line in (proc.stdout + proc.stderr).splitlines():
        s = line.strip()
        if not s:
            continue
        if ("SKIPPED" in s or "REFUSING" in s or "SPEND CAP" in s
                or s.startswith("!!") or "reached a backend" in s
                or s.startswith("wrote ")):
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladders", default=None,
                    help="comma-separated subset, e.g. wide,deepseek. "
                         "Default: every ladder in models.LADDERS.")
    ap.add_argument("--keep-going", action="store_true",
                    help="carry on to the next ladder after a failure. Off by "
                         "default: a ladder that cannot be served usually means "
                         "the cache is missing something, and the next ladder "
                         "will fail the same way.")
    ap.add_argument("--skip-plots", action="store_true",
                    help="data files only, no SVGs")
    ap.add_argument("--log", default=str(REPO / "run_all_ladders.log"))
    args = ap.parse_args()

    available = ladders_available()
    if args.ladders:
        wanted = [x.strip() for x in args.ladders.split(",") if x.strip()]
        unknown = [x for x in wanted if x not in available]
        if unknown:
            sys.exit(f"unknown ladder(s): {', '.join(unknown)}\n"
                     f"  available: {', '.join(available)}")
    else:
        wanted = available

    log = Path(args.log)
    log.write_text("", encoding="utf-8")

    mode = os.environ.get("ROUTER_MODE", "mock")
    print(f"mode={mode}  ladders={', '.join(wanted)}")
    if mode == "real":
        print("  !! REAL MODE - these runs will spend money. The per-run cap is\n"
              "     models.MAX_SPEND_USD, currently "
              f"${os.environ.get('ROUTER_MAX_SPEND_USD', '5.0')}.")
    print(f"  full output -> {log}")

    failed = []
    for ladder in wanted:
        env = dict(os.environ)
        env["ROUTER_LADDER"] = ladder

        steps = [
            (["run_eval.py", "--out", f"results.{ladder}.jsonl"], "run_eval"),
            (["frontier.py", "--out", f"frontier.{ladder}.jsonl"], "frontier"),
            (["sweep_degraded.py", "--out", f"sweep_degraded.{ladder}.jsonl"],
             "sweep_degraded"),
        ]
        if not args.skip_plots:
            steps.append((
                ["plot.py",
                 "--frontier", f"frontier.{ladder}.jsonl",
                 "--sweep", f"sweep_degraded.{ladder}.jsonl",
                 "--suffix", f".{ladder}"],
                "plot",
            ))

        print(f"\n--- {ladder} " + "-" * (60 - len(ladder)))
        for cmd, name in steps:
            proc, elapsed = run(cmd, env, log)
            status = "ok " if proc.returncode == 0 else "FAIL"
            print(f"  {status} {name:<16} {elapsed:5.1f}s")
            for line in interesting(proc):
                print(f"       {line}")
            if proc.returncode != 0:
                failed.append((ladder, name))
                if not args.keep_going:
                    print(f"\nstopped at {ladder}/{name}. Full output in {log}.")
                    print("  Re-run with --keep-going to push past it.")
                    return 1
                break

    # The canonical ladder also owns the unsuffixed names, because README,
    # STATUS, tests/ and router_agent/findings.py all read `results.jsonl` and
    # `frontier.jsonl`. Copy rather than symlink: this runs on Windows, and a
    # tracked file that is sometimes a link is a portability problem nobody
    # needs.
    if CANONICAL in wanted and not any(l == CANONICAL for l, _ in failed):
        for stem in ("results", "frontier", "sweep_degraded"):
            src = REPO / f"{stem}.{CANONICAL}.jsonl"
            if src.exists():
                shutil.copyfile(src, REPO / f"{stem}.jsonl")
        for stem in ("frontier", "degradation"):
            src = REPO / "figures" / f"{stem}.{CANONICAL}.svg"
            if src.exists():
                shutil.copyfile(src, REPO / "figures" / f"{stem}.svg")
        print(f"\ncopied {CANONICAL} -> the unsuffixed names "
              f"(results.jsonl, frontier.jsonl, sweep_degraded.jsonl, figures/)")

    if failed:
        print("\nfailed steps:")
        for ladder, name in failed:
            print(f"  {ladder}/{name}")
        print(f"Full output in {log}.")
        return 1

    print(f"\nall {len(wanted)} ladder(s) complete.")
    print("  paired tests are per-ladder - different ladders are different")
    print("  models, so run stats.py once per file rather than across them:")
    for ladder in wanted:
        print(f"    python stats.py --results results.{ladder}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
