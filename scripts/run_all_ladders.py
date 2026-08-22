#!/usr/bin/env python3
"""Run the whole pipeline once per ladder, writing one set of files per ladder.

WHY THIS EXISTS
---------------
The thesis is that the sign of the cascading-vs-routing result depends on the
PRICE RATIO between rungs, and a price ratio is a property of a ladder. Testing
it means holding three ladders' numbers side by side, so they cannot share a
filename - a single fixed `results.jsonl` was correct with one measured ladder
and wrong the moment there were three. (A `deepseek` run once found cached data
for one policy, dropped the other eight, and overwrote a complete nine-policy
`wide` run with 47 rows.) Each ladder gets its own file, so
`run_eval.guard_regression` compares like with like and there is nothing to
clobber.

WHAT IT DOES

    for each ladder:
        -m llm_routing.run_eval       --out runs/results.<ladder>.jsonl
        -m llm_routing.frontier       --out runs/frontier.<ladder>.jsonl

    and, on the one ladder the degradation experiment is reported for:

        -m llm_routing.sweep_degraded --out runs/sweep_degraded.<ladder>.jsonl
        -m llm_routing.plot           --frontier ... --sweep ... --suffix .<ladder>

There is no unsuffixed copy. Every reader - docs/RESULTS.md, the tests, and
`router_agent.findings` - names the ladder it wants, so there is exactly one
file per ladder and no second copy that can disagree with it.

EACH LADDER RUNS IN ITS OWN SUBPROCESS. `models.LADDER` is read at import time
and baked into module-level constants, so switching ladder inside one process
means reloading half the repo. A subprocess is the honest version, and one
ladder failing cannot leave global state behind for the next.

    python scripts/run_all_ladders.py                     # all three
    python scripts/run_all_ladders.py --ladders wide      # just one
    python scripts/run_all_ladders.py --keep-going        # don't stop on failure

MODE IS NOT SET HERE, DELIBERATELY. It comes from the environment or `.env`,
exactly as it does for a direct run - which defaults to replay, so the bare
command reproduces every published artefact from the committed responses for
$0.00. This script has no --go gate and no spend cap of its own because it adds
neither: it shells out to the same entry points, which carry
`models.MAX_SPEND_USD` and the replay guards. Running it with ROUTER_MODE=real
spends money, and the individual scripts are what stop it. Running it with
ROUTER_MODE=mock fails on the first step, because every entry point below calls
`models.require_measured_mode`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The degradation sweep and its figures are produced for this ladder only.
# docs/RESULTS.md reports the experiment on it; see the note in main().
DEGRADATION_LADDER = "wide"


def ladders_available():
    """Read the ladder names from models rather than hard-coding them.

    Imported lazily and in a subprocess-free way: this only touches LADDERS,
    which does not depend on which ladder is selected.
    """
    from llm_routing import models
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
    miss is the difference between a full comparison and a one-policy file
    that looks like one.
    """
    out = []
    for line in (proc.stdout + proc.stderr).splitlines():
        s = line.strip()
        if not s:
            continue
        if ("SKIPPED" in s or "REFUSING" in s or "SPEND CAP" in s
                or s.startswith("!!") or "reached a backend" in s
                or s.startswith("wrote ") or s.startswith("skipped ")):
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
    ap.add_argument("--log", default=str(REPO / "runs" / "run_all_ladders.log"))
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
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("", encoding="utf-8")

    mode = os.environ.get("ROUTER_MODE", "replay")
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

        # `-m` rather than a path: the research modules are a package now, and
        # `python llm_routing/run_eval.py` would put llm_routing/ on sys.path
        # instead of the repo root, so `from llm_routing import models` would
        # not resolve. cwd=REPO below is what makes -m find the package with
        # nothing installed.
        steps = [
            (["-m", "llm_routing.run_eval",
              "--out", f"runs/results.{ladder}.jsonl"], "run_eval"),
            (["-m", "llm_routing.frontier",
              "--out", f"runs/frontier.{ladder}.jsonl"], "frontier"),
            # The cross-tab and the scorecard are here because the figures need
            # them, and because runs/README.md claims that deleting the whole
            # directory and running this script puts it back. They used to be
            # absent, and the failure was silent in the worst way: `plot` skips
            # a chart whose artefact is missing, so a from-scratch run drew
            # seven of the nine figures and said so only in a line nobody read.
            (["-m", "llm_routing.routable", "--ladders", ladder],
             "routable", f"runs/routable.{ladder}.txt"),
            (["-m", "llm_routing.scorecard",
              "--results", f"runs/results.{ladder}.jsonl",
              "--json", f"runs/scorecard.{ladder}.json"],
             "scorecard", f"runs/scorecard.{ladder}.txt"),
        ]

        # The degradation sweep and the rendered curves run on ONE ladder.
        #
        # The sweep holds the ladder fixed and varies verifier fidelity inside
        # the code domain - the ladder is the control, not the variable - so a
        # second ladder's curve answers no question the first has not. Three
        # ladders exist for the price-ratio finding, and `frontier.<ladder>.jsonl`
        # is still written for each because `router_agent.findings` reads its own
        # ladder's economics and declines when there is none.
        if ladder == DEGRADATION_LADDER:
            steps.append((
                ["-m", "llm_routing.sweep_degraded",
                 "--out", f"runs/sweep_degraded.{ladder}.jsonl"], "sweep_degraded"))
            if not args.skip_plots:
                steps.append((
                    ["-m", "llm_routing.plot",
                     "--frontier", f"runs/frontier.{ladder}.jsonl",
                     "--sweep", f"runs/sweep_degraded.{ladder}.jsonl",
                     "--suffix", f".{ladder}", "--no-summaries"],
                    "plot",
                ))

        print(f"\n--- {ladder} " + "-" * (60 - len(ladder)))
        for step in steps:
            cmd, name = step[0], step[1]
            capture = step[2] if len(step) > 2 else None
            proc, elapsed = run(cmd, env, log)
            if capture and proc.returncode == 0:
                # These two report to stdout rather than to a file, and
                # the committed artefact IS that transcript. Writing it
                # here keeps the driver the single way to regenerate
                # everything under runs/.
                (REPO / capture).write_text(proc.stdout, encoding="utf-8",
                                            newline="")
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

    # The cross-ladder charts read every ladder's artefacts, so they are drawn
    # once at the end rather than redrawn identically inside each ladder's loop.
    # They are also the ones that fail quietly: a summary is missing when one
    # ladder's artefact is, and that must not look like a clean run.
    if not args.skip_plots and not failed:
        proc, elapsed = run(["-m", "llm_routing.plot", "--only-summaries"],
                            dict(os.environ), log)
        drawn = [line for line in proc.stdout.splitlines() if "wrote " in line]
        print(f"\n  {'ok ' if proc.returncode == 0 else 'FAIL'} "
              f"summaries        {elapsed:5.1f}s  ({len(drawn)} figures)")
        for line in interesting(proc):
            if "skipped " in line:
                print(f"       {line}")

    if failed:
        print("\nfailed steps:")
        for ladder, name in failed:
            print(f"  {ladder}/{name}")
        print(f"Full output in {log}.")
        return 1

    print(f"\nall {len(wanted)} ladder(s) complete.")
    print("  paired tests are per-ladder - different ladders are different")
    print("  models, so run stats once per file rather than across them:")
    for ladder in wanted:
        print(f"    python -m llm_routing.stats "
              f"--results runs/results.{ladder}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
