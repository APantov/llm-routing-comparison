#!/usr/bin/env python3
"""Physically remove the quarantined tasks from every artefact on disk.

WHY THIS EXISTS, AND WHY IT IS DESTRUCTIVE ON PURPOSE
-----------------------------------------------------
Five MBPP+ tasks have expected answers that cannot be derived from their prompt
(`build_taskset.QUARANTINED`). The first fix kept their
recorded responses and filtered them out at every point of use: a filter in
`build_taskset`, another in `router_agent.findings.load_probe`, and a standing
rule that every future reader had to remember.

That was the wrong trade. It spread five broken tasks across the whole codebase
as permanent complexity, and what it bought was $0.17 of API responses that can
never be used for anything - the tasks are unpassable, so no rerun, no ladder
and no future experiment will ever want them. Deleting the data deletes the
filters with it.

So this script removes them for good. It is the auditable record of what was
deleted, which is why it is committed rather than run from a shell one-liner.

WHAT IS NOT LOST
----------------
Git. Commit 24302ba is the last one containing every purged row, so the
responses remain recoverable and the diagnosis in docs/METHOD.md (the quarantine rule)
stays reproducible from history. The evidence for the quarantine also survives
in prose: each entry in QUARANTINED carries the specific input that breaks it,
and a test asserts that it does.

    python scripts/provenance/purge_quarantined.py            # report only
    python scripts/provenance/purge_quarantined.py --go       # rewrite the files
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from llm_routing import paths                                        # noqa: E402
from llm_routing import routable as routable_mod                          # noqa: E402
from llm_routing.build_taskset import QUARANTINED                    # noqa: E402
from scripts.provenance.redraw_decisive import observed_cells, reestimate   # noqa: E402

Q = set(QUARANTINED)

# Every file that stores a task id. Mock caches are absent deliberately: they
# are gitignored and regenerate for free, so they are deleted outright rather
# than filtered.
#
# The per-ladder results files are named explicitly rather than left to a
# default: this list used to end in a single `runs/results.jsonl`, which the
# per-ladder restructure replaced with one file per ladder. A purge that
# still named the old path would have reported success while leaving every
# quarantined row in all three live results files.
JSONL_TARGETS = [
    "cache/raw_calls.wide.jsonl",
    "cache/raw_calls.claude.jsonl",
    "cache/raw_calls.deepseek.jsonl",
    "cache/routellm_scores.jsonl",
    "runs/results.probe.jsonl",
    "runs/results.wide.jsonl",
    "runs/results.claude.jsonl",
    "runs/results.deepseek.jsonl",
]
MOCK_CACHES = "cache/*.mock.jsonl"
REDRAW = "runs/redraw.{ladder}.json"


def purge_jsonl(path: Path, go: bool):
    """Drop every row naming a quarantined task. Returns (kept, dropped, $)."""
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [r for r in rows if r.get("task_id") not in Q]
    dropped = len(rows) - len(kept)
    spend = sum(r.get("cost_usd", 0.0) for r in rows if r.get("task_id") in Q)
    if go and dropped:
        shutil.copy2(path, path.with_suffix(path.suffix + ".prepurge"))
        # newline="" so the file stays byte-identical across platforms, matching
        # how build_taskset and response_cache write.
        with path.open("w", encoding="utf-8", newline="") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
    return len(kept), dropped, spend


def purge_redraw(path: Path, go: bool, n_total: int):
    """Drop the quarantined tasks from p_hat and RE-DERIVE the summary.

    The summary cannot be edited by hand: `expected` and `reproducible` are sums
    over the redrawn cells divided by the whole task set, so removing five tasks
    changes both the numerator and the denominator. reestimate() is imported
    from redraw_decisive rather than reimplemented, so the published numbers
    keep coming from exactly one formula.

    The CELLS are recomputed from the probe rather than inferred from p_hat.
    Inferring them is tempting and wrong: `observed` counts the single-draw
    verdict, so math-96 belongs in `routable` (one draw, cheap missed) even
    though its redrawn p_hat is cheap 1.0 / expensive 1.0. Reading the split
    off the probabilities moves tasks between cells and silently changes
    `observed`, which must not move at all here - every purged task was
    both_fail, so the routable count is untouched.
    """
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    p_hat = {k: v for k, v in d.get("p_hat", {}).items() if k not in Q}
    dropped = len(d.get("p_hat", {})) - len(p_hat)

    tasks = routable_mod.load_tasks(paths.TASKSET)
    cells = observed_cells(tasks, d.get("ladder", "wide"))
    routable, both_fail = cells["routable"], cells["both_fail"]

    # A redraw file only re-estimates the task set it was drawn on. When the
    # task set is REBUILT rather than trimmed, the current cross-tab contains
    # tasks this file never drew, and reestimate() would raise KeyError on the
    # first one - which is what happened when the code half went from 35 tasks
    # to 366.
    #
    # Refusing here rather than re-estimating over the overlap is deliberate.
    # `reproducible` is a correction to a PUBLISHED number, and computing it
    # from whichever tasks happen to appear in both files would produce a
    # figure that looks like the same quantity and is not.
    missing = [t["id"] for t in routable + both_fail if t["id"] not in p_hat]
    if missing:
        return {
            "stale": True, "dropped": dropped, "missing": len(missing),
            "example": missing[:3], "path": path,
            "drawn_for": len(d.get("p_hat", {})),
        }

    out = reestimate(cells, p_hat, n_total, d.get("tau", 0.2))

    before = {k: d.get(k) for k in ("observed", "expected", "reproducible", "noise_share")}
    if go:
        shutil.copy2(path, path.with_suffix(".json.prepurge"))
        d.update(p_hat=p_hat, n_graded=n_total, **out)
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return dropped, before, out, len(routable), len(both_fail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="rewrite the files")
    ap.add_argument("--ladder", default="wide")
    args = ap.parse_args()

    print(f"quarantined tasks: {', '.join(sorted(Q))}\n")

    total_spend, total_dropped = 0.0, 0
    for rel in JSONL_TARGETS:
        res = purge_jsonl(REPO / rel, args.go)
        if res is None:
            print(f"  {rel:<34} (absent)")
            continue
        kept, dropped, spend = res
        # Only the caches count towards discarded SPEND. A results row carries
        # attributed cost - what the policy would pay to serve alone - and
        # several policies are attributed the same underlying call, so adding
        # those to the cache total would count the same dollar repeatedly.
        if rel.startswith("cache/raw_calls"):
            total_spend += spend
        total_dropped += dropped
        flag = "  <- purged" if dropped else ""
        money = f"${spend:.4f}" if rel.startswith("cache/raw_calls") else "     -"
        print(f"  {rel:<34} keep {kept:>5}   drop {dropped:>4}   {money}{flag}")

    mocks = sorted((REPO / "cache").glob("*.mock.jsonl"))
    for m in mocks:
        print(f"  {m.relative_to(REPO).as_posix():<34} DELETE (derived, regenerates free)")
        if args.go:
            m.unlink()

    # n_total is the size of the task set the fraction is expressed over. Read
    # it rather than assumed, so this stays right if the task set changes again.
    n_total = sum(1 for l in paths.TASKSET.read_text(
        encoding="utf-8").splitlines() if l.strip())
    r = purge_redraw(REPO / REDRAW.format(ladder=args.ladder), args.go, n_total)
    if isinstance(r, dict) and r.get("stale"):
        print(f"\n  redraw.{args.ladder}.json  NOT UPDATED - it was drawn for a "
              f"different task set.")
        print(f"     it holds p_hat for {r['drawn_for']} task(s); the current "
              f"cross-tab has {r['missing']} task(s) it never drew,")
        print(f"     e.g. {', '.join(r['example'])}.")
        print(f"     `reproducible` is a correction to a published number, and "
              f"re-estimating it over\n     whichever tasks appear in both "
              f"files would produce a figure that looks like the\n     same "
              f"quantity and is not. Archive this file and redraw:")
        print(f"       ROUTER_MODE=real python scripts/provenance/redraw_decisive.py "
              f"--cells decisive --go")
    elif r:
        dropped, before, after, n_rout, n_bf = r
        print(f"\n  redraw.{args.ladder}.json  dropped {dropped} of "
              f"{dropped + n_rout + n_bf} p_hat entries; "
              f"{n_rout} routable + {n_bf} both_fail remain, over n={n_total}")
        for k in ("observed", "expected", "reproducible", "noise_share"):
            print(f"     {k:<14} {before[k]:.4f} -> {after[k]:.4f}")

    print(f"\n  total rows dropped: {total_dropped}")
    print(f"  real spend discarded: ${total_spend:.4f}")
    if not args.go:
        print("\nReport only. Add --go to rewrite. Originals are copied to *.prepurge,")
        print("and commit 24302ba is the last one holding every purged row.")


if __name__ == "__main__":
    main()
