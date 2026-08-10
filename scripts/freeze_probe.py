#!/usr/bin/env python3
"""Re-freeze the probe figures the docs quote, after a deliberate task-set change.

WHY THIS EXISTS
---------------
`tests/test_findings.py` used to carry the probe's magnitudes as literals -
`assert probe.n == 95`, `routable_pct == 15.8`. They went stale twice: the
6 August 2026 task-set rebuild moved every one of them, and the 10 August code-
half rebuild (35 tasks to 366) moved them again. Both times the test failed for
the right reason and taught nothing, because a pinned literal cannot tell
"the measurement moved because the task set moved" from "the measurement broke".

test_experiment.py's docstring already states the rule:

    These target the arithmetic and the invariants, not the findings. A test
    that pinned an accuracy figure would just re-create the staleness problem
    the repo already hit once.

So the magnitudes live in `tests/frozen_probe.json`, and this script is the
deliberate act of moving them. The test then asserts drift rather than value:
it fails when the probe changes WITHOUT someone having decided it should.

WHEN TO RUN IT
--------------
After a task-set change you intended - a rebuild, a quarantine, a reversal - and
only once the adjudication behind that change is finished. Running it to make a
red test green is the one use that defeats the point: `both_fail` moving is how
you find out that unpassable tasks have entered the set, and this script would
bless them.

    python scripts/freeze_probe.py            # show the diff, write nothing
    python scripts/freeze_probe.py --go       # write it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SNAPSHOT = REPO / "tests" / "frozen_probe.json"

# The rate attributes carried in the snapshot. Kept explicit rather than
# reflected off the object so that adding a field to Probe does not silently
# widen what the test pins.
RATES = ("routable_pct", "ceiling_pct", "rescue_rate", "cheap_acc",
         "expensive_acc")


def snapshot():
    from router_agent import findings

    probe = findings.load_probe()
    if probe is None:
        sys.exit("no measured results.probe.jsonl to freeze")
    lo, hi = probe.ci95()
    return {
        "note": "Regenerate with scripts/freeze_probe.py after a DELIBERATE "
                "task-set change. See that script for why this is a file "
                "rather than literals in the test.",
        "cells": {
            "n": probe.n, "both_ok": probe.both_ok, "routable": probe.routable,
            "both_fail": probe.both_fail, "inverted": probe.inverted,
        },
        "rates": {k: getattr(probe, k) for k in RATES},
        "ci95": [lo, hi],
        "by_domain": {
            d: {"routable_pct": v["routable_pct"], "both_fail": v["both_fail"]}
            for d, v in probe.by_domain.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="write the snapshot")
    args = ap.parse_args()

    new = snapshot()
    old = (json.loads(SNAPSHOT.read_text(encoding="utf-8"))
           if SNAPSHOT.exists() else None)

    print(f"{'field':<28}{'frozen':>14}{'now':>14}")
    print("-" * 56)
    moved = False
    for section in ("cells", "rates", "by_domain"):
        for key, val in new[section].items():
            was = (old or {}).get(section, {}).get(key)
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    w2 = (was or {}).get(k2)
                    flag = "" if w2 == v2 else "  <-- moved"
                    moved = moved or w2 != v2
                    print(f"{section}.{key}.{k2:<12}"[:28].ljust(28)
                          + f"{str(w2):>14}{str(v2):>14}{flag}")
            else:
                v = round(val, 4) if isinstance(val, float) else val
                w = round(was, 4) if isinstance(was, float) else was
                flag = "" if w == v else "  <-- moved"
                moved = moved or w != v
                print(f"{key:<28}{str(w):>14}{str(v):>14}{flag}")

    if not moved and old is not None:
        print("\nnothing moved; snapshot already current.")
        return 0

    if old is not None:
        bf_old = old["cells"]["both_fail"]
        bf_new = new["cells"]["both_fail"]
        if bf_new > bf_old:
            print(f"\n  !! both_fail rose {bf_old} -> {bf_new}. Every one of "
                  f"those is either\n     genuinely hard or unpassable-by-spec, "
                  f"and an unpassable task silently\n     caps every policy. "
                  f"Adjudicate before freezing:\n"
                  f"       python scripts/triage_both_fail.py")

    if not args.go:
        print("\nplan only - nothing written. Add --go once the moves above are "
              "ones you meant.")
        return 0

    SNAPSHOT.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {SNAPSHOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
