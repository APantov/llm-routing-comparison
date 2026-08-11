"""Calibration / evaluation split.

Every threshold in this project is a free parameter: the agreement threshold,
RouteLLM's score threshold, the random baseline's rate, cascade_routing's quality
estimates. Fitting those on the same 100 tasks the results are then reported on
makes the results a measure of the fitting, not of the method. That is the
textbook version of the mistake, and an evaluation repo has no excuse for it.

So: split the task set once, deterministically, and keep the halves apart.

    CALIBRATION   tune thresholds, fit estimators, look at whatever you like
    EVALUATION    report numbers from here, and only here

STRATIFIED BY (domain, difficulty), not uniformly at random. The task set is only
100 items, and difficulty is the variable everything in this project turns on. An
unstratified coin flip could easily put most of the level-5 maths in one half,
which would make the two halves incomparable and the split worse than useless.

DETERMINISTIC, and keyed on task id rather than on file position, so the split
survives a rebuild of taskset.jsonl and does not silently change when the shuffle
order does.

    python -m llm_routing.splits            # show the split and check it is balanced
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from llm_routing import paths

# Fraction of tasks in the calibration half. A half-and-half split costs the most
# evaluation power; a 30/70 split keeps more tasks for reporting but calibrates on
# fewer. 0.5 is chosen because at n=100 the calibration estimates are already the
# noisier half of the problem, and cascade_routing's estimator tables are
# conditional probabilities that need the data more than the report does.
CALIBRATION_FRACTION = 0.5

# Bumping this reshuffles the split. It exists so that "we tried three splits and
# reported the best" is impossible to do by accident: changing it is a visible
# edit to a named constant.
SPLIT_SEED = 1


def _bucket(task):
    """The stratum a task belongs to: its domain and its difficulty band.

    Three difficulty bands rather than the raw proxy, because the raw proxy has
    only three distinct values on maths and a long tail on code. Bands make the
    two domains stratify the same way.
    """
    pct = task.get("difficulty_pct", 0.5)
    band = 0 if pct < 0.34 else (1 if pct < 0.67 else 2)
    return f"{task['domain']}-{band}"


def _rank(task):
    """A stable pseudo-random number in [0, 1) for one task.

    A hash rather than a shuffle, so a task's side of the split depends only on
    its own id and the seed. Adding or removing other tasks cannot move it.
    """
    key = f"{SPLIT_SEED}|{task['id']}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16) / float(1 << 64)


def split(tasks, fraction=None):
    """Return (calibration, evaluation), stratified and deterministic."""
    fraction = CALIBRATION_FRACTION if fraction is None else fraction
    by_bucket = {}
    for t in tasks:
        by_bucket.setdefault(_bucket(t), []).append(t)

    calib_ids = set()
    for _name, group in sorted(by_bucket.items()):
        group = sorted(group, key=_rank)
        n = int(round(fraction * len(group)))
        calib_ids.update(t["id"] for t in group[:n])

    # Preserve the caller's ordering in both halves, so downstream code that
    # assumes taskset order still behaves.
    calibration = [t for t in tasks if t["id"] in calib_ids]
    evaluation = [t for t in tasks if t["id"] not in calib_ids]
    return calibration, evaluation


def describe(calibration, evaluation):
    """Lines describing the split, for a run header."""
    out = [f"split: {len(calibration)} calibration / {len(evaluation)} evaluation "
           f"(seed {SPLIT_SEED}, stratified by domain and difficulty band)"]
    for name, half in (("calib", calibration), ("eval ", evaluation)):
        c = Counter(_bucket(t) for t in half)
        out.append(f"  {name} " + "  ".join(f"{k}={c[k]}" for k in sorted(c)))
    return out


def main():
    path = paths.TASKSET
    if not path.exists():
        sys.exit(f"{path.name} not found. Build it first: python -m llm_routing.build_taskset")
    with path.open(encoding="utf-8") as f:
        tasks = [json.loads(l) for l in f if l.strip()]

    calibration, evaluation = split(tasks)
    print("\n".join(describe(calibration, evaluation)))

    # A split is only useful if the halves are actually comparable, so check the
    # thing that would ruin them rather than assume stratification worked.
    print()
    print("balance check (the halves must be comparable, or the split is worse")
    print("than no split at all):")
    ok = True
    for domain in ("math", "code"):
        for name, half in (("calibration", calibration), ("evaluation", evaluation)):
            sub = [t for t in half if t["domain"] == domain]
            if not sub:
                print(f"  {domain:<5} {name:<12} EMPTY - cannot report on this domain")
                ok = False
                continue
            mean = sum(t["difficulty_pct"] for t in sub) / len(sub)
            print(f"  {domain:<5} {name:<12} n={len(sub):<3} mean difficulty_pct={mean:.3f}")
    for domain in ("math", "code"):
        halves = [[t for t in h if t["domain"] == domain] for h in (calibration, evaluation)]
        if all(halves):
            means = [sum(t["difficulty_pct"] for t in h) / len(h) for h in halves]
            gap = abs(means[0] - means[1])
            verdict = "ok" if gap < 0.10 else "LOPSIDED - try another SPLIT_SEED"
            if gap >= 0.10:
                ok = False
            print(f"  {domain:<5} difficulty gap between halves: {gap:.3f}  {verdict}")
    if not ok:
        sys.exit("\nsplit is not usable as-is; see above")


if __name__ == "__main__":
    main()
