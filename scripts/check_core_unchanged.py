#!/usr/bin/env python3
"""Prove the agent layer's edit to `models.py` cannot move a benchmark number.

Adding the serving layer required exactly one change to the research core:
`models.py` gained a `general` domain prompt, a `code_untested` prompt, and a
branch in `_mock_call` for live queries that have no ground truth to perturb.

The claim is that none of that is reachable from `build_taskset.py` output -
every task in `taskset.jsonl` is `math` or `code`, every code task carries
asserts, and no task carries the `_live` marker. This script checks the claim
rather than asserting it, by fingerprinting every mock response the task set
can produce and comparing against the same fingerprint computed from the
version of `models.py` at a chosen git revision.

    python scripts/check_core_unchanged.py                # vs HEAD
    python scripts/check_core_unchanged.py --ref v0.1.0   # vs a tag

Exits non-zero on any divergence, so CI fails loudly if a future edit to the
serving path leaks into the experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLE_INDICES = (0, 1, 2, 3)

# The research core is a flat set of modules at the repo root, and `models`
# imports `response_cache` from there. Running from scripts/ puts scripts/ on
# the path instead, so the root has to be added explicitly.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fingerprint(mod, tasks: list[dict]) -> str:
    """Hash every mock response the task set can produce on the loaded ladder.

    Covers the prompt text as well as the response, because `build_prompt` is
    the other function the edit touched - a changed prompt would change the
    response-cache key and silently invalidate every cached real response.
    """
    h = hashlib.sha256()
    for task in tasks:
        h.update(mod.build_prompt(task, kind="answer").encode("utf-8"))
        h.update(mod.build_prompt(task, kind="route").encode("utf-8"))
        for tier in mod.TIERS:
            for sample_idx in SAMPLE_INDICES:
                for temperature in (0.0, 0.8):
                    r = mod._mock_call(
                        tier, mod.build_prompt(task), task, temperature, sample_idx
                    )
                    h.update(
                        f"{task['id']}|{tier}|{temperature}|{sample_idx}|"
                        f"{r.text}|{r.tokens_in}|{r.tokens_out}|{r.cost_usd:.10f}"
                        .encode("utf-8")
                    )
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="HEAD",
                    help="git revision to compare against (default: HEAD)")
    args = ap.parse_args()

    taskset = REPO / "taskset.jsonl"
    if not taskset.exists():
        print("taskset.jsonl not found; run: python build_taskset.py",
              file=sys.stderr)
        return 1
    tasks = [json.loads(l) for l in taskset.open(encoding="utf-8") if l.strip()]

    # Sanity: the whole argument rests on these being true of the task set.
    domains = {t["domain"] for t in tasks}
    if not domains <= {"math", "code"}:
        print(f"task set contains unexpected domains: {domains - {'math', 'code'}}\n"
              "The 'serving code is unreachable' argument no longer holds.",
              file=sys.stderr)
        return 1
    if any(t.get("_live") for t in tasks):
        print("a task carries the `_live` marker; that is a serving-only flag",
              file=sys.stderr)
        return 1
    for t in tasks:
        if t["domain"] == "code" and not t.get("grader_payload", {}).get("tests"):
            print(f"code task {t['id']} has no asserts, so it would reach the "
                  f"serving-only `code_untested` prompt", file=sys.stderr)
            return 1

    try:
        original = subprocess.run(
            ["git", "show", f"{args.ref}:models.py"],
            cwd=REPO, capture_output=True, check=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"could not read models.py at {args.ref}: {exc}", file=sys.stderr)
        return 1

    failures = []
    with tempfile.TemporaryDirectory() as td:
        ref_path = Path(td) / "models_ref.py"
        ref_path.write_text(original, encoding="utf-8")

        import os
        for ladder in ("claude", "deepseek", "wide"):
            os.environ["ROUTER_LADDER"] = ladder
            os.environ["ROUTER_MODE"] = "mock"
            for name in list(sys.modules):
                if name.startswith("models"):
                    del sys.modules[name]

            ref_mod = _load(ref_path, f"models_ref_{ladder}")
            cur_mod = _load(REPO / "models.py", f"models_cur_{ladder}")

            a, b = _fingerprint(ref_mod, tasks), _fingerprint(cur_mod, tasks)
            status = "identical" if a == b else "DIVERGED"
            print(f"  {ladder:<9} {status}  {b[:16]}")
            if a != b:
                failures.append(ladder)

    print()
    if failures:
        print(f"FAIL: mock output changed on {', '.join(failures)}.\n"
              f"An edit intended for the serving path has reached the "
              f"experiment. Every number in the repository is now suspect.",
              file=sys.stderr)
        return 1

    print(f"OK: {len(tasks)} tasks x 3 ladders x {len(SAMPLE_INDICES)} samples "
          f"x 2 temperatures - byte-identical to {args.ref}.")
    print("The serving-only branches are unreachable from the task set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
