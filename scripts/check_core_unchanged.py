#!/usr/bin/env python3
"""Prove the agent layer's edit to `models.py` cannot move a benchmark number.

Adding the serving layer required exactly one change to the research core:
`models.py` gained a `general` domain prompt, a `code_untested` prompt, and a
branch in `_mock_call` for live queries that have no ground truth to perturb.

The claim is that none of that is reachable from `build_taskset.py` output -
every task in `taskset.jsonl` is `math` or `code`, every code task carries
asserts, and no task carries the `_live` marker. This script checks the claim
rather than asserting it, by fingerprinting every mock response the task set
can produce and comparing against a frozen baseline.

    python scripts/check_core_unchanged.py                # vs the frozen baseline
    python scripts/check_core_unchanged.py --ref v0.1.0   # vs a git revision
    python scripts/check_core_unchanged.py --update       # re-freeze, deliberately

WHY THE BASELINE IS A FILE AND NOT `HEAD`.

This used to default to `--ref HEAD`, which made it useless in the one place it
was supposed to run. CI checks out a commit, so the working tree IS HEAD, the
reference copy and the current copy were byte-identical by construction, and the
comparison could never fail. It only ever caught uncommitted local drift, while
its own docstring promised CI would fail loudly. A guard that cannot fail is
worse than no guard, because it is read as evidence.

The baseline now lives in `tests/frozen_mock_fingerprints.json`, on the same
principle as `tests/frozen_probe.json`: moving it is a deliberate edit that
shows up in review, rather than something that happens silently whenever
somebody commits.

`--ref` is kept, because comparing against a named revision is genuinely useful
when bisecting - it is just the wrong default.

Exits non-zero on any divergence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLE_INDICES = (0, 1, 2, 3)
LADDERS = ("claude", "deepseek", "wide")
BASELINE = REPO / "tests" / "frozen_mock_fingerprints.json"

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


def _fingerprint_all(models_path: Path, tasks: list[dict], tag: str) -> dict:
    """{ladder: sha256} for one copy of models.py, across every ladder.

    The ladder is baked into module-level constants at import time, so each one
    needs a fresh exec with the environment already set.
    """
    out = {}
    for ladder in LADDERS:
        os.environ["ROUTER_LADDER"] = ladder
        os.environ["ROUTER_MODE"] = "mock"
        for name in list(sys.modules):
            if name.startswith("models"):
                del sys.modules[name]
        out[ladder] = _fingerprint(_load(models_path, f"models_{tag}_{ladder}"), tasks)
    return out


def _load_tasks() -> list[dict] | None:
    taskset = REPO / "data" / "taskset.jsonl"
    if not taskset.exists():
        print("taskset.jsonl not found; run: python -m llm_routing.build_taskset",
              file=sys.stderr)
        return None
    tasks = [json.loads(l) for l in taskset.open(encoding="utf-8") if l.strip()]

    # Sanity: the whole argument rests on these being true of the task set.
    domains = {t["domain"] for t in tasks}
    if not domains <= {"math", "code"}:
        print(f"task set contains unexpected domains: {domains - {'math', 'code'}}\n"
              "The 'serving code is unreachable' argument no longer holds.",
              file=sys.stderr)
        return None
    if any(t.get("_live") for t in tasks):
        print("a task carries the `_live` marker; that is a serving-only flag",
              file=sys.stderr)
        return None
    for t in tasks:
        if t["domain"] == "code" and not t.get("grader_payload", {}).get("tests"):
            print(f"code task {t['id']} has no asserts, so it would reach the "
                  f"serving-only `code_untested` prompt", file=sys.stderr)
            return None
    return tasks


def _reference_from_git(ref: str, tasks: list[dict]) -> dict | None:
    """Fingerprint the copy of models.py at a git revision."""
    # models.py moved from the repository root into llm_routing/ during the
    # package restructure. Both locations are tried, newest first, so `--ref`
    # still reaches revisions from either side of that move.
    original = None
    for rel in ("llm_routing/models.py", "models.py"):
        try:
            original = subprocess.run(
                ["git", "show", f"{ref}:{rel}"],
                cwd=REPO, capture_output=True, check=True, text=True,
            ).stdout
            break
        except subprocess.CalledProcessError:
            continue
        except FileNotFoundError as exc:
            print(f"git is not available: {exc}", file=sys.stderr)
            return None
    if original is None:
        print(f"could not read models.py at {ref} from either "
              f"llm_routing/models.py or models.py", file=sys.stderr)
        return None

    # A pre-move reference copy opens with a flat `import response_cache`, which
    # no longer resolves. Alias it to the current module, which is exactly what
    # that line used to bind to when response_cache.py sat next to models.py.
    # Nothing in the fingerprint comes from the cache - _mock_call is pure - so
    # the alias cannot change what is being compared.
    from llm_routing import response_cache as _rc
    sys.modules.setdefault("response_cache", _rc)

    with tempfile.TemporaryDirectory() as td:
        ref_path = Path(td) / "models_ref.py"
        ref_path.write_text(original, encoding="utf-8")
        return _fingerprint_all(ref_path, tasks, "ref")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default=None,
                    help="compare against a git revision instead of the frozen "
                         "baseline (e.g. --ref v0.1.0)")
    ap.add_argument("--update", action="store_true",
                    help="re-freeze the baseline from the working tree")
    args = ap.parse_args()

    tasks = _load_tasks()
    if tasks is None:
        return 1

    current = _fingerprint_all(REPO / "llm_routing" / "models.py", tasks, "cur")

    if args.update:
        BASELINE.write_text(
            json.dumps({"_comment": "Frozen mock fingerprints. See "
                                    "scripts/check_core_unchanged.py.",
                        "sample_indices": list(SAMPLE_INDICES),
                        "fingerprints": current}, indent=2) + "\n",
            encoding="utf-8")
        print(f"re-froze {BASELINE.relative_to(REPO)}:")
        for ladder, fp in current.items():
            print(f"  {ladder:<9} {fp}")
        print("\nCommit this deliberately, and say in the message why the mock moved.")
        return 0

    if args.ref:
        expected = _reference_from_git(args.ref, tasks)
        if expected is None:
            return 1
        source = f"models.py at {args.ref}"
    else:
        if not BASELINE.exists():
            print(f"{BASELINE.relative_to(REPO)} is missing. Create it with:\n"
                  f"  python scripts/check_core_unchanged.py --update",
                  file=sys.stderr)
            return 1
        expected = json.loads(BASELINE.read_text(encoding="utf-8"))["fingerprints"]
        source = str(BASELINE.relative_to(REPO))

    failures = []
    for ladder in LADDERS:
        want, got = expected.get(ladder), current[ladder]
        status = "identical" if want == got else "DIVERGED"
        print(f"  {ladder:<9} {status}  {got[:16]}")
        if want != got:
            failures.append(ladder)

    print()
    if failures:
        print(f"FAIL: mock output changed on {', '.join(failures)}, against "
              f"{source}.\n"
              f"An edit intended for the serving path has reached the "
              f"experiment. Every number in the repository is now suspect.\n"
              f"If the mock was changed on purpose, re-freeze with:\n"
              f"  python scripts/check_core_unchanged.py --update",
              file=sys.stderr)
        return 1

    print(f"OK: {len(tasks)} tasks x {len(LADDERS)} ladders x "
          f"{len(SAMPLE_INDICES)} samples x 2 temperatures - byte-identical to "
          f"{source}.")
    print("The serving-only branches are unreachable from the task set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
