"""Download MBPP+ and convert it to the shape build_taskset.py expects.

WHY THIS SCRIPT EXISTS
----------------------
MBPP+ (evalplus/mbppplus) is the same 378 MBPP problems with roughly 35x more
test cases each. The problems are unchanged; what changes is how hard it is to
pass them, because the original MBPP suites are thin enough that a wrong solution
routinely slips through.

Demonstrated on task 3 (`is_not_prime`): a solution that forgets the n == 1 case
passes all four original asserts and fails the expanded suite. Under MBPP that
model scores a point it did not earn. Under MBPP+ it does not. Since this project
measures a cheap model's failure rate, and that rate was 0/10 against original
MBPP, tightening the tests is the cheapest way to make the task set discriminate
again.

    python scripts/provenance/fetch_mbppplus.py     # writes data/mbppplus.json
    python -m llm_routing.build_taskset
    python -m llm_routing.sanity_check

NEEDS THE `datasets` PACKAGE, and only this script does. It is a one-time
conversion: after it runs, data/mbppplus.json is plain JSON and the rest of the
repo reads it with the standard library alone.

    pip install datasets

THE NUMPY CAVEAT, which is a real cost and is not hidden. Every MBPP+ test
program begins `import numpy as np` and uses `np.allclose` to compare float
results. So grading the code half now requires numpy at runtime. Mock mode still
needs nothing for the maths half, but `graders.grade_test_program` will fail
without numpy. That is the price of using the official suites verbatim rather
than a rewritten approximation of them, and verbatim is worth more here: a
hand-modified benchmark is no longer the benchmark.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from llm_routing import paths                                        # noqa: E402

OUT = paths.DATA / "mbppplus.json"

# Seconds per validation run. The expanded suites take about 0.1s each; this is
# headroom for a slow machine rather than a tuned value.
VALIDATE_TIMEOUT_S = 30

# task_id 1-10 are MBPP's canonical few-shot prompt examples, excluded for the
# same reason build_taskset excludes them from plain MBPP: they are the examples
# a model may legitimately have been shown, so they measure recall rather than
# capability.
FEWSHOT_IDS = set(range(1, 11))


def reference_passes(row):
    """Does this task's own reference solution pass its own expanded suite?

    Run HERE, on the machine that will run the evaluation, because the answer is
    not the same everywhere. MBPP+'s assertion helper only reaches np.allclose
    when the expected value is a float or a flat sequence of floats; anything
    else - a nested tuple, a complex number - falls through to exact `==`. For a
    handful of tasks that makes the verdict depend on whether the platform's libm
    returns bit-identical results to the machine the expectations were generated
    on.

    Task 590 (`polar_rect`) is the worked example: its expected value is
    `(tuple, complex)`, so `is_floats` is False, atol stays 0, and correctness
    comes down to exact float equality on `cmath.polar`. It passes on Linux and
    fails on Windows. Neither verdict is about the model.
    """
    program = row["code"] + "\n" + row["test"]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "ref.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True, timeout=VALIDATE_TIMEOUT_S,
            )
            if proc.returncode == 0:
                return True, ""
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            return False, (tail[-1][:120] if tail else "non-zero exit")
        except subprocess.TimeoutExpired:
            return False, f"timed out after {VALIDATE_TIMEOUT_S}s"
        except OSError as e:
            return False, f"could not run: {e}"


def validate(rows, quiet=False):
    """Drop tasks whose own reference solution fails, and say which and why.

    This is not tidying. A task whose reference cannot pass is actively harmful
    in two directions:

      - the MOCK emits `_ref_code` as its "correct" answer, so a mock-correct
        answer would be graded WRONG and every mock number on that task is
        silently corrupted;
      - a REAL model that produces the canonically correct answer is also marked
        wrong, which reads as a capability result rather than a broken task.

    sanity_check.py catches this after the fact and refuses to proceed. Filtering
    here means it never gets that far.
    """
    kept, dropped = [], []
    for i, row in enumerate(rows, 1):
        ok, why = reference_passes(row)
        (kept if ok else dropped).append(row if ok else (row["task_id"], why))
        if not quiet and i % 50 == 0:
            print(f"  validated {i}/{len(rows)} ...", file=sys.stderr)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument(
        "--no-validate", action="store_true",
        help="skip the reference-solution check. Not recommended: sanity_check.py "
             "will then fail instead, later and less informatively.",
    )
    ap.add_argument(
        "--validate-only", action="store_true",
        help="re-validate an existing data/mbppplus.json and rewrite it, without "
             "downloading anything. Use this if the download already happened.",
    )
    args = ap.parse_args()

    if args.validate_only:
        path = Path(args.out)
        if not path.exists():
            sys.exit(f"{path} not found - nothing to validate. Run without "
                     f"--validate-only to download it first.")
        with path.open(encoding="utf-8") as f:
            rows = json.load(f)
        print(f"validating {len(rows)} existing tasks on THIS machine "
              f"({sys.platform}) ...", file=sys.stderr)
        kept, dropped = validate(rows)
        _write(path, kept, dropped)
        return

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "\nThis script needs the `datasets` package (only this script does).\n"
            "  pip install datasets\n\n"
            "  Alternative without it: download the parquet by hand from\n"
            "  https://huggingface.co/datasets/evalplus/mbppplus/tree/main/data\n"
            "  and convert the columns task_id, prompt, code, test to JSON.\n"
        )

    print("downloading evalplus/mbppplus ...", file=sys.stderr)
    ds = load_dataset("evalplus/mbppplus", split="test")

    rows = []
    for r in ds:
        if r["task_id"] in FEWSHOT_IDS:
            continue
        if not r.get("test"):
            continue
        rows.append({
            "task_id": r["task_id"],
            "prompt": r["prompt"],
            # The reference solution. Used only by the mock model, which needs
            # something that genuinely passes, and by sanity_check.py.
            "code": r["code"],
            # The EXPANDED suite: a self-contained program that defines its own
            # assertion helper and runs every input case. This is the whole point
            # of MBPP+ and it is what the grader executes.
            "test": r["test"],
            # The original thin asserts, kept for reference and for the
            # difficulty proxy. NOT used for grading - grading them would throw
            # away everything this dataset is for.
            "test_list": list(r.get("test_list") or []),
        })

    dropped = []
    if not args.no_validate:
        print(f"validating {len(rows)} reference solutions on THIS machine "
              f"({sys.platform}) ...", file=sys.stderr)
        rows, dropped = validate(rows)

    _write(Path(args.out), rows, dropped)


def _write(out, rows, dropped):
    out.parent.mkdir(parents=True, exist_ok=True)
    # newline="" and LF, matching every other artefact this repo writes, so the
    # file is byte-identical on Windows and Linux.
    with out.open("w", encoding="utf-8", newline="") as f:
        json.dump(rows, f, indent=1)
        f.write("\n")

    sizes = sorted(len(r["test"]) for r in rows)
    print(f"wrote {len(rows)} tasks -> {out}")
    print(f"  expanded test programs: min {sizes[0]}, median {sizes[len(sizes)//2]}, "
          f"max {sizes[-1]} characters")

    if dropped:
        print(f"\n  DROPPED {len(dropped)} task(s) whose own reference solution "
              f"fails its own expanded suite on this machine:")
        for task_id, why in dropped:
            print(f"    {task_id}: {why}")
        print("\n  These are not model failures. A task whose own reference answer")
        print("  cannot pass cannot measure a model, and it actively corrupts the")
        print("  mock, which emits that reference as its 'correct' answer.")
        print("\n  One known cause, seen on task 590 (`polar_rect`): MBPP+'s assertion")
        print("  helper only reaches np.allclose when the expected value is a float")
        print("  or a flat float sequence. For a nested tuple or a complex number it")
        print("  falls through to exact `==`, so the verdict turns on whether your")
        print("  platform's libm matches the machine the expectations came from.")
        print("  That one is genuinely platform-specific - it passes on Linux and")
        print("  fails on Windows - which is why this check runs locally rather than")
        print("  shipping a fixed exclusion list.")
        print("\n  Other causes are possible; the error above is what actually")
        print("  happened, and is the thing to read rather than this explanation.")

    print("\nnext:  python -m llm_routing.build_taskset && python -m llm_routing.sanity_check")


if __name__ == "__main__":
    main()
