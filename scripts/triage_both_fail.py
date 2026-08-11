#!/usr/bin/env python3
r"""Sort the `both_fail` tasks into "the model was wrong" and "the task is broken".

WHY THIS EXISTS
---------------
STATUS.md section 6 (the quarantine rule) found five MBPP+ tasks whose expected answers cannot be
derived from their prompt: they score a candidate against whatever the MBPP
reference happened to return on inputs the natural-language prompt never
describes. Those five were ALL of `always_expensive`'s failures on the eval
split, so leaving them in capped every policy in the project at 92% instead of
100%. The recorded lesson is the general one:

    "both rungs failed" reads as "hard" and is equally consistent with "broken".

At 365 code tasks the expected queue is 40-45 of these, and `STATUS.md` flags
the cost honestly: THE REAL COST OF B IS HUMAN, NOT FINANCIAL. This script does
not replace that judgement. It gathers the evidence mechanically and names the
specific inputs in dispute, so the human time goes to deciding rather than to
diffing hundred-element fuzzed input lists by eye.

THE DISCRIMINATOR, AND TWO THAT DO NOT WORK
-------------------------------------------
Every MBPP+ task carries both suites, which is the whole design of the swap (see
`build_taskset.load_mbppplus`):

    grader_payload["tests"]         the original thin asserts, SHOWN IN THE
                                    PROMPT by models.build_prompt. This is the
                                    specification the model was given.
    grader_payload["test_program"]  the expanded fuzzed suite. Grader only. The
                                    model never sees it.

*Rejected: "passes the prompt's asserts but fails the expanded suite".* That is
not a broken task, it is the entire point of MBPP+. Its own docstring gives the
example: a solution that forgets `n == 1` passes all four original asserts of
`is_not_prime` and fails the expanded suite, and under plain MBPP that was a
point the model did not earn. Measured here: 3 of 5 genuine capability failures
satisfy every prompt assert.

*Rejected: "mismatches only a few hidden inputs".* Measured against the five
adjudicated tasks, the broken ones span 1/106 to 119/123 mismatches - and
`codeplus-741`, a genuine failure the expensive rung then solved, sits at 1/104.
Broken and hard are indistinguishable on this axis.

*What works: independent candidates failing on THE SAME inputs.* Run every
prompt-conformant candidate the project has bought - the cheap rung's draws and
the expensive rung's answer, which are different model families - and intersect
the sets of hidden inputs each one gets "wrong". On the five adjudicated tasks
that intersection is non-empty every time, and on four of the five it is the
whole set (Jaccard 1.00): DeepSeek and Claude Opus, independently, disagree with
the reference on exactly the same cases. When two unrelated solutions that both
satisfy the written specification disagree with the reference in the same place,
the specification does not determine that place.

That intersection is also precisely the artefact the standing quarantine rule
demands - the specific input that breaks the task - so this prints those inputs
with their expected and actual values.

WHAT THIS IS AND IS NOT
-----------------------
It is a READING QUEUE, ordered by strength of evidence, with the disputed inputs
already extracted. It is not an oracle, and the buckets are priorities rather
than verdicts. Two measured limits, both found by checking it against the five
tasks a human adjudicated on 8 August 2026:

  It misses an internally inconsistent reference. `codeplus-305` gives mutually
  inconsistent expectations for identically-shaped inputs, so no two candidates
  fail in the same place and the intersection is empty. It is a broken task that
  looks exactly like a hard one here. That is why "all candidates fail, on
  disjoint inputs" is filed as `needs_read` rather than kept.

  A shared disputed input is not proof on its own. Every candidate failing on
  `[]` can mean the specification says nothing about the empty case - or that
  every model forgot it and the reference did not. Only the prompt settles that,
  which is why the printed line ends in a question for the reader rather than a
  verdict.

It will not quarantine anything. Quarantining is a deliberate edit to
`build_taskset.QUARANTINED` followed by `scripts/purge_quarantined.py --go`, and
the evidence sentence is written by a person who read the prompt.

    python scripts/triage_both_fail.py
    python scripts/triage_both_fail.py --json triage.json
    python scripts/triage_both_fail.py --show 3      # more disputed inputs each

Reads only files already on disk. No API calls, no spend, no network.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Generated code runs here. Same posture as graders.py - a subprocess with a
# timeout, because generated code loops forever often enough to matter.
TIMEOUT_S = 60

# The two shapes MBPP+ test programs end in. 367 of 370 use the first; tasks 88
# and 255 use the second, which recomputes the expectation from a reference
# function at test time rather than reading a precomputed list.
LOOP_MARKERS = (
    "for i, (inp, exp) in enumerate(zip(inputs, results)):",
    "for i, inp in enumerate(inputs):",
)


def _strip_fences(text):
    """Same rule graders.py applies, so this scores what the grader scored."""
    if text is None:
        return ""
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return fence.group(1) if fence else text


def _run(program):
    """Execute a program; return (returncode, stdout), or (None, "") if it hung."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True, text=True, timeout=TIMEOUT_S,
            )
            return proc.returncode, proc.stdout
        except subprocess.TimeoutExpired:
            return None, ""
        except OSError:
            return None, ""


def passes_shipped_asserts(code, payload):
    """Does the candidate satisfy the asserts the PROMPT showed it?

    `graders.grade` never asks this for an MBPP+ task - it goes straight to the
    expanded suite - so the information is on disk and has never been read. It is
    the filter, not the verdict: a candidate that fails here never had a claim on
    the task in the first place, so its disagreements carry no information about
    whether the specification is complete.
    """
    parts = [_strip_fences(code), payload.get("setup", "") or ""]
    parts.extend(payload.get("tests") or [])
    rc, _ = _run("\n".join(parts))
    return rc == 0


def disputed_inputs(code, test_program):
    """Which hidden cases does this candidate get "wrong", and what did it return?

    The shipped grader answers pass/fail, because it runs the suite as written and
    the first failing assertion ends the program. Deciding whether a task is
    underdetermined needs the whole set, so the final loop is replaced by one that
    catches per case and reports.

    Returns (bad_index_set, total, detail_by_index) or None when the program has a
    shape this cannot instrument - in which case the caller falls back to a human
    read rather than guessing.
    """
    marker = next((m for m in LOOP_MARKERS if m in test_program), None)
    if marker is None:
        return None

    head, _, body = test_program.rpartition(marker)
    body_lines = [l for l in body.splitlines() if l.strip()]
    if not body_lines:
        return None
    # The original loop body, verbatim, one level deeper so it can sit in a try.
    # Verbatim matters: it names the task's own function and carries its own
    # float tolerance, and rewriting either would change what is measured.
    indented = "\n".join("    " + l for l in body_lines)
    call = body_lines[0].strip()
    # `assertion(foo(*inp), exp, 0)` -> `foo(*inp)`, so the actual return value
    # can be reported next to the expected one.
    m = re.search(r"assertion\(\s*(.+?)\s*,\s*(?:exp|ref_func\(\*inp\))", call)
    expr = m.group(1) if m else None

    harness = f"""{head}
_has_results = 'results' in dir()
_seq = list(enumerate(zip(inputs, results))) if _has_results \\
       else list(enumerate(inputs))
_bad, _detail = [], {{}}
for _it in _seq:
    i = _it[0]
    if _has_results:
        inp, exp = _it[1]
    else:
        inp = _it[1]
        exp = None
    try:
{indented}
    except BaseException:
        _bad.append(i)
        try:
            _got = repr({expr if expr else 'None'})[:200]
        except BaseException as _e:
            _got = '<raised ' + type(_e).__name__ + '>'
        try:
            _want = repr(exp)[:200] if exp is not None else '<computed>'
        except BaseException:
            _want = '<unreprable>'
        try:
            _in = repr(inp)[:200]
        except BaseException:
            _in = '<unreprable>'
        _detail[i] = {{'input': _in, 'expected': _want, 'got': _got}}
import json as _json
print('__TRIAGE__' + _json.dumps(
    {{'bad': _bad, 'total': len(_seq), 'detail': _detail}}))
"""
    rc, out = _run(_strip_fences(code) + "\n" + harness)
    if rc is None:
        return None
    for line in out.splitlines():
        if line.startswith("__TRIAGE__"):
            d = json.loads(line[len("__TRIAGE__"):])
            return set(d["bad"]), d["total"], d["detail"]
    return None


def load_pool():
    """task_id -> raw MBPP+ record, for the reference solution.

    From data/mbppplus.json rather than taskset.jsonl, because `build_taskset`
    drops `_ref_code` before writing.
    """
    path = REPO / "data" / "mbppplus.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {f"codeplus-{r['task_id']}": r for r in json.load(f)}


def all_candidates(tasks, ladder):
    """task_id -> [(label, text)], every independent attempt on disk.

    Every greedy answer at temperature 0, at every rung and every draw index. The
    cheap rung's extra draws come from the screener, and they are independent in
    the way that matters here: DeepSeek is not deterministic at temperature 0, so
    repeat draws are genuinely different solutions rather than one solution
    counted three times.
    """
    from llm_routing import models

    path = REPO / "cache" / f"raw_calls.{ladder}.jsonl"
    if not path.exists():
        return {}
    by_id = {t["id"]: t for t in tasks}
    out = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("kind") != "answer" or d.get("temperature") not in (0, 0.0):
                continue
            task = by_id.get(d["task_id"])
            if task is None:
                continue
            # Same two exclusions routable.real_verdicts applies. A truncated
            # response is unmeasured, not wrong, and adjudicating a task on one
            # would blame the task for the token cap.
            if not models.is_reachable(d, task) or models.is_truncated(d):
                continue
            label = f"{d['tier']}#{d.get('sample_idx') or 0}"
            out[d["task_id"]].append((label, d.get("text") or ""))
    return out


def triage_one(task, cands, ref_code):
    """Gather the evidence for one task and put it in a bucket."""
    payload = task["grader_payload"]
    tp = payload["test_program"]

    # A candidate that does not satisfy the prompt's own examples has no claim on
    # the task, so its disagreements say nothing about whether the specification
    # is complete. Filtering here is what stops the "is_not_prime forgot n == 1"
    # case being read as a broken task.
    conformant = [(lbl, c) for lbl, c in cands
                  if passes_shipped_asserts(c, payload)]

    rec = {
        "id": task["id"],
        "n_candidates": len(cands),
        "n_prompt_conformant": len(conformant),
        "reference_passes_own_asserts": (
            passes_shipped_asserts(ref_code, payload) if ref_code else None),
    }

    if rec["reference_passes_own_asserts"] is False:
        rec["bucket"] = "broken"
        rec["why"] = ("the task's own reference solution fails the asserts its "
                      "prompt shows. Nothing can pass this.")
        return rec

    if not conformant:
        rec["bucket"] = "likely_hard"
        rec["why"] = ("no candidate satisfied even the asserts the prompt showed, "
                      "so this is the model being wrong rather than the task "
                      "being underdetermined.")
        return rec

    sets, details, uninstrumentable = [], {}, False
    for lbl, code in conformant:
        got = disputed_inputs(code, tp)
        if got is None:
            uninstrumentable = True
            continue
        bad, total, detail = got
        sets.append((lbl, bad))
        rec["total_hidden"] = total
        details[lbl] = detail

    if uninstrumentable and not sets:
        rec["bucket"] = "ambiguous"
        rec["why"] = ("satisfies the prompt, but the expanded suite could not be "
                      "instrumented. Read it.")
        return rec

    if any(not bad for _, bad in sets):
        # A prompt-conformant candidate that disputes nothing passed the whole
        # expanded suite, so the task was solvable and is not both_fail on this
        # evidence.
        rec["bucket"] = "likely_hard"
        rec["why"] = "a candidate passed the expanded suite outright."
        return rec

    core = set.intersection(*(bad for _, bad in sets)) if sets else set()
    rec["per_candidate_mismatches"] = {lbl: len(bad) for lbl, bad in sets}
    rec["core_disputed"] = sorted(core)

    if len(sets) < 2:
        rec["bucket"] = "needs_read"
        rec["why"] = (f"only one prompt-conformant candidate "
                      f"({sets[0][0]}), which is not independent evidence. "
                      f"It disputes {len(sets[0][1])} of "
                      f"{rec.get('total_hidden')} hidden inputs. Read it, or "
                      f"buy another draw.")
        return rec

    if not core:
        # NOT filed as hard. Several independent good-faith solutions failing,
        # each in its own place, is what a hard task looks like - and it is also
        # what `codeplus-305` looks like, whose reference gives mutually
        # inconsistent expectations for identically-shaped inputs, so no two
        # candidates fail identically. That one was adjudicated broken by a
        # human and this rule cannot tell it from hard, so it goes to a person
        # rather than being kept silently.
        rec["bucket"] = "needs_read"
        rec["why"] = (f"{len(sets)} independent prompt-conformant candidates all "
                      f"fail, on DISJOINT inputs. Usually that means each is "
                      f"wrong in its own way (a hard task) - but an internally "
                      f"inconsistent reference looks the same, so read it.")
        return rec

    # Name the disputed cases, taking the detail from whichever candidate
    # reported them.
    examples = []
    for i in sorted(core):
        for lbl in details:
            if str(i) in details[lbl]:
                d = details[lbl][str(i)]
                examples.append({"index": i, **d})
                break
    rec["examples"] = examples
    rec["bucket"] = "broken_evidence"
    rec["why"] = (
        f"{len(sets)} independent prompt-conformant candidates disagree with the "
        f"reference on the SAME {len(core)} of {rec.get('total_hidden')} hidden "
        f"inputs. THE QUESTION FOR YOU: does the prompt say what should happen "
        f"for the input(s) below? If not, the task is underdetermined. If it "
        f"does, every candidate simply missed the same edge case and the task "
        f"is fine.")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default="wide")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="also write the full triage as JSON")
    ap.add_argument("--show", type=int, default=2,
                    help="disputed inputs to print per broken task")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from llm_routing import models
    from llm_routing import run_eval
    from llm_routing.graders import grade

    tasks = run_eval.load_tasks(domain="code")
    pool = load_pool()
    cands = all_candidates(tasks, args.ladder)

    # both_fail, computed exactly as routable.py computes it: every rung's greedy
    # answer graded, task kept only when all of them failed.
    targets = []
    for t in tasks:
        greedy = {}
        for lbl, text in cands.get(t["id"], []):
            tier, _, idx = lbl.partition("#")
            if idx == "0":
                greedy[tier] = text
        if not all(tier in greedy for tier in models.TIERS):
            continue
        if any(grade(t, greedy[tier]) for tier in models.TIERS):
            continue
        targets.append(t)

    if args.limit:
        targets = targets[:args.limit]

    print(f"ladder {args.ladder}  |  {len(tasks)} code tasks  |  "
          f"{len(targets)} both_fail to triage")
    if not targets:
        print("\nnothing to adjudicate - no task failed at every rung.")
        return 0
    print()

    buckets = defaultdict(list)
    records = []
    for t in targets:
        ref = pool.get(t["id"], {}).get("code")
        rec = triage_one(t, cands.get(t["id"], []), ref)
        records.append(rec)
        buckets[rec["bucket"]].append(rec["id"])
        n = rec["n_prompt_conformant"]
        print(f"  {rec['id']:<18} {rec['bucket']:<15} "
              f"[{n}/{rec['n_candidates']} conformant] {rec['why']}")
        for ex in (rec.get("examples") or [])[:args.show]:
            print(f"       input {ex['input']}  ->  reference expects "
                  f"{ex['expected']}, every candidate returned {ex['got']}")

    print()
    print("=" * 74)
    print("  READING QUEUE, strongest evidence first. These are priorities,")
    print("  not verdicts - see the module docstring for what each can miss.")
    print("=" * 74)
    labels = {
        "broken_evidence": "read first  - independent candidates dispute the "
                           "same inputs",
        "needs_read":      "read next   - conformant candidates, no shared "
                           "disputed input",
        "likely_hard":     "read last   - nothing even satisfied the prompt's "
                           "own asserts",
    }
    for b in ("broken_evidence", "needs_read", "likely_hard"):
        print(f"  {b:<17}{len(buckets[b]):>4}   {labels[b]}")
        if buckets[b]:
            print(f"{'':<21}   {', '.join(buckets[b][:8])}"
                  f"{' ...' if len(buckets[b]) > 8 else ''}")
    print("=" * 74)
    print("\nValidated against the five tasks a human adjudicated on 8 August:")
    print("  4 of 5 land in `broken_evidence`, and the inputs printed for them")
    print("  match the evidence sentences in build_taskset.QUARANTINED. The")
    print("  fifth (codeplus-305, an internally inconsistent reference) lands in")
    print("  `needs_read` - which is why that bucket is read rather than kept.")
    print("\nNothing here is applied. Quarantining is a deliberate edit to")
    print("build_taskset.QUARANTINED with the disputed input named as evidence,")
    print("then `python scripts/purge_quarantined.py --go`.")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"ladder": args.ladder, "records": records}, indent=2),
            encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
