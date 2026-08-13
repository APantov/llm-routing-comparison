"""`llm-router` - route one query and show the work.

The trace is the product. Anyone can call a model; the reason to run a router
is to see which rung answered, what verification concluded, whether it
escalated and what that cost. So the default output is the trace, and
`--json` is there for machines.

    llm-router "What is 17 * 23?"
    llm-router --demo                       # real cached data, no key, $0
    llm-router --estimate "..."             # price every policy, no calls
    llm-router --findings                   # what the benchmark measured
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _force_utf8_stdout() -> None:
    """Make stdout able to print a model's answer on Windows.

    Windows consoles default to cp1252, which cannot encode the characters
    real answers are full of - `√`, `±`, `≤`, every Greek letter.
    Printing an Opus answer to a default Windows terminal raises
    UnicodeEncodeError partway through, which turns a working router into a
    traceback for anyone on Windows.

    `errors="replace"` rather than `"strict"`: a mangled character is a
    cosmetic problem, a crash in the middle of the deliverable is not.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # already wrapped, or not a real stream (pytest capture)


_force_utf8_stdout()

# ANSI, disabled when not a tty or when NO_COLOR is set (informal standard,
# https://no-color.org). Piping the trace into a file should not produce
# escape sequences.
_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


DIM = lambda s: _c("2", s)
BOLD = lambda s: _c("1", s)
GREEN = lambda s: _c("32", s)
YELLOW = lambda s: _c("33", s)
RED = lambda s: _c("31", s)
CYAN = lambda s: _c("36", s)

NODE_STYLE = {
    "classify": CYAN, "answer": BOLD, "verify": YELLOW,
    "escalate": RED, "finalize": GREEN,
}


def _render_event(ev: dict) -> str:
    style = NODE_STYLE.get(ev["node"], str)
    return f"  {style(ev['node'].ljust(9))} {ev['detail']}"


def _print_estimate(est: dict) -> None:
    """Render a cost projection for a human. `--json` prints the payload raw.

    The numbers here are unrounded floats carrying full binary noise
    (0.00011270000000000002), which is correct to keep in the payload and wrong
    to put in front of a reader. Formatting happens at the edge, so nothing
    downstream inherits a rounded figure.
    """
    print()
    print(BOLD(f"  cost projection  ") + DIM(f"({est['domain']} domain, "
                                             f"{est['ladder']} ladder)"))
    print(DIM("  no model is called; this is arithmetic over the price table"))
    print()
    for tier, model in est["rungs"].items():
        print(f"    {DIM(tier.ljust(10))} {model}")
    print()
    for e in est["estimates"]:
        lo, hi = e["min_usd"], e["max_usd"]
        span = f"${lo:.6f}" if lo == hi else f"${lo:.6f} - ${hi:.6f}"
        print(f"    {e['policy'].ljust(18)} {span}")

    v = est.get("recommended_policy") or {}
    print()
    if v.get("verdict"):
        print(BOLD("  recommended policy   ") + GREEN(v["verdict"])
              + DIM(f"        (measured on the {v['ladder']} ladder)"))
        pct = v.get("cascade_vs_always_best_pct")
        if pct is not None:
            print(f"  cascade vs always-best, at matched accuracy   {pct:+.1f}%")
        print(DIM(f"  source: {v.get('economics_source')}"))
    else:
        print(YELLOW("  recommended policy   unknown"))
        print(DIM(f"  {v.get('note', 'no frontier run for this ladder')}"))
    print()
    print(DIM("  " + est["basis"].replace(". ", ".\n  ")))
    print()


def _print_outcome(out, show_answer: bool = True) -> None:
    d = out.to_dict()

    print()
    for ev in d["trace"]:
        print(_render_event(ev))
    print()

    verdict = GREEN("VERIFIED") if d["verified"] else YELLOW("UNVERIFIED")
    conf = "" if d["confidence"] is None else f" ({d['confidence']:.0%} agreement)"
    print(f"  {BOLD('verdict')}   {verdict}{conf}")
    print(f"            {DIM(d['verified_meaning'])}")
    print(f"  {BOLD('answered')}  {d['final_tier']} - {d['final_model']}")

    # Built separately rather than nested inside the f-string: nesting the same
    # quote character inside an f-string expression is a syntax error before
    # Python 3.12, and this package supports 3.10.
    breakdown = "({} calls, {} escalation(s))".format(
        d["n_calls"], d["escalations"]
    )
    print(f"  {BOLD('cost')}      ${d['cost_usd']:.6f}  {DIM(breakdown)}")
    print(f"  {BOLD('stopped')}   {d['stop_reason']}")

    if d["simulated"]:
        print()
        print(RED(
            "  SIMULATED. ROUTER_MODE=mock fabricates responses; the answer "
            "text is a placeholder\n  and means nothing about any model. Use "
            "--demo for real cached output, or\n  ROUTER_MODE=real with a key."
        ))

    if show_answer and d["answer"]:
        print()
        print(BOLD("  answer"))
        for line in d["answer"].strip().splitlines():
            print(f"    {line}")
    print()


def _demo(args) -> int:
    """Route a real benchmark question through the real cached responses.

    The whole point: a reviewer with no API key, no account and no money can
    watch the router escalate over genuine model output, because the 5,075
    responses this project paid for are committed to the repository.
    """
    import json as _json
    from pathlib import Path

    from llm_routing import paths

    taskset = paths.TASKSET
    if not taskset.exists():
        print(
            "data/taskset.jsonl not found. Build it first:\n"
            "    python -m llm_routing.build_taskset",
            file=sys.stderr,
        )
        return 1

    rows = [_json.loads(l) for l in taskset.open(encoding="utf-8") if l.strip()]

    # Declare which ids are live before the cache is first read. Without this,
    # response_cache cannot tell a conflict that invalidates a comparison from
    # one on a task that no longer exists, so it hedges with a note - and that
    # note landed in the middle of the demo trace, which is the one place in the
    # repository where a reader is watching the output line by line. The demo
    # has just loaded the task set, so it can answer the question instead.
    from llm_routing import response_cache as _rc
    _rc.LIVE_TASK_IDS = {r["id"] for r in rows}

    wanted = args.domain if args.domain in ("math", "code") else "math"
    pool = [r for r in rows if r["domain"] == wanted]
    if not pool:
        print(f"no {wanted} tasks in taskset.jsonl", file=sys.stderr)
        return 1
    task = pool[args.demo_index % len(pool)]

    # The wide ladder is the only one with real cached responses.
    os.environ["ROUTER_LADDER"] = "wide"
    os.environ["ROUTER_MODE"] = "replay"

    from router_agent.config import RouterConfig
    from router_agent.engine import route

    cfg = RouterConfig(mode="replay", ladder="wide", policy=args.policy)

    print(BOLD(f"\n  demo - benchmark task {task['id']}, replayed from cache"))
    print(DIM("  ladder=wide (DeepSeek v4-flash -> Opus 5)  mode=replay  $0.00 spent"))
    print(DIM(f"  {task['prompt'][:150]}"))

    # Hand the code half its asserts. Without them `live.synthesize_task` builds
    # the serving-only `code_untested` prompt, which no paid run ever recorded,
    # so replay raised ReplayMiss and `--demo --domain code` crashed - including
    # in CI, which runs exactly this line. Passing them reproduces the benchmark
    # prompt byte for byte, which is what the committed responses answer.
    #
    # It is also the honest demo. A benchmark code question IS its asserts, and
    # `scripts/demo.py` trace 3 makes the same point explicitly: the perfect
    # verifier is available only because the CALLER brought the tests.
    tests = task.get("grader_payload", {}).get("tests") if wanted == "code" else None
    out = route(task["prompt"], cfg=cfg, domain=wanted, tests=tests)
    _print_outcome(out, show_answer=not args.quiet)

    print(DIM(
        "  These are real model responses, paid for once on 6 August 2026 and\n"
        "  committed to cache/raw_calls.wide.jsonl. `cost` is what serving this\n"
        "  query would cost in production; this run spent nothing."
    ))
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="llm-router",
        description=(
            "Route a query through a cost-aware cascade and show the trace. "
            "Policy defaults come from the benchmark in this repository."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  llm-router --demo                       real cached data, no key, $0\n"
            "  llm-router \"What is 17 * 23?\"\n"
            "  llm-router --estimate \"prove X\"          price every policy, no calls\n"
            "  llm-router --findings                   what the benchmark measured\n"
        ),
    )
    p.add_argument("query", nargs="?", help="the question to route")
    p.add_argument("--policy", default="cascade",
                   choices=["cascade", "predictive", "always_cheap", "always_expensive"])
    p.add_argument("--ladder", default=None, help="claude | deepseek | wide")
    p.add_argument("--mode", default=None, help="mock | real | replay")
    p.add_argument("--domain", default="auto", help="auto | math | code | general")
    p.add_argument("--verifier", default="auto",
                   choices=["auto", "self_consistency", "tests", "none"])
    p.add_argument("-k", "--samples", type=int, default=None,
                   help="self-consistency samples (default 5)")
    p.add_argument("--agreement", type=float, default=None,
                   help="agreement threshold to accept (default 0.8)")
    p.add_argument("--max-cost", type=float, default=None,
                   help="hard USD ceiling for this query (default 0.50)")
    p.add_argument("--approve-above", type=float, default=None,
                   help="pause for approval when an escalation exceeds this USD")
    p.add_argument("--test", action="append", dest="tests", default=None,
                   help="an assert the answer must pass; repeatable")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--estimate", action="store_true",
                   help="project every policy's cost without calling a model")
    p.add_argument("--findings", action="store_true",
                   help="print what the benchmark measured, then exit")
    p.add_argument("--demo", action="store_true",
                   help="replay a benchmark task through the real cached responses")
    p.add_argument("--demo-index", type=int, default=0, help="which demo task")
    p.add_argument("--quiet", "-q", action="store_true", help="trace only, no answer")

    args = p.parse_args(argv)

    # Env must be set before `models` is imported: it reads ROUTER_LADDER at
    # module scope and builds the ladder once.
    if args.ladder:
        os.environ["ROUTER_LADDER"] = args.ladder
    if args.mode:
        os.environ["ROUTER_MODE"] = args.mode

    if args.findings:
        from router_agent import findings
        ladder = os.environ.get("ROUTER_LADDER", "claude")
        print(json.dumps(findings.summary(ladder), indent=2))
        return 0

    if args.demo:
        return _demo(args)

    if not args.query:
        p.print_help()
        return 2

    from router_agent.config import RouterConfig
    from router_agent.engine import estimate, resume, route

    cfg = RouterConfig.from_env(
        policy=args.policy, domain=args.domain, verifier=args.verifier,
        self_consistency_k=args.samples, agreement_threshold=args.agreement,
        max_cost_usd=args.max_cost, require_approval_above_usd=args.approve_above,
    )

    if args.estimate:
        est = estimate(args.query, cfg, tests=args.tests)
        if args.json:
            print(json.dumps(est, indent=2))
        else:
            _print_estimate(est)
        return 0

    try:
        out = route(args.query, cfg=cfg, tests=args.tests)
    except KeyError as exc:
        # The characteristic replay failure: this exact prompt was never paid
        # for. Worth an explanation rather than a traceback, because it is the
        # first thing anyone hits when they try replay with their own question.
        print(
            RED("replay mode has no cached response for this query.\n") +
            "Replay can only serve prompts that were actually paid for - the "
            "100 benchmark\ntasks. Try:\n"
            "    llm-router --demo                (a benchmark task, $0)\n"
            "    ROUTER_MODE=real llm-router ...  (your own query, costs money)\n"
            f"\ndetail: {str(exc)[:300]}",
            file=sys.stderr,
        )
        return 1

    # Human approval, interactively.
    while out.interrupted:
        payload = out.interrupted
        print()
        print(YELLOW(f"  APPROVAL NEEDED  {payload.get('question')}"))
        print(DIM(f"  spent so far ${payload.get('spent_so_far_usd', 0):.4f}"))
        try:
            reply = input("  escalate? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            reply = "n"
        out = resume(out.thread_id, approved=reply in ("y", "yes"), cfg=cfg)

    if args.json:
        print(json.dumps(out.to_dict(), indent=2))
        return 0

    _print_outcome(out, show_answer=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
