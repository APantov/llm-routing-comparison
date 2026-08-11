"""The experiment: 16 modules that build the task set, run the policies, and
work out whether the differences between them are real.

Each one is still runnable on its own and still reads top to bottom - that has
not changed by their becoming a package:

    python -m llm_routing.build_taskset
    python -m llm_routing.run_eval --limit 10
    python -m llm_routing.stats --results runs/results.wide.jsonl

DELIBERATELY EMPTY OF IMPORTS. `models` reads ROUTER_MODE and ROUTER_LADDER at
module scope and builds its price table once, so importing it has to stay an
explicit act by the caller. A convenience re-export here would make
`import llm_routing.paths` construct the ladder as a side effect, and the test
suite's `use_ladder` fixture - which reloads `models` after changing the
environment - would be reloading a module something else had already fixed.

The serving layer that this package's findings argue for is `router_agent/`. It
depends on this package; nothing here depends on it, and CI has a job whose only
purpose is to keep that arrow pointing one way.
"""
