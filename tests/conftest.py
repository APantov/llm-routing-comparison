"""Shared fixtures.

One wrinkle dominates this file: `models.py` reads `ROUTER_LADDER` at MODULE
SCOPE and builds `MODELS`/`TIERS` once, at import. That is the right design for
an evaluation - the ladder is fixed for a run, and a ladder that could change
underneath a policy would make the results meaningless - but it means a test
that wants a different ladder has to reload the module and everything that
imported symbols from it.

`use_ladder` does that reloading in the correct dependency order, and restores
the original afterwards so tests cannot leak state into each other.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set before the first import of `models`, not after.
os.environ.setdefault("ROUTER_MODE", "mock")
os.environ.setdefault("ROUTER_LADDER", "claude")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: grades the real task set end to end. Deselected by default - the "
        "suite is meant to run in seconds so it actually gets run. Enable with "
        "`pytest -m slow`, and note it scales with the task set: the same test "
        "took 3 seconds at 95 tasks and 168 at 426.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip `slow` unless asked for, by -m slow or by name."""
    if config.getoption("-m") or config.getoption("-k"):
        return
    skip = pytest.mark.skip(reason="slow: run with `pytest -m slow`")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


# Reload order matters: `models` first, then anything holding references to it.
# `router_agent.graph` caches a compiled graph via lru_cache, so its cache has
# to be cleared or a reloaded ladder would still be routed by a graph bound to
# the old one.
_RELOAD_ORDER = [
    "llm_routing.models",
    "llm_routing.policies",
    "router_agent.pricing",
    "router_agent.verifiers",
    "router_agent.nodes",
    "router_agent.engine",
]


def _reload_stack():
    for name in _RELOAD_ORDER:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
    if "router_agent.graph" in sys.modules:
        graph = sys.modules["router_agent.graph"]
        importlib.reload(graph)
        graph.compiled.cache_clear()


@pytest.fixture
def use_ladder(monkeypatch):
    """Rebuild the model ladder for one test, then put it back."""
    original = os.environ.get("ROUTER_LADDER", "claude")

    def _apply(name: str):
        monkeypatch.setenv("ROUTER_LADDER", name)
        _reload_stack()
        from llm_routing import models
        assert models.LADDER == name
        return models

    yield _apply

    os.environ["ROUTER_LADDER"] = original
    _reload_stack()


@pytest.fixture
def cfg():
    """A mock-mode config factory with sane test defaults."""
    from router_agent.config import RouterConfig

    def _make(**kw):
        base = dict(
            mode="mock",
            ladder=os.environ.get("ROUTER_LADDER", "claude"),
            policy="cascade",
        )
        base.update(kw)
        return RouterConfig(**base)

    return _make


@pytest.fixture(scope="module")
def wide_verdicts():
    """`routable.real_verdicts` over the MATHS half of the committed `wide` cache.

    Maths only, and module-scoped, both for speed. Grading the code half means
    executing 35 expanded MBPP+ suites, which takes ~39 seconds in a suite that
    otherwise finishes in two - and buys nothing here, because every question
    this fixture is used to answer is about truncation. Code answers average 55
    output tokens against a 4096 cap; not one is within an order of magnitude
    of it. All three truncations on disk are maths.
    """
    import json
    from llm_routing import routable
    path = REPO_ROOT / "data" / "taskset.jsonl"
    if not path.exists():
        pytest.skip("taskset.jsonl not built; run python -m llm_routing.build_taskset")
    tasks = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    return routable.real_verdicts([t for t in tasks if t["domain"] == "math"],
                                  "wide")


@pytest.fixture
def benchmark_task():
    """A real row from the task set, for replay tests.

    Skips rather than fails when the task set has not been built: a fresh
    clone has no taskset.jsonl until `python -m llm_routing.build_taskset` runs, and a
    missing artefact is not a broken test.
    """
    import json
    path = REPO_ROOT / "data" / "taskset.jsonl"
    if not path.exists():
        pytest.skip("taskset.jsonl not built; run python -m llm_routing.build_taskset")
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if not rows:
        pytest.skip("taskset.jsonl is empty")
    return rows
