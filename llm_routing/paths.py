"""Every filesystem location this project reads or writes, resolved once.

One anchor, imported rather than derived. `Path(__file__).parent` in any module
here resolves to `llm_routing/`, finding no `data/`, no `cache/` and no `.env` -
and silently, because most callers treat a missing file as "not built yet" rather
than an error, so the failure looks like an empty run instead of a broken path.

THE THREE CLASSES OF PATH, which are kept apart on purpose:

    data/     inputs. Source corpora, plus the built task set - committed,
              because a task set that cannot be reproduced byte for byte is a
              task set that cannot be argued with.
    cache/    the real model responses. What makes ROUTER_MODE=replay free.
    runs/     everything derived. Deleting the whole directory and regenerating
              it is the standard way to check that a published figure really
              does come from the committed data.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
CACHE = ROOT / "cache"
RUNS = ROOT / "runs"
FIGURES = ROOT / "figures"
ARCHIVE = ROOT / "archive"

# The task set every module joins against. Built by `python -m
# llm_routing.build_taskset`, and committed, so a clone can run before it builds.
TASKSET = DATA / "taskset.jsonl"

# Gitignored; .env.example is committed in its place. models.load_dotenv reads
# this, and a real environment variable always beats it.
ENV_FILE = ROOT / ".env"


def default_ladder():
    """The ladder name a run will use, without importing `models`.

    Duplicated from `models.LADDER` on purpose, and only here: `stats` and
    `scorecard` need to name a default results file *before* they know which
    ladder produced it, and `scorecard` in particular must not import `models`
    early - it reads the ladder out of the results file and sets the
    environment before the first import, because `models` builds its price
    table at module scope. `TestPathsDefaultLadder` keeps the two in step.
    """
    return os.environ.get("ROUTER_LADDER", "claude")


def ensure_runs():
    """Create runs/ if it is missing, and return it.

    Called at the point of writing rather than at import, because importing a
    module must not touch the filesystem - the test suite imports every one of
    these to inspect constants, and an import with a side effect turns a
    read-only test run into one that leaves directories behind.
    """
    RUNS.mkdir(exist_ok=True)
    return RUNS
