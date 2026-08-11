"""Every filesystem location this project reads or writes, resolved once.

Before the 11 August 2026 restructure the research modules sat at the repository
root, so `Path(__file__).parent` *was* the root and each module spelled out its
own anchor. Now they sit one directory down, and that idiom would silently
resolve to `llm_routing/` - finding no `data/`, no `cache/`, and no `.env`.
Silently, because most of the callers treat a missing file as "not built yet"
rather than as an error, so the failure would look like an empty run instead of
a broken path.

One anchor, therefore, and every module imports it rather than deriving it.
Moving a directory is then a one-line edit here instead of a hunt through
sixteen modules.

THE THREE CLASSES OF PATH, which are kept apart on purpose:

    data/     inputs. Source corpora, plus the built task set - committed,
              because a task set that cannot be reproduced byte for byte is a
              task set that cannot be argued with.
    cache/    the real model responses. What makes ROUTER_MODE=replay free.
    runs/     everything derived. Deleting the whole directory and regenerating
              it is the standard way to check that a published figure really
              does come from the committed data.
"""

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


def ensure_runs():
    """Create runs/ if it is missing, and return it.

    Called at the point of writing rather than at import, because importing a
    module must not touch the filesystem - the test suite imports every one of
    these to inspect constants, and an import with a side effect turns a
    read-only test run into one that leaves directories behind.
    """
    RUNS.mkdir(exist_ok=True)
    return RUNS
