"""Content-addressed cache of raw model responses.

WHY THIS EXISTS, because it is easy to mistake for a speed optimisation and it
is not one.

Every policy calls the models independently. Per task, the *same* greedy cheap
call at temperature 0 is made several separate times over: by always_cheap, by
cascade, by oracle, and by every one-shot router that happens to route cheap.
run_eval prints the exact counts at the end of a run ("requested" versus
"reached a backend"), and the gap is large.

In mock mode those duplicates are byte-identical for free, because the mock is a
pure hash of its inputs. In real mode they would be hundreds of extra paid API
calls returning hundreds of *different* answers.

That second fact is the real problem, and it is a validity problem rather than a
cost one. Every paired statistic this project wants - McNemar, a paired
bootstrap, a discordance table - assumes the policies are compared on the same
model outputs. Without this cache they are not: always_cheap and cascade would
disagree partly because of decoding noise rather than because of policy, and the
oracle would be bounding a different set of draws than the ones the other
policies actually received, which is not a bound at all.

So: one draw per distinct (mode, model, prompt, temperature, sample_idx, seed),
stored on disk, shared by every policy and every rerun.


WHAT THE CACHE DOES *NOT* DO
----------------------------
It does not change what any policy costs. A cache hit returns the full
ModelResponse including cost_usd, and the policy is charged for it exactly as if
it had made the call. This is deliberate: cost_usd answers "what would this
policy cost in production", and in production there is no cross-policy cache
because only one policy is running. The cache reduces what *this harness* spends
to measure that, not what the policy would spend to serve it.

Both numbers are reported separately by run_eval:

    "total attributed cost"   sum over policies, unchanged by caching
    "reached a backend"       what this run would actually have paid for


KEY
---
Hashed over everything that determines a response and nothing that does not:

    mode            a mock response and a real response are different objects
    model id        not the tier name; the tier -> model mapping can change
    prompt          the full prompt text, so editing PROMPTS invalidates cleanly
    temperature
    sample_idx      which independent draw this is
    max_tokens      it decides truncation, which grades as a wrong answer
    mock_seed       mock only; None in real mode

Deliberately NOT in the key: task_id and tier. Both are stored alongside for
grepping, but two tasks that somehow produced an identical prompt should share a
response, and the prompt is what the model actually saw.
"""

import hashlib
import json
import os
import threading
from pathlib import Path

from llm_routing import paths

CACHE_DIR = Path(os.environ.get("ROUTER_CACHE_DIR", paths.CACHE))

# Real and mock responses live in SEPARATE files even though `mode` is in the
# hash and they could not collide: the real file is what gets committed after a
# paid run and should hold only responses that were paid for, and a fabricated
# response should never be one careless `git add` from being published.
#
# Both are PLACEHOLDERS - configure() folds the ladder name in, giving
# cache/raw_calls.<ladder>[.mock].jsonl. Nothing should read them before then.
REAL_PATH = CACHE_DIR / "raw_calls.jsonl"
MOCK_PATH = CACHE_DIR / "raw_calls.mock.jsonl"

# Set by configure() from the selected ladder. The cache key already contains the
# model id, so two ladders can never return each other's responses; the suffix is
# about keeping the FILES separate, so that deleting one ladder's mock cache does
# not disturb another's, and so a committed real cache says which ladder paid for
# it. Both are practical concerns rather than correctness ones.
LADDER = ""

# Off switch, for measuring the un-cached call count and for debugging.
#
# MOCK MODE ONLY, enforced in configure(). With it off, put() discards its
# argument, so a real run would pay for every response and write none to disk -
# the most expensive failure this file can have. Replay IS a cache read, so with
# the cache off every lookup misses and the run measures nothing. Both were one
# environment variable away, which is too close on the spend path.
ENABLED = os.environ.get("ROUTER_CACHE", "1") not in ("0", "false", "no")

# Set by configure(). PATH is written to; READ_PATHS are all read, in order, with
# later files winning. That ordering is what puts the real cache last when replay
# is allowed to fall back to the mock one, so a real response always wins.
PATH = REAL_PATH
READ_PATHS = [REAL_PATH]

# Fields that go into the key, in a fixed order. Order matters: the hash is the
# identity of a stored response and must not change because a dict happened to
# iterate differently.
_KEY_FIELDS = ("mode", "model", "prompt", "temperature", "sample_idx", "max_tokens", "mock_seed")

_store = None          # key -> record dict
_lock = threading.Lock()
_conflicts = []        # keys seen twice on disk with different payloads

# The task ids a caller considers live, set before the first cache read. None
# means "the caller has not said", and a conflict is then reported as a note
# rather than as a warning. `run_eval.load_tasks` sets it; the serving layer
# never does, because a conflict on a benchmark id cannot affect a live query.
LIVE_TASK_IDS = None


# Real caches belonging to OTHER ladders, read before this ladder's own.
#
# There is no ladder in the cache key, and the prompt is built from the task
# alone rather than the tier - verified on the committed data, where all 112
# tasks with both rungs cached have identical prompt_sha256 across tiers. So a
# response is identified by what was asked of which model, and the ladder only
# decided which FILE it landed in.
#
# That matters for money: `wide` shares Opus 5 with `claude` and v4-flash with
# `deepseek`, so its 980 responses cover much of both. Without this, running
# `claude` re-buys every Opus call - about $1.91. See docs/ARCHITECTURE.md
# (standing invariants).
#
# Only REAL caches are shared. Mock caches stay per-ladder: a mock response is a
# function of MOCK_SEED and stipulated skill, so pooling them would let one
# ladder's fabricated answer serve another's.
_KNOWN_LADDERS = ("wide", "claude", "deepseek")


def _sibling_real_paths(ladder: str):
    """Other ladders' real caches, oldest-priority first."""
    return [
        CACHE_DIR / f"raw_calls.{other}.jsonl"
        for other in _KNOWN_LADDERS
        if other != ladder and (CACHE_DIR / f"raw_calls.{other}.jsonl").exists()
    ]


def configure(mode: str, ladder: str = ""):
    """Point the cache at the right file(s) for the run mode and ladder."""
    global PATH, READ_PATHS, REAL_PATH, MOCK_PATH, LADDER, _store
    if not ENABLED and mode != "mock":
        raise SystemExit(
            f"\nROUTER_CACHE is off and ROUTER_MODE is {mode!r}. Refusing.\n"
            f"  real:   every response would be paid for and then discarded -\n"
            f"          put() is a no-op with the cache off.\n"
            f"  replay: replay is nothing but a cache read, so every lookup\n"
            f"          would miss and the run would measure nothing.\n\n"
            f"  The switch exists to count un-deduplicated calls in mock mode.\n"
            f"  Unset ROUTER_CACHE, or run with ROUTER_MODE=mock.\n"
        )
    LADDER = ladder
    suffix = f".{ladder}" if ladder else ""
    REAL_PATH = CACHE_DIR / f"raw_calls{suffix}.jsonl"
    MOCK_PATH = CACHE_DIR / f"raw_calls{suffix}.mock.jsonl"
    # This ladder's own file goes LAST in every list below, so that on the
    # (impossible-by-key, but cheap to guarantee) event of a collision, the
    # ladder actually being run wins.
    siblings = _sibling_real_paths(ladder) if ladder else []
    if mode == "mock":
        PATH, READ_PATHS = MOCK_PATH, [MOCK_PATH]
    elif mode == "replay":
        # Replay reads the real cache only, unless the fallback is explicitly
        # asked for. Read as an env var rather than imported from models, because
        # models imports this module and the dependency must not run both ways.
        if os.environ.get("ROUTER_REPLAY_FALLBACK", "0") in ("1", "true", "yes"):
            # Mock first so that a real response always wins over a simulated one.
            PATH, READ_PATHS = REAL_PATH, [MOCK_PATH, *siblings, REAL_PATH]
        else:
            PATH, READ_PATHS = REAL_PATH, [*siblings, REAL_PATH]
    else:
        PATH, READ_PATHS = REAL_PATH, [*siblings, REAL_PATH]
    _store = None  # force a reload against the new paths


def make_key(*, mode, model, prompt, temperature, sample_idx, max_tokens, mock_seed):
    payload = {
        "mode": mode,
        "model": model,
        "prompt": prompt,
        "temperature": float(temperature),
        "sample_idx": int(sample_idx),
        "max_tokens": int(max_tokens),
        "mock_seed": mock_seed,
    }
    blob = json.dumps({k: payload[k] for k in _KEY_FIELDS}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load():
    """Read the cache file(s) once, into memory.

    Tolerant of a truncated final line: a real run that was killed mid-write
    still holds hundreds of responses that were paid for, and refusing to load
    them because the last one is half-written would be the worst possible
    failure mode for this file.
    """
    global _store
    if _store is not None:
        return _store
    _store = {}
    del _conflicts[:]
    for path in READ_PATHS:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  !! response_cache: skipping unparseable line {lineno} in {path.name}")
                    continue
                key = rec.get("key")
                if key is None:
                    continue
                prev = _store.get(key)
                if prev is not None and prev.get("text") != rec.get("text"):
                    # Same inputs, different output. In mock mode this is
                    # impossible unless the mock itself changed; in real mode it
                    # means the file was appended to by two runs that saw
                    # different model behaviour. Either way it silently
                    # invalidates every paired comparison, so say so rather than
                    # quietly picking one.
                    _conflicts.append((key, prev.get("task_id"), rec.get("task_id")))
                _store[key] = rec
    if _conflicts:
        # A conflict only invalidates a comparison if the task is still IN the
        # task set. A duplicate on a task that no longer exists is dead weight,
        # not a threat - and this module deliberately knows nothing about the
        # task set, so the caller declares what is live via LIVE_TASK_IDS and
        # this answers its own question instead of asking the reader to.
        #
        # Printed unconditionally, the loud form opens every run and every demo
        # with a multi-line warning about stranded ids from superseded task sets -
        # recorded into two ladder caches whose texts differ because DeepSeek is
        # not deterministic at temperature 0. A warning that is always on is a
        # warning nobody reads.
        ids = sorted({t for _, t, _ in _conflicts if t} | {t for _, _, t in _conflicts if t})
        live = sorted(set(ids) & LIVE_TASK_IDS) if LIVE_TASK_IDS is not None else None
        if live:
            print(
                f"  !! response_cache: {len(live)} LIVE task(s) have conflicting "
                f"responses across {', '.join(p.name for p in READ_PATHS)}. "
                f"Later entries won.\n"
                f"     affected: {', '.join(live)}\n"
                f"     Their paired comparisons are NOT valid until resolved."
            )
        elif live is None and ids:
            print(f"  note: {len(_conflicts)} stale cache key(s) from a superseded "
                  f"task set ({', '.join(ids)}); later entries won.")
    return _store


def get(key):
    """Return the stored record for `key`, or None."""
    if not ENABLED:
        return None
    return _load().get(key)


def put(key, record):
    """Append a record and make it visible immediately.

    Appended and flushed per call rather than batched at the end of the run: a
    real run that dies halfway through must keep every response it paid for.
    """
    if not ENABLED:
        return
    store = _load()
    with _lock:
        store[key] = record
        PATH.parent.mkdir(parents=True, exist_ok=True)
        # newline="" so this file is byte-identical on Windows and Linux.
        with PATH.open("a", encoding="utf-8", newline="") as f:
            f.write(json.dumps({"key": key, **record}, ensure_ascii=False) + "\n")


def size():
    return len(_load())
