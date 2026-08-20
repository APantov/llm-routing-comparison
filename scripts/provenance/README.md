# scripts/provenance/ — how the committed data was made

**You do not need to run anything in here.** Every artefact these scripts
produce is already committed, and every published number replays from it for
$0.00. They are kept because a measurement whose provenance is not in the
repository is a measurement you have to take on trust.

**Exactly two can reach a backend**: `record_missing.py` and
`redraw_decisive.py`, the only two that call `models.call`. Both default to
planning and pricing the work and require an explicit `--go` before a single
call is made. The rest read what is already on disk.

`purge_quarantined.py` and `freeze_probe.py` also take `--go`, for a different
reason: they rewrite committed artefacts, so the flag guards destruction rather
than spend.

| script | what it does | can spend |
|---|---|---|
| `fetch_mbppplus.py` | downloads MBPP+, validates every reference solution against its own expanded suite, writes `data/mbppplus.json` | no (needs `datasets`) |
| `redraw_decisive.py` | redraws a cross-tab cell at k draws to get a per-rung p̂, or screens a candidate task pool | **yes**, `--go` |
| `record_missing.py` | buys exactly the calls a full replay is missing, and nothing else | **yes**, `--go` |
| `triage_both_fail.py` | gathers the evidence for a quarantine decision, and refuses to make it | no — reads the cache and runs candidates locally |
| `purge_quarantined.py` | deletes a quarantined task's rows from every artefact on disk | no (`--go` guards the rewrite) |
| `resample_vs_reroute.py` | asks whether a gain is the routing signal or decoding noise | no — reads draws already on disk |
| `freeze_probe.py` | freezes the probe cross-tab that `router_agent/findings.py` is tested against | no (`--go` guards the rewrite) |

## The order they were used in

1. `fetch_mbppplus.py` — build the code half's source
2. `redraw_decisive.py --screen` — is the candidate pool hard enough to route?
3. `record_missing.py --go` — buy the run
4. `triage_both_fail.py` — for tasks both rungs failed, gather evidence
5. edit `build_taskset.QUARANTINED`, then `purge_quarantined.py --go`
6. `freeze_probe.py --go` — re-freeze the probe the serving layer reads

## Why these are not in `scripts/`

`scripts/` holds the four things a reader might actually run: the demo, the
full-replay driver, and the two CI guards. These seven are operator tools for
acquiring paid data against a task set that is already built. Separating them
keeps the first list short enough to read.

Each one still anchors on the repository root, so they run from anywhere:

```bash
python scripts/provenance/redraw_decisive.py --help
```
