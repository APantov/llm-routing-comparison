# data/ — the inputs

| file | what it is | source |
|---|---|---|
| `math500.jsonl` | MATH-500, the full set. Only level 5 is sampled. | third party, committed |
| `mbppplus.json` | MBPP+ with its expanded evalplus test suites. | fetched by `scripts/provenance/fetch_mbppplus.py`, committed |
| `taskset.jsonl` | **the 417 tasks every module joins against** | built, and committed |

## Licensing, since both are redistributed here

Neither dataset is mine, and both are committed to this repository rather than
fetched at run time — which makes redistribution terms a thing this README has
to state rather than assume.

| file | upstream | upstream licence |
|---|---|---|
| `math500.jsonl` | MATH-500, the 500-problem subset of Hendrycks et al., *Measuring Mathematical Problem Solving With the MATH Dataset* (2021) | MIT |
| `mbppplus.json` | [`evalplus/mbppplus`](https://huggingface.co/datasets/evalplus/mbppplus) — EvalPlus' expanded suites over Google's MBPP (Austin et al., 2021) | Apache-2.0 (EvalPlus); MBPP itself CC-BY-4.0 |

`taskset.jsonl` is derived from both and inherits their terms. The MIT licence
on this repository covers the code and the measurements, not the upstream task
content.

`taskset.jsonl` is the odd one: it is *derived* from the two above rather than
downloaded, and it still lives here rather than in `runs/` because everything in
this repository treats it as an input. It is committed so that a fresh clone can
run before it builds anything, and because a task set that cannot be reproduced
byte for byte is a task set that cannot be argued with.

It rebuilds byte-identically, on any platform, from the committed sources:

```bash
python -m llm_routing.build_taskset
git diff --stat -- data/taskset.jsonl     # expect no output
```

The defaults reproduce the committed file exactly. That is load-bearing rather
than tidy: the default once built a 96-task sample instead, so
the quickstart in the README silently replaced the artefact every published
number is joined against.

Line endings are pinned to LF by `.gitattributes` and by `newline=""` at every
writer, because the same code producing byte-different files on Windows and
Linux makes a hash-based regression gate impossible.
