# figures/ — every claim that is easier to see than to read

Nothing here is an input, and nothing here is drawn by hand. Every figure is
written by `llm_routing/plot.py` from an artefact in `runs/`, so deleting the
whole directory and regenerating it is the standard way to check that a
published picture really does come from the committed data:

```bash
ROUTER_MODE=replay python scripts/run_all_ladders.py
```

No matplotlib. The research core reproduces on a bare interpreter with no
network and no installs, and a plotting dependency would trade that away to draw
nine charts. SVG is text, so `plot.py` writes it directly — sharp at any size.

## What each one carries

| figure | the claim | read from |
|---|---|---|
| `ladders.svg` | cascade against always-paying-for-the-best, in accuracy **and** money, on all three ladders — [§2.1](../docs/RESULTS.md#21-the-cascade-beats-always-paying-for-the-best-on-two-ladders-of-three) | `results.<ladder>.jsonl` |
| `routable.svg` | what there is to route at all: the cheap/top cross-tab per ladder — [§2.4](../docs/RESULTS.md#24-the-third-ladder-has-almost-nothing-to-route) | `scorecard.<ladder>.json` |
| `ratio.svg` | the price ratio does not decide whether to cascade — [§2.5](../docs/RESULTS.md#25-the-price-ratio-does-not-decide-whether-to-cascade) | `frontier.<ladder>.jsonl` |
| `predictive.svg` | predictive routing does not beat a coin flip, six of six — [§2.3](../docs/RESULTS.md#23-predictive-routing-does-not-beat-a-coin-flip--six-of-six) | `results.*` (McNemar) + `frontier.*` (AUC) |
| `scorecard.svg` | what each policy did with its escalations — [§3](../docs/RESULTS.md#3-what-each-policy-got-right-and-wrong) | `scorecard.wide.json` |
| `noise.svg` | a sixth of the routing opportunity does not reproduce — [§2.6](../docs/RESULTS.md#26-a-sixth-of-the-routing-opportunity-is-noise) | `redraw.wide.json` |
| `frontier.<ladder>.svg` | cost against accuracy: policies are curves, not points | `frontier.<ladder>.jsonl` |
| `degradation.<ladder>.svg` | the experiment — cascade quality *and* cost against verifier quality — [§2.8](../docs/RESULTS.md#28-the-cascade-degrades-smoothly-with-its-verifier--the-experiment) | `sweep_degraded.<ladder>.jsonl` |
| `graph.svg` | the product half: the LangGraph state machine that ships | `router_agent/graph.py` |

The two per-ladder figures carry the ladder in the filename because there is one
per sweep; the rest read every ladder at once and are written once. Only the
`wide` ladder has a degradation sweep — that experiment holds the ladder fixed
and varies verifier fidelity, so the ladder is its control rather than its
variable.

`graph.svg` is the one figure with no run behind it, and it is still not drawn
from memory: `plot.read_state_machine` parses `router_agent/graph.py` with `ast`
and lays out the nodes and edges it finds there. Reading the source text is not
importing it, so the one-way dependency the architecture rests on —
`router_agent` imports `llm_routing`, never the reverse — is untouched.

## Five conventions that are not cosmetic

**Money is per 1,000 queries.** Per task these costs run from $0.00005 to
$0.005, and four leading zeros is not a number anyone can compare by eye.

**Cost axes are logarithmic.** The rungs are two orders of magnitude apart on
`wide`. On a linear axis every cascade setting collapses into the left eighth of
the frame, which is what the first version of the frontier chart did.

**Where the accuracy axis is zoomed, the mark is a dot.** The differences here
are three or four points on a base of ninety, so the axis has to be zoomed to
show them — and a zoomed axis and a bar chart cannot both be honest, because bar
length then encodes distance from an arbitrary floor rather than accuracy. Each
figure that zooms says where its axis starts.

**Ticks land on numbers people count in.** 1, 2, 2.5 or 5 times a power of ten,
via `plot.nice_ticks`. The scorecard's axis used to read 0, 53, 106, 159, 212 —
the data's own maximum cut in quarters — and `noise.svg` printed *round* labels
on gridlines that were not: lines at 4.9% and 9.8% wearing the labels 5% and
10%, so every bar read slightly taller against them than it was.

**Text is measured, not estimated.** `plot.text_width` carries Adobe's advance
widths for Helvetica and Helvetica-Bold, so a label can be right-aligned inside
a frame, a legend column can be sized to its longest entry, and two annotations
can be told not to print on top of each other. The estimate this replaced —
0.56em a character, applied to everything — is why the McNemar row labels ran
underneath their own legend and the counts at the right edge were cropped.

## The layout is a test, not a judgement call

A chart whose labels print on top of each other is still valid SVG, still
regenerates byte-identically, and still passes every other test here. That is
how this directory came to publish a cross-tab with `height="0.0"` on all
twelve of its bars — the counts floating in white space where the bars should
have been — and a McNemar panel whose key named two colours that appeared
nowhere on it. Both had been wrong in every committed copy.

So [tests/test_figures.py](../tests/test_figures.py) re-measures each committed
figure with the same advance widths that drew it, and fails on the four things
that actually went wrong: markup that is not well-formed, text outside the
canvas, two labels substantially overlapping, and a bar drawn with zero extent.
It cannot catch an ugly figure. It catches an unreadable one.

## Readable on either background

Every neutral — paper, ink, gridlines, the frontier's reference marks — is
emitted with a class as well as its literal colour, and a `prefers-color-scheme`
rule in each file restates those for a dark viewer. The data colours do not
move: Okabe-Ito holds its identity on either ground, and a reader who has seen
a figure in one mode should recognise it in the other.

This is additive and cannot break anything. Every element still carries its
light colour as a presentation attribute, and a CSS rule outranks a
presentation attribute — so a renderer that ignores the stylesheet draws
exactly the figure that was there before.

Each file also carries `role="img"` with a `<title>` and a `<desc>`. A README
caption is not alt text.

## The label that matters most

Every figure carries a provenance line under its title reading `measured on
real models`, and it is the one line here that is checked rather than written.
`plot._read_jsonl` and `plot._read_json` refuse to load an artefact whose rows
are stamped `simulated: true`, so a chart drawn from one cannot exist.

It used to *say so* instead, in a subtitle reading **SIMULATED (mock mode)**.
That was the wrong shape of guard for this directory specifically. A figure is
the most portable thing this repository produces — it gets cropped, pasted into
a slide, screenshotted without its caption — and a label that survives none of
that was the only thing distinguishing a result from a picture of
`models.MOCK_SKILL`. Since `models.require_measured_mode`, no run can write a
simulated artefact in the first place, so a chart that cannot be drawn is a
stronger statement than a chart that says it should not be trusted.

The committed figures are all measured, and regenerating them is how you check
it: deleting this directory and running the command at the top of this file
returns all nine byte-identical.
