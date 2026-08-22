"""Figures, from the standard library only.

No matplotlib. That is a deliberate constraint rather than an inconvenience: the
whole repo reproduces on a bare interpreter with no network and no installs, and
adding a plotting dependency to draw charts would trade that away for very
little. SVG is text, so writing it directly is a few hundred lines and the
output is sharp at any size and diffable in git.

WHAT IS DRAWN, AND WHICH CLAIM EACH ONE CARRIES
-----------------------------------------------
Every figure is one claim from docs/RESULTS.md, recomputed here from a committed
artefact in runs/. Nothing on any of them is transcribed - a figure with a
number typed into it is a figure that goes stale silently, which is a failure
this repository has already had twice.

    ladders.svg      2.1  cascade vs always-best on all three ladders, in BOTH
                          accuracy and money            <- results.<ladder>.jsonl
    routable.svg     2.4  what there is to route at all: the cheap/top cross-tab
                                                        <- scorecard.<ladder>.json
    ratio.svg        2.5  the price ratio does not decide whether to cascade
                                                        <- frontier.<ladder>.jsonl
    predictive.svg   2.3  predictive routing does not beat a coin flip, 6 of 6
                                        <- results.* (McNemar) + frontier.* (AUC)
    scorecard.svg    3    what each policy did with its escalations
                                                        <- scorecard.wide.json
    noise.svg        2.6  a sixth of the routing opportunity does not reproduce
                                                        <- redraw.wide.json
    frontier.<l>.svg 2.5  cost against accuracy: policies are curves, not points
                                                        <- frontier.<ladder>.jsonl
    degradation.<l>.svg
                     2.8  the experiment: cascade quality AND cost against
                          verifier quality             <- sweep_degraded.<l>.jsonl
    graph.svg        --   the product half: the LangGraph state machine
                                                        <- router_agent/graph.py

`graph.svg` is the one figure with no run behind it, and it still is not drawn
from memory: it parses `router_agent/graph.py` with `ast` and lays out the nodes
and edges it finds. Reading the source text is not importing it, so the one-way
arrow the architecture depends on (router_agent -> llm_routing, never the
reverse) is intact and CI's check still holds. A hand-drawn state machine would
be a second source of truth for the graph, and this repository has been bitten
by every one of those it has had.

THE LAYOUT IS ASSERTED, NOT EYEBALLED

A chart whose labels print on top of each other is still valid SVG and still
regenerates byte-identically, so nothing in this repository noticed that
`routable.svg` was drawing every bar with `height="0.0"` - `Chart.hbar`
subtracted its corners in a fixed order, half these charts run their category
axis downwards, and `rect` clamped the negative height at zero. Twelve bars,
four rows, two figures, wrong in every committed copy.

tests/test_figures.py re-measures each figure with the same advance widths that
drew it and fails on well-formedness, text outside the canvas, substantially
overlapping labels, and bars with zero extent.

TWO CONVENTIONS THAT ARE NOT COSMETIC

  Money is drawn per 1,000 queries, not per task. A per-task cost on this
  workload runs from $0.00005 to $0.005, and four leading zeros is not a number
  a reader can compare by eye. Per 1,000 queries it is $0.05 to $5.02.

  Cost axes are logarithmic. The rungs are two orders of magnitude apart on
  `wide`; on a linear axis every cascade setting collapses into the leftmost
  eighth of the frame, which is exactly what the first frontier chart did.

AND ONE THAT IS A CORRECTION. Accuracy differences here are three or four
points on a base of ninety, so any accuracy axis has to be zoomed to show them.
A zoomed axis and a bar chart cannot both be honest - a bar drawn from a
baseline of 76% encodes 92.3% and 95.7% as lengths in the ratio 16:20, which
overstates a 3.4-point difference as a 24% one. So wherever the accuracy axis
is zoomed the mark is a DOT, whose position is read against the axis and whose
length encodes nothing.

    python -m llm_routing.sweep_degraded && python -m llm_routing.frontier
    python -m llm_routing.plot
"""

import argparse
import ast
import json
import math
import sys
from pathlib import Path

from llm_routing import paths

# Colour-blind-safe qualitative palette (Okabe-Ito). Chosen over a default cycle
# because the frontier chart puts six series on one pair of axes and red/green
# alone would make it unreadable for a fair share of readers.
PALETTE = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]
BLUE, ORANGE, GREEN, PINK, AMBER, SKY = PALETTE

INK = "#222222"
MUTED = "#666666"
FAINT = "#8a8a8a"
GRID = "#dddddd"
WASH = "#f0f0f0"
BASELINE = "#8d8d8d"
PAPER = "#ffffff"

# Text printed ON TOP of a fill that does not follow the theme - a count inside
# a bar, a label on a swatch - must not follow it either. These two are
# deliberately absent from NEUTRAL_CLASS below, so they stay literal in both
# modes and keep their contrast against the fill they sit on rather than
# against the paper.
ON_LIGHT = "#1b1b1b"
ON_DARK = "#fbfbfb"

# Every neutral above is emitted with a class as well as its literal colour, so
# the stylesheet in `Figure.render` can restate it for a dark viewer. The
# literal stays on the element: a renderer that ignores the stylesheet draws
# exactly the light figure it drew before, and a CSS rule always outranks a
# presentation attribute, so one that honours it gets the dark one. Deriving
# the class from the colour rather than passing it at every call site keeps
# this out of the 300-odd drawing calls that would otherwise each need it.
NEUTRAL_CLASS = {INK: "ink", MUTED: "muted", FAINT: "faint", GRID: "grid",
                 WASH: "wash", BASELINE: "base", PAPER: "paper",
                 "white": "paper", "#efefef": "grid", "#f7f7f7": "wash",
                 # The frontier's reference marks. These are near-black and
                 # near-grey because they are ink - `always_expensive` is a
                 # landmark, not a series - so they invert with the page. Left
                 # literal, the two black ones vanish on a dark ground, which
                 # is a legend entry for a mark the reader cannot find.
                 "#111111": "ink", "#5f5f5f": "muted", "#9a9a9a": "faint"}


def _class(colour, attr="fill"):
    """The class name a neutral is restated under, or "" for a data colour."""
    name = NEUTRAL_CLASS.get(colour)
    return f'{"s" if attr == "stroke" else "f"}-{name}' if name else ""


def _cls(colour, attr="fill"):
    """The same, as a ready-to-append attribute.

    Callers that need to combine two roles on one element ask `_class` and
    join the names themselves - two `class` attributes on one element is not
    well-formed XML, and the whole file then renders as nothing.
    """
    name = _class(colour, attr)
    return f' class="{name}"' if name else ""

# The three ladders get one colour each and keep it across every figure, so a
# reader who learns the key on one chart can read the next one without it.
LADDER_COLOUR = {"wide": BLUE, "claude": ORANGE, "deepseek": GREEN}

FIG_TITLE = 15
FIG_SUB = 11
PANEL_TITLE = 12
AXIS = 11
NOTE = 10


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def per_1k(cost_per_task):
    """Per-task USD -> USD per 1,000 queries. See the module docstring."""
    return cost_per_task * 1000.0


def money(v, places=2):
    return f"${v:,.{places}f}"


# ---------------------------------------------------------------------------
# Text metrics
#
# SVG has no layout engine. A figure that wants to right-align a label against
# a frame, size a legend column, or refuse to overprint two annotations has to
# know how wide its own text is, and the previous answer was an average: 0.56em
# per character, applied to everything. An average is wrong in both directions
# and both directions cost something. `LLM-as-router  vs  cost-matched coin
# flip` at 10.5px measures 24px wider than the estimate, which is how it came
# to run underneath the legend beside it; `0/3` measures narrower, which is how
# the counts at the right edge came to be clipped by a margin sized for the
# estimate rather than the text.
#
# So the widths are the real ones: Adobe's published advance widths for
# Helvetica and Helvetica-Bold, in 1/1000 em. Every renderer that substitutes
# Arial, Nimbus Sans or Liberation Sans matches these to within a percent,
# because being metrically compatible with Helvetica is what those faces are
# for. Two tables of 95 numbers are not a dependency.
# ---------------------------------------------------------------------------

def _widths(groups):
    out = {}
    for width, chars in groups:
        for ch in chars:
            out[ch] = width
    return out


_HELVETICA = _widths([
    (191, "'"), (222, "ijl"), (260, "|"), (278, " !,./:;I[]ft"),
    (333, "()-`r"), (334, "{}"), (355, '"'), (389, "*"), (469, "^"),
    (500, "Jcksvxyz"), (556, "#$0123456789?_Labdeghnopqu"), (584, "+<=>~"),
    (611, "FTZ"), (667, "&ABEKSVXY"), (722, "CDHNRUw"), (778, "GOQ"),
    (833, "Mm"), (889, "%"), (944, "W"), (1015, "@"),
])

_HELVETICA_BOLD = _widths([
    (238, "'"), (278, " ,./Iil"), (280, "|"), (333, "!():;[]`ft"),
    (389, "*r{}"), (474, '"'), (500, "z"), (556, "#$0123456789_Jacksvxy"),
    (584, "+<=>^~"), (611, "?FLTZbdghnopqu"), (667, "EPSVXY"),
    (722, "ABCDHKNRU"), (778, "GOQw"), (833, "M"), (889, "%m"), (944, "W"),
    (975, "@"),
])

# Written apart from the tables above so neither table needs an escape in it.
_HELVETICA[chr(92)] = 278
_HELVETICA_BOLD[chr(92)] = 278

# The handful of non-ASCII glyphs these figures actually use. Helvetica proper
# has no arrow, so `→` is whatever the renderer substitutes; 1000 is the width
# of the em dash it is usually sourced beside, and erring wide only ever wraps
# a line early.
_EXTRA = {"→": 1000, "—": 1000, "–": 556, "−": 584, "·": 278, "⟲": 1000,
          "“": 333, "”": 333, "’": 222, "≈": 549, "×": 584, "τ": 500,
          "§": 556, "±": 584, "≤": 549, "≥": 549}


def text_width(s, size, weight=None):
    """Rendered width of `s` in px, at `size` px in Helvetica.

    Everything on these figures that has to fit inside something else is
    measured through here.
    """
    table = _HELVETICA_BOLD if weight == "bold" else _HELVETICA
    total = 0
    for ch in str(s):
        total += table.get(ch) or _EXTRA.get(ch, 556)
    return total * size / 1000.0


def wrap_px(text, px, size, weight=None):
    """Greedy wrap of `text` to lines that measure at most `px` wide.

    A word longer than the line is left to overhang rather than broken: every
    such word in this repository is an identifier like `always_expensive`, and
    a hyphen in the middle of one reads as part of the name.
    """
    out, line = [], ""
    for word in str(text).split():
        trial = f"{line} {word}".strip()
        if line and text_width(trial, size, weight) > px:
            out.append(line)
            line = word
        else:
            line = trial
    if line:
        out.append(line)
    return out


def ellipsize(s, px, size, weight=None):
    """`s`, shortened with an ellipsis until it fits `px`. Never returns wider."""
    if text_width(s, size, weight) <= px:
        return s
    s = str(s)
    while s and text_width(s + "…", size, weight) > px:
        s = s[:-1]
    return (s + "…") if s else ""


# ---------------------------------------------------------------------------
# Axis ticks
#
# `[top * i / 4 for i in range(5)]` is a defensible thing to write and it is
# how the scorecard's axis came to be labelled 0, 53, 106, 159, 212, and the
# degradation sweep's $0.36, $0.63, $0.91, $1.19, $1.47. Neither is a number a
# reader holds in their head, and an axis whose labels cannot be held in the
# head is an axis nobody reads values off - which is the entire job of the
# labels. Ticks land on 1, 2, 2.5 or 5 times a power of ten instead.
# ---------------------------------------------------------------------------

def nice_step(lo, hi, target=5):
    """The roundest step that puts about `target` intervals across [lo, hi].

    Candidates are 1, 2, 2.5 and 5 times a power of ten - the multiples people
    count in. Picking the closest to `target` rather than the first one at
    least as coarse matters: a span of 212 asks for a step of 53, and "first
    at least as coarse" answers 100, which is two labels on a whole axis.
    """
    span = float(hi - lo)
    if span <= 0:
        return 1.0
    best = None
    e = math.floor(math.log10(span)) - 2
    while e <= math.ceil(math.log10(span)) + 1:
        for mult in (1, 2, 2.5, 5):
            step = mult * 10.0 ** e
            count = span / step
            if not 1 <= count <= 40:
                continue
            # Ties to the coarser step: fewer, rounder labels read better than
            # more of them, and the tick count is a preference not a contract.
            score = (abs(count - target), -step)
            if best is None or score < best[0]:
                best = (score, step)
        e += 1
    return best[1] if best else span / max(1, target)


def nice_ticks(lo, hi, target=5):
    """Round tick values lying inside [lo, hi]."""
    step = nice_step(lo, hi, target)
    out, i = [], math.ceil(lo / step - 1e-9)
    while True:
        v = round(i * step, 10)
        if v > hi + step * 1e-9:
            break
        out.append(v)
        i += 1
        if len(out) > 60:
            break
    return out or [lo, hi]


def nice_bounds(lo, hi, target=5):
    """[lo, hi] widened outwards to round numbers, with the ticks spanning it.

    Used where the axis limits are the figure's to choose. Where they are not -
    a percentage axis that must end at 100%, a log cost axis - `nice_ticks`
    alone lands ticks inside the limits the data already fixed.
    """
    step = nice_step(lo, hi, target)
    lo2 = math.floor(lo / step + 1e-9) * step
    hi2 = math.ceil(hi / step - 1e-9) * step
    n = int(round((hi2 - lo2) / step))
    ticks = [round(lo2 + i * step, 10) for i in range(n + 1)]
    return (round(lo2, 10), round(hi2, 10)), ticks


def times(a, b):
    """`a` against `b` as a human ratio: '4.2x cheaper', '1.24x dearer', 'same'.

    A ratio rather than a difference, because the difference degenerates. The
    first version of ladders.svg printed the `deepseek` cost delta as
    `$-0.00000/task`, which reads as a bug and tells the reader nothing; the
    same fact as a ratio is '-0.2%', which is small AND legible.
    """
    if b <= 0:
        return "n/a"
    r = a / b
    if abs(r - 1.0) < 0.005:
        return "no change"
    if r < 1:
        return (f"{100 * (1 - r):.1f}% cheaper" if r > 0.5
                else f"{1 / r:.1f}x cheaper")
    return (f"{100 * (r - 1):.1f}% dearer" if r < 2
            else f"{r:.1f}x dearer")


class Figure:
    """A canvas: a title, some panels, a key, a footnote.

    Panels rather than one chart per file, because several of these findings are
    two quantities that only mean something together - an accuracy is not a
    result without the cost that bought it, and the first version of
    `ladders.svg` had to print that cost as 9px white text inside a bar because
    there was nowhere else to put it.
    """

    def __init__(self, w, h, title, subtitle="", notes=(), margin=26):
        self.w, self.h = w, h
        self.title, self.subtitle = title, subtitle
        self.notes = [notes] if isinstance(notes, str) else list(notes)
        self.margin = margin
        self.parts = []
        # Boxes later labels must not print on top of.
        self.taken = []

    # -- raw canvas drawing, in px ----------------------------------------
    def raw(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=AXIS, anchor="start", colour=INK, weight=None,
             italic=False, claim=False):
        """Draw one run of text. Returns the box it covers.

        With `claim`, the box is also registered as occupied, so a later
        `place` will route around it instead of printing on top of it.
        """
        w = f' font-weight="{weight}"' if weight else ""
        i = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-size="{size}" fill="{colour}"{_cls(colour)}{w}{i}>'
            f'{esc(s)}</text>')
        box = self.text_box(x, y, s, size, anchor, weight)
        if claim:
            self.taken.append(box)
        return box

    # -- knowing where the ink already is -----------------------------------
    def text_box(self, x, y, s, size, anchor="start", weight=None):
        """The rectangle a run of text covers, `y` being its baseline.

        Cap height and descender rather than the full em box: 0.75em above the
        baseline and 0.22em below is where Helvetica actually puts ink, and
        reserving the whole em makes stacked annotations look loose.
        """
        w = text_width(s, size, weight)
        x0 = x if anchor == "start" else (x - w if anchor == "end" else x - w / 2)
        return (x0, y - size * 0.75, x0 + w, y + size * 0.22)

    def occupy(self, box):
        self.taken.append(tuple(box))
        return box

    def free(self, box, pad=2.0):
        """Is `box` clear of everything already claimed?"""
        x0, y0, x1, y1 = box
        for a0, b0, a1, b1 in self.taken:
            if x0 < a1 + pad and a0 < x1 + pad and y0 < b1 + pad and b0 < y1 + pad:
                return False
        return True

    def inside(self, box, bounds=None):
        x0, y0, x1, y1 = box
        b0, c0, b1, c1 = bounds or (self.margin - 4, 8, self.w - self.margin + 4,
                                    self.h - 8)
        return x0 >= b0 and x1 <= b1 and y0 >= c0 and y1 <= c1

    def place(self, x, y, s, size=AXIS, colour=INK, weight=None, italic=False,
              offsets=(), bounds=None, leader=None, pad=2.0):
        """Text near (x, y), moved to the first offset that lands on nothing.

        `offsets` are (dx, dy, anchor) candidates in preference order. The
        first one that is both clear of claimed ink and inside `bounds` wins;
        if none is, the least-bad candidate is used rather than dropping the
        label, because a slightly crowded number still beats a missing one.

        This exists because five annotations on ratio.svg were computed from
        the data, placed at fixed offsets from their points, and printed on top
        of each other - the figure asserted a finding its own labels made
        unreadable.
        """
        offsets = list(offsets) or [(0, -10, "middle"), (0, 16, "middle"),
                                    (9, 4, "start"), (-9, 4, "end")]
        best, best_score = None, None
        for dx, dy, anchor in offsets:
            box = self.text_box(x + dx, y + dy, s, size, anchor, weight)
            ok_in = self.inside(box, bounds)
            score = (0 if self.free(box, pad) else 1) + (0 if ok_in else 2)
            if score == 0:
                best = (dx, dy, anchor, box)
                break
            if best_score is None or score < best_score:
                best, best_score = (dx, dy, anchor, box), score
        dx, dy, anchor, box = best
        if leader is not None and (abs(dx) > leader or abs(dy) > leader):
            self.raw(f'<line x1="{x:.1f}" y1="{y:.1f}" '
                     f'x2="{x + dx * 0.75:.1f}" y2="{y + dy * 0.7:.1f}" '
                     f'stroke="{colour}" stroke-width="0.9" opacity="0.55"/>')
        self.text(x + dx, y + dy, s, size=size, anchor=anchor, colour=colour,
                  weight=weight, italic=italic)
        return self.occupy(box)

    def vtext(self, x, y, s, size=AXIS, anchor="start", colour=INK, weight=None,
              italic=False, claim=False):
        """Text running up the page. A label ON a vertical reference line
        cannot collide with the horizontal labels around it, which is the only
        reliable berth for one on a crowded plot."""
        w = f' font-weight="{weight}"' if weight else ""
        i = ' font-style="italic"' if italic else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-size="{size}" fill="{colour}"{_cls(colour)}{w}{i} '
            f'transform="rotate(-90 {x:.1f} {y:.1f})">{esc(s)}</text>')
        # rotate(-90) maps an offset (dx, dy) to (dy, -dx): the run of text
        # grows UPWARDS from the rotation origin, and the em box grows to the
        # LEFT of it. Which end of the run sits at `y` is what the anchor
        # decides.
        tw = text_width(s, size, weight)
        if anchor == "start":
            ylo, yhi = y - tw, y
        elif anchor == "end":
            ylo, yhi = y, y + tw
        else:
            ylo, yhi = y - tw / 2, y + tw / 2
        box = (x - size * 0.75, ylo, x + size * 0.22, yhi)
        if claim:
            self.taken.append(box)
        return box

    def stroke(self, x1, y1, x2, y2, colour=GRID, width=1.0, dash=None,
               opacity=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{colour}"{_cls(colour, "stroke")} '
            f'stroke-width="{width}"{d}{o}/>')

    def rect(self, x, y, w, h, fill, stroke=None, rx=0, opacity=None):
        s = (f' stroke="{stroke}"{_cls(stroke, "stroke")}'
             if stroke else "")
        o = f' opacity="{opacity}"' if opacity is not None else ""
        r = f' rx="{rx}"' if rx else ""
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(0.0, w):.1f}" '
            f'height="{max(0.0, h):.1f}" fill="{fill}"{_cls(fill)}{s}{r}{o}/>')

    def chart(self, box, xlim, ylim, xscale="linear", yscale="linear"):
        return Chart(self, box, xlim, ylim, xscale, yscale)

    # -- the key -----------------------------------------------------------
    def glyph(self, x, y, colour, kind, w=22):
        """One legend swatch, drawn as the mark it actually stands for.

        The reason this takes a `kind` at all: the previous legend drew every
        entry as a line segment, so `always_cheap` and `always_expensive` -
        which are distinguished on the plot by circle against square, and share
        an ink - appeared in the key as two identical black dashes with nothing
        to tell them apart.
        """
        if kind in ("line", "dash"):
            self.stroke(x, y, x + w, y, colour, 2.5,
                        dash="6,4" if kind == "dash" else None)
        elif kind == "swatch":
            self.rect(x, y - 5, w, 10, colour)
        else:
            marker(self, x + w / 2, y, colour, kind)

    def block(self, x, y, text, width_px, title=None, size=9.5, colour=MUTED,
              lead=14):
        """A paragraph in the margin, wrapped to the room it actually has.

        Returns the y it finished at, so a caller can stack two of them without
        counting lines by hand.
        """
        if title:
            self.text(x, y, title, size=11, weight="bold")
            y += 20
        lines = wrap_px(text, width_px, size)
        for i, line in enumerate(lines):
            self.text(x, y + lead * i, line, size=size, colour=colour)
        return y + lead * len(lines)

    def key(self, x, y, entries, gap=18, title=None, size=AXIS):
        """entries: (label, colour, kind). `kind` picks the swatch.

        Returns the y it finished at, so a caller can put something under it
        without counting entries by hand.
        """
        if title:
            self.text(x, y - 15, title, size=10, colour=MUTED, weight="bold")
        for i, (label, colour, kind) in enumerate(entries):
            cy = y + i * gap
            self.glyph(x, cy, colour, kind)
            self.text(x + 30, cy + 4, label, size=size)
        return y + gap * max(0, len(entries) - 1)

    def keyrow(self, x, y, entries, width, gap=26, size=AXIS, lead=18,
               title=None):
        """The same key, laid out across the page and wrapped to `width`.

        Measured rather than positioned. Every legend in this module used to be
        two `key` calls at two hand-chosen x values, which is fine until an
        entry is reworded - and then either the columns collide or a gap opens
        that nothing fills. Returns the y it finished at.
        """
        if title:
            self.text(x, y, title, size=10, colour=MUTED, weight="bold")
            y += 17
        cx = x
        for label, colour, kind in entries:
            w = 30 + text_width(label, size)
            if cx > x and cx + w > x + width:
                cx, y = x, y + lead
            self.glyph(cx, y, colour, kind)
            self.text(cx + 30, y + 4, label, size=size)
            cx += w + gap
        return y

    # Restated neutrals for a viewer in dark mode. Only the paper and the greys
    # move: the Okabe-Ito data colours are picked to hold their identity on
    # either ground, and swapping those would mean a reader who has seen the
    # figure in one mode cannot recognise it in the other.
    #
    # This is additive and cannot break a renderer that ignores it. Every
    # element still carries its literal light colour as a presentation
    # attribute, and CSS outranks a presentation attribute, so no stylesheet
    # support means exactly the figure that was there before.
    DARK_CSS = (
        "@media (prefers-color-scheme:dark){"
        ".f-paper{fill:#0f1216}"
        ".f-ink{fill:#e9ecef}.f-muted{fill:#a8b0b8}.f-faint{fill:#8b939b}"
        ".f-grid{fill:#2a3038}.f-wash{fill:#1b2027}.f-base{fill:#98a1aa}"
        ".s-ink{stroke:#e9ecef}.s-muted{stroke:#a8b0b8}"
        ".s-faint{stroke:#8b939b}.s-grid{stroke:#333b45}"
        ".s-base{stroke:#98a1aa}.s-paper{stroke:#0f1216}"
        "}")

    def describe(self):
        """The alt text. Every figure is embedded in a README with a caption,
        and a caption is not a description of what the chart shows."""
        bits = [self.subtitle] + [n for n in self.notes if n]
        return " ".join(b.rstrip(".") + "." for b in bits if b)

    def render(self):
        head = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
            f'font-family="Helvetica,Arial,sans-serif" role="img" '
            f'aria-labelledby="t d">',
            f'<title id="t">{esc(self.title)}</title>',
            f'<desc id="d">{esc(self.describe())}</desc>',
            f'<style>{self.DARK_CSS}</style>',
            f'<rect width="{self.w}" height="{self.h}" fill="{PAPER}" '
            f'class="f-paper"/>',
            f'<text x="{self.margin}" y="26" font-size="{FIG_TITLE}" '
            f'font-weight="bold" fill="{INK}" class="f-ink">'
            f'{esc(self.title)}</text>',
        ]
        if self.subtitle:
            head.append(
                f'<text x="{self.margin}" y="43" font-size="{FIG_SUB}" '
                f'fill="{MUTED}" class="f-muted">{esc(self.subtitle)}</text>')
        # Wrapped here rather than at the call site, so a note is written as a
        # sentence and the canvas decides where it breaks. Hand-broken lines go
        # wrong the first time a figure changes width, and four of them did.
        lines = []
        for note in self.notes:
            # Capped rather than run to the full canvas: 940px of 10px text
            # is a 190-character measure, and a line that long is read by
            # losing your place in it. 780 is about 120 characters.
            lines.extend(wrap_px(note, min(self.w - 2 * self.margin, 780),
                                 NOTE))
        tail = []
        base = self.h - 12 - 13 * (len(lines) - 1)
        for i, line in enumerate(lines):
            tail.append(
                f'<text x="{self.margin}" y="{base + 13 * i}" font-size="{NOTE}" '
                f'fill="{FAINT}" class="f-faint">{esc(line)}</text>')
        return "\n".join(head + self.parts + tail + ["</svg>"])


def marker(fig, px, py, colour, shape, size=5):
    """A point glyph in canvas coordinates. Shared by the charts and the key."""
    if shape == "diamond":
        fig.raw(f'<polygon points="{px:.1f},{py - size - 1:.1f} '
                f'{px + size + 1:.1f},{py:.1f} {px:.1f},{py + size + 1:.1f} '
                f'{px - size - 1:.1f},{py:.1f}" fill="{colour}"{_cls(colour)}/>')
    elif shape == "square":
        fig.raw(f'<rect x="{px - size + 0.5:.1f}" y="{py - size + 0.5:.1f}" '
                f'width="{2 * size - 1:.1f}" height="{2 * size - 1:.1f}" '
                f'fill="{colour}"{_cls(colour)}/>')
    elif shape == "ring":
        # The hole takes the paper colour, not white: on a dark page a white
        # hole is a bright dot, which is the opposite of what a ring means.
        # One class attribute carrying both roles - two of them on an element
        # is not well-formed XML, and the file then renders as nothing.
        ring = " ".join(x for x in ("f-paper", _class(colour, "stroke")) if x)
        fig.raw(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{size - 0.6:.1f}" '
                f'fill="{PAPER}" stroke="{colour}" stroke-width="2.2" '
                f'class="{ring}"/>')
    else:
        fig.raw(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{size:.1f}" '
                f'fill="{colour}"{_cls(colour)}/>')


class Chart:
    """One pair of axes inside a box on a Figure.

    Data coordinates in, canvas coordinates out. Either axis can be logarithmic,
    which every cost axis here is - see the module docstring for why that is not
    a style choice.
    """

    def __init__(self, fig, box, xlim, ylim, xscale="linear", yscale="linear"):
        self.f = fig
        self.x, self.y, self.w, self.h = box
        self.xscale, self.yscale = xscale, yscale
        self.x0, self.x1 = self._guard(xlim, xscale)
        self.y0, self.y1 = self._guard(ylim, yscale)

    @staticmethod
    def _guard(lim, scale):
        a, b = lim
        if scale == "log":
            a = max(a, 1e-12)
            b = max(b, a * 1.0000001)
        elif b == a:
            b = a + 1e-9
        return a, b

    def _f(self, v, lo, hi, scale):
        if scale == "log":
            v = max(float(v), 1e-12)
            span = math.log10(hi) - math.log10(lo)
            return (math.log10(v) - math.log10(lo)) / span if span else 0.0
        return (v - lo) / (hi - lo)

    def px(self, v):
        return self.x + self._f(v, self.x0, self.x1, self.xscale) * self.w

    def py(self, v):
        return self.y + self.h - self._f(v, self.y0, self.y1, self.yscale) * self.h

    # -- frame -------------------------------------------------------------
    def frame(self, colour=INK):
        self.f.rect(self.x, self.y, self.w, self.h, "none", stroke=colour)

    def band(self, y_lo, y_hi, colour, opacity=0.3):
        self.f.rect(self.x, self.py(y_hi), self.w,
                    self.py(y_lo) - self.py(y_hi), colour, opacity=opacity)

    def vband(self, x_lo, x_hi, colour, opacity=0.3):
        self.f.rect(self.px(x_lo), self.y, self.px(x_hi) - self.px(x_lo),
                    self.h, colour, opacity=opacity)

    def ygrid(self, values, fmt=str, size=AXIS, labels=True, colour=GRID):
        for v in values:
            py = self.py(v)
            self.f.stroke(self.x, py, self.x + self.w, py, colour)
            if labels:
                self.f.text(self.x - 8, py + 4, fmt(v), size=size, anchor="end")

    def xgrid(self, values, fmt=str, size=AXIS, labels=True, colour=GRID):
        for v in values:
            px = self.px(v)
            self.f.stroke(px, self.y, px, self.y + self.h, colour)
            if labels:
                self.f.text(px, self.y + self.h + 16, fmt(v), size=size,
                            anchor="middle")

    def xlabels(self, values, labels, size=AXIS, dy=16, colour=INK, weight=None):
        for v, s in zip(values, labels):
            self.f.text(self.px(v), self.y + self.h + dy, s, size=size,
                        anchor="middle", colour=colour, weight=weight)

    def axis_title(self, text, which="x", size=12, dy=36, dx=48):
        if which == "x":
            self.f.text(self.x + self.w / 2, self.y + self.h + dy, text,
                        size=size, anchor="middle")
        else:
            self.f.vtext(self.x - dx, self.y + self.h / 2, text, size=size,
                         anchor="middle")

    def panel_title(self, text, size=PANEL_TITLE, colour=INK, dy=-11):
        self.f.text(self.x, self.y + dy, text, size=size, colour=colour,
                    weight="bold")

    # -- marks -------------------------------------------------------------
    def line(self, pts, colour, width=2.2, dashed=False, opacity=None):
        pts = list(pts)
        if len(pts) < 2:
            return
        d = " ".join(f"{'M' if i == 0 else 'L'} {self.px(a):.1f} {self.py(b):.1f}"
                     for i, (a, b) in enumerate(pts))
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        o = f' opacity="{opacity}"' if opacity is not None else ""
        self.f.raw(f'<path d="{d}" fill="none" stroke="{colour}"'
                   f'{_cls(colour, "stroke")} '
                   f'stroke-width="{width}"{dash}{o} stroke-linejoin="round" '
                   f'stroke-linecap="round"/>')

    def dots(self, pts, colour, shape="dot", size=3, opacity=None):
        if opacity is not None:
            self.f.raw(f'<g opacity="{opacity}">')
        for a, b in pts:
            marker(self.f, self.px(a), self.py(b), colour, shape, size)
        if opacity is not None:
            self.f.raw("</g>")

    def hline(self, y, colour=FAINT, dashed=True, width=1.2):
        self.f.stroke(self.x, self.py(y), self.x + self.w, self.py(y),
                      colour, width, dash="5,4" if dashed else None)

    def vline(self, x, colour=FAINT, dashed=True, width=1.2):
        self.f.stroke(self.px(x), self.y, self.px(x), self.y + self.h,
                      colour, width, dash="5,4" if dashed else None)

    def connector(self, y, x_a, x_b, colour, width=3, arrow=True):
        """A dumbbell's shaft, with a head at the `b` end.

        The head is the point of the mark: it says which of the two values is
        the policy under test, so the reader gets the direction of the change
        without consulting the key.
        """
        pa, pb, py = self.px(x_a), self.px(x_b), self.py(y)
        if abs(pb - pa) < 1.5:
            return
        s = 1 if pb > pa else -1
        end = pb - (8 if arrow else 0) * s
        self.f.stroke(pa, py, end, py, colour, width)
        if arrow:
            self.f.raw(
                f'<polygon points="{pb:.1f},{py:.1f} {pb - 9 * s:.1f},'
                f'{py - 5:.1f} {pb - 9 * s:.1f},{py + 5:.1f}" fill="{colour}"'
                f'{_cls(colour)}/>')

    # Both of these take their corners with min/abs rather than by subtracting
    # in a fixed order, because half the charts here run their category axis
    # DOWNWARDS - ylim=(n-0.4, -0.6), so that row 0 is the top row. On such an
    # axis `py(y - half) - py(y + half)` is negative, `rect` clamped it at
    # `max(0.0, h)`, and every bar came out `height="0.0"`. That is how
    # routable.svg published four empty rows with the counts floating in white
    # space, and how predictive.svg's AUC panel published a key for two colours
    # that appeared nowhere on it.
    def hbar(self, y, x_lo, x_hi, half, colour, opacity=None):
        xa, xb = self.px(x_lo), self.px(x_hi)
        ya, yb = self.py(y - half), self.py(y + half)
        self.f.rect(min(xa, xb), min(ya, yb), abs(xb - xa), abs(yb - ya),
                    colour, opacity=opacity)

    def vbar(self, x, half, y_lo, y_hi, colour, opacity=None):
        xa, xb = self.px(x - half), self.px(x + half)
        ya, yb = self.py(y_lo), self.py(y_hi)
        self.f.rect(min(xa, xb), min(ya, yb), abs(xb - xa), abs(yb - ya),
                    colour, opacity=opacity)

    def at(self, x, y, text, size=10, anchor="middle", colour=INK, dy=0,
           weight=None, italic=False):
        self.f.text(self.px(x), self.py(y) + dy, text, size=size, anchor=anchor,
                    colour=colour, weight=weight, italic=italic)


# ---------------------------------------------------------------------------
# Reading the artefacts
#
# Every loader returns None for a file that is not there, and the caller prints
# the command that would produce it. A missing run is a normal state for a fresh
# clone, not an error - `plot` draws what it can and says what it could not.
# ---------------------------------------------------------------------------

LADDERS = ("wide", "claude", "deepseek")

# The order policies are ranked in wherever they share an axis. Best-behaved
# first, so the eye reads down a staircase rather than hunting.
POLICY_ORDER = ["oracle", "cascade", "cascade_routing", "always_expensive",
                "always_mid", "llm_router", "routellm", "random_matched",
                "always_cheap"]


# EVERY artefact this module draws from enters through one of these two, which
# is why the provenance check lives here and nowhere else. It used to be
# threaded: each loader returned a `simulated` flag, every aggregation carried
# it, every figure function passed it down to `_provenance`. That was fourteen
# sites plumbing a boolean whose only possible destination was a refusal - the
# same "guarding an unreachable state" this change removed from the banners, in
# a different costume. Check it where the bytes are read; nothing downstream
# needs to know.


def _refuse_if_simulated(path, simulated):
    from llm_routing import models
    models.refuse_simulated_artefact("plot", bool(simulated), Path(path).name)


def _read_jsonl(path):
    if not Path(path).exists():
        return None
    with Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    _refuse_if_simulated(path, any(r.get("simulated") for r in rows))
    return rows


def _read_json(path):
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # `.get`, not `[...]`: the redraw, screen and triage records predate the
    # field and legitimately do not carry it.
    _refuse_if_simulated(path, isinstance(data, dict) and data.get("simulated"))
    return data


def _absent(what, fix):
    print(f"  skipped {what} - {fix}", file=sys.stderr)
    return False


def _provenance(detail=""):
    """The subtitle every figure carries.

    THIS USED TO BE A CONDITIONAL LABEL. A mock run produced a chart that looked
    exactly like a measured one, so the subtitle read `SIMULATED (mock mode)`
    instead, and it was the only thing on the page distinguishing a result from
    a restatement of `models.MOCK_SKILL`.

    A figure is the most portable thing this repository produces. It gets
    cropped, pasted into slides and screenshotted without its caption, and at
    that point the label is gone while the chart still looks like evidence. So
    the label stopped being conditional and the condition moved upstream: a run
    cannot write a simulated artefact (`models.require_measured_mode`), and
    `_read_jsonl` refuses to load one. By the time anything reaches here the
    only truthful subtitle is the one below.
    """
    return (f"measured on real models  |  {detail}" if detail
            else "measured on real models")


def _short(model_id):
    """`claude-haiku-4-5-20251001` -> `haiku-4-5`. Ladder rungs on one line."""
    for vendor in ("claude-", "deepseek-"):
        if model_id.startswith(vendor):
            model_id = model_id[len(vendor):]
    parts = model_id.split("-")
    if parts and len(parts[-1]) == 8 and parts[-1].isdigit():
        parts = parts[:-1]
    return "-".join(parts)


def _rungs(ladder):
    from llm_routing import models
    ids = models.LADDERS.get(ladder)
    return " → ".join(_short(i) for i in ids) if ids else ""


def _effective_ratio(ladder):
    """Top rung's price over the bottom's, with the tokenizer asymmetry folded in.

    The same arithmetic `router_agent.findings.price_ratios` reports, over the
    same table. It is arithmetic over `models.MODEL_SPECS` rather than anything
    a run produced, so it cannot drift from the ladder it describes.
    """
    from llm_routing import models
    ids = models.LADDERS.get(ladder)
    if not ids:
        return None
    bottom, top = models.MODEL_SPECS[ids[0]], models.MODEL_SPECS[ids[-1]]
    return ((top["price_in"] * top["tokenizer_factor"])
            / (bottom["price_in"] * bottom["tokenizer_factor"]))


def _policy_stats(ladder):
    """Per-policy accuracy, cost and per-task outcomes from one ladder's run."""
    rows = _read_jsonl(paths.RUNS / f"results.{ladder}.jsonl")
    if not rows:
        return None
    by = {}
    for r in rows:
        if r.get("split") not in (None, "eval"):
            continue
        by.setdefault(r["policy"], []).append(r)
    out = {}
    for name, g in by.items():
        out[name] = {
            "accuracy": sum(bool(r["correct"]) for r in g) / len(g),
            "cost_per_task": sum(r["cost_usd"] for r in g) / len(g),
            "n": len(g),
            "correct": {r["task_id"]: bool(r["correct"]) for r in g},
        }
    return {"policies": out}


def _frontier_economics(ladder):
    """`frontier.economics` for one ladder, or None if it has no sweep."""
    rows = _read_jsonl(paths.RUNS / f"frontier.{ladder}.jsonl")
    if not rows:
        return None
    from llm_routing import frontier as frontier_mod
    econ = frontier_mod.economics(rows)
    if econ is not None:
        econ["n"] = rows[0].get("n")
    return econ


def log_ticks(lo, hi):
    """1-2-5 ticks spanning [lo, hi]. Enough labels to read, few enough to fit."""
    out, e = [], math.floor(math.log10(lo))
    while 10 ** e <= hi * 1.001:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo * 0.999 <= v <= hi * 1.001:
                out.append(v)
        e += 1
    return out


def money_tick(v):
    return f"${v:,.0f}" if v >= 1 else f"${v:.2f}"


def pct_tick(v):
    return f"{v:.0%}"


# ---------------------------------------------------------------------------
# 2.1  Cascade against always paying for the best model, on all three ladders
# ---------------------------------------------------------------------------

def plot_ladders(out, ladders=LADDERS):
    """The headline, and the correction to the chart that used to carry it.

    Two things go wrong when this is a bar chart, and both did:

    1. The accuracy axis has to be zoomed - the differences are three to four
       points on a base of ninety - and a bar drawn from a baseline of 76%
       encodes 92.3% against 95.7% as lengths in the ratio 16:20. That reads as
       a quarter more accurate. It is 3.4 points. So the mark here is a dot,
       whose position is read against the axis and whose length means nothing.

    2. Cost is half the finding and there was nowhere to put it, so it went in
       as 9px white text inside the bar. On `claude` the cascade wins accuracy
       and LOSES on cost, and that is the whole reason the router reads the
       ladder instead of assuming - it cannot be a caption. It gets its own
       panel, on a log axis, because the two ladders' costs are 20x apart.
    """
    groups = []
    for ladder in ladders:
        stats = _policy_stats(ladder)
        if not stats:
            continue
        pol = stats["policies"]
        if not {"cascade", "always_expensive"} <= set(pol):
            continue
        groups.append((ladder, pol))

    if not groups:
        return _absent("ladders.svg", "no per-ladder results; run: "
                                      "ROUTER_MODE=replay python scripts/run_all_ladders.py")

    accs = [pol[p]["accuracy"] for _, pol in groups
            for p in ("cascade", "always_expensive")]
    costs = [per_1k(pol[p]["cost_per_task"]) for _, pol in groups
             for p in ("cascade", "always_expensive")]
    a_lo = math.floor(min(accs) * 20 - 0.5) / 20          # down to a 5% mark
    c_lo, c_hi = min(costs) / 2.2, max(costs) * 2.2

    n = groups[0][1]["cascade"]["n"]

    # The case against drawing this as bars, computed from the axis this figure
    # actually chose rather than asserted. Bar length would encode distance from
    # the axis floor, so the worst-distorted row is the honest example.
    worst = max(
        ((pol["cascade"]["accuracy"] - pol["always_expensive"]["accuracy"],
          (pol["cascade"]["accuracy"] - a_lo) / (pol["always_expensive"]["accuracy"] - a_lo),
          ladder)
         for ladder, pol in groups
         if pol["always_expensive"]["accuracy"] > a_lo),
        key=lambda t: t[1], default=None)

    fig = Figure(
        940, 452,
        "Cascade against always paying for the best model",
        _provenance(f"n={n} held-out tasks per ladder · both panels share "
                    f"their rows, so each ladder is one line across the figure"),
        notes=[
            (f"Dots, not bars: the accuracy axis starts at {a_lo:.0%}, so a bar "
             f"drawn from it would encode the {worst[2]} ladder's "
             f"{100 * worst[0]:.1f}-point gain as a bar {worst[1] - 1:.0%} longer."
             if worst else
             "Dots, not bars: the accuracy axis is zoomed, and bar length from a "
             "false baseline would overstate every difference on it."),
            f"Cost is what serving the same {n} tasks cost, scaled to 1,000 "
            f"queries. The arrow runs from always-expensive to the cascade.",
        ])

    ylim = (len(groups) - 0.45, -0.55)
    acc = fig.chart((186, 104, 300, 186), (a_lo, 1.0), ylim)
    cost = fig.chart((610, 104, 296, 186), (c_lo, c_hi), ylim, xscale="log")

    acc.panel_title("accuracy on the held-out half")
    cost.panel_title("cost per 1,000 queries")

    acc.ygrid([i for i in range(len(groups))], labels=False, colour="#efefef")
    cost.ygrid([i for i in range(len(groups))], labels=False, colour="#efefef")
    ticks = [a_lo + 0.05 * i for i in range(int(round((1.0 - a_lo) / 0.05)) + 1)]
    acc.xgrid(ticks, pct_tick)
    cost.xgrid(log_ticks(c_lo, c_hi), money_tick)
    acc.frame()
    cost.frame()

    for i, (ladder, pol) in enumerate(groups):
        colour = LADDER_COLOUR.get(ladder, BLUE)
        fig.text(170, acc.py(i) - 2, ladder, size=13, anchor="end", weight="bold",
                 colour=colour)
        fig.text(170, acc.py(i) + 13, _rungs(ladder), size=9, anchor="end",
                 colour=MUTED)

        a_exp = pol["always_expensive"]["accuracy"]
        a_cas = pol["cascade"]["accuracy"]
        acc.connector(i, a_exp, a_cas, BLUE)
        acc.dots([(a_exp, i)], BASELINE, shape="ring", size=5.5)
        acc.dots([(a_cas, i)], BLUE, shape="dot", size=5.5)
        # Clear of the ring by the ring's own radius. Anchored exactly at the
        # point, the last digit printed underneath the mark it labels.
        fig.text(acc.px(a_exp) - 11, acc.py(i) + 4, f"{a_exp:.1%}", size=9.5,
                 anchor="end", colour=MUTED)
        fig.text(acc.px(a_cas) + 10, acc.py(i) + 4, f"{a_cas:.1%}", size=10,
                 weight="bold", colour=INK)
        acc.at((a_exp + a_cas) / 2, i, f"+{100 * (a_cas - a_exp):.1f} pts",
               size=9, dy=-11, colour=BLUE)

        c_exp = per_1k(pol["always_expensive"]["cost_per_task"])
        c_cas = per_1k(pol["cascade"]["cost_per_task"])
        # Ratio, not difference. A difference of -$0.00000 is what the old chart
        # printed for `deepseek`, which reads as a bug rather than as "no change".
        verdict = times(c_cas, c_exp)
        tone = (GREEN if c_cas < c_exp * 0.995 else
                ORANGE if c_cas > c_exp * 1.005 else FAINT)
        if abs(cost.px(c_cas) - cost.px(c_exp)) < 5:
            cost.dots([(c_exp, i)], BASELINE, shape="ring", size=5.5)
            cost.dots([(c_exp, i)], FAINT, shape="dot", size=2.5)
            fig.text(cost.px(c_exp) + 12, cost.py(i) + 4,
                     f"{money(c_exp)} either way", size=10, colour=MUTED)
            cost.at(c_exp, i, verdict, size=9, dy=-11, colour=tone)
        else:
            cost.connector(i, c_exp, c_cas, tone)
            cost.dots([(c_exp, i)], BASELINE, shape="ring", size=5.5)
            cost.dots([(c_cas, i)], tone, shape="dot", size=5.5)
            left, right = sorted((c_exp, c_cas))
            fig.text(cost.px(left) - 10, cost.py(i) + 4, money(left), size=9.5,
                     anchor="end", colour=MUTED)
            fig.text(cost.px(right) + 10, cost.py(i) + 4, money(right), size=10,
                     weight="bold", colour=INK)
            cost.at(math.sqrt(c_exp * c_cas), i, verdict, size=9, dy=-11,
                    colour=tone)

    acc.axis_title("accuracy", dy=34)
    cost.axis_title("USD per 1,000 queries  (log scale)", dy=34)

    fig.key(186, 372, [
        ("always_expensive — pay the top rung on every query", BASELINE, "ring"),
        ("cascade — verify at the cheap rung, escalate when it fails", BLUE, "dot"),
    ])
    fig.key(610, 372, [
        ("cascade also costs less", GREEN, "line"),
        ("cascade buys the accuracy at a premium", ORANGE, "line"),
    ])

    Path(out).write_text(fig.render(), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# 2.4  What there is to route at all
# ---------------------------------------------------------------------------

CELLS = [
    ("both_ok", "#d9d9d9", "both rungs right — nothing to win"),
    ("routable", BLUE, "cheap wrong, top right — the only winnable cell"),
    ("inverted", AMBER, "cheap right, top wrong — escalating loses"),
    # Mid grey rather than near-black. `both_fail` has to read as the dark
    # cell against `both_ok`'s light one on a white page AND stay visible on a
    # dark one, and #4f4f4f disappeared into the second.
    ("both_fail", "#6b6b6b", "both rungs wrong — hopeless"),
]


def plot_routable(out, ladders=LADDERS):
    """Why no policy wins on `deepseek`: there is almost nothing there to win.

    The cross-tab is the precondition for every other result on this page, and
    it had no figure. A router can only ever convert the `routable` cell; on
    `deepseek` that cell is a third the size of the hopeless one, which is the
    whole explanation for a ladder where `always_expensive` lands below a coin
    flip.
    """
    groups = []
    for ladder in ladders:
        card = _read_json(paths.RUNS / f"scorecard.{ladder}.json")
        if not card or not card.get("cells"):
            continue
        groups.append((ladder, card["cells"]))

    if not groups:
        return _absent("routable.svg", "no scorecards; run: python -m "
                                       "llm_routing.scorecard --json runs/scorecard.wide.json")

    n = sum(groups[0][1].values())
    fig = Figure(
        900, 424,
        "What there is to route: the cheap rung against the top rung",
        _provenance(f"n={n} held-out tasks per ladder · every task falls in "
                    f"exactly one cell"),
        notes=[
            "A router can only ever convert the blue cell. Grey either way is "
            "denominator: both rungs agree, so no routing decision changes the "
            "outcome.",
            "Discordant pairs — blue plus amber — are what a routing "
            "benchmark should be sized by, not its task count (RESULTS §2.2).",
        ])

    ylim = (len(groups) - 0.4, -0.6)
    ch = fig.chart((190, 104, 570, 168), (0, 1), ylim)
    ch.xgrid([0, 0.25, 0.5, 0.75, 1.0], lambda v: f"{v:.0%}")
    ch.frame()
    ch.axis_title("share of the held-out set", dy=34)

    for i, (ladder, cells) in enumerate(groups):
        total = sum(cells.values()) or 1
        colour = LADDER_COLOUR.get(ladder, BLUE)
        fig.text(174, ch.py(i) - 2, ladder, size=13, anchor="end", weight="bold",
                 colour=colour)
        fig.text(174, ch.py(i) + 13, _rungs(ladder), size=9, anchor="end",
                 colour=MUTED)

        base = 0.0
        for key, fill, _label in CELLS:
            v = cells.get(key, 0)
            if not v:
                continue
            frac = v / total
            ch.hbar(i, base, base + frac, 0.19, fill)
            width = ch.px(base + frac) - ch.px(base)
            if width > 15:
                ink = ON_DARK if key == "both_fail" else ON_LIGHT
                ch.at(base + frac / 2, i, str(v), size=10, dy=4, colour=ink,
                      weight="bold" if key == "routable" else None)
            base += frac

        routable, hopeless = cells.get("routable", 0), cells.get("both_fail", 0)
        disc = routable + cells.get("inverted", 0)
        fig.text(775, ch.py(i) - 2,
                 f"{routable} routable · {hopeless} hopeless", size=10,
                 colour=INK)
        fig.text(775, ch.py(i) + 13, f"{disc} discordant pairs", size=9,
                 colour=MUTED)

    # The one row that needs saying out loud, and only when the data still says
    # it: a ladder whose hopeless cell outweighs its routable one has no
    # routing problem to solve.
    for i, (ladder, cells) in enumerate(groups):
        if cells.get("both_fail", 0) > cells.get("routable", 0):
            ch.at(0.5, i, "more hopeless than routable — no policy can win "
                          "what is not there", size=9.5, dy=26, colour=ORANGE,
                  italic=True)
            break

    fig.keyrow(190, 330, [(label, fill, "swatch") for _k, fill, label in CELLS],
               680, size=10.5)

    Path(out).write_text(fig.render(), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# 2.5  The price ratio does not decide whether to cascade
# ---------------------------------------------------------------------------

def plot_ratio(out, ladders=LADDERS):
    """The project's own hypothesis, and the measurement that refuses it.

    The design set out to find a price-ratio crossover: cascade above it, route
    below. Three measured ladders come back non-monotonic in the ratio, so no
    threshold on the x axis can separate the points on the y axis - which is a
    result rather than a null, because it is why the shipped router reads a
    frontier instead of computing a ratio.
    """
    pts = []
    for ladder in ladders:
        econ = _frontier_economics(ladder)
        ratio = _effective_ratio(ladder)
        if not econ or ratio is None:
            continue
        if econ["cascade_vs_always_best_pct"] is None:
            continue
        pts.append((ladder, ratio, econ["cascade_vs_always_best_pct"]))

    if len(pts) < 2:
        return _absent("ratio.svg", "needs a frontier run per ladder; run: "
                                    "ROUTER_MODE=replay python scripts/run_all_ladders.py")

    pts.sort(key=lambda p: p[1])
    ys = [p[2] for p in pts]
    y_lo = min(-100.0, math.floor(min(ys) / 25) * 25)
    y_hi = max(25.0, math.ceil(max(ys) / 25) * 25)

    fig = Figure(
        900, 486,
        "The price ratio does not decide whether to cascade",
        _provenance("each ladder's own frontier run, n=209 held-out tasks"),
        notes=[
            "Read down the x axis: cheap ladder cascades, middle ladder does not, "
            "dear ladder cascades. No threshold on this axis separates those, "
            "which is what non-monotonic means.",
            "What decides it instead is what verification costs on that ladder, "
            "and how much better the top rung actually is (RESULTS §2.5).",
        ])

    ch = fig.chart((118, 96, 546, 250), (2.4, 62), (y_lo, y_hi), xscale="log")
    ch.band(y_lo, 0, "#dff0e6", opacity=0.85)
    ch.band(0, y_hi, "#fbe6da", opacity=0.85)
    ch.ygrid([v for v in range(int(y_lo), int(y_hi) + 1, 25)],
             lambda v: f"{v:+.0f}%")
    ch.xgrid([3, 5, 10, 20, 50], lambda v: f"{v:.0f}x")
    ch.frame()
    ch.hline(0, colour=INK, dashed=False, width=1.5)

    # The two half-planes are named in the corners the data leaves empty, and
    # claimed before anything else is placed so the point labels below route
    # around them. Every one of these five annotations is computed from the
    # run; the previous version placed all five at fixed offsets and three of
    # them landed on each other, which made a figure about a clean separation
    # illegible at exactly the point it was making.
    frame_box = (ch.x + 4, ch.y + 4, ch.x + ch.w - 4, ch.y + ch.h - 4)
    # Clear of the rule-of-thumb line, which owns the bottom-left corner.
    fig.text(ch.px(3.0) + 10, ch.py(y_lo) - 12,
             "cascade is CHEAPER at matched accuracy",
             size=10, colour="#1c7a4a", weight="bold", claim=True)
    fig.text(ch.x + ch.w - 10, ch.py(y_hi) + 18,
             "cascade is DEARER at matched accuracy", size=10, anchor="end",
             colour="#a8461a", weight="bold", claim=True)

    # The literature's rule of thumb, drawn because the project set out to test
    # it. It is not used to compute anything - see findings.CROSSOVER_RATIO.
    ch.vline(3.0, colour=MUTED)
    fig.vtext(ch.px(3.0) - 5, ch.y + ch.h - 8, "rule of thumb: cascade above ~3x",
              size=9.5, colour=MUTED, italic=True, claim=True)

    ch.line([(r, y) for _l, r, y in pts], FAINT, width=1.4, dashed=True)
    # The dots go down first and are claimed, so no label can sit on a point.
    for _ladder, ratio, y in pts:
        fig.occupy((ch.px(ratio) - 9, ch.py(y) - 9, ch.px(ratio) + 9,
                    ch.py(y) + 9))
    for ladder, ratio, y in pts:
        colour = LADDER_COLOUR.get(ladder, BLUE)
        ch.dots([(ratio, y)], colour, shape="dot", size=7)
        head, sub = f"{ladder}  {y:+.1f}%", f"{ratio:.2f}x · {_rungs(ladder)}"
        # Eight candidate berths per label, tried in order of how naturally
        # they read: directly above, directly below, then out to either side.
        cands = [(0, -30, "middle"), (0, 40, "middle"),
                 (14, -6, "start"), (-14, -6, "end"),
                 (14, 18, "start"), (-14, 18, "end"),
                 (0, -46, "middle"), (0, 56, "middle")]
        box = fig.place(ch.px(ratio), ch.py(y), head, size=11, weight="bold",
                        colour=colour, offsets=cands, bounds=frame_box)
        # The rungs line goes immediately under whichever berth the name took,
        # so the two always read as one label.
        anchor = ("start" if abs(box[0] - ch.px(ratio)) < 2 else
                  "end" if abs(box[2] - ch.px(ratio)) < 2 else "middle")
        sx = {"start": box[0], "end": box[2], "middle": (box[0] + box[2]) / 2}[anchor]
        fig.text(sx, box[3] + 12, ellipsize(sub, ch.w * 0.55, 9), size=9,
                 anchor=anchor, colour=MUTED, claim=True)

    ch.axis_title("effective price ratio, top rung over bottom  (log scale)",
                  dy=36)
    ch.axis_title("cascade cost vs always-best, at matched accuracy", which="y",
                  dx=62)

    side = 690
    y = fig.block(side, 112,
                  "A rule of the form “cascade when the ratio exceeds τ” must "
                  "put every ladder above τ on the same side of zero.",
                  fig.w - side - 24, title="Why a threshold fails")
    fig.block(side, y + 4,
              f"These {len(pts)} do not sort that way, so no τ gets more than "
              f"two of the three right.", fig.w - side - 24, colour=INK)

    Path(out).write_text(fig.render(), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# 2.3  Predictive routing does not beat a coin flip
# ---------------------------------------------------------------------------

# The pre-registered comparisons. The first two are the six that matter: a
# router that cannot beat a coin flip spending the same money has no skill,
# whatever its accuracy looks like next to `always_cheap`.
COMPARISONS = [
    ("llm_router", "random_matched", "LLM-as-router  vs  cost-matched coin flip"),
    ("routellm", "random_matched", "RouteLLM (pretrained BERT)  vs  same coin flip"),
    ("cascade", "llm_router", "cascade  vs  LLM-as-router"),
]

ALPHA = 0.05


def plot_predictive(out, ladders=LADDERS):
    """Six comparisons, no skill - and the same thing seen a second way.

    Two panels because two objections have to be closed at once. The p-values
    answer "did it beat the null on the tasks", and a reader can fairly reply
    that a single operating point is whatever its owner tuned it to. The AUC
    panel answers that: it integrates each policy's achievable frontier across
    the whole budget range, and RouteLLM sits *below* the null on all three
    ladders at every budget, not just at one.
    """
    from llm_routing.stats import mcnemar_exact

    rows, econ = {}, {}
    for ladder in ladders:
        stats = _policy_stats(ladder)
        if stats:
            rows[ladder] = stats
        e = _frontier_economics(ladder)
        if e:
            econ[ladder] = e
    if not rows:
        return _absent("predictive.svg", "no per-ladder results; run: "
                                         "ROUTER_MODE=replay python scripts/run_all_ladders.py")

    def p_of(ladder, a, b):
        pol = rows[ladder]["policies"]
        if a not in pol or b not in pol:
            return None
        ids = sorted(set(pol[a]["correct"]) & set(pol[b]["correct"]))
        if not ids:
            return None
        return mcnemar_exact([pol[a]["correct"][i] for i in ids],
                             [pol[b]["correct"][i] for i in ids])[2]

    grid = [[(lad, p_of(lad, a, b)) for lad in rows] for a, b, _ in COMPARISONS]
    predictive = [p for row in grid[:2] for _l, p in row if p is not None]
    beat = sum(1 for p in predictive if p < ALPHA)

    n = next(iter(rows.values()))["policies"]["cascade"]["n"]

    # The left gutter is measured, not guessed. `LLM-as-router  vs
    # cost-matched coin flip` is 193px at 10.5, the gutter was 292px with the
    # ladder key parked inside it, and the two printed on top of each other on
    # every render of this figure that has ever been published.
    row_labels = [c[2] for c in COMPARISONS]
    gutter = max(text_width(s, 10.5, "bold") for s in row_labels)
    left = 26 + gutter + 14
    width = 920 - left - 76          # room at the right for the k/3 counts

    fig = Figure(
        920, 556,
        "Predictive routing does not beat a coin flip",
        _provenance(f"exact McNemar over paired outcomes, n={n} "
                    f"held-out tasks per ladder"),
        notes=[
            "`random_matched` flips a coin at the LLM router's own escalation "
            "rate, so the comparison holds spend roughly fixed and isolates skill "
            "rather than budget.",
            "The distinction is when the decision is made: a predictive router "
            "commits before seeing an attempt, a cascade decides after verifying "
            "one (RESULTS §2.3).",
        ])

    # -- panel 1: the p-values -------------------------------------------
    ch = fig.chart((left, 100, width, 116), (0.0018, 1.35), (2.6, -0.6),
                   xscale="log")
    ch.panel_title("Is the difference detectable at all?")
    ch.vband(0.0018, ALPHA, "#dff0e6", opacity=0.9)
    ch.xgrid([0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
             lambda v: f"{v:g}".rstrip("."))
    ch.frame()
    ch.vline(ALPHA, colour="#1c7a4a")
    # Both of these used to sit above the frame, where the panel title already
    # was. The band names itself from the inside instead.
    fig.text(ch.px(ALPHA) - 6, ch.y + ch.h - 7, "detectable — p < 0.05",
             size=9.5, anchor="end", colour="#1c7a4a", italic=True, claim=True)

    for i, ((a, b, label), row) in enumerate(zip(COMPARISONS, grid)):
        skill = a == "cascade"
        fig.text(left - 14, ch.py(i) + 4, label, size=10.5, anchor="end",
                 weight="bold" if skill else None,
                 colour=INK if skill else MUTED)
        for ladder, p in row:
            if p is None:
                continue
            ch.dots([(min(p, 1.28), i)], LADDER_COLOUR.get(ladder, BLUE),
                    shape="dot" if skill else "ring", size=6)
        got = [p for _l, p in row if p is not None]
        if got:
            fig.text(ch.x + ch.w + 10, ch.py(i) + 4,
                     f"{sum(1 for p in got if p < ALPHA)}/{len(got)}",
                     size=11, weight="bold",
                     colour="#1c7a4a" if all(p < ALPHA for p in got) else ORANGE)
    ch.axis_title("McNemar p  (log scale)", dy=32)

    # Clear of the axis title below the frame, which it used to print across.
    fig.text(left, ch.y + ch.h + 58,
             f"{beat} of {len(predictive)} predictive-routing comparisons clear "
             f"p<0.05. The cascade clears it on every ladder.",
             size=11.5, weight="bold", colour=ORANGE if beat == 0 else INK)

    # -- panel 2: the same question at every budget ----------------------
    gains = {}
    for ladder, e in econ.items():
        for fam in ("routellm", "cascade"):
            if fam in e["auc_gain"]:
                gains.setdefault(ladder, {})[fam] = e["auc_gain"][fam]

    bottom = ch.y + ch.h + 58
    if gains:
        vals = [v for d in gains.values() for v in d.values()]
        (lo, hi), xticks = nice_bounds(min(vals + [-0.02]), max(vals + [0.02]), 4)
        ladders_here = [l for l in ladders if l in gains]

        ch2 = fig.chart((left, 322, width, 132), (lo, hi),
                        (len(ladders_here) - 0.4, -0.6))
        ch2.panel_title("...and at every budget, not just the tuned one")
        ch2.vband(lo, 0, "#fbe6da", opacity=0.8)
        ch2.xgrid(xticks, lambda v: f"{v:+.2f}")
        ch2.frame()
        ch2.vline(0, colour=INK, dashed=False, width=1.4)
        fig.text(ch2.px(0) - 6, ch2.y + ch2.h - 7, "worse than the null",
                 size=9.5, anchor="end", colour="#a8461a", italic=True,
                 claim=True)

        for i, ladder in enumerate(ladders_here):
            fig.text(left - 14, ch2.py(i) + 4, ladder, size=12, anchor="end",
                     weight="bold", colour=LADDER_COLOUR.get(ladder, BLUE))
            for j, (fam, fill) in enumerate((("routellm", PINK),
                                             ("cascade", BLUE))):
                if fam not in gains[ladder]:
                    continue
                v = gains[ladder][fam]
                y = i + (j - 0.5) * 0.34
                ch2.hbar(y, min(0, v), max(0, v), 0.14, fill)
                # These bars are 1-3px long at this scale, so the number IS the
                # mark. Placed rather than offset by a constant: on `wide` the
                # two families differ by 0.0006 and their labels landed on each
                # other.
                # Outward from the bar's own end, so the label never lands on
                # the fill it belongs to: a bar running left wants its number
                # further left, not tucked inside it against the zero line.
                away = -1 if v < 0 else 1
                near, far = ("start", "end") if away > 0 else ("end", "start")
                fig.place(ch2.px(v), ch2.py(y) + 4, f"{v:+.4f}", size=9.5,
                          colour=INK,
                          offsets=[(9 * away, 0, near), (9 * away, -11, near),
                                   (9 * away, 11, near), (-9 * away, 0, far)],
                          bounds=(ch2.x - 60, ch2.y, ch2.x + ch2.w + 60,
                                  ch2.y + ch2.h))
        ch2.axis_title("frontier AUC minus the cost-matched null  "
                       "(accuracy points)", dy=32)
        bottom = ch2.y + ch2.h + 48

    # One legend row across the foot of the figure, sized to what it holds.
    # It used to be three separate keys, two of them stacked in the left gutter
    # underneath the row labels above.
    entries = [(l, LADDER_COLOUR.get(l, BLUE), "dot") for l in rows]
    entries += [("hollow — no skill shown", MUTED, "ring"),
                ("filled — skill shown", INK, "dot")]
    if gains:
        entries += [("routellm AUC", PINK, "swatch"),
                    ("cascade AUC", BLUE, "swatch")]
    fig.keyrow(26, bottom, entries, 920 - 52, size=10.5)

    Path(out).write_text(fig.render(), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# 3  What each policy did with its escalations
# ---------------------------------------------------------------------------

def plot_scorecard(out, src=None):
    """Accuracy cannot distinguish escalating the right ten tasks from
    escalating everything, and those are very different products.

    MISTAKES ONLY, so "shorter is better" is literally true. An earlier version
    stacked `rescued` in here too, which made the oracle's bar the tallest on a
    chart captioned "shorter is better" - the good outcome and the bad ones
    cannot share an axis. Rescues and wasted spend are printed instead.
    """
    src = Path(src) if src else paths.RUNS / "scorecard.wide.json"
    data = _read_json(src)
    if not data:
        return _absent("scorecard.svg", f"{src.name} not found; run: python -m "
                                        f"llm_routing.scorecard --json {src}")
    pols = data["policies"]
    order = [p for p in POLICY_ORDER if p in pols]

    buckets = [
        ("missed_rescue", ORANGE, "missed — stayed cheap, answer was wrong"),
        ("wasted_escalation", "#bbbbbb", "wasted — escalated, cheap rung already had it"),
        ("harmful_escalation", PINK, "harmful — escalated a right answer into a wrong one"),
    ]
    totals = [sum(pols[p]["all"].get(k, 0) for k, _c, _l in buckets) for p in order]
    # Headroom for the two lines of annotation each bar carries, then rounded
    # up to a tick: an axis that stops at 212 is an axis whose labels are 0, 53,
    # 106, 159, 212, and nobody reads a value off those.
    (_z, top), yticks = nice_bounds(0, (max(totals) or 1) * 1.24, 4)

    n = pols[order[0]]["all"]["n"]

    # The money sentence, recomputed rather than transcribed. It is the whole
    # point of the chart and it moves whenever the run does.
    exp = pols.get("always_expensive", {}).get("all", {})
    cas = pols.get("cascade", {}).get("all", {})
    verdict = ""
    if exp and cas and cas.get("wasted_cost"):
        verdict = (
            f"`always_expensive` burns {money(exp['wasted_cost'], 3)} on "
            f"escalations that could not improve the answer, to buy "
            f"{exp.get('correct_rescue', 0)} rescues. `cascade` gets "
            f"{cas.get('correct_rescue', 0)} of them and wastes "
            f"{exp['wasted_cost'] / cas['wasted_cost']:.0f}x less, because "
            f"verification tells it which escalations are worth making.")

    fig = Figure(
        900, 520,
        "What each policy did with its escalations",
        _provenance(f"{data['ladder']} ladder, n={n} held-out tasks · every "
                    f"escalation is joined against what the two rungs could "
                    f"actually do"),
        notes=[
            "Shorter is better: every unit of bar is a task got wrong, or an "
            "escalation paid for that could not have improved the answer.",
            verdict,
        ])

    ch = fig.chart((150, 108, 700, 244), (-0.6, len(order) - 0.4), (0, top))
    ch.ygrid(yticks, lambda v: f"{v:.0f}")
    ch.frame()
    ch.axis_title("tasks", which="y", dx=44)

    for i, name in enumerate(order):
        g = pols[name]["all"]
        base = 0.0
        for key, fill, _label in buckets:
            v = g.get(key, 0)
            if v:
                ch.vbar(i, 0.3, base, base + v, fill)
            base += v
        head = top * 0.035
        ch.at(i, base + head, f"{g['accuracy']:.1%}", size=10.5, weight="bold")
        ch.at(i, base + head * 2.5, f"{g.get('correct_rescue', 0)} rescued",
              size=9, colour=GREEN)

    ch.xlabels(range(len(order)), [p.replace("_", " ") for p in order], size=9,
               dy=16)
    for i, name in enumerate(order):
        g = pols[name]["all"]
        fig.text(ch.px(i), ch.y + ch.h + 31, money(per_1k(g["cost_per_task"])),
                 size=9, anchor="middle", colour=MUTED)
        fig.text(ch.px(i), ch.y + ch.h + 45,
                 f"{money(g.get('wasted_cost', 0.0), 3)} wasted", size=9,
                 anchor="middle",
                 colour=ORANGE if g.get("wasted_cost", 0) > 0.3 else MUTED)
    fig.text(142, ch.y + ch.h + 31, "per 1,000 queries", size=9, anchor="end",
             colour=MUTED)
    fig.text(142, ch.y + ch.h + 45, "wasted on escalations", size=9,
             anchor="end", colour=MUTED)

    fig.keyrow(150, 424, [(label, fill, "swatch")
                          for _k, fill, label in buckets], 720, size=10.5)

    Path(out).write_text(fig.render(), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# 2.6  A sixth of the routing opportunity does not reproduce
# ---------------------------------------------------------------------------

def plot_noise(out, ladder=None):
    """What a one-draw-per-cell probe over-reports, measured by redrawing.

    The only figure here whose data cost money to produce and cannot be
    regenerated from anything cheaper: it needs fresh draws at both rungs on
    the decisive tasks, which is what `scripts/provenance/redraw_decisive.py`
    bought.
    """
    ladder = ladder or "wide"
    d = _read_json(paths.RUNS / f"redraw.{ladder}.json")
    if not d:
        return _absent("noise.svg", f"redraw.{ladder}.json not found; it is a "
                                    f"PAID artefact - see scripts/provenance/redraw_decisive.py")

    steps = [
        # One hue across all three. These are the same quantity measured three
        # ways, not three quantities, and giving each its own colour said the
        # opposite - the eye reads three categories where the finding is a
        # single number shrinking. Weight carries the emphasis instead: the
        # first bar is what a probe would publish, the last is what survives.
        ("observed", d.get("observed"), BLUE,
         "one draw per cell — what a probe publishes"),
        ("expected", d.get("expected"), "#5ba3d0",
         "mean over fresh draws"),
        ("reproducible", d.get("reproducible"), "#9ec9e4",
         "cheap reliably fails AND top rung reliably succeeds"),
    ]
    steps = [s for s in steps if s[1] is not None]
    if len(steps) < 2:
        return _absent("noise.svg", "redraw file carries no rates")

    # Round bounds, not data times a fraction. The old axis ran to 19.6% with
    # gridlines at 4.9, 9.8, 14.7 - which `.0%` printed as 5%, 10%, 15%. The
    # labels were round; the lines they sat on were not, so every bar read
    # slightly taller against them than it was.
    (_z, hi), yticks = nice_bounds(0, max(v for _n, v, _c, _d in steps) * 1.45, 4)
    fig = Figure(
        760, 442,
        "A sixth of the routing opportunity does not reproduce",
        _provenance(f"{ladder} ladder · {len(d.get('p_hat', {}))} "
                    f"decisive tasks redrawn {d.get('draws')}x at both "
                    f"rungs · n={d.get('n_graded')} graded"),
        notes=[
            "A router credited against the observed fraction is being paid for "
            "mass it cannot capture twice running.",
            "This is a LOWER bound on the correction: `both_ok` and `inverted` "
            "were not redrawn, so flakiness hidden in them is still uncounted "
            "(RESULTS §2.6).",
        ])

    ch = fig.chart((150, 108, 420, 208), (-0.65, len(steps) - 0.35), (0, hi))
    ch.ygrid(yticks, lambda v: f"{v:.0%}")
    ch.frame()
    ch.axis_title("routable fraction of the task set", which="y", dx=48)

    for i, (name, v, fill, _sub) in enumerate(steps):
        ch.vbar(i, 0.28, 0, v, fill)
        ch.at(i, v, f"{v:.1%}", size=13, weight="bold", dy=-10)
    ch.xlabels(range(len(steps)), [s[0] for s in steps], size=11.5, dy=17)
    for i, (_n, _v, _c, sub) in enumerate(steps):
        for j, part in enumerate(wrap_px(sub, 150, 8.5)):
            fig.text(ch.px(i), ch.y + ch.h + 33 + 12 * j, part, size=8.5,
                     anchor="middle", colour=MUTED)

    # The two drops, which are the finding - the bars alone just descend.
    for i in range(len(steps) - 1):
        a, b = steps[i][1], steps[i + 1][1]
        x = i + 0.5
        ch.at(x, max(a, b), f"−{100 * (a - b):.1f} pts", size=10, dy=-30,
              colour=ORANGE, weight="bold")
        fig.raw(f'<line x1="{ch.px(i + 0.29):.1f}" y1="{ch.py(max(a, b)) - 24:.1f}" '
                f'x2="{ch.px(i + 0.71):.1f}" y2="{ch.py(max(a, b)) - 24:.1f}" '
                f'stroke="{ORANGE}" stroke-width="1.2"/>')

    total = steps[0][1] - steps[-1][1]
    fig.text(596, 150, "The headline", size=11, weight="bold")
    for j, line in enumerate([
            f"{steps[0][1]:.1%} → {steps[-1][1]:.1%}",
            f"a drop of {100 * total:.1f} points,",
            f"or {total / steps[0][1]:.0%} of the",
            "apparent opportunity.",
    ]):
        fig.text(596, 172 + 16 * j, line, size=10.5,
                 colour=INK if j == 0 else MUTED,
                 weight="bold" if j == 0 else None)

    Path(out).write_text(fig.render(), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# 2.5  Cost against accuracy: policies are curves, not points
# ---------------------------------------------------------------------------

# One colour per family, fixed by name rather than by position, so `cascade` is
# the same blue on all three ladders' frontiers even when a ladder is missing a
# family (`cascade_routing` is not replayable from the `deepseek` cache).
FAMILY_COLOUR = {
    "cascade": BLUE,
    "cascade_routing": SKY,
    "routellm": PINK,
    "llm_router": AMBER,
    "random": GREEN,
}

# Reference points get a glyph rather than a colour, because two of them share
# an ink. That was invisible in the old key, which drew every entry as a line
# segment - `always_cheap` and `always_expensive` appeared as two identical
# black dashes with nothing to tell them apart.
REFS = {
    "always_cheap": ("#111111", "circle", "always_cheap — the bottom rung"),
    "always_mid": ("#5f5f5f", "ring", "always_mid — the middle rung"),
    "always_expensive": ("#111111", "square", "always_expensive — the top rung"),
    "oracle": ("#9a9a9a", "diamond", "oracle — the bound, not deployable"),
}


def plot_frontier(out, src=None):
    """The achievable frontier, which is what "frontier" is supposed to mean.

    Two corrections to the chart this replaces:

    * It joined every swept point in cost order, dominated ones included, so
      `cascade_routing` visibly dived and doubled back and the picture was
      sweep noise rather than a curve. The bold line here is `upper_hull` - the
      same function `findings` uses to compute the verdict the router ships -
      and the raw settings stay on the page as faint marks behind it.
    * Its cost axis was linear from zero, which put every cascade setting in
      the left third of the frame. The rungs are two orders of magnitude apart.
    """
    src = Path(src) if src else paths.RUNS / f"frontier.{paths.default_ladder()}.jsonl"
    rows = _read_jsonl(src)
    if not rows:
        return _absent(src.name, "run: python -m llm_routing.frontier")

    from llm_routing import frontier as frontier_mod

    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    curves = {k: v for k, v in fams.items() if v[0].get("knob") is not None}
    fixed = {k: v[0] for k, v in fams.items() if v[0].get("knob") is None}

    costs = [per_1k(r["cost_per_task"]) for r in rows]
    accs = [r["accuracy"] for r in rows]
    c_lo, c_hi = min(costs) / 1.7, max(costs) * 1.7
    a_lo = math.floor(min(accs) * 20 - 0.5) / 20
    a_hi = min(1.0, math.ceil(max(accs) * 20 + 0.5) / 20)

    ladder = rows[0].get("ladder") or "?"
    econ = frontier_mod.economics(rows)

    fig = Figure(
        860, 540,
        f"Cost-quality frontier on the `{ladder}` ladder: policies are curves, "
        f"not points",
        _provenance(f"{_rungs(ladder)} · n={rows[0].get('n')} held-out tasks · "
                    f"every policy swept across its whole knob range"),
        notes=[
            "Up and to the left is better. The bold line is each family's "
            "achievable frontier (upper hull); the faint marks behind it are the "
            "individual settings, including the ones a mix of two others beats.",
            "Comparing two routers at one setting each lets whoever set the knobs "
            "pick the winner, which is why this sweeps them instead.",
        ])

    # The legend column is as wide as its widest entry, and the plot takes
    # what is left. It used to be a fixed 240px holding `always_expensive —
    # the top rung`, which is 176px of text before its swatch.
    legend_w = max(text_width(t, AXIS) for _c, _s, t in REFS.values()) + 46
    ch = fig.chart((78, 104, 860 - 78 - legend_w - 34, 306), (c_lo, c_hi),
                   (a_lo, a_hi), xscale="log")
    ch.ygrid([a_lo + (a_hi - a_lo) * i / 5 for i in range(6)], pct_tick)
    ch.xgrid(log_ticks(c_lo, c_hi), money_tick)
    ch.frame()
    ch.axis_title("cost per 1,000 queries  (log scale)", dy=34)
    ch.axis_title("accuracy on the held-out half", which="y", dx=48)

    entries = []
    for name in sorted(curves):
        colour = FAMILY_COLOUR.get(name, PALETTE[len(entries) % len(PALETTE)])
        pts = [(per_1k(p["cost_per_task"]), p["accuracy"]) for p in curves[name]]
        hull = frontier_mod.upper_hull(pts)
        # The swept settings are context, the hull is the claim, so the
        # weight between them is widened: the two used to read as equally
        # important and the picture came out as a cloud with a line in it.
        ch.dots(sorted(set(pts)), colour, size=2.4, opacity=0.3)
        ch.line(hull, colour, width=2.8, dashed=(name == "random"))
        entries.append((f"{name}{' — the null to beat' if name == 'random' else ''}",
                        colour, "dash" if name == "random" else "line"))

    for name, (colour, shape, label) in REFS.items():
        if name in fixed:
            r = fixed[name]
            ch.dots([(per_1k(r["cost_per_task"]), r["accuracy"])], colour,
                    shape=shape, size=5.5)
            entries.append((label, colour, shape))

    # The headline this chart exists to produce, drawn on it rather than left
    # for the caption: the cheapest cascade setting that matches always-paying
    # for the top rung, against what that rung costs.
    if econ and econ.get("cascade_matched") and "always_expensive" in fixed:
        best = econ["always_expensive"]
        match = econ["cascade_matched"]
        y = best["accuracy"]
        ch.hline(y, colour=MUTED)
        bx, mx = per_1k(best["cost_per_task"]), per_1k(match["cost_per_task"])
        band = y + (a_hi - a_lo) * 0.045
        ch.connector(band, bx, mx, INK, width=1.6)
        pct = econ["cascade_vs_always_best_pct"]
        # Both of these are placed against ink that is already down. The
        # matched-accuracy callout used to print across the `cascade_routing`
        # hull, and the reference-line label used to print across the
        # `always_expensive` square it names.
        fig.occupy((ch.px(bx) - 8, ch.py(y) - 8, ch.px(bx) + 8, ch.py(y) + 8))
        fig.place(ch.px(math.sqrt(bx * mx)), ch.py(band), 
                  f"{pct:+.1f}% at matched accuracy", size=10.5, weight="bold",
                  colour=GREEN if pct < 0 else ORANGE,
                  offsets=[(0, -9, "middle"), (0, -24, "middle"),
                           (0, 20, "middle"), (-12, -9, "end")],
                  bounds=(ch.x + 2, ch.y + 2, ch.x + ch.w - 2,
                          ch.y + ch.h - 2))
        fig.text(ch.x + 6, ch.py(y) - 6, "always-expensive accuracy", size=9,
                 colour=MUTED, claim=True)

    fig.key(614, 116, entries, title="policy families")
    Path(out).write_text(fig.render(), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# 2.8  The experiment: cascade quality AND cost against verifier quality
# ---------------------------------------------------------------------------

def plot_degradation(out, src=None):
    """The only part of the project that manipulates a variable.

    Cost was missing from this figure, and it is a third of the finding. The
    trap §2.8 exposes lives entirely in the bottom panel: at p=1.00 a worthless
    verifier still sends half the traffic to the cheap rung, so cost-per-correct
    still looks respectable. Only the accuracy panel above it says otherwise.
    """
    src = (Path(src) if src else
           paths.RUNS / f"sweep_degraded.{paths.default_ladder()}.jsonl")
    rows = _read_jsonl(src)
    if not rows:
        return _absent(src.name, "run: python -m llm_routing.sweep_degraded")

    by_p = {}
    for r in rows:
        by_p.setdefault(r["verifier_corruption"], []).append(r)
    ps = sorted(by_p)
    acc, esc, cost, bars = [], [], [], []
    for p in ps:
        g = by_p[p]
        n = len(g)
        a = sum(bool(x["correct"]) for x in g) / n
        acc.append((p, a))
        esc.append((p, sum(bool(x["escalated"]) for x in g) / n))
        cost.append((p, per_1k(sum(x["cost_usd"] for x in g) / n)))
        # Binomial standard error, as a visual reminder that these are
        # estimates. The sweep's own table reports the across-draw spread,
        # which is the better number; this is the within-draw one.
        bars.append((p, a, (a * (1 - a) / n) ** 0.5))

    ladder = rows[0].get("ladder") or paths.default_ladder()
    n = len(by_p[ps[0]])
    fig = Figure(
        900, 620,
        "The experiment: cascade quality and cost against verifier quality",
        _provenance(f"{ladder} ladder, {n} code tasks, seed 0 · models, prompts, "
                    f"grader and tasks all held fixed"),
        notes=[
            "p is the probability the verifier ignores the test result and "
            "guesses. p=0 is a perfect verifier; p=1 is a coin flip.",
            "No cliff: there is no threshold below which verifier quality stops "
            "mattering, so it has to be budgeted for rather than assumed "
            "(RESULTS §2.8).",
        ])

    top = fig.chart((88, 100, 560, 190), (0, 1), (0, 1))
    top.panel_title("what the verifier buys, and what it costs in escalations")
    top.ygrid([i / 5 for i in range(6)], pct_tick)
    top.xgrid(ps, lambda v: "", labels=False)
    top.frame()
    top.line(acc, BLUE)
    top.dots(acc, BLUE)
    for x, y, e in bars:
        if e > 0:
            fig.raw(f'<line x1="{top.px(x):.1f}" y1="{top.py(y - e):.1f}" '
                    f'x2="{top.px(x):.1f}" y2="{top.py(y + e):.1f}" '
                    f'stroke="{BLUE}" stroke-width="1.2"/>')
    top.line(esc, ORANGE, dashed=True)
    top.dots(esc, ORANGE)
    # The endpoints of both series, placed rather than offset by a constant.
    # Accuracy starts at 96.1% on a 0-100% axis, so a fixed -10px put the first
    # label through the top of the frame.
    inner = (top.x + 2, top.y + 2, top.x + top.w - 2, top.y + top.h - 2)
    for (px_, py_), txt, colour, right in (
            (acc[0], f"{acc[0][1]:.1%}", BLUE, False),
            (acc[-1], f"{acc[-1][1]:.1%}", BLUE, True),
            (esc[-1], f"{esc[-1][1]:.1%}", ORANGE, True)):
        side = "end" if right else "start"
        fig.place(top.px(px_), top.py(py_), txt, size=10, weight="bold",
                  colour=colour, bounds=inner,
                  offsets=[(0 if right else 0, -10, side),
                           (0, 17, side),
                           (-9 if right else 9, 4, "end" if right else "start"),
                           (0, 26, side)])
    top.axis_title("rate", which="y", dx=46)

    c_vals = [c for _p, c in cost]
    pad = (max(c_vals) - min(c_vals)) * 0.3 or 0.1
    # Rounded outwards to a tick rather than to the data plus a fraction, which
    # is what labelled this axis $0.36, $0.63, $0.91, $1.19, $1.47.
    (c0, c1), cticks = nice_bounds(max(0.0, min(c_vals) - pad),
                                   max(c_vals) + pad, 4)
    bot = fig.chart((88, 360, 560, 150), (0, 1), (max(0.0, c0), c1))
    bot.panel_title("what it costs in money")
    bot.ygrid(cticks, lambda v: money(v))
    bot.xgrid(ps, lambda v: f"{v:.2f}")
    bot.frame()
    bot.line(cost, PINK)
    bot.dots(cost, PINK)
    bot.at(ps[0], cost[0][1], money(cost[0][1]), size=10, anchor="start", dy=-10,
           colour=PINK)
    bot.at(ps[-1], cost[-1][1], money(cost[-1][1]), size=10, anchor="end",
           dy=-10, colour=PINK, weight="bold")
    bot.axis_title("verifier corruption p", dy=34)
    bot.axis_title("USD per 1,000 queries", which="y", dx=46)

    fig.key(690, 116, [
        ("accuracy", BLUE, "line"),
        ("escalation rate", ORANGE, "dash"),
        ("cost", PINK, "line"),
    ], title="measured")

    # Alongside the money panel, which is the panel it is about. It used to
    # sit beside the accuracy panel, 130px above the evidence for it.
    fig.block(690, 372,
              "A cascade with a worthless verifier still looks good on cost "
              "per correct answer: at p=1.00 a coin flip sends half the traffic "
              "to the cheap rung and saves money doing it. Only the "
              "accuracy-matched comparison separates a good verifier from no "
              "verifier, and the sweep prints both.",
              fig.w - 690 - 24, title="The trap")

    Path(out).write_text(fig.render(), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# The product half: the LangGraph state machine
# ---------------------------------------------------------------------------

GRAPH_SRC = paths.ROOT / "router_agent" / "graph.py"
NODES_SRC = paths.ROOT / "router_agent" / "nodes.py"


def read_state_machine(src=GRAPH_SRC):
    """Node and edge lists, parsed out of `build_graph` with `ast`.

    Parsed rather than imported, and rather than drawn from memory. Importing
    would need langgraph installed and would invert the one-way dependency the
    architecture rests on - `router_agent` imports `llm_routing`, never the
    reverse, and CI has a job whose only purpose is to keep it that way.
    Reading the source text does neither.

    A hand-drawn diagram would be a second source of truth for the graph, and
    this repository has been bitten by every one of those it has kept.
    """
    src = Path(src)
    if not src.exists():
        return [], []

    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):          # START / END
            return node.id
        return None

    nodes, edges = [], []
    for call in ast.walk(ast.parse(src.read_text(encoding="utf-8"))):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        what, args = call.func.attr, call.args
        if what == "add_node" and args:
            name = literal(args[0])
            if name and name not in nodes:
                nodes.append(name)
        elif what == "add_edge" and len(args) >= 2:
            a, b = literal(args[0]), literal(args[1])
            if a and b:
                edges.append((a, b, None))
        elif what == "add_conditional_edges" and len(args) >= 3:
            a, mapping = literal(args[0]), args[2]
            if a and isinstance(mapping, ast.Dict):
                for k, v in zip(mapping.keys, mapping.values):
                    label, target = literal(k), literal(v)
                    if target:
                        edges.append((a, target, label))
    return nodes, edges


def plot_graph(out, src=GRAPH_SRC):
    """The half of this repository that ships, which had no figure at all.

    The benchmark's nine policies are all decided by one of these: the shipped
    router is the `cascade` row, running as a state machine with the escalation
    step behind a human-approval interrupt.
    """
    nodes, edges = read_state_machine(src)
    if not nodes:
        return _absent("graph.svg", f"{Path(src).name} not found or has no "
                                    f"add_node calls to read")

    lane = ["START"] + nodes + ["END"]
    order = {name: i for i, name in enumerate(lane)}

    # Parallel edges are collapsed onto one arrow carrying both labels.
    # `verify` routes to `finalize` under two different branch names, and drawn
    # separately they land on identical arcs - one arrow perfectly hiding the
    # other, with the two labels overprinted.
    merged = {}
    for a, b, label in edges:
        if a not in order or b not in order:
            continue
        merged.setdefault((a, b), [])
        if label and label not in merged[(a, b)]:
            merged[(a, b)].append(label)
    edges = [(a, b, " / ".join(labels)) for (a, b), labels in merged.items()]

    interrupts = ("interrupt(" in NODES_SRC.read_text(encoding="utf-8")
                  if NODES_SRC.exists() else False)

    fig = Figure(
        940, 470,
        "The router that ships: classify → answer → verify → escalate ⟲",
        f"parsed from router_agent/graph.py — {len(nodes)} nodes, "
        f"{len(edges)} edges, read with `ast` rather than drawn from memory",
        notes=[
            "The loop is the product. `verify` decides whether the cheap answer "
            "stands; only a failed verification spends money on the next rung, "
            "and `escalate` can refuse on budget, approval or top-of-ladder.",
            "Every dollar figure this graph reports comes from the same price "
            "table and response cache as the benchmark tables, which is what "
            "makes the two comparable.",
        ])

    box_h, box_w = 46, 104
    row1, row2 = 148, 256

    # A node that sends an edge BACKWARDS drops out of the main lane onto a
    # second row. Everything ugly about the previous drawing came from keeping
    # all five in one line: `verify -> finalize` had to arc over `escalate`,
    # that arc crossed the human-approval marker's leader, and its arrowhead
    # landed on precisely the same pixel as `escalate -> finalize`'s, so two
    # different branches appeared to be one arrow. Off the lane, every edge
    # here is a straight segment between two boxes.
    sunk = {a for a, b, _l in edges if order[b] < order[a]}
    main = [n for n in lane if n not in sunk]
    span = 940 - 2 * 56
    xs = {n: 56 + span * i / max(1, len(main) - 1) for i, n in enumerate(main)}
    ys = {n: row1 for n in main}
    for n in sunk:
        near = ([a for a, b, _l in edges if b == n and a in xs]
                + [b for a, b, _l in edges if a == n and b in xs
                   and order[b] > order[n]])
        xs[n] = (sum(xs[m] for m in near) / len(near) if near else 470)
        ys[n] = row2

    def half(name):
        return (21, 21) if name in ("START", "END") else (box_w / 2, box_h / 2)

    def border(name, tx, ty, pad=0.0):
        """Where the segment towards (tx, ty) leaves `name`'s outline."""
        cx, cy, (hw, hh) = xs[name], ys[name], half(name)
        dx, dy = tx - cx, ty - cy
        dist = math.hypot(dx, dy) or 1.0
        if name in ("START", "END"):
            t = (hw + pad) / dist
        else:
            t = min(hw / abs(dx) if dx else 9e9, hh / abs(dy) if dy else 9e9)
            t += pad / dist
        return cx + dx * t, cy + dy * t

    def arrow(x0, y0, x1, y1, colour):
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        bx, by = x1 - 9 * ux, y1 - 9 * uy
        fig.stroke(x0, y0, bx, by, colour, 1.8)
        fig.raw(f'<polygon points="{x1:.1f},{y1:.1f} '
                f'{bx - 4.6 * uy:.1f},{by + 4.6 * ux:.1f} '
                f'{bx + 4.6 * uy:.1f},{by - 4.6 * ux:.1f}" fill="{colour}"'
                f'{_cls(colour)}/>')

    # Boxes are claimed before any edge label is placed, so no label can land
    # on a node name.
    for name in lane:
        hw, hh = half(name)
        fig.occupy((xs[name] - hw - 3, ys[name] - hh - 3,
                    xs[name] + hw + 3, ys[name] + hh + 3))

    cycle_y = row2 + box_h / 2 + 56
    for a, b, label in edges:
        back = order[b] < order[a]
        colour = ORANGE if back else INK
        if back:
            # Routed around the outside rather than straight through the middle:
            # this is the edge that makes the thing a cascade, and it reads as
            # a return path only if it is drawn as one.
            ax, ay = xs[a], ys[a] + half(a)[1]
            bx, by = xs[b], ys[b] + half(b)[1]
            fig.raw(f'<path d="M {ax:.1f} {ay:.1f} L {ax:.1f} {cycle_y:.1f} '
                    f'L {bx:.1f} {cycle_y:.1f} L {bx:.1f} {by + 11:.1f}" '
                    f'fill="none" stroke="{colour}" stroke-width="1.8" '
                    f'stroke-linejoin="round"/>')
            arrow(bx, by + 20, bx, by + 2, colour)
            if label:
                fig.text((ax + bx) / 2, cycle_y + 15, label, size=9.5,
                         anchor="middle", colour=colour, weight="bold",
                         claim=True)
            continue
        x0, y0 = border(a, xs[b], ys[b])
        x1, y1 = border(b, xs[a], ys[a], pad=1.0)
        arrow(x0, y0, x1, y1, colour)
        if label:
            fig.place((x0 + x1) / 2, (y0 + y1) / 2, label, size=9.5,
                      colour=colour,
                      offsets=[(0, -7, "middle"), (0, 15, "middle"),
                               (10, 4, "start"), (-10, 4, "end")])

    for name in lane:
        cx, cy = xs[name], ys[name]
        if name in ("START", "END"):
            # One `class`, carrying both roles: two class attributes on one
            # element is not well-formed XML and the whole file fails to parse.
            fig.raw(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="21" '
                    f'fill="{PAPER}" stroke="{INK}" stroke-width="1.8" '
                    f'class="f-paper s-ink"/>')
            fig.text(cx, cy + 4, name, size=9.5, anchor="middle", weight="bold")
            continue
        fill = "#e8f1f8" if name != "escalate" else "#fdeee2"
        stroke = BLUE if name != "escalate" else ORANGE
        fig.rect(cx - box_w / 2, cy - box_h / 2, box_w, box_h, fill,
                 stroke=stroke, rx=7)
        fig.text(cx, cy + 4, name, size=12, anchor="middle", weight="bold",
                 colour=ON_LIGHT)

    if interrupts:
        gate = "escalate" if "escalate" in xs else nodes[-1]
        gx = xs[gate] + box_w / 2
        fig.stroke(gx + 2, ys[gate], gx + 20, ys[gate], ORANGE, 1.2, dash="4,3")
        fig.text(gx + 26, ys[gate] - 2, "human approval", size=10,
                 weight="bold", colour=ORANGE)
        fig.text(gx + 26, ys[gate] + 12, "interrupt() suspends here", size=9,
                 colour=MUTED)

    fig.text(56, cycle_y + 42,
             "the cycle — `escalate` bumps the rung and re-enters `answer`, "
             "which is what makes this a cascade rather than a router",
             size=10, colour=ORANGE)

    Path(out).write_text(fig.render(), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Draw every figure in figures/ from the artefacts in runs/.")
    ap.add_argument("--outdir", default=str(paths.FIGURES))
    ap.add_argument("--frontier", metavar="PATH", default=None,
                    help="read this frontier file instead of "
                         "frontier.<ladder>.jsonl")
    ap.add_argument("--sweep", metavar="PATH", default=None,
                    help="read this degradation sweep instead of "
                         "sweep_degraded.<ladder>.jsonl")
    ap.add_argument("--suffix", default="",
                    help="appended to each per-ladder figure's stem, e.g. "
                         "--suffix .wide gives figures/frontier.wide.svg. "
                         "Without it a second ladder's figures overwrite the "
                         "first ladder's.")
    ap.add_argument("--no-summaries", dest="summaries", action="store_false",
                    help="skip the cross-ladder charts, which read every "
                         "ladder's artefacts rather than the files named above")
    ap.add_argument("--only-summaries", action="store_true",
                    help="draw only the cross-ladder charts. They are the same "
                         "picture whichever ladder is loaded, so the driver "
                         "draws them once after the last ladder rather than "
                         "three identical times.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    made = []

    # Per-ladder: these read one ladder's run and take the suffix.
    if not args.only_summaries:
        for fn, stem, src in (
                (plot_frontier, "frontier", args.frontier),
                (plot_degradation, "degradation", args.sweep)):
            svg = outdir / f"{stem}{args.suffix}.svg"
            if fn(svg, Path(src) if src else None):
                made.append(svg)

    # Cross-ladder: written once, not per ladder, so they take no suffix - a
    # --suffix run would otherwise redraw the same chart three times under
    # three names.
    if args.summaries or args.only_summaries:
        for fn, stem in ((plot_ladders, "ladders"),
                         (plot_routable, "routable"),
                         (plot_ratio, "ratio"),
                         (plot_predictive, "predictive"),
                         (plot_scorecard, "scorecard"),
                         (plot_noise, "noise"),
                         (plot_graph, "graph")):
            svg = outdir / f"{stem}.svg"
            if fn(svg):
                made.append(svg)

    if not made:
        sys.exit("no figures written; generate the data files in runs/ first")
    for p in made:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
