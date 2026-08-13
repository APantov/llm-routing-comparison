"""Figures, from the standard library only.

No matplotlib. That is a deliberate constraint rather than an inconvenience: the
whole repo reproduces on a bare interpreter with no network and no installs, and
adding a plotting dependency to draw two charts would trade that away for very
little. SVG is text, so writing it directly is a hundred lines and the output is
sharp at any size and diffable in git.

Two figures:

  frontier.svg    cost against accuracy, one line per policy family. This is the
                  figure that makes the point a table cannot: policies are curves,
                  and which one wins depends on the budget.

  degradation.svg accuracy and escalation rate against verifier corruption, with
                  error bars over the repeat draws. The experiment.

    python -m llm_routing.sweep_degraded && python -m llm_routing.frontier   # produce the data
    python -m llm_routing.plot                                    # draw both
"""

import argparse
import json
import sys
from pathlib import Path

from llm_routing import paths

W, H = 760, 470
PAD_L, PAD_R, PAD_T, PAD_B = 78, 168, 46, 62

# Colour-blind-safe qualitative palette (Okabe-Ito). Chosen over a default cycle
# because the frontier chart puts five lines on one pair of axes and red/green
# alone would make it unreadable for a fair share of readers.
PALETTE = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]
INK = "#222222"
GRID = "#dddddd"


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class Chart:
    """Minimal linear-axes plotter. Data coordinates in, SVG out."""

    def __init__(self, title, x_label, y_label, xlim, ylim, subtitle=""):
        self.parts = []
        self.title = title
        self.subtitle = subtitle
        self.x_label, self.y_label = x_label, y_label
        (self.x0, self.x1), (self.y0, self.y1) = xlim, ylim
        if self.x1 <= self.x0:
            self.x1 = self.x0 + 1e-9
        if self.y1 <= self.y0:
            self.y1 = self.y0 + 1e-9
        self.legend = []

    def px(self, x):
        return PAD_L + (x - self.x0) / (self.x1 - self.x0) * (W - PAD_L - PAD_R)

    def py(self, y):
        return H - PAD_B - (y - self.y0) / (self.y1 - self.y0) * (H - PAD_T - PAD_B)

    def axes(self, x_fmt, y_fmt, n_x=5, n_y=5, x_ticks=True):
        for i in range(n_y + 1):
            y = self.y0 + (self.y1 - self.y0) * i / n_y
            py = self.py(y)
            self.parts.append(
                f'<line x1="{PAD_L}" y1="{py:.1f}" x2="{W - PAD_R}" y2="{py:.1f}" '
                f'stroke="{GRID}" stroke-width="1"/>')
            self.parts.append(
                f'<text x="{PAD_L - 9}" y="{py + 4:.1f}" text-anchor="end" '
                f'font-size="11" fill="{INK}">{esc(y_fmt(y))}</text>')
        if x_ticks:
            for i in range(n_x + 1):
                x = self.x0 + (self.x1 - self.x0) * i / n_x
                px = self.px(x)
                self.parts.append(
                    f'<line x1="{px:.1f}" y1="{PAD_T}" x2="{px:.1f}" '
                    f'y2="{H - PAD_B}" stroke="{GRID}" stroke-width="1"/>')
                self.parts.append(
                    f'<text x="{px:.1f}" y="{H - PAD_B + 18}" text-anchor="middle" '
                    f'font-size="11" fill="{INK}">{esc(x_fmt(x))}</text>')
        self.parts.append(
            f'<rect x="{PAD_L}" y="{PAD_T}" width="{W - PAD_L - PAD_R}" '
            f'height="{H - PAD_T - PAD_B}" fill="none" stroke="{INK}" stroke-width="1"/>')

    def line(self, pts, colour, label, dashed=False, markers=True):
        if not pts:
            return
        d = " ".join(f"{'M' if i == 0 else 'L'} {self.px(x):.1f} {self.py(y):.1f}"
                     for i, (x, y) in enumerate(pts))
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2.2"{dash}/>')
        if markers:
            for x, y in pts:
                self.parts.append(
                    f'<circle cx="{self.px(x):.1f}" cy="{self.py(y):.1f}" r="3" '
                    f'fill="{colour}"/>')
        self.legend.append((label, colour, dashed))

    def errorbars(self, pts, colour):
        """pts are (x, y, half_height) in data units."""
        for x, y, e in pts:
            if e <= 0:
                continue
            px = self.px(x)
            self.parts.append(
                f'<line x1="{px:.1f}" y1="{self.py(y - e):.1f}" x2="{px:.1f}" '
                f'y2="{self.py(y + e):.1f}" stroke="{colour}" stroke-width="1.2"/>')

    def bar(self, x, half_width, y_lo, y_hi, colour, label=None):
        """A rectangle in data coordinates, for the categorical charts.

        Bars rather than lines wherever the x axis is a name (a ladder, a
        policy) rather than a quantity. A line between two ladders would imply
        the space between them means something, and it does not.
        """
        x0, x1 = self.px(x - half_width), self.px(x + half_width)
        y0, y1 = self.py(y_hi), self.py(y_lo)
        self.parts.append(
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" '
            f'height="{max(0.0, y1 - y0):.1f}" fill="{colour}"/>')
        if label and (label, colour, False) not in self.legend:
            self.legend.append((label, colour, False))

    def label_at(self, x, y, text, size=10, anchor="middle", colour=None):
        self.parts.append(
            f'<text x="{self.px(x):.1f}" y="{self.py(y):.1f}" '
            f'text-anchor="{anchor}" font-size="{size}" '
            f'fill="{colour or INK}">{esc(text)}</text>')

    def tick_labels(self, positions, labels, size=11):
        """Replace the numeric x axis with category names."""
        for x, text in zip(positions, labels):
            self.parts.append(
                f'<text x="{self.px(x):.1f}" y="{H - PAD_B + 18}" '
                f'text-anchor="middle" font-size="{size}" '
                f'fill="{INK}">{esc(text)}</text>')

    def point(self, x, y, colour, label, shape="diamond"):
        px, py = self.px(x), self.py(y)
        if shape == "diamond":
            self.parts.append(
                f'<polygon points="{px:.1f},{py - 6:.1f} {px + 6:.1f},{py:.1f} '
                f'{px:.1f},{py + 6:.1f} {px - 6:.1f},{py:.1f}" fill="{colour}"/>')
        elif shape == "square":
            self.parts.append(
                f'<rect x="{px - 4.5:.1f}" y="{py - 4.5:.1f}" width="9" height="9" '
                f'fill="{colour}"/>')
        else:
            self.parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{colour}"/>')
        self.legend.append((label, colour, False))

    def render(self, note=""):
        head = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}" font-family="Helvetica,Arial,sans-serif">',
            f'<rect width="{W}" height="{H}" fill="white"/>',
            f'<text x="{PAD_L}" y="24" font-size="15" font-weight="bold" '
            f'fill="{INK}">{esc(self.title)}</text>',
        ]
        if self.subtitle:
            head.append(
                f'<text x="{PAD_L}" y="40" font-size="11" fill="#666666">'
                f'{esc(self.subtitle)}</text>')
        tail = []
        if self.x_label:
            tail.append(
                f'<text x="{PAD_L + (W - PAD_L - PAD_R) / 2:.0f}" y="{H - 30}" '
                f'text-anchor="middle" font-size="12" fill="{INK}">'
                f'{esc(self.x_label)}</text>')
        tail += [
            f'<text x="18" y="{PAD_T + (H - PAD_T - PAD_B) / 2:.0f}" font-size="12" '
            f'fill="{INK}" transform="rotate(-90 18 '
            f'{PAD_T + (H - PAD_T - PAD_B) / 2:.0f})" text-anchor="middle">'
            f'{esc(self.y_label)}</text>',
        ]
        ly = PAD_T + 6
        for label, colour, dashed in self.legend:
            dash = ' stroke-dasharray="6,4"' if dashed else ""
            tail.append(
                f'<line x1="{W - PAD_R + 12}" y1="{ly}" x2="{W - PAD_R + 36}" '
                f'y2="{ly}" stroke="{colour}" stroke-width="2.5"{dash}/>')
            tail.append(
                f'<text x="{W - PAD_R + 42}" y="{ly + 4}" font-size="11" '
                f'fill="{INK}">{esc(label)}</text>')
            ly += 19
        if note:
            tail.append(
                f'<text x="{PAD_L}" y="{H - 9}" font-size="10" fill="#888888">'
                f'{esc(note)}</text>')
        return "\n".join(head + self.parts + tail + ["</svg>"])


def _read(path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _mark(rows):
    """Loud label if the underlying numbers are simulated."""
    if rows and any(r.get("simulated") for r in rows):
        return "SIMULATED (mock mode) - restates models.MOCK_SKILL, measures no model"
    return "measured"


def plot_frontier(out, src=None):
    src = src or paths.RUNS / f"frontier.{paths.default_ladder()}.jsonl"
    rows = _read(src)
    if not rows:
        print(f"  {src.name} not found - run: python -m llm_routing.frontier", file=sys.stderr)
        return False

    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    fixed = {k: v[0] for k, v in fams.items() if v[0].get("knob") is None}
    curves = {k: v for k, v in fams.items() if v[0].get("knob") is not None}

    costs = [r["cost_per_task"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    lo, hi = min(accs), max(accs)
    pad = max(0.02, (hi - lo) * 0.12)

    c = Chart(
        "Cost-quality frontier: policies are curves, not points",
        "cost per task (USD)", "accuracy",
        (0, max(costs) * 1.05), (max(0.0, lo - pad), min(1.0, hi + pad)),
        subtitle=_mark(rows),
    )
    c.axes(lambda v: f"${v:.4f}", lambda v: f"{v:.0%}")

    for i, (name, pts) in enumerate(sorted(curves.items())):
        pts = sorted(((p["cost_per_task"], p["accuracy"]) for p in pts))
        c.line(pts, PALETTE[i % len(PALETTE)], name,
               dashed=(name == "random"))

    # Reference points get a distinct glyph, because they are single settings
    # rather than curves; drawing them as one-point lines would imply a sweep.
    # Distinct greys rather than one shared ink, so the legend can tell them apart.
    refs = {
        "always_cheap": ("#111111", "circle"),
        "always_mid": ("#666666", "circle"),
        "always_expensive": ("#111111", "square"),
        # The oracle is a diamond and a lighter grey because it is not a policy
        # anyone can deploy. It marks the headroom, not a competitor.
        "oracle": ("#999999", "diamond"),
    }
    for name, (colour, shape) in refs.items():
        if name in fixed:
            r = fixed[name]
            label = name + (" (not deployable)" if name == "oracle" else "")
            c.point(r["cost_per_task"], r["accuracy"], colour, label, shape=shape)

    out.write_text(c.render(
        "up and to the left is better; the dashed line is the random baseline "
        "every router must beat"), encoding="utf-8")
    return True


def plot_degradation(out, src=None):
    src = src or paths.RUNS / f"sweep_degraded.{paths.default_ladder()}.jsonl"
    rows = _read(src)
    if not rows:
        print(f"  {src.name} not found - run: python -m llm_routing.sweep_degraded",
              file=sys.stderr)
        return False

    # sweep_degraded writes per-task rows for seed 0 only, so aggregate here.
    by_p = {}
    for r in rows:
        by_p.setdefault(r["verifier_corruption"], []).append(r)
    pts_acc, pts_esc, bars = [], [], []
    for p in sorted(by_p):
        g = by_p[p]
        n = len(g)
        acc = sum(bool(x["correct"]) for x in g) / n
        esc = sum(bool(x["escalated"]) for x in g) / n
        pts_acc.append((p, acc))
        pts_esc.append((p, esc))
        # Binomial standard error, as a visual reminder that these points are
        # estimates. The sweep's own table reports the across-draw spread, which
        # is the better number; this is the within-draw one.
        bars.append((p, acc, (acc * (1 - acc) / n) ** 0.5))

    c = Chart(
        "The experiment: cascade quality against verifier quality",
        "verifier corruption p  (probability the verifier ignores the tests)",
        "rate",
        (0, 1), (0, 1),
        subtitle=f"{_mark(rows)}  |  n={len(by_p[sorted(by_p)[0]])} code tasks, seed 0",
    )
    c.axes(lambda v: f"{v:.2f}", lambda v: f"{v:.0%}")
    c.line(pts_acc, PALETTE[0], "accuracy")
    c.errorbars(bars, PALETTE[0])
    c.line(pts_esc, PALETTE[1], "escalation rate", dashed=True)

    out.write_text(c.render(
        "p=0 is a perfect verifier; p=1 is a coin flip. Everything else is held "
        "fixed."), encoding="utf-8")
    return True


def plot_ladders(out, ladders=("wide", "claude", "deepseek")):
    """The headline, across all three ladders at once.

    The one chart that carries the finding: the cascade's accuracy against
    always-paying-for-the-best, on each ladder, with the cost difference
    printed on the bar. Everything else in figures/ is per-ladder detail.
    """
    groups, missing = [], []
    for ladder in ladders:
        rows = _read(paths.RUNS / f"results.{ladder}.jsonl")
        if not rows:
            missing.append(ladder)
            continue
        by = {}
        for r in rows:
            if r.get("split") not in (None, "eval"):
                continue
            by.setdefault(r["policy"], []).append(r)
        if not {"cascade", "always_expensive"} <= set(by):
            missing.append(ladder)
            continue
        stat = {}
        for name in ("cascade", "always_expensive"):
            g = by[name]
            stat[name] = (sum(bool(r["correct"]) for r in g) / len(g),
                          sum(r["cost_usd"] for r in g) / len(g))
        groups.append((ladder, stat, any(r.get("simulated") for r in rows)))

    if not groups:
        print("  no per-ladder results found - run: "
              "ROUTER_MODE=replay python scripts/run_all_ladders.py", file=sys.stderr)
        return False

    accs = [a for _, s, _ in groups for a, _ in s.values()]
    lo = min(accs) - 0.08
    c = Chart(
        "Cascade vs always paying for the best model",
        "", "accuracy on the held-out half",
        (-0.5, len(groups) - 0.5), (max(0.0, lo), 1.0),
        subtitle=("SIMULATED" if any(m for _, _, m in groups)
                  else "measured on real models, n=209 per ladder"),
    )
    c.axes(lambda v: "", lambda v: f"{v:.0%}", x_ticks=False)

    for i, (ladder, stat, _) in enumerate(groups):
        for j, (name, colour) in enumerate((("always_expensive", "#999999"),
                                            ("cascade", PALETTE[0]))):
            acc, cost = stat[name]
            x = i + (j - 0.5) * 0.28
            c.bar(x, 0.12, max(0.0, lo), acc, colour, name)
            # Accuracy above the bar, cost inside it. The cost label used to sit
            # further above and ran off the top of the frame on `claude`, where
            # the cascade reaches 96.7%.
            c.label_at(x, acc + 0.010, f"{acc:.1%}")
            c.label_at(x, acc - 0.022, f"${cost:.5f}", size=9, colour="#ffffff")
        # The cost delta is the second half of the finding and belongs on the
        # chart: on `claude` the cascade wins accuracy and LOSES on cost, which
        # a bare accuracy chart would hide.
        d = stat["cascade"][1] - stat["always_expensive"][1]
        c.label_at(i, max(0.0, lo) + 0.012,
                   f"{'+' if d > 0 else ''}${d:.5f}/task", size=10,
                   colour="#d55e00" if d > 0 else "#009e73")

    c.tick_labels(range(len(groups)), [g[0] for g in groups], size=13)
    out.write_text(c.render(
        "green = the cascade also costs less; orange = it buys the accuracy at a "
        "premium"), encoding="utf-8")
    return True


def plot_scorecard(out, src=None):
    """Where each policy's spend goes, in tasks rather than dollars.

    Accuracy alone cannot distinguish a router that escalated the right tasks
    from one that escalated everything, and those are very different products.
    """
    src = src or paths.RUNS / "scorecard.wide.json"
    if not src.exists():
        print(f"  {src.name} not found - run: python -m llm_routing.scorecard "
              f"--json {src}", file=sys.stderr)
        return False
    data = json.loads(src.read_text(encoding="utf-8"))
    pols = data["policies"]

    order = ["oracle", "cascade", "cascade_routing", "always_expensive",
             "llm_router", "routellm", "random_matched", "always_cheap"]
    order = [p for p in order if p in pols]

    # MISTAKES ONLY, so "shorter is better" is literally true. An earlier
    # version stacked `rescued` in here too, which made the oracle's bar the
    # tallest on a chart captioned "shorter is better" - the good outcome and
    # the bad ones cannot share an axis. Rescues are printed as a number
    # instead.
    buckets = [
        ("missed_rescue", PALETTE[1], "missed (stayed cheap, was wrong)"),
        ("wasted_escalation", "#bbbbbb", "wasted (escalated, cheap already had it)"),
        ("harmful_escalation", PALETTE[3], "harmful (escalated and lost the answer)"),
    ]
    totals = [sum(pols[p]["all"].get(k, 0) for k, _, _ in buckets) for p in order]

    c = Chart(
        "What each policy did with its escalations",
        "", "tasks",
        (-0.6, len(order) - 0.4), (0, max(totals) * 1.15),
        subtitle=(f"{'SIMULATED' if data.get('simulated') else 'measured'}"
                  f"  |  {data['ladder']} ladder, n={pols[order[0]]['all']['n']}"),
    )
    c.axes(lambda v: "", lambda v: f"{v:.0f}", n_y=4, x_ticks=False)

    for i, name in enumerate(order):
        g = pols[name]["all"]
        base = 0.0
        for key, colour, label in buckets:
            v = g.get(key, 0)
            if v:
                c.bar(i, 0.3, base, base + v, colour, label)
            base += v
        head = max(totals) * 0.04
        c.label_at(i, base + head, f"{g['accuracy']:.1%}", size=10)
        c.label_at(i, base + head * 2.4,
                   f"{g.get('correct_rescue', 0)} rescued", size=9,
                   colour=PALETTE[2])

    c.tick_labels(range(len(order)), [p.replace("_", " ") for p in order], size=9)
    out.write_text(c.render(
        "shorter is better - every bar is a task got wrong, or an escalation paid "
        "for and not needed. Accuracy and rescues above each bar."),
        encoding="utf-8")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(paths.FIGURES))
    ap.add_argument("--frontier", metavar="PATH", default=None,
                    help="read this frontier file instead of frontier.jsonl")
    ap.add_argument("--sweep", metavar="PATH", default=None,
                    help="read this degradation sweep instead of "
                         "sweep_degraded.jsonl")
    ap.add_argument("--suffix", default="",
                    help="appended to each figure's stem, e.g. --suffix .claude "
                         "gives figures/frontier.claude.svg. Without it a second "
                         "ladder's figures overwrite the first ladder's.")
    ap.add_argument("--no-summaries", dest="summaries", action="store_false",
                    help="skip the two cross-ladder charts (ladders.svg, "
                         "scorecard.svg), which read every ladder's results "
                         "rather than the files named above")
    ap.add_argument("--only-summaries", action="store_true",
                    help="draw only the two cross-ladder charts. They are the "
                         "same picture whichever ladder is loaded, so the "
                         "driver draws them once after the last ladder rather "
                         "than three identical times.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frontier_svg = outdir / f"frontier{args.suffix}.svg"
    degradation_svg = outdir / f"degradation{args.suffix}.svg"

    made = []
    if not args.only_summaries:
        if plot_frontier(frontier_svg,
                         Path(args.frontier) if args.frontier else None):
            made.append(frontier_svg)
        if plot_degradation(degradation_svg,
                            Path(args.sweep) if args.sweep else None):
            made.append(degradation_svg)

    # The two cross-ladder summaries are written once, not per ladder, so they
    # take no suffix - a --suffix run would otherwise redraw the same chart
    # three times under three names.
    if args.summaries or args.only_summaries:
        for fn, name in ((plot_ladders, "ladders"), (plot_scorecard, "scorecard")):
            svg = outdir / f"{name}.svg"
            if fn(svg):
                made.append(svg)

    if not made:
        sys.exit("no figures written; generate the data files first")
    for p in made:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
