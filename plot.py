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

    python3 sweep_degraded.py && python3 frontier.py   # produce the data
    python3 plot.py                                    # draw both
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent

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

    def axes(self, x_fmt, y_fmt, n_x=5, n_y=5):
        for i in range(n_y + 1):
            y = self.y0 + (self.y1 - self.y0) * i / n_y
            py = self.py(y)
            self.parts.append(
                f'<line x1="{PAD_L}" y1="{py:.1f}" x2="{W - PAD_R}" y2="{py:.1f}" '
                f'stroke="{GRID}" stroke-width="1"/>')
            self.parts.append(
                f'<text x="{PAD_L - 9}" y="{py + 4:.1f}" text-anchor="end" '
                f'font-size="11" fill="{INK}">{esc(y_fmt(y))}</text>')
        for i in range(n_x + 1):
            x = self.x0 + (self.x1 - self.x0) * i / n_x
            px = self.px(x)
            self.parts.append(
                f'<line x1="{px:.1f}" y1="{PAD_T}" x2="{px:.1f}" y2="{H - PAD_B}" '
                f'stroke="{GRID}" stroke-width="1"/>')
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
        tail = [
            f'<text x="{PAD_L + (W - PAD_L - PAD_R) / 2:.0f}" y="{H - 30}" '
            f'text-anchor="middle" font-size="12" fill="{INK}">{esc(self.x_label)}</text>',
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


def plot_frontier(out):
    rows = _read(HERE / "frontier.jsonl")
    if not rows:
        print("  frontier.jsonl not found - run: python3 frontier.py", file=sys.stderr)
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


def plot_degradation(out):
    rows = _read(HERE / "sweep_degraded.jsonl")
    if not rows:
        print("  sweep_degraded.jsonl not found - run: python3 sweep_degraded.py",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(HERE / "figures"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    made = []
    if plot_frontier(outdir / "frontier.svg"):
        made.append(outdir / "frontier.svg")
    if plot_degradation(outdir / "degradation.svg"):
        made.append(outdir / "degradation.svg")

    if not made:
        sys.exit("no figures written; generate the data files first")
    for p in made:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
