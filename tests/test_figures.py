"""The figures, checked as geometry rather than as pictures.

Everything in `figures/` is written by hand-placed coordinates in
`llm_routing.plot`, and a chart whose labels print on top of each other is
still a valid SVG, still regenerates byte-identically, and still passes every
other test in this suite. That is how this repository came to publish a
cross-tab with `height="0.0"` on every bar, a McNemar panel whose row labels
ran underneath its own legend, and a price-ratio chart with five annotations
stacked in one corner - defects that only a person looking at the file could
catch, and only if they thought to look.

So the layout is asserted. These tests re-measure each committed figure with
the same Helvetica advance widths that drew it and fail on the three things
that actually went wrong:

    parseable      an SVG with two `class` attributes on one element is not
                   well-formed, renders as nothing, and looks fine in a diff
    inside         no text may fall outside the canvas it was sized for
    legible        no two runs of text may substantially overlap, and no bar
                   may be drawn with zero extent

This cannot catch an ugly figure. It catches an unreadable one, which is the
class of bug that got published.
"""

import math
import re
import xml.etree.ElementTree as ET

import pytest

from llm_routing import paths, plot

SVG = "{http://www.w3.org/2000/svg}"
FIGURES = sorted(paths.FIGURES.glob("*.svg"))


def _figures():
    if not FIGURES:
        pytest.skip("no figures drawn yet; run python -m llm_routing.plot")
    return FIGURES


def _boxes(root):
    """Every run of text in the file, as (box, string).

    `y` is the baseline; the box is cap height above it and descender below,
    matching `Figure.text_box`, so what is asserted here is what the drawing
    code believed it was reserving.
    """
    out = []
    for el in root.iter(f"{SVG}text"):
        s = "".join(el.itertext()).strip()
        if not s:
            continue
        x, y = float(el.get("x", 0)), float(el.get("y", 0))
        size = float(el.get("font-size", 11))
        weight = el.get("font-weight")
        anchor = el.get("text-anchor", "start")
        w = plot.text_width(s, size, weight)
        transform = el.get("transform", "")
        if "rotate(-90" in transform:
            # Grows upwards from (x, y); the em box grows to its left.
            if anchor == "start":
                ylo, yhi = y - w, y
            elif anchor == "end":
                ylo, yhi = y, y + w
            else:
                ylo, yhi = y - w / 2, y + w / 2
            box = (x - size * 0.75, ylo, x + size * 0.22, yhi)
        else:
            x0 = x if anchor == "start" else (x - w if anchor == "end"
                                              else x - w / 2)
            box = (x0, y - size * 0.75, x0 + w, y + size * 0.22)
        out.append((box, s))
    return out


def _overlap(a, b):
    """Area shared by two boxes."""
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


@pytest.mark.parametrize("path", _figures(), ids=lambda p: p.name)
class TestFigure:

    def test_is_well_formed_xml(self, path):
        """A renderer reads these; a diff does not.

        Two `class` attributes on one element is the specific mistake, and it
        costs the whole file: browsers refuse a malformed SVG outright rather
        than skipping the element, so the figure renders as blank space in a
        README that otherwise looks correct.
        """
        ET.parse(path)

    def test_carries_its_own_description(self, path):
        """Every figure is embedded in a README with a caption, and a caption
        is not alt text. `role="img"` plus title and desc is what a screen
        reader gets."""
        root = ET.parse(path).getroot()
        assert root.get("role") == "img"
        title = root.find(f"{SVG}title")
        desc = root.find(f"{SVG}desc")
        assert title is not None and (title.text or "").strip()
        assert desc is not None and len((desc.text or "").strip()) > 40

    def test_nothing_is_drawn_outside_the_canvas(self, path):
        """Text that overruns the viewBox is not clipped by SVG - it is simply
        gone, because the renderer crops to the box. The counts on
        `predictive.svg` were lost this way."""
        root = ET.parse(path).getroot()
        w, h = float(root.get("width")), float(root.get("height"))
        for (x0, y0, x1, y1), s in _boxes(root):
            assert x0 >= -1 and x1 <= w + 1, f"{s!r} runs off {path.name} at x"
            assert y0 >= -1 and y1 <= h + 1, f"{s!r} runs off {path.name} at y"

    def test_no_bar_is_drawn_with_zero_extent(self, path):
        """`rect` clamps a negative height at zero, so a bar computed on an
        inverted axis silently disappears instead of failing. Four rows of
        `routable.svg` and six bars of `predictive.svg` shipped like this."""
        root = ET.parse(path).getroot()
        for el in root.iter(f"{SVG}rect"):
            if el.get("fill") in (None, "none"):
                continue
            w, h = float(el.get("width", 0)), float(el.get("height", 0))
            # A rect with one zero side is either a collapsed bar or a hairline
            # nobody meant to draw. Both are bugs; neither is ever intended.
            assert not (w > 0.5 and h == 0.0), (
                f"{path.name}: zero-height rect {el.get('fill')} at "
                f"x={el.get('x')} y={el.get('y')}")
            assert not (h > 0.5 and w == 0.0), (
                f"{path.name}: zero-width rect {el.get('fill')} at "
                f"x={el.get('x')} y={el.get('y')}")

    def test_no_two_labels_print_on_top_of_each_other(self, path):
        """The one that would have caught `ratio.svg`.

        Substantial overlap only: labels are allowed to touch, and a wrapped
        paragraph's lines share an em box by design. A quarter of the smaller
        label's area is well past touching - at that point one label is sitting
        on another, and the reader has to guess which characters belong to
        which.
        """
        boxes = _boxes(ET.parse(path).getroot())
        bad = []
        for i, (a, sa) in enumerate(boxes):
            for b, sb in boxes[i + 1:]:
                area = _overlap(a, b)
                if not area:
                    continue
                smaller = min((a[2] - a[0]) * (a[3] - a[1]),
                              (b[2] - b[0]) * (b[3] - b[1])) or 1.0
                if area / smaller > 0.25:
                    bad.append(f"{sa!r} over {sb!r}")
        assert not bad, f"{path.name}: overlapping labels: " + "; ".join(bad[:6])


def test_every_published_claim_still_has_a_figure():
    """figures/README.md names them; a rename that lands in one and not the
    other is a broken image in the docs, which is invisible until someone
    opens the page."""
    listed = set(re.findall(r"`([a-z_]+(?:\.<ladder>|\.wide)?\.svg)`",
                            (paths.FIGURES / "README.md").read_text(
                                encoding="utf-8")))
    have = {p.name for p in _figures()}
    for name in listed:
        candidates = {name, name.replace("<ladder>", "wide")}
        assert candidates & have, f"{name} is documented but not drawn"
