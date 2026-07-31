"""Cairo dest compositing should match Porter-Duff over on white."""

from __future__ import annotations

import os
import tempfile

import numpy as onp
from PIL import Image

from chalk import rectangle
from chalk.style import to_color


def _stacked():
    frame = (
        rectangle(64, 64)
        .line_width(0)
        .fill_opacity(0)
        .translate(32, 32)
    )
    green = (
        rectangle(48, 28)
        .line_width(0)
        .fill_color(to_color("green"))
        .fill_opacity(0.55)
        .translate(32, 40)
    )
    brown = (
        rectangle(10, 18)
        .line_width(0)
        .fill_color(onp.array([0.35, 0.18, 0.08]))
        .fill_opacity(0.95)
        .translate(32, 44)
    )
    return (green + brown).with_envelope(frame), (brown + green).with_envelope(frame)


def test_cairo_paints_small_on_top():
    front, back = _stacked()
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "front.png")
        b = os.path.join(td, "back.png")
        front.render(a, height=64, width=64)
        back.render(b, height=64, width=64)
        pa = onp.asarray(Image.open(a).convert("RGB"), dtype="float64") / 255.0
        pb = onp.asarray(Image.open(b).convert("RGB"), dtype="float64") / 255.0
    cx, cy = 32, 44
    # Small brown last → pixel closer to brown than to green.
    brown = onp.array([0.35, 0.18, 0.08])
    green = onp.asarray(to_color("green"))
    assert onp.linalg.norm(pa[cy, cx] - brown) < onp.linalg.norm(pa[cy, cx] - green)
    # Reversed draw order covers brown with green.
    assert onp.linalg.norm(pb[cy, cx] - green) < onp.linalg.norm(pb[cy, cx] - brown)


def test_cairo_dest_is_opaque_white_outside():
    d, _ = _stacked()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "out.png")
        d.render(path, height=64, width=64)
        im = Image.open(path).convert("RGB")
        px = onp.asarray(im)
    assert tuple(px[2, 2]) == (255, 255, 255)
