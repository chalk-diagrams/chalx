"""Thin raster glue: trace_measure + composite."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

import jax.numpy as jnp

from chalk.measure import trace_measure
from chalk.style import composite
from chalk.transform import V2


def scanline_origins(n_lines: int, *, axis: str = "x", pixel: float = 1.0):
    """Origins for ``n_lines`` scan rays along ``axis`` ('x' or 'y')."""
    ts = (jnp.arange(n_lines) + 0.5) * pixel
    z = jnp.zeros((n_lines,))
    o = jnp.ones((n_lines,))
    if axis == "x":
        return jnp.stack([z, ts, o], axis=-1)[..., None]
    if axis == "y":
        return jnp.stack([ts, z, o], axis=-1)[..., None]
    raise ValueError(axis)


def scanline_direction(axis: str = "x"):
    if axis == "x":
        return V2(1.0, 0.0)
    if axis == "y":
        return V2(0.0, 1.0)
    raise ValueError(axis)


def rasterize(
    shapes: Union[Any, Iterable[Any]],
    height: int,
    width: int,
    *,
    kernel: int = 11,
    background: float = 1.0,
    paint: Optional[Any] = None,
    axes: tuple[str, ...] = ("x",),
):
    """Rasterize diagram(s) via ``trace_measure`` + ``composite``."""
    if not isinstance(shapes, (list, tuple)):
        shapes = [shapes]
    img = jnp.ones((height, width, 3)) * background
    for shape in shapes:
        color = paint
        if color is None:
            color = getattr(shape, "get_style", lambda: None)()
        if color is None:
            from chalk.style import to_color

            color = to_color("black")
        for axis in axes:
            if axis == "x":
                p, n = scanline_origins(height, axis="x"), width
                α = trace_measure(shape, p, scanline_direction("x"), n, kernel=kernel)
            else:
                p, n = scanline_origins(width, axis="y"), height
                α = trace_measure(shape, p, scanline_direction("y"), n, kernel=kernel).T
            img = composite(img, α, color)
    return img
