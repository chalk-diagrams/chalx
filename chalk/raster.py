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


def trace_measure_xy(
    shape,
    height: int,
    width: int,
    *,
    kernel: int = 11,
    pixel: float = 1.0,
    boundary: bool = True,
):
    """Average coverage from a left-to-right scan and a top-to-bottom scan.

    1D AA on one axis misses edges parallel to that axis; averaging the two
    directions matches the DiffRast plug. Also covers rows/cols that one
    scan drops when a hit starts off-frame.
    """
    # Use raw homogeneous vectors (not hijax V2) so this is safe under jit+scan.
    vx = jnp.array([[1.0], [0.0], [0.0]])
    vy = jnp.array([[0.0], [1.0], [0.0]])
    ax = trace_measure(
        shape,
        scanline_origins(height, axis="x", pixel=pixel),
        vx,
        width,
        kernel=kernel,
        pixel=pixel,
        boundary=boundary,
    )
    ay = trace_measure(
        shape,
        scanline_origins(width, axis="y", pixel=pixel),
        vy,
        height,
        kernel=kernel,
        pixel=pixel,
        boundary=boundary,
    )
    return 0.5 * (ax + ay.T)


def rasterize(
    shapes: Union[Any, Iterable[Any]],
    height: int,
    width: int,
    *,
    kernel: int = 11,
    background: float = 1.0,
    paint: Optional[Any] = None,
    axes: tuple[str, ...] = ("x", "y"),
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
        if tuple(axes) == ("x", "y") or tuple(axes) == ("y", "x"):
            α = trace_measure_xy(shape, height, width, kernel=kernel)
        elif axes == ("x",):
            α = trace_measure(
                shape,
                scanline_origins(height, axis="x"),
                scanline_direction("x"),
                width,
                kernel=kernel,
            )
        elif axes == ("y",):
            α = trace_measure(
                shape,
                scanline_origins(width, axis="y"),
                scanline_direction("y"),
                height,
                kernel=kernel,
            ).T
        else:
            raise ValueError(axes)
        img = composite(img, α, color)
    return img
