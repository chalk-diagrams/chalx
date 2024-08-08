from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import chalk.backend.patch
import chalk.transform as tx
from chalk.backend.patch import Patch, order_patches
from chalk.types import Diagram

PyCairoContext = Any


def write_style(d: Dict[str, Any], ctx: PyCairoContext) -> None:
    if "facecolor" in d:
        ctx.set_source_rgba(*d["facecolor"], d.get("alpha", 1))
        ctx.fill_preserve()
    if "edgecolor" in d:
        ctx.set_source_rgb(*d["edgecolor"])
    if "linewidth" in d:
        ctx.set_line_width(d["linewidth"])


def to_cairo(patch: Patch, ctx: PyCairoContext, ind: Tuple[int, ...]) -> None:
    # Render Curves
    v, c = patch.vert[ind], patch.command[ind]
    i = 0
    while i < c.shape[0] - 1:
        if c[i] == chalk.backend.patch.Command.MOVETO.value:
            ctx.move_to(v[i, 0], v[i, 1])
            i += 1
        elif c[i] == chalk.backend.patch.Command.LINETO.value:
            ctx.line_to(v[i, 0], v[i, 1])
            i += 1
        elif c[i] == chalk.backend.patch.Command.CURVE3.value:
            c1 = v[i - 1] + 2 / 3 * (v[i] - v[i - 1])
            c2 = v[i + 1] + 2 / 3 * (v[i] - v[i + 1])
            ctx.curve_to(
                c1[0],
                c1[0],
                c2[0],
                c2[1],
                v[i + 1, 0],
                v[i + 1, 1],
            )
            i += 2
        elif c[i] == chalk.backend.patch.Command.CLOSEPOLY.value:
            # ctx.stroke()
            i += 1
        elif c[i] == chalk.backend.patch.Command.CURVE4.value:
            ctx.curve_to(
                v[i, 0],
                v[i, 1],
                v[i + 1, 0],
                v[i + 1, 1],
                v[i + 2, 0],
                v[i + 2, 1],
            )
            i += 3


def render_cairo_patches(
    patches: List[Patch], ctx: PyCairoContext
) -> None:
    # Order the primitives
    for ind, patch, style in order_patches(patches):
        to_cairo(patch, ctx, ind)
        write_style(style, ctx)
        ctx.stroke()


def patches_to_file(
    patches: List[Patch],
    path: str,
    height: tx.IntLike,
    width: tx.IntLike,
) -> None:
    import cairo

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(width), int(height))
    ctx = cairo.Context(surface)
    render_cairo_patches(patches, ctx)
    surface.write_to_png(path)


def render(
    self: Diagram, path: str, height: int = 128, width: Optional[int] = None,
    draw_height: Optional[int]=None
) -> None:
    """Render the diagram to a PNG file.

    Args:
        self (Diagram): Given ``Diagram`` instance.
        path (str): Path of the .png file.
        height (int, optional): Height of the rendered image.
                                Defaults to 128.
        width (Optional[int], optional): Width of the rendered image.
                                         Defaults to None.
    """

    patches, h, w = self.layout(height, width, draw_height)
    patches_to_file(patches, path, h, w)  # type: ignore
