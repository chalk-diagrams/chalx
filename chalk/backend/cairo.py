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
    while i < c.shape[0]:
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
            ctx.close_path()
            i += 1
        elif c[i] == chalk.backend.patch.Command.SKIP.value:
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
    patches: List[Patch], ctx: PyCairoContext, time: Tuple[int, ...]
) -> None:
    # Order the primitives
    for ind, patch, style in order_patches(patches, time):
        to_cairo(patch, ctx, ind)
        write_style(style, ctx)
        ctx.stroke()


def patches_to_file(
    patches: List[Patch],
    path: str,
    height: tx.IntLike,
    width: tx.IntLike,
    time: Tuple[int, ...] = (),
) -> None:
    import cairo

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(width), int(height))
    ctx = cairo.Context(surface)
    render_cairo_patches(patches, ctx, time)
    surface.write_to_png(path)


def render(
    self: Diagram,
    path: str,
    height: int = 128,
    width: Optional[int] = None,
    draw_height: Optional[int] = None,
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


def animate(
    self: Diagram,
    path: str,
    height: int = 128,
    width: Optional[int] = None,
    draw_height: Optional[int] = None,
) -> None:
    shape = self.shape

    assert len(shape) == 1, f"Must be one time dimension {shape}"

    patches, h, w = self.layout(height, width, draw_height)
    h = tx.np.max(h)
    w = tx.np.max(w)
    path_frame = "/tmp/frame-{:d}.png"
    import imageio

    with imageio.get_writer(path, fps=20, loop=0) as writer:
        for i in range(shape[0]):
            path = path_frame.format(i)
            patches_to_file(patches, path, h, w, (i,))
            from PIL import Image

            png = Image.open(path).convert('RGBA')
            background = Image.new('RGBA', png.size, (255,255,255))

            alpha_composite = Image.alpha_composite(background, png)
            alpha_composite.save(path, 'PNG')
            
            image = imageio.imread(path)
            
            writer.append_data(image) # type: ignore
