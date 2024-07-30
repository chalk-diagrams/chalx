from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import chalk.backend.patch
import chalk.transform as tx
from chalk.backend.patch import Patch
from chalk.shapes import (
    ArrowHead,
    Image,
    Latex,
    Path,
    Segment,
    Spacer,
    Text,
    from_pil,
)
from chalk.style import Style, StyleHolder
from chalk.transform import Affine
from chalk.types import Diagram
from chalk.visitor import ShapeVisitor

if TYPE_CHECKING:
    from chalk.core import Primitive


PyCairoContext = Any
EMPTY_STYLE = StyleHolder.empty()


def tx_to_cairo(affine: Affine) -> Any:
    import cairo

    def convert(a, b, c, d, e, f):  # type: ignore
        return cairo.Matrix(a, d, b, e, c, f)  # type: ignore

    return convert(*affine[0, 0], *affine[0, 1])  # type: ignore


def write_style(d: Dict[str, Any], ctx) -> None:
    if "facecolor" in d:
        ctx.set_source_rgba(*d["facecolor"], d.get("alpha", 1))
        ctx.fill_preserve()
    if "edgecolor" in d:
        ctx.set_source_rgb(*d["edgecolor"])
    if "linewidth" in d:
        ctx.set_line_width(d["linewidth"])


class ToCairoShape(ShapeVisitor[None]):

    def render_segment(self, seg: Segment, ctx: PyCairoContext) -> None:
        q, angle, dangle = (
            seg.q,
            tx.to_radians(seg.angle),
            tx.to_radians(seg.dangle),
        )
        end = angle + dangle

        for i in range(q.shape[0]):
            if tx.X.np.abs(dangle[i]) < 1:
                ctx.line_to(q[i, 0, 0], q[i, 1, 0])
            else:
                ctx.save()
                matrix = tx_to_cairo(seg.t[i : i + 1])
                ctx.transform(matrix)
                if dangle[i] < 0:
                    ctx.arc_negative(0.0, 0.0, 1.0, angle[i], end[i])
                else:
                    ctx.arc(0.0, 0.0, 1.0, angle[i], end[i])
                ctx.restore()

    def visit_path(
        self,
        path: Path,
        ctx: PyCairoContext = None,
        style: StyleHolder = EMPTY_STYLE,
    ) -> None:

        for loc_trail in path.loc_trails:
            p = loc_trail.location
            ctx.move_to(p[0, 0, 0], p[0, 1, 0])
            segments = loc_trail.located_segments()
            self.render_segment(segments, ctx)
            if loc_trail.trail.closed == 1:
                ctx.close_path()

    def visit_latex(
        self,
        shape: Latex,
        ctx: PyCairoContext = None,
        style: StyleHolder = EMPTY_STYLE,
    ) -> None:
        raise NotImplementedError("Latex is not implemented")

    def visit_text(
        self,
        shape: Text,
        ctx: PyCairoContext = None,
        style: StyleHolder = EMPTY_STYLE,
    ) -> None:
        ctx.select_font_face("sans-serif")
        if shape.font_size is not None:
            ctx.set_font_size(shape.font_size)
        extents = ctx.text_extents(shape.text)

        ctx.move_to(-(extents.width / 2), (extents.height / 2))
        ctx.text_path(shape.text)

    def visit_spacer(
        self,
        shape: Spacer,
        ctx: PyCairoContext = None,
        style: StyleHolder = EMPTY_STYLE,
    ) -> None:
        return

    def visit_arrowhead(
        self,
        shape: ArrowHead,
        ctx: PyCairoContext = None,
        style: StyleHolder = EMPTY_STYLE,
    ) -> None:

        assert style.output_size
        scale = 0.01 * (15 / 500) * style.output_size
        render_cairo_prims(
            shape.arrow_shape.scale(scale).get_primitives(), ctx
        )

    def visit_image(
        self,
        shape: Image,
        ctx: PyCairoContext = None,
        style: StyleHolder = EMPTY_STYLE,
    ) -> None:
        surface = from_pil(shape.im)
        ctx.set_source_surface(
            surface, -(shape.width / 2), -(shape.height / 2)
        )
        ctx.paint()


def to_cairo(patch: Patch, ctx: PyCairoContext, ind: Tuple[int, ...]) -> None:

    v, c = patch.vert[ind], patch.command[ind]
    i = 0
    while i < c.shape[0] - 1:
        if c[i] == chalk.backend.patch.Command.MOVETO.value:
            ctx.move_to(v[i, 0], v[i, 1])
            i += 1
        if c[i] == chalk.backend.patch.Command.CURVE4.value:
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
    patches: List[Patch], ctx: PyCairoContext, even_odd: bool = False
) -> None:

    import cairo
    import numpy as onp

    if even_odd:
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    # shape_renderer = ToCairoShape()

    style = Style()
    # Order the primitives
    d = {}

    for patch in patches:
        for ind, i in tx.onp.ndenumerate(onp.asarray(patch.order)):  # type: ignore
            assert i not in d, f"Order {i} assigned twice"
            d[i] = (patch, ind)

    d = tx.tree_map(onp.asarray, d)
    for j in sorted(d.keys()):
        patch, ind = d[j]
        to_cairo(patch, ctx, ind)
        style_new = patch.get_style(ind)
        write_style(style_new, ctx)
        ctx.stroke()

    # Order the primitives
    # d = {}
    # for prim in prims:
    #     prim_order: onp.ndarray = onp.asarray(prim.order)
    #     for ind in rproduct(prim_order.shape): # type: ignore
    #         n = prim_order[ind]
    #         assert n not in d, "Order assigned twice"
    #         d[prim_order[ind]] = (prim, ind)

    # for j in sorted(d.keys()):
    #     prim, ind = d[j]
    #     prim = prim.split(ind)
    #     for i in range(prim.transform.shape[0]):
    #         # apply transformation
    #         # matrix = tx_to_cairo(prim.transform[i : i + 1])
    #         # ctx.transform(matrix)
    #         # ps: StyleHolder = prim.style
    #         # if ps is None:
    #         #     ps = Style()

    #         # prim.shape.accept(shape_renderer, ctx=ctx, style=ps)

    #         # # undo transformation
    #         # matrix.invert()
    #         # ctx.transform(matrix)

    #         # style = ps
    #         # if (
    #         #     hasattr(prim.shape, "loc_trails")
    #         #     and prim.shape.loc_trails[0].trail.closed != 1
    #         # ):
    #         #     style = style.merge(Style(fill_opacity_=0))


def patches_to_file(
    patches: List[Patch],
    path: str,
    height: float,
    width: float,
    even_odd: bool = False,
) -> None:
    import cairo

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(width), int(height))
    ctx = cairo.Context(surface)
    render_cairo_patches(patches, ctx, even_odd)
    surface.write_to_png(path)


def render(
    self: Diagram, path: str, height: int = 128, width: Optional[int] = None
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

    patches, h, w = self.layout(height, width)
    patches_to_file(patches, path, h, w)  # type: ignore
