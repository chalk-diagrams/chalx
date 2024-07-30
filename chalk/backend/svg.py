from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import svgwrite
from svgwrite import Drawing
from svgwrite.base import BaseElement
from svgwrite.shapes import Rect

import chalk.backend.patch
from chalk import transform as tx
from chalk.backend.patch import Patch
from chalk.shapes import ArrowHead, Image, Latex, Path, Segment, Spacer, Text
from chalk.style import StyleHolder
from chalk.types import Diagram
from chalk.visitor import ShapeVisitor

if TYPE_CHECKING:
    from chalk.core import Primitive


EMPTY_STYLE = StyleHolder.empty()


def tx_to_svg(affine: tx.Affine) -> str:
    def convert(
        a: tx.Floating,
        b: tx.Floating,
        c: tx.Floating,
        d: tx.Floating,
        e: tx.Floating,
        f: tx.Floating,
    ) -> str:
        return f"matrix({a}, {d}, {b}, {e}, {c}, {f})"

    return convert(*affine[0, 0], *affine[0, 1])


class Raw(Rect):  # type: ignore
    """Shape class.

    A fake SVG node for importing latex.
    """

    def __init__(self, st: str):
        ET.register_namespace("", "http://www.w3.org/2000/svg")
        self.xml = ET.fromstring(st)

    def get_xml(self) -> ET.Element:
        return self.xml


class ToSVGShape(ShapeVisitor[BaseElement]):
    def __init__(self, dwg: Drawing):
        self.dwg = dwg

    # def render_segment(self, seg: Segment) -> Iterator[str]:
    #     # https://www.w3.org/TR/SVG/implnote.html#ArcConversionCenterToEndpoint
    #     dangle = seg.dangle
    #     f_A = abs(dangle) > 180
    #     det: float = tx.X.np.linalg.det(seg.t)  # type: ignore
    #     f_S = det * dangle > 0
    #     r_x, r_y, rot, q = seg.r_x, seg.r_y, seg.rot, seg.q

    #     for i in range(q.shape[0]):
    #         if tx.X.np.abs(dangle[i]) > 1:
    #             yield f"""
    #             A {r_x[i]} {r_y[i]} {rot[i]} {int(f_A[i])} {int(f_S[i])}
    #               {q[i, 0, 0]} {q[i, 1, 0]}"""
    #         else:
    #             yield f"L {q[i, 0, 0]} {q[i, 1, 0]}"

    # def visit_path(
    #     self, path: Path, style: StyleHolder = EMPTY_STYLE
    # ) -> BaseElement:
    #     extra_style = ""
    #     if not path.loc_trails[0].trail.closed:
    #         extra_style = "fill:none;"
    #     line = self.dwg.path(
    #         style="vector-effect: non-scaling-stroke;" + extra_style,
    #     )
    #     for loc_trail in path.loc_trails:
    #         p = loc_trail.location
    #         line.push(f"M {p[0, 0, 0]} {p[0, 1, 0]}")
    #         segments = loc_trail.located_segments()
    #         for seg in self.render_segment(segments):
    #             line.push(seg)
    #         if loc_trail.trail.closed:
    #             line.push("Z")
    #     return line

    def visit_latex(
        self, shape: Latex, style: StyleHolder = EMPTY_STYLE
    ) -> BaseElement:
        dx, dy = -shape.width / 2, -shape.height / 2
        g = self.dwg.g(transform=f"scale(0.05) translate({dx} {dy})")
        g.add(Raw(shape.content))
        return g

    def visit_text(
        self, shape: Text, style: StyleHolder = EMPTY_STYLE
    ) -> BaseElement:
        dx = -(shape.get_bounding_box().width / 2)
        return self.dwg.text(
            shape.text,
            transform=f"translate({dx[0]}, 0)",
            style=f"""text-align:center; text-anchor:middle; dominant-baseline:middle;
                      font-family:sans-serif; font-weight: bold;
                      font-size:{shape.font_size}px;
                      vector-effect: non-scaling-stroke;""",
        )

        raise NotImplementedError("Text is not implemented")

    def visit_spacer(
        self, shape: Spacer, style: StyleHolder = EMPTY_STYLE
    ) -> BaseElement:
        return self.dwg.g()

    def visit_arrowhead(
        self, shape: ArrowHead, style: StyleHolder = EMPTY_STYLE
    ) -> BaseElement:
        assert style.output_size

        scale = 0.01 * (15 / 500) * style.output_size
        return render_svg_prims(
            shape.arrow_shape.scale(scale).get_primitives(), self.dwg, style
        )

    def visit_image(
        self, shape: Image, style: StyleHolder = EMPTY_STYLE
    ) -> BaseElement:
        dx = -shape.width / 2
        dy = -shape.height / 2
        return self.dwg.image(
            href=shape.url_path, transform=f"translate({dx}, {dy})"
        )


def to_svg(patch: Patch, dwg: Drawing, ind: Tuple[int, ...]) -> BaseElement:
    line = dwg.path(style="vector-effect: non-scaling-stroke;")
    v, c = patch.vert[ind], patch.command[ind]
    i = 0
    while i < c.shape[0] - 1:
        if c[i] == chalk.backend.patch.Command.MOVETO.value:
            line.push(f"M {v[i, 0]} {v[i, 1]}")
            i += 1
        if c[i] == chalk.backend.patch.Command.CURVE4.value:
            line.push(
                f"C {v[i, 0]} {v[i, 1]} {v[i+1, 0]} {v[i+1, 1]} {v[i+2, 0]} {v[i+2, 1]}"
            )
            i += 3
    return line


def write_style(d: Dict[str, Any]) -> str:
    out = ""
    up = {
        "facecolor": "fill",
        "edgecolor": "stroke",
        "linewidth": "stroke-width",
        "alpha": "fill-opacity",
    }
    for k, v in d.items():
        if "color" in k:
            v = v * 256
            v = f"rgb({v[0]} {v[1]} {v[2]})"
        out += f"{up[k]}: {v};"
    return out


def render_svg_patches(
    patches: List[Patch], dwg: Drawing, style: StyleHolder
) -> None:
    outer = dwg.g(style="fill:white;")
    # Arrow marker
    marker = dwg.marker(
        id="arrow", refX=5.0, refY=1.7, size=(5, 3.5), orient="auto"
    )
    marker.add(dwg.polygon([(0, 0), (5, 1.75), (0, 3.5)]))
    dwg.defs.add(marker)

    dwg.add(outer)
    shape_renderer = ToSVGShape(dwg)

    # Order the primitives
    d = {}

    import numpy as onp

    patches = tx.tree_map(onp.asarray, patches)

    for patch in patches:
        # assert prim.order.shape == patch.command.shape[:-1]
        for ind, i in onp.ndenumerate(onp.asarray(patch.order)):  # type: ignore
            assert i not in d, f"Order {i} assigned twice"
            d[i] = (patch, ind)

    for j in sorted(d.keys()):
        patch, ind = d[j]
        style_new = patch.get_style(ind)
        inner = to_svg(patch, dwg, ind)
        g = dwg.g(style=write_style(style_new))
        g.add(inner)
        dwg.add(g)


def patches_to_file(
    patches: List[Patch], path: str, height: float, width: float
) -> None:

    dwg = svgwrite.Drawing(path, size=(int(width), int(height)))
    style = StyleHolder.root(output_size=height)
    render_svg_patches(patches, dwg, style)
    dwg.save()


def render(
    self: Diagram,
    path: str,
    height: int = 128,
    width: Optional[int] = None,
    draw_height: Optional[int] = None,
) -> None:
    """Render the diagram to an SVG file.

    Args:
        self (Diagram): Given ``Diagram`` instance.
        path (str): Path of the .svg file.
        height (int, optional): Height of the rendered image.
                                Defaults to 128.
        width (Optional[int], optional): Width of the rendered image.
                                         Defaults to None.
        draw_height (Optional[int], optional): Override the height for
                                               line width.

    """

    patches, h, w = self.layout(height, width)
    patches_to_file(patches, path, h, w)  # type: ignore

    # pad = 0.05
    # envelope = self.get_envelope()

    # # infer width to preserve aspect ratio
    # assert envelope is not None
    # width = width or int(height * envelope.width / envelope.height)
    # # determine scale to fit the largest axis in the target frame size
    # if envelope.width - width <= envelope.height - height:
    #     α = height / ((1 + pad) * envelope.height)
    # else:
    #     α = width / ((1 + pad) * envelope.width)
    # s = self.center_xy().pad(1 + pad).scale(α)
    # e = s.get_envelope()
    # assert e is not None
    # s = s.translate(e(-unit_x), e(-unit_y))
    # if draw_height is None:
    #     draw_height = max(height, width)
    # style = StyleHolder.root(output_size=draw_height)
    # outer.add(to_svg(s, dwg, style))
    # dwg.save()
