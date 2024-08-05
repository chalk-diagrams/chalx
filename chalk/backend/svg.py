from __future__ import annotations

from typing import Any, Dict, List, Optional

import svgwrite
from svgwrite import Drawing
from svgwrite.base import BaseElement

import chalk.backend.patch
from chalk.backend.patch import Patch
from chalk.types import Diagram


def to_svg(patch: Patch, dwg: Drawing, ind: int) -> BaseElement:
    line = dwg.path(style="vector-effect: non-scaling-stroke;")
    v, c = patch.vert[ind], patch.command[ind]
    if v.shape[0] == 0:
        return dwg.g()
    i = 0

    while i < c.shape[0] - 1:
        if c[i] == chalk.backend.patch.Command.MOVETO.value:
            line.push(f"M {v[i, 0]} {v[i, 1]}")
            i += 1
        elif c[i] == chalk.backend.patch.Command.LINETO.value:
            line.push(f"L {v[i, 0]} {v[i, 1]}")
            i += 1
        elif c[i] == chalk.backend.patch.Command.CURVE3.value:
            line.push(f"Q {v[i, 0]} {v[i, 1]} {v[i+1, 0]} {v[i+1, 1]}")
            i += 2
        elif c[i] == chalk.backend.patch.Command.CLOSEPOLY.value:
            line.push("Z")
            i += 1
        elif c[i] == chalk.backend.patch.Command.CURVE4.value:
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


def render_svg_patches(patches: List[Patch], dwg: Drawing, height) -> None:
    for ind, patch, style_new in chalk.backend.patch.order_patches(
        patches, height
    ):
        inner = to_svg(patch, dwg, ind)
        g = dwg.g(style=write_style(style_new))
        g.add(inner)
        dwg.add(g)


def patches_to_file(
    patches: List[Patch], path: str, height: float, width: float, draw_height
) -> None:
    dwg = svgwrite.Drawing(path, size=(int(width), int(height)))
    render_svg_patches(
        patches, dwg, draw_height if draw_height is not None else height
    )
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
    assert self.size() == (), "Must be a size () diagram"
    patches, h, w = self.layout(height, width)
    patches_to_file(patches, path, h, w, draw_height)  # type: ignore
