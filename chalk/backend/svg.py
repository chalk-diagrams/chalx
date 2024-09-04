from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import chalk.backend.patch
import chalk.transform as tx
from chalk.backend.patch import Patch
from chalk.types import Diagram


def to_svg(patch: Patch, ind: Tuple[int, ...]) -> str:
    v, c = patch.vert[ind], patch.command[ind]
    if v.shape[0] == 0:
        return "<g></g>"
    parts = []
    i = 0
    while i < c.shape[0]:
        if c[i] == chalk.backend.patch.Command.MOVETO.value:
            parts.append(f"M {v[i, 0]:.2f} {v[i, 1]:.2f}")
            i += 1
        elif c[i] == chalk.backend.patch.Command.LINETO.value:
            parts.append(f"L {v[i, 0]} {v[i, 1]}")
            i += 1
        elif c[i] == chalk.backend.patch.Command.CURVE3.value:
            parts.append(f"Q {v[i, 0]} {v[i, 1]} {v[i+1, 0]} {v[i+1, 1]}")
            i += 2
        elif c[i] == chalk.backend.patch.Command.CLOSEPOLY.value:
            parts.append("Z")
            i += 1
        elif c[i] == chalk.backend.patch.Command.SKIP.value:
            i += 1
        elif c[i] == chalk.backend.patch.Command.CURVE4.value:
            parts.append(
                f"C {v[i, 0]:.2f} {v[i, 1]:.2f} {v[i+1, 0]:.2f} {v[i+1, 1]:.2f} {v[i+2, 0]:.2f} {v[i+2, 1]:.2f}"
            )
            i += 3
    return " ".join(parts)


def write_style(d: Dict[str, Any]) -> Dict[str, str]:
    out = {}
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
        out[up[k]] = str(v)
    return out


def render_svg_patches(
    patches: List[Patch], animate: bool = False, time_steps: int = 0
) -> str:
    if animate:
        out = ""
        new_patches = [
            chalk.backend.patch.order_patches(patches, (step,))
            for step in range(time_steps)
        ]
        for v in zip(*new_patches):
            out += "\n\n <path>\n"
            lines = []
            css = {}

            for ind, patch, style_new in v:
                lines.append(to_svg(patch, ind))
                s = write_style(style_new)
                for k, v in s.items():
                    css.setdefault(k, []).append(v)
            s = set(lines)
            if len(s) == 1:
                out += f"""
                <set attributeName="d" to="{list(s)[0]}"/>
                """
            else:
                values = ";".join(lines)
                out += f"""
                <animate attributeName="d" values="{values}" dur="2s" repeatCount="indefinite"/>
                """
            for k, v in css.items():
                s = set(v)
                if len(s) == 1:
                    out += f"""<set attributeName="{k}" to="{list(s)[0]}"/>"""

                else:
                    out += f"""
                <animate attributeName="{k}" values="{';'.join(v)}" dur="2s" repeatCount="indefinite"/>
        """
            out += "</path>\n\n"
        return out
    else:
        out = ""
        for ind, patch, style_new in chalk.backend.patch.order_patches(patches):
            inner = to_svg(patch, ind)
            style_t = ";".join([f"{k}:{v}" for k, v in write_style(style_new).items()])
            out += f"""
            <g style="{style_t}">
                <path d="{inner}" />
            </g>"""
        return out


def patches_to_file(
    patches: List[Patch],
    path: str,
    height: tx.IntLike,
    width: tx.IntLike,
    animate: bool = False,
    time_steps: int = 0,
) -> None:
    with open(path, "w") as f:
        f.write(f"""<?xml version="1.0" encoding="utf-8" ?>
<svg baseProfile="full" height="{int(height)}" version="1.1" width="{int(width)}" xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink">
    """)

        out = render_svg_patches(patches, animate, time_steps)
        f.write(out)
        f.write("</svg>")


def render(
    self: Diagram,
    path: str,
    height: int = 128,
    width: Optional[int] = None,
    draw_height: Optional[int] = None,
) -> None:
    """Render the diagram to an SVG file.

    Args:
    ----
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
    patches, h, w = self._layout(height, width, draw_height)
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

    patches, h, w = self._layout(height, width, draw_height)
    h = tx.np.max(h)
    w = tx.np.max(w)
    patches_to_file(patches, path, h, w, animate=True, time_steps=shape[0])


__all__ = []
