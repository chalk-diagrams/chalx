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
    line = ""
    i = 0
    while i < c.shape[0]:
        if c[i] == chalk.backend.patch.Command.MOVETO.value:
            line += f"M {v[i, 0]} {v[i, 1]}"
            i += 1
        elif c[i] == chalk.backend.patch.Command.LINETO.value:
            line += f"L {v[i, 0]} {v[i, 1]}"
            i += 1
        elif c[i] == chalk.backend.patch.Command.CURVE3.value:
            line += f"Q {v[i, 0]} {v[i, 1]} {v[i+1, 0]} {v[i+1, 1]}"
            i += 2
        elif c[i] == chalk.backend.patch.Command.CLOSEPOLY.value:
            line += "Z"
            i += 1
        elif c[i] == chalk.backend.patch.Command.SKIP.value:
            i += 1
        elif c[i] == chalk.backend.patch.Command.CURVE4.value:
            line += f"C {v[i, 0]} {v[i, 1]} {v[i+1, 0]} {v[i+1, 1]} {v[i+2, 0]} {v[i+2, 1]}"
            i += 3
    return line


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


def render_svg_patches(patches: List[Patch], animate:bool =False, time_steps=0) -> str:
    out = ""
    patches = [chalk.backend.patch.order_patches(patches, (step,)) 
               for step in range(time_steps)]
    for v in zip(*patches):
        out += f"""
    <path>
"""     
        lines = []
        css = {}
        for ind, patch, style_new in v:
            lines.append(to_svg(patch, ind))
            s = write_style(style_new)
            for k, v in s.items():
                css.setdefault(k, []).append(v)
        
        values = ";".join(lines)
        out += f"""
        <animate attributeName="d" values="{values}" dur="2s" repeatCount="indefinite"/>
        """
        for k, v in css.items():
            out += f"""
        <animate attributeName="{k}" values="{';'.join(v)}" dur="2s" repeatCount="indefinite"/>
"""
        out += """
    </path>
    """
    return out

def patches_to_file(
    patches: List[Patch], path: str, height: tx.IntLike, 
    width: tx.IntLike,
    animate: bool = False,
    time_steps=0
) -> None:
    dwg = f"""<?xml version="1.0" encoding="utf-8" ?>
<svg baseProfile="full" height="{int(height)}" version="1.1" width="{int(width)}" xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink">
    """
    dwg += render_svg_patches(patches, animate, time_steps)
    dwg += "</svg>"
    with open(path, "w") as f:
        f.write(dwg)


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
    patches_to_file(patches, path, h, w, animate=True, time_steps=shape[0])
