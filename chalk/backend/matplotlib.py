from __future__ import annotations

from typing import List, Optional

import matplotlib.axes
import matplotlib.collections
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.path import Path

import chalk.transform as tx
from chalk.backend.patch import Patch, order_patches
from chalk.style import StyleHolder
from chalk.types import Diagram

EMPTY_STYLE = StyleHolder.empty()


def render_patches(patches: List[Patch], ax: matplotlib.axes.Axes) -> None:
    ps = []
    for ind, patch, style_new in order_patches(patches):
        ps.append(
            matplotlib.patches.PathPatch(
                Path(patch.vert[ind] * [1, -1], patch.command[ind]),
                **style_new,
            )
        )

    collection = matplotlib.collections.PatchCollection(
        ps, match_original=True
    )
    ax.add_collection(collection)


def patches_to_file(
    patches: List[Patch], path: str, height: tx.IntLike, width: tx.IntLike
) -> None:
    fig, ax = plt.subplots()
    render_patches(patches, ax)
    ax.set_xlim((0, width)) # type: ignore
    ax.set_ylim((-height, 0)) # type: ignore
    ax.set_aspect("equal")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_axis_off()
    fig.savefig(path, dpi=400)


def render(
    self: Diagram,
    path: str,
    height: int = 128,
    width: Optional[int] = None,
    draw_height: Optional[int] = None,
) -> None:
    prims, h, w = self.layout(height, width, draw_height)
    patches_to_file(prims, path, h, w)  # type: ignore
