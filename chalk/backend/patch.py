from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from matplotlib.text import TextPath

import chalk.transform as tx
from chalk.path import Path
from chalk.style import StyleHolder

if TYPE_CHECKING:
    from chalk.core import Primitive


class Command(Enum):
    MOVETO = 1
    LINETO = 2
    CURVE3 = 3
    CURVE4 = 4
    CLOSEPOLY = 79
    SKIP = 0


@partial(tx.vectorize, signature="(3,3),(2)->(a,3,1)")
def segment_to_curve(transform: tx.Affine, angles: tx.Angles) -> tx.V2_tC:
    angle = angles[..., 0]
    end = angle + angles[..., 1]
    path = tx.arc_to_bezier(angle, end)
    return transform[..., None, :, :] @ path  # type: ignore


@partial(tx.np.vectorize, signature="(3,1),(a,c,3,1),(3,3)->(b,3,1),(b)")
def close(p: tx.P2_t, vert: tx.V2_t, trans: tx.Affine) -> Tuple[tx.V2_tC, tx.IntLikeC]:
    # vert = vert.reshape(*vert.shape[:-4], vert.shape[-4] * vert.shape[-3],
    #                 vert.shape[-2], vert.shape[-1])
    vert = vert.reshape(-1, 3, 1)
    command = tx.np.full(vert.shape[0] + 1, Command.CURVE4.value)
    command = tx.index_update(command, (Ellipsis, 0), Command.MOVETO.value)
    vert = tx.np.concatenate([p[None], vert], axis=0)
    vert = trans @ vert
    return vert, command


def order_patches(
    patches: List[Patch], time: Tuple[int, ...] = ()
) -> List[Tuple[Tuple[int, ...], Patch, Dict[str, Any]]]:
    import numpy as onp

    if tx.JAX_MODE:
        patches = tx.tree_map(onp.asarray, patches)

    d = {}
    for patch in patches:
        for ind, i in tx.onp.ndenumerate(patch.order[time]):  # type: ignore
            assert i not in d, f"Order {i} assigned twice"
            d[i] = (patch, time + ind)
    return [(d[k][1], d[k][0], d[k][0].get_style(d[k][1])) for k in sorted(d.keys())]


@dataclass
class Patch:
    vert: tx.Array
    command: tx.Array
    style: Dict[str, Any]
    order: tx.Array
    height: tx.IntLike
    closed: tx.BoolLike

    def get_style(self, ind: Tuple[int, ...]) -> Dict[str, Any]:
        d = {k: v[ind] for k, v in self.style.items()}
        normalizer = self.height * (15 / 500)
        if "linewidth" in d:
            d["linewidth"] = d["linewidth"] * normalizer
        else:
            d["linewidth"] = 0.1 * normalizer
        if not self.closed:
            d["alpha"] = 0
        return d

    @staticmethod
    def from_path(
        path: Path,
        transform: tx.Affine,
        style: Dict[str, Any],
        order: tx.Array,
        height: tx.IntLike,
    ) -> Patch:
        np = tx.np
        vert = np.empty((0, 3, 1))
        command = np.empty((0))
        closed = True
        for loc_trail in path.loc_trails:
            p = loc_trail.location
            segments = loc_trail.located_segments()
            vert = segment_to_curve(segments.transform, segments.angles)
            if path.scale_invariant is not None:
                scale = height / 20
                transform = tx.remove_scale(transform) @ tx.scale(tx.V2(scale, scale))

            vert, command = close(p, vert, transform)
            closed = loc_trail.trail.closed.all()

            # Closing
            extra = tx.np.zeros(vert.shape)
            vert = np.concatenate([vert, extra[..., -1:, :, :]], axis=-3)
            extra = tx.np.full(
                command.shape,
                np.where(closed, Command.CLOSEPOLY.value, Command.SKIP.value),
            )
            command = np.concatenate([command, extra[..., -1:]], -1)

        vert = vert[..., :2, 0]
        # Text rendering
        if path.text is not None:
            text_path = TextPath((0, 0), path.text.to_str(), size=1, usetex=True)
            command = np.concatenate([command, text_path.codes], -1)
            v = text_path.vertices
            # Center
            t1 = np.max(v[..., 0]) / 2
            t2 = np.max(v[..., 1]) / 2
            new_verts: tx.Array = (
                transform
                @ tx.translation(tx.V2(-t1, -t2))
                @ tx.P2(v[..., 0], v[..., 1])  # type: ignore
            )
            vert = np.concatenate([vert, new_verts[..., :2, 0]], -2)

        return Patch(vert, command, style, order, height, closed)


@tx.jit  # type: ignore
def patch_from_prim(prim: Primitive, style: StyleHolder, height: tx.IntLike) -> Patch:
    size = prim.size()
    style = prim.style if prim.style is not None else style
    assert isinstance(prim.prim_shape, Path)
    assert prim.order is not None
    in_style = (
        style.to_mpl()
    )  # tx.multi_vmap(style.to_mpl.__func__, len(size))(style),  # type: ignore
    patch = Patch.from_path(
        prim.prim_shape,
        prim.transform,
        in_style,
        prim.order,
        height,
    )
    assert size == patch.command.shape[:-1]
    return patch
