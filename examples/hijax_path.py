"""Mock: opaque hijax Path vs pytree-batched chalk Path.

Run with::

    python examples/hijax_path.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from chalk.hijax_path import (
    PathSpec,
    beside,
    bounds,
    circle,
    from_points,
    rotate,
    square,
    to_points,
    translate,
)


def _polys(path, closed=True):
    pts = jax.device_get(to_points(path))
    if pts.ndim == 2:
        return [pts]
    return list(pts)


def add_path(ax, path, facecolor, edgecolor="black", lw=1.2, alpha=0.85):
    for poly in _polys(path):
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=lw,
                alpha=alpha,
            )
        )


def main():
    p = circle(1.0)
    print("eager circle:", jax.typeof(p))
    print("vertices shape:", to_points(p).shape)

    print("\n--- jit jaxpr (path is a single 'path[32]' value) ---")
    print(jax.jit(lambda r: translate(circle(r), 1.0, 0.5)).trace(1.0).jaxpr)

    rs = jnp.arange(1.0, 5.0)
    ps = jax.vmap(circle, out_axes=PathSpec())(rs)
    print("\nvmapped circles:", jax.typeof(ps), "verts", to_points(ps).shape)

    shifted = jax.vmap(
        lambda path: translate(path, 0.5, 0.0),
        in_axes=PathSpec(),
        out_axes=PathSpec(),
        axis_size=4,
    )(ps)
    print("vmapped translate:", jax.typeof(shifted))

    logo = jax.jit(lambda: beside(circle(1.0), rotate(square(1.4), 15.0)))()
    print("\nbeside(circle, rotated square):", jax.typeof(logo), "bounds", bounds(logo))

    def loss(pts):
        path = from_points(pts, closed=True)
        verts = to_points(translate(path, 1.0, 0.0))
        return jnp.sum(verts**2)

    pts0 = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=jnp.float32)
    print("\ngrad(loss)(triangle) =\n", jax.grad(loss)(pts0))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))

    ax = axes[0]
    add_path(ax, circle(1.0), "#4C78A8")
    add_path(ax, translate(square(1.1), 2.6, 0.0), "#F58518")
    ax.set_title("circle + translated square")

    ax = axes[1]
    colors = ["#4C78A8", "#54A24B", "#EECA3B", "#E45756"]
    verts = jax.device_get(to_points(ps))
    for i, poly in enumerate(verts):
        ax.add_patch(
            Polygon(
                poly,
                closed=True,
                facecolor="none",
                edgecolor=colors[i],
                linewidth=1.8,
            )
        )
    ax.set_title("vmap(circle) → path[4;32]")

    ax = axes[2]
    add_path(ax, circle(1.0), "#72B7B2", alpha=0.9)
    add_path(ax, translate(rotate(square(1.4), 15.0), 2.4, 0.0), "#B279A2", alpha=0.9)
    ax.set_title("beside-style layout")

    for ax in axes:
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.margins(0.15)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#f7f7f7")

    fig.suptitle("hijax Path mock", y=1.02)
    fig.tight_layout()
    out = "examples/output/hijax_path.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
