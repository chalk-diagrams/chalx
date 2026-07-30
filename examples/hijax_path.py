"""Mock: opaque hijax Path vs pytree-batched chalk Path.

Run with::

    python examples/hijax_path.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from chalk.hijax_path import (
    PathSpec,
    beside,
    bounds,
    circle,
    concat,
    from_points,
    rotate,
    square,
    to_points,
    translate,
)


def draw(paths, path, color="C0", lw=1.5):
    pts = jax.device_get(to_points(path))
    if pts.ndim == 2:
        pts = pts[None, ...]
    for poly in pts:
        closed = bool(jax.device_get(path.closed).reshape(-1)[0]) if pts.shape[0] == 1 else True
        xs, ys = poly[:, 0], poly[:, 1]
        if closed:
            xs = list(xs) + [xs[0]]
            ys = list(ys) + [ys[0]]
        paths.plot(xs, ys, color=color, lw=lw)


def main():
    p = circle(1.0)
    print("eager circle:", jax.typeof(p))
    print("vertices shape:", to_points(p).shape)

    # jit: path is one opaque value, not a spray of array leaves
    def move(r):
        return to_points(translate(circle(r), 1.0, 0.5))

    print("\n--- jit jaxpr (path is a single 'path[32]' value) ---")
    print(jax.jit(lambda r: translate(circle(r), 1.0, 0.5)).trace(1.0).jaxpr)

    print("\njit(move)(1.25)[0] =", jax.jit(move)(1.25)[0])

    # vmap over radius produces a batched path
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

    # combinator built from primitives (works under jit)
    logo = jax.jit(lambda: beside(circle(1.0), rotate(square(1.4), 15.0)))()
    print("\nbeside(circle, rotated square):", jax.typeof(logo))
    print("bounds:", bounds(logo))

    # autodiff: tangent of a path is a vertex array
    def loss(pts):
        path = from_points(pts, closed=True)
        verts = to_points(translate(path, 1.0, 0.0))
        return jnp.sum(verts**2)

    pts0 = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=jnp.float32)
    print("\ngrad(loss)(triangle) =\n", jax.grad(loss)(pts0))

    # scan over a stack of paths
    def step(total, path):
        lo, hi = bounds(path)
        return total + jnp.sum(hi - lo), ()

    total, _ = jax.lax.scan(step, 0.0, ps, length=4)
    print("scan sum of bbox sizes:", float(total))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    draw(ax, circle(1.0), "C0")
    draw(ax, translate(square(1.2), 2.5, 0.0), "C1")
    draw(ax, concat(circle(0.4), translate(circle(0.4), 3.5, 1.2)), "C2")
    ax.set_aspect("equal")
    ax.set_title("hijax Path mock")
    fig.tight_layout()
    out = "examples/output/hijax_path.png"
    fig.savefig(out, dpi=120)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
