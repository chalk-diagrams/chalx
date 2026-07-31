"""Fit 50 ellipses to Sasha's portrait from https://rush-nlp.com/."""

from __future__ import annotations

import random
import urllib.request

import jax
import jax.numpy as jnp
import numpy as onp
from PIL import Image

from chalk import circle
from chalk.measure import trace_measure
from chalk.raster import scanline_direction, scanline_origins
from chalk.style import composite
from chalk.trace import transform_trace

H = W = 80
KERNEL = 11
N = 50
STEPS = 500
LR = 0.04
PHOTO_URL = "https://avatars0.githubusercontent.com/u/35882?s=460&v=4"

tr0 = circle(1.0).get_trace()
_p = scanline_origins(H, axis="x")
_v = scanline_direction("x")


def _ellipse_affine(lx, ly, rx, ry, rot):
    ca, sa = jnp.cos(-rot), jnp.sin(-rot)
    eye = jnp.eye(3)
    R = eye.at[0, 0].set(ca).at[0, 1].set(-sa).at[1, 0].set(sa).at[1, 1].set(ca)
    S = eye.at[0, 0].set(rx).at[1, 1].set(ry)
    T = eye.at[0, 2].set(lx).at[1, 2].set(ly)
    return T @ S @ R


def load_goal():
    path = "/tmp/rush_avatar.jpg"
    urllib.request.urlretrieve(PHOTO_URL, path)
    im = Image.open(path).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    return jnp.asarray(onp.asarray(im).astype("float64") / 255.0)


def reduce_color(y):
    return jnp.floor(y * 40.0) / 40.0


def render(params):
    loc, radii, color, rot = params
    As = jax.vmap(_ellipse_affine)(
        loc[:, 0], loc[:, 1], radii[:, 0], radii[:, 1], rot[:, 0]
    )
    paints = jax.nn.sigmoid(color)

    def cover(A):
        return trace_measure(transform_trace(tr0, A), _p, _v, W, kernel=KERNEL)

    alphas = jax.lax.map(cover, As)

    def paint_over(img, xs):
        alpha, paint = xs
        return composite(img, alpha, paint), None

    img, _ = jax.lax.scan(paint_over, jnp.ones((H, W, 3)), (alphas, paints))
    return img


def loss_fn(params, goal):
    return jnp.sum((render(params) - goal) ** 2)


def to_uint8(img):
    return (onp.clip(onp.asarray(img), 0.0, 1.0) * 255).astype("uint8")


def to_png(img, path):
    Image.fromarray(to_uint8(img)).save(path)


def write_gif(imgs, path, duration=0.08):
    import imageio

    imageio.mimsave(path, [to_uint8(im) for im in imgs], loop=0, duration=duration)


def init_params(seed=1):
    random.seed(seed)
    loc = jnp.array(
        [[8.0 + (W - 16.0) * random.random(), 8.0 + (H - 16.0) * random.random()] for _ in range(N)]
    )
    radii = 1.2 + 0.28 * jnp.arange(N, 0, -1)[:, None] * jnp.ones((N, 2))
    color = jnp.zeros((N, 3))
    rot = jnp.zeros((N, 1))
    return (loc, radii, color, rot)


def main():
    goal = reduce_color(load_goal())
    to_png(goal, "/opt/cursor/artifacts/fit_rush_target.png")
    params = init_params()
    start_img = render(params)
    to_png(start_img, "/opt/cursor/artifacts/fit_rush_start.png")

    loss_jit = jax.jit(loss_fn)
    grad_jit = jax.jit(jax.grad(loss_fn))
    render_jit = jax.jit(render)

    history = [params]
    best, best_loss = params, float("inf")
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for i in range(1, STEPS + 1):
        g = grad_jit(params, goal)

        def adam(pi, gi, mi, vi):
            mi = b1 * mi + (1 - b1) * gi
            vi = b2 * vi + (1 - b2) * gi * gi
            mh = mi / (1 - b1**i)
            vh = vi / (1 - b2**i)
            return pi - LR * mh / (jnp.sqrt(vh) + eps), mi, vi

        new, new_m, new_v = [], [], []
        for pi, gi, mi, vi in zip(params, g, m, v):
            po, mo, vo = adam(pi, gi, mi, vi)
            new.append(po)
            new_m.append(mo)
            new_v.append(vo)
        params, m, v = tuple(new), tuple(new_m), tuple(new_v)
        loc, radii, color, rot = params
        radii = jnp.clip(jnp.abs(radii), 0.5, float(H) * 0.4)
        loc = jnp.clip(loc, -5.0, float(W + 5))
        color = jnp.clip(color, -6.0, 6.0)
        params = (loc, radii, color, rot)
        cur_loss = float(loss_jit(params, goal))
        if cur_loss < best_loss:
            best_loss, best = cur_loss, params
        if i % 8 == 0 or i == 1:
            history.append(params)
        if i == 1 or i % 50 == 0 or i == STEPS:
            print(f"step {i:4d}  loss={cur_loss:.1f}  best={best_loss:.1f}")

    params = best
    final = render_jit(params)
    to_png(final, "/opt/cursor/artifacts/fit_rush_final.png")
    to_png(
        jnp.concatenate([goal, start_img, final], axis=1),
        "/opt/cursor/artifacts/fit_rush_compare.png",
    )
    frames = [jnp.concatenate([goal, render_jit(p)], axis=1) for p in history]
    write_gif(frames, "/opt/cursor/artifacts/fit_rush.gif", duration=0.08)
    print("wrote /opt/cursor/artifacts/fit_rush.gif")


if __name__ == "__main__":
    main()
