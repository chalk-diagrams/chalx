"""L2-fit a parameterized circle to a target raster (trace_measure)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as onp
from PIL import Image

from chalk import circle
from chalk.raster import scanline_direction, scanline_origins
from chalk.style import composite, to_color
from chalk.trace import transform_trace
from chalk.measure import trace_measure

H = W = 64
KERNEL = 11
PAINT = to_color("#ff9700")
STEPS = 80
LR = 0.35

tr0 = circle(1.0).get_trace()
_p = scanline_origins(H, axis="x")
_v = scanline_direction("x")


def _affine(cx, cy, r):
    return jnp.array([[r, 0.0, cx], [0.0, r, cy], [0.0, 0.0, 1.0]])


def render(params):
    cx, cy, r = params
    tr = transform_trace(tr0, _affine(cx, cy, r))
    img = jnp.ones((H, W, 3))
    alpha = trace_measure(tr, _p, _v, W, kernel=KERNEL)
    return composite(img, alpha, PAINT)


def loss_fn(params, goal):
    return jnp.sum((render(params) - goal) ** 2)


loss_jit = jax.jit(loss_fn)
grad_jit = jax.jit(jax.grad(loss_fn))


def to_uint8(img):
    arr = onp.clip(onp.asarray(img), 0.0, 1.0)
    return (arr * 255).astype("uint8")


def to_png(img, path):
    Image.fromarray(to_uint8(img)).save(path)


def write_gif(imgs, path, duration=0.08):
    import imageio

    frames = [to_uint8(im) for im in imgs]
    imageio.mimsave(path, frames, loop=0, duration=duration)


def main():
    target = jnp.array([32.0, 32.0, 16.0])
    params = jnp.array([44.0, 20.0, 9.0])
    goal = render(target)
    render_jit = jax.jit(render)

    to_png(goal, "/opt/cursor/artifacts/fit_circle_target.png")
    start_img = render(params)
    to_png(start_img, "/opt/cursor/artifacts/fit_circle_start.png")

    history = [params]
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for i in range(1, STEPS + 1):
        g = grad_jit(params, goal)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1**i)
        vh = v / (1 - b2**i)
        params = params - LR * mh / (jnp.sqrt(vh) + eps)
        params = params.at[2].set(jnp.maximum(params[2], 1.0))
        history.append(params)
        if i == 1 or i % 20 == 0 or i == STEPS:
            print(
                f"step {i:3d}  loss={float(loss_jit(params, goal)):.3f}  "
                f"params={onp.asarray(params)}"
            )

    frames = []
    for p in history:
        cur = render_jit(p)
        frames.append(jnp.concatenate([goal, cur], axis=1))
    write_gif(frames, "/opt/cursor/artifacts/fit_circle.gif", duration=0.07)

    final = frames[-1][:, W:, :]
    to_png(final, "/opt/cursor/artifacts/fit_circle_final.png")
    to_png(jnp.concatenate([goal, start_img, final], axis=1), "/opt/cursor/artifacts/fit_circle_compare.png")
    print("target", onp.asarray(target), "final", onp.asarray(params))
    print("wrote /opt/cursor/artifacts/fit_circle.gif")


if __name__ == "__main__":
    main()
