"""Fit ellipses to Sasha's portrait: one diagram, trace raster, Cairo PNG."""

from __future__ import annotations

import random
import urllib.request

import jax
import jax.numpy as jnp
import numpy as onp
from PIL import Image, ImageDraw, ImageFont

from chalk import circle
from chalk.measure import trace_measure
from chalk.raster import scanline_origins
from chalk.style import composite

H = W = 80
KERNEL = 11
N = 300
STEPS = 500
MIN_SIZE = 2.5
LR = 0.03
LOSS_EVERY = 25
PHOTO_URL = "https://avatars0.githubusercontent.com/u/35882?s=460&v=4"
LIB_HEIGHT = 400

_unit = circle(1.0).line_width(0)
_px = scanline_origins(H, axis="x")
_vx = jnp.array([[1.0], [0.0], [0.0]])


def load_goal():
    path = "/tmp/rush_avatar.jpg"
    urllib.request.urlretrieve(PHOTO_URL, path)
    im = Image.open(path).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    return jnp.asarray(onp.asarray(im).astype("float64") / 255.0)


def reduce_color(y):
    return jnp.floor(y * 40.0) / 40.0


def sort_params(params):
    loc, radii, rots, color = params
    order = jnp.argsort(-(radii[:, 0] * radii[:, 1]))
    return loc[order], radii[order], rots[order], color[order]


def diagram(params):
    """One batched diagram: unit circle, scaled/rotated/translated/filled."""
    loc, radii, rots, color = sort_params(params)
    paints = jax.nn.sigmoid(color)
    return (
        _unit.fill_color(paints)
        .scale_x(radii[:, 0])
        .scale_y(radii[:, 1])
        .rotate_rad(rots[:, 0])
        .translate(loc[:, 0], loc[:, 1])
    )


def raster(params):
    loc, radii, rots, color = sort_params(params)
    paints = jax.nn.sigmoid(color)

    def cover(_, xs):
        (lx, ly), (rx, ry), (rot,) = xs
        d = _unit.scale_x(rx).scale_y(ry).rotate_rad(rot).translate(lx, ly)
        α = trace_measure(d, _px, _vx, W, kernel=KERNEL, boundary=False)
        return None, α

    _, alphas = jax.lax.scan(cover, None, (loc, radii, rots))

    def paint_over(img, xs):
        alpha, paint = xs
        return composite(img, alpha, paint), None

    img, _ = jax.lax.scan(paint_over, jnp.ones((H, W, 3)), (alphas, paints))
    return img


def loss_fn(params, goal):
    return jnp.sum((raster(params) - goal) ** 2)


def to_uint8(img):
    return (onp.clip(onp.asarray(img), 0.0, 1.0) * 255).astype("uint8")


def to_png(img, path):
    Image.fromarray(to_uint8(img)).save(path)


def write_gif(imgs, path, duration=0.08):
    import imageio

    imageio.mimsave(path, [to_uint8(im) for im in imgs], loop=0, duration=duration)


def init_params(seed=42):
    random.seed(seed)
    loc = jnp.array(
        [[8.0 + (W - 16.0) * random.random(), 8.0 + (H - 16.0) * random.random()] for _ in range(N)]
    )
    radii = []
    for _ in range(N):
        size = MIN_SIZE + 12.0 * random.random() ** 1.7
        aspect = 0.35 + 0.65 * random.random()
        if random.random() < 0.5:
            radii.append([size, max(MIN_SIZE, size * aspect)])
        else:
            radii.append([max(MIN_SIZE, size * aspect), size])
    radii = jnp.array(radii)
    rots = jnp.array([[random.uniform(0.0, 2.0 * jnp.pi)] for _ in range(N)])
    color = jnp.array([[random.uniform(-2.0, 2.0) for _ in range(3)] for _ in range(N)])
    return (loc, radii, rots, color)


def _letterbox(im: Image.Image, size: int) -> Image.Image:
    im = im.convert("RGB")
    im = im.copy()
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def write_compare_strip(raster_img, library_path, goal, out_path):
    lib = Image.open(library_path).convert("RGB")
    size = max(lib.height, LIB_HEIGHT)
    raster_im = Image.fromarray(to_uint8(raster_img)).resize(
        (size, size), Image.Resampling.NEAREST
    )
    goal_im = Image.fromarray(to_uint8(goal)).resize(
        (size, size), Image.Resampling.NEAREST
    )
    lib_im = _letterbox(lib, size)
    label_h = 36
    strip = Image.new("RGB", (size * 3, size + label_h), (20, 20, 20))
    for i, im in enumerate([raster_im, lib_im, goal_im]):
        strip.paste(im, (i * size, label_h))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    labels = [
        "scanline (optimizer saw this)",
        "cairo of the same diagram",
        "target photo (not composited)",
    ]
    for i, label in enumerate(labels):
        draw.text((i * size + 10, 8), label, fill=(255, 255, 255), font=font)
    strip.save(out_path)


def main():
    print(f"N={N} STEPS={STEPS} {H}x{W} ellipses (one diagram)", flush=True)
    goal = reduce_color(load_goal())
    to_png(goal, "/opt/cursor/artifacts/fit_rush_target.png")
    params = init_params()

    print("jit compile…", flush=True)
    raster_jit = jax.jit(raster)
    loss_jit = jax.jit(loss_fn)
    grad_jit = jax.jit(jax.grad(loss_fn))
    import time as _time

    t0 = _time.perf_counter()
    start_img = raster_jit(params).block_until_ready()
    print(f"  raster compiled in {_time.perf_counter() - t0:.1f}s", flush=True)
    to_png(start_img, "/opt/cursor/artifacts/fit_rush_start.png")
    t0 = _time.perf_counter()
    jax.tree.map(lambda x: x.block_until_ready(), grad_jit(params, goal))
    print(f"  grad compiled in {_time.perf_counter() - t0:.1f}s", flush=True)

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
        loc, radii, rots, color = params
        radii = jnp.clip(jnp.abs(radii), MIN_SIZE, float(H) * 0.4)
        loc = jnp.clip(loc, -5.0, float(W + 5))
        color = jnp.clip(color, -6.0, 6.0)
        params = (loc, radii, rots, color)
        if i % 25 == 0 or i == 1:
            history.append(params)
        if i == 1 or i % LOSS_EVERY == 0 or i == STEPS:
            cur_loss = float(loss_jit(params, goal))
            if cur_loss < best_loss:
                best_loss, best = cur_loss, params
            print(f"step {i:4d}  loss={cur_loss:.1f}  best={best_loss:.1f}")

    params = best
    onp.savez(
        "/opt/cursor/artifacts/fit_rush_best.npz",
        loc=onp.asarray(params[0]),
        radii=onp.asarray(params[1]),
        rots=onp.asarray(params[2]),
        color=onp.asarray(params[3]),
    )
    final = raster_jit(params)
    to_png(final, "/opt/cursor/artifacts/fit_rush_final.png")
    to_png(
        jnp.concatenate([goal, start_img, final], axis=1),
        "/opt/cursor/artifacts/fit_rush_compare.png",
    )
    frames = [jnp.concatenate([goal, raster_jit(p)], axis=1) for p in history]
    write_gif(frames, "/opt/cursor/artifacts/fit_rush.gif", duration=0.08)
    print("wrote /opt/cursor/artifacts/fit_rush.gif")

    dia = diagram(params).concat().scale_y(-1).center_xy()
    lib_path = "/opt/cursor/artifacts/rush_ell_library.png"
    dia.render(lib_path, height=LIB_HEIGHT)
    print(f"wrote cairo of the fitted diagram {lib_path}")
    write_compare_strip(
        final,
        lib_path,
        goal,
        "/opt/cursor/artifacts/rush_ell_strip.png",
    )
    to_png(start_img, "/opt/cursor/artifacts/rush_ell_start.png")
    to_png(final, "/opt/cursor/artifacts/rush_ell_final.png")
    to_png(goal, "/opt/cursor/artifacts/rush_ell_target.png")
    print("wrote /opt/cursor/artifacts/rush_ell_strip.png")


if __name__ == "__main__":
    main()
