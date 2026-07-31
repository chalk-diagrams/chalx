"""Fit ellipses to a photo with a learnable differentiable z-order.

Each ellipse keeps a scalar ``z`` (low = behind). Scanline and Cairo both
hard-sort by ``z`` and gate fill opacity with ``modulate_opacity(z)``.
Coverage is the average of a left-to-right and top-to-bottom scan.
Init is size-ordered so ``z`` starts as large-behind / small-in-front.
"""

from __future__ import annotations

import random

import jax
import jax.numpy as jnp
import numpy as onp
from PIL import Image, ImageDraw, ImageFont

from chalk import circle, rectangle
from chalk.raster import trace_measure_xy
from chalk.style import composite_by_z, modulate_opacity

W, H = 96, 72  # 4:3, matches the highland-cow photo
N = 100
STEPS = 750
MIN_SIZE = 1.0
LR0 = 0.03
LOSS_EVERY = 10
DECAY_START = 500  # last 250 steps taper LR and kernel 7→5→3
PHOTO = "/home/ubuntu/.cursor/projects/workspace/assets/019fb880-e393-7efc-a666-974299a8fcb2.jpg"
LIB_HEIGHT = 360
LIB_WIDTH = 480
OUT = "/opt/cursor/artifacts"
PREFIX = "cow5"
KERNELS = (7, 5, 3)

_unit = circle(1.0).line_width(0)


def load_goal():
    im = Image.open(PHOTO).convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    return jnp.asarray(onp.asarray(im).astype("float64") / 255.0)


def reduce_color(y):
    return jnp.floor(y * 40.0) / 40.0


def lr_at(step: int) -> float:
    if step <= DECAY_START:
        return LR0
    t = (step - DECAY_START) / max(STEPS - DECAY_START, 1)
    return LR0 * (1.0 - 0.9 * t)


def kernel_at(step: int) -> int:
    if step <= DECAY_START:
        return 7
    if step <= DECAY_START + (STEPS - DECAY_START) // 2:
        return 5
    return 3


def split_geom(params):
    loc, radii, rots, color, opacity, _z = params
    return loc, radii, rots, color, opacity


def diagram(params):
    loc, radii, rots, color, opacity, z = params
    paints = jax.nn.sigmoid(color)
    opac = modulate_opacity(jax.nn.sigmoid(opacity), z)
    return (
        _unit.fill_color(paints)
        .fill_opacity(opac)
        .scale_x(radii[:, 0])
        .scale_y(radii[:, 1])
        .rotate_rad(rots[:, 0])
        .translate(loc[:, 0], loc[:, 1])
    )


def diagram_stacked(params):
    """Cairo: deterministic paint order = argsort(z), low z behind."""
    loc, radii, rots, color, opacity, z = params
    order = jnp.argsort(z)
    ordered = (loc[order], radii[order], rots[order], color[order], opacity[order], z[order])
    frame = (
        rectangle(float(W), float(H))
        .line_width(0)
        .fill_opacity(0)
        .translate(float(W) / 2.0, float(H) / 2.0)
    )
    return diagram(ordered).concat().with_envelope(frame)


def make_raster(kernel: int):
    def raster(params):
        loc, radii, rots, color, opacity, z = params
        paints = jax.nn.sigmoid(color)
        opac = jax.nn.sigmoid(opacity)

        def cover(_, xs):
            (lx, ly), (rx, ry), (rot,) = xs
            d = _unit.scale_x(rx).scale_y(ry).rotate_rad(rot).translate(lx, ly)
            α = trace_measure_xy(d, H, W, kernel=kernel)
            return None, α

        _, alphas = jax.lax.scan(cover, None, (loc, radii, rots))
        return composite_by_z(jnp.ones((H, W, 3)), alphas, (paints, opac), z, hard=True)

    def loss_fn(params, goal):
        return jnp.sum((raster(params) - goal) ** 2)

    return raster, loss_fn


def to_uint8(img):
    return (onp.clip(onp.asarray(img), 0.0, 1.0) * 255).astype("uint8")


def to_png(img, path):
    Image.fromarray(to_uint8(img)).save(path)


def init_params(seed=42):
    random.seed(seed)
    cx, cy = W / 2.0, H / 2.0
    locs = []
    for _ in range(N):
        if random.random() < 0.72:
            x = cx + random.gauss(0.0, W * 0.16)
            y = cy + random.gauss(0.0, H * 0.16)
            x = min(max(x, 4.0), float(W - 4))
            y = min(max(y, 4.0), float(H - 4))
        else:
            x = 6.0 + (W - 12.0) * random.random()
            y = 6.0 + (H - 12.0) * random.random()
        locs.append([x, y])
    loc = jnp.array(locs)
    radii = []
    for _ in range(N):
        size = MIN_SIZE + 14.0 * random.random() ** 1.6
        aspect = 0.35 + 0.65 * random.random()
        if random.random() < 0.5:
            radii.append([size, max(MIN_SIZE, size * aspect)])
        else:
            radii.append([max(MIN_SIZE, size * aspect), size])
    radii = jnp.array(radii)
    rots = jnp.array([[random.uniform(0.0, 2.0 * jnp.pi)] for _ in range(N)])
    color = jnp.array([[random.uniform(-2.0, 2.0) for _ in range(3)] for _ in range(N)])
    opacity = jnp.array([random.uniform(-0.2, 1.8) for _ in range(N)])
    # Size-order once: slot 0 = largest = behind. Learnable z starts aligned.
    order = jnp.argsort(-(radii[:, 0] * radii[:, 1]))
    loc, radii, rots, color, opacity = (
        loc[order],
        radii[order],
        rots[order],
        color[order],
        opacity[order],
    )
    z = jnp.linspace(-1.0, 1.0, N)
    return (loc, radii, rots, color, opacity, z)


def write_compare_strip(raster_img, library_path, goal, out_path, k_label: int):
    lib = Image.open(library_path).convert("RGB")
    tw, th = lib.size
    raster_im = Image.fromarray(to_uint8(raster_img)).resize((tw, th), Image.Resampling.NEAREST)
    goal_im = Image.fromarray(to_uint8(goal)).resize((tw, th), Image.Resampling.NEAREST)
    label_h = 36
    strip = Image.new("RGB", (tw * 3, th + label_h), (20, 20, 20))
    for i, im in enumerate([raster_im, lib, goal_im]):
        strip.paste(im.convert("RGB"), (i * tw, label_h))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:
        font = ImageFont.load_default()
    labels = [
        f"scanline xy+z (k={k_label})",
        "cairo argsort(z)+α(z)",
        "target photo",
    ]
    for i, label in enumerate(labels):
        draw.text((i * tw + 10, 8), label, fill=(255, 255, 255), font=font)
    strip.save(out_path)


def main():
    print(
        f"{PREFIX} N={N} STEPS={STEPS} {W}x{H} z-order+α(z) xy-scan stills-only "
        f"LR/kernel 7→5→3 after {DECAY_START}",
        flush=True,
    )
    goal = reduce_color(load_goal())
    to_png(goal, f"{OUT}/{PREFIX}_target.png")
    params = init_params()

    print("jit compile kernels", KERNELS, "…", flush=True)
    fns = {}
    import time as _time

    for k in KERNELS:
        raster, loss_fn = make_raster(k)
        t0 = _time.perf_counter()
        raster_jit = jax.jit(raster)
        loss_jit = jax.jit(loss_fn)
        grad_jit = jax.jit(jax.grad(loss_fn))
        start_img = raster_jit(params).block_until_ready()
        jax.tree.map(lambda x: x.block_until_ready(), grad_jit(params, goal))
        print(f"  kernel={k} compiled in {_time.perf_counter() - t0:.1f}s", flush=True)
        fns[k] = (raster_jit, loss_jit, grad_jit)
        if k == KERNELS[0]:
            to_png(start_img, f"{OUT}/{PREFIX}_start.png")
            start_img0 = start_img

    curve = []
    best, best_loss = params, float("inf")
    prev_kern = kernel_at(1)
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for i in range(1, STEPS + 1):
        lr = lr_at(i)
        kern = kernel_at(i)
        raster_jit, loss_jit, grad_jit = fns[kern]
        g = grad_jit(params, goal)

        def adam(pi, gi, mi, vi):
            mi = b1 * mi + (1 - b1) * gi
            vi = b2 * vi + (1 - b2) * gi * gi
            mh = mi / (1 - b1**i)
            vh = vi / (1 - b2**i)
            return pi - lr * mh / (jnp.sqrt(vh) + eps), mi, vi

        new, new_m, new_v = [], [], []
        for pi, gi, mi, vi in zip(params, g, m, v):
            po, mo, vo = adam(pi, gi, mi, vi)
            new.append(po)
            new_m.append(mo)
            new_v.append(vo)
        params, m, v = tuple(new), tuple(new_m), tuple(new_v)
        loc, radii, rots, color, opacity, z = params
        radii = jnp.clip(jnp.abs(radii), MIN_SIZE, float(min(H, W)) * 0.45)
        loc = jnp.stack(
            [
                jnp.clip(loc[:, 0], -8.0, float(W + 8)),
                jnp.clip(loc[:, 1], -8.0, float(H + 8)),
            ],
            axis=1,
        )
        color = jnp.clip(color, -6.0, 6.0)
        opacity = jnp.clip(opacity, -6.0, 6.0)
        z = jnp.clip(z, -8.0, 8.0)
        params = (loc, radii, rots, color, opacity, z)
        if i == 1 or i % LOSS_EVERY == 0 or i == STEPS:
            if kern != prev_kern:
                best_loss = float("inf")
                prev_kern = kern
            cur_loss = float(loss_jit(params, goal))
            if cur_loss < best_loss:
                best_loss, best = cur_loss, params
            curve.append((i, cur_loss, best_loss, lr, kern))
            print(
                f"step {i:4d}  loss={cur_loss:.1f}  best={best_loss:.1f}  "
                f"lr={lr:.4f}  k={kern}",
                flush=True,
            )

    params = best
    onp.savez(
        f"{OUT}/{PREFIX}_best.npz",
        loc=onp.asarray(params[0]),
        radii=onp.asarray(params[1]),
        rots=onp.asarray(params[2]),
        color=onp.asarray(params[3]),
        opacity=onp.asarray(params[4]),
        z=onp.asarray(params[5]),
        curve=onp.asarray(curve),
    )
    final_k = kernel_at(STEPS)
    raster_jit, loss_jit, _ = fns[final_k]
    final = raster_jit(params)
    to_png(final, f"{OUT}/{PREFIX}_final.png")
    to_png(jnp.concatenate([goal, start_img0, final], axis=1), f"{OUT}/{PREFIX}_compare.png")

    dia = diagram_stacked(params)
    lib_path = f"{OUT}/{PREFIX}_cairo.png"
    dia.render(lib_path, height=LIB_HEIGHT, width=LIB_WIDTH)
    dia.render_svg(f"{OUT}/{PREFIX}.svg", height=LIB_HEIGHT)
    write_compare_strip(final, lib_path, goal, f"{OUT}/{PREFIX}_strip.png", final_k)
    to_png(start_img0, f"{OUT}/{PREFIX}_start.png")
    to_png(goal, f"{OUT}/{PREFIX}_target.png")
    z = onp.asarray(params[5])
    area = onp.asarray(params[1][:, 0] * params[1][:, 1])
    print(
        f"wrote {OUT}/{PREFIX}_strip.png best={best_loss:.1f} "
        f"zcorr(area)={float(onp.corrcoef(z, -onp.log(area + 1e-6))[0,1]):.3f}",
        flush=True,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps, losses, bests, lrs, ks = zip(*curve)
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
        ax.plot(steps, losses, color="#4c78a8", lw=1.5, marker="o", ms=3, label="loss")
        ax.plot(steps, bests, color="#f58518", lw=2, label="best")
        ax.axvline(DECAY_START, color="#54a24b", ls="--", lw=1, label="decay start")
        ax.set_xlabel("Adam step")
        ax.set_ylabel("L2")
        ax.set_title("100 ellipses · cow · z+α(z) · xy scan · 7→5→3")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{OUT}/{PREFIX}_curve.png")
    except Exception as e:
        print("curve skip", e, flush=True)


if __name__ == "__main__":
    main()
