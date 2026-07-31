"""Fit triangles to Sasha's portrait: one diagram, trace raster, Cairo + video."""

from __future__ import annotations

import random
import subprocess
import urllib.request

import jax
import jax.numpy as jnp
import numpy as onp
from PIL import Image, ImageDraw, ImageFont

from chalk import rectangle, triangle
from chalk.measure import trace_measure
from chalk.raster import scanline_origins
from chalk.style import composite

H = W = 80
KERNEL = 5
N = 500
STEPS = 300
MIN_SIZE = 1.0
LR = 0.03
LOSS_EVERY = 10
GIF_EVERY = 10
PHOTO_URL = "https://avatars0.githubusercontent.com/u/35882?s=460&v=4"
LIB_HEIGHT = 400
OUT = "/opt/cursor/artifacts"

_unit = triangle(1.0).line_width(0)
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
    loc, sizes, rots, color, opacity = params
    order = jnp.argsort(-sizes)
    return loc[order], sizes[order], rots[order], color[order], opacity[order]


def diagram(params):
    loc, sizes, rots, color, opacity = sort_params(params)
    paints = jax.nn.sigmoid(color)
    opac = jax.nn.sigmoid(opacity)
    return (
        _unit.fill_color(paints)
        .fill_opacity(opac)
        .rotate_rad(rots[:, 0])
        .scale(sizes)
        .translate(loc[:, 0], loc[:, 1])
    )


def diagram_stacked(params):
    frame = (
        rectangle(float(W), float(H))
        .line_width(0)
        .fill_opacity(0)
        .translate(float(W) / 2.0, float(H) / 2.0)
    )
    return diagram(params).concat().with_envelope(frame)


def raster(params):
    loc, sizes, rots, color, opacity = sort_params(params)
    paints = jax.nn.sigmoid(color)
    opac = jax.nn.sigmoid(opacity)

    def cover(_, xs):
        (lx, ly), size, (rot,) = xs
        d = _unit.rotate_rad(rot).scale(size).translate(lx, ly)
        α = trace_measure(d, _px, _vx, W, kernel=KERNEL)
        return None, α

    _, alphas = jax.lax.scan(cover, None, (loc, sizes, rots))

    def paint_over(img, xs):
        coverage, paint, a = xs
        return composite(img, coverage, (paint, a)), None

    img, _ = jax.lax.scan(paint_over, jnp.ones((H, W, 3)), (alphas, paints, opac))
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


def write_video(frames_uint8, path, fps=12):
    """Encode RGB uint8 frames to mp4 via ffmpeg."""
    h, w = frames_uint8[0].shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        path,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    assert proc.stdin is not None and proc.stderr is not None
    try:
        for fr in frames_uint8:
            proc.stdin.write(onp.ascontiguousarray(fr).tobytes())
    finally:
        proc.stdin.close()
    err = proc.stderr.read()
    code = proc.wait()
    if code != 0:
        raise RuntimeError(err.decode("utf-8", errors="replace")[-2000:])


def init_params(seed=42):
    random.seed(seed)
    loc = jnp.array(
        [[8.0 + (W - 16.0) * random.random(), 8.0 + (H - 16.0) * random.random()] for _ in range(N)]
    )
    sizes = jnp.array([MIN_SIZE + 12.0 * random.random() ** 1.7 for _ in range(N)])
    rots = jnp.array([[random.uniform(0.0, 2.0 * jnp.pi)] for _ in range(N)])
    color = jnp.array([[random.uniform(-2.0, 2.0) for _ in range(3)] for _ in range(N)])
    opacity = jnp.array([random.uniform(-0.5, 1.5) for _ in range(N)])
    return (loc, sizes, rots, color, opacity)


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
        f"scanline triangles (kernel={KERNEL})",
        "cairo of the same diagram",
        "target photo (not composited)",
    ]
    for i, label in enumerate(labels):
        draw.text((i * size + 10, 8), label, fill=(255, 255, 255), font=font)
    strip.save(out_path)


def cairo_frame(params, height=LIB_HEIGHT):
    path = "/tmp/rush_tri_frame.png"
    diagram_stacked(params).render(path, height=height, width=height)
    return onp.asarray(Image.open(path).convert("RGB"))


def main():
    print(f"N={N} STEPS={STEPS} MIN_SIZE={MIN_SIZE} KERNEL={KERNEL} triangles {H}x{W}", flush=True)
    goal = reduce_color(load_goal())
    to_png(goal, f"{OUT}/fit_rush_target.png")
    params = init_params()

    print("jit compile…", flush=True)
    raster_jit = jax.jit(raster)
    loss_jit = jax.jit(loss_fn)
    grad_jit = jax.jit(jax.grad(loss_fn))
    import time as _time

    t0 = _time.perf_counter()
    start_img = raster_jit(params).block_until_ready()
    print(f"  raster compiled in {_time.perf_counter() - t0:.1f}s", flush=True)
    to_png(start_img, f"{OUT}/fit_rush_start.png")
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
        loc, sizes, rots, color, opacity = params
        sizes = jnp.clip(jnp.abs(sizes), MIN_SIZE, float(H) * 0.4)
        loc = jnp.clip(loc, -5.0, float(W + 5))
        color = jnp.clip(color, -6.0, 6.0)
        opacity = jnp.clip(opacity, -6.0, 6.0)
        params = (loc, sizes, rots, color, opacity)
        if i % GIF_EVERY == 0 or i == 1:
            history.append(params)
        if i == 1 or i % LOSS_EVERY == 0 or i == STEPS:
            cur_loss = float(loss_jit(params, goal))
            if cur_loss < best_loss:
                best_loss, best = cur_loss, params
            print(f"step {i:4d}  loss={cur_loss:.1f}  best={best_loss:.1f}", flush=True)

    params = best
    onp.savez(
        f"{OUT}/fit_rush_best.npz",
        loc=onp.asarray(params[0]),
        sizes=onp.asarray(params[1]),
        rots=onp.asarray(params[2]),
        color=onp.asarray(params[3]),
        opacity=onp.asarray(params[4]),
    )
    final = raster_jit(params)
    to_png(final, f"{OUT}/fit_rush_final.png")
    to_png(
        jnp.concatenate([goal, start_img, final], axis=1),
        f"{OUT}/fit_rush_compare.png",
    )

    scan_side = [jnp.concatenate([goal, raster_jit(p)], axis=1) for p in history]
    write_gif(scan_side, f"{OUT}/rush_tri500.gif", duration=0.08)
    scan_u8 = [
        onp.asarray(
            Image.fromarray(to_uint8(fr)).resize((480, 240), Image.Resampling.NEAREST)
        )
        for fr in scan_side
    ]
    write_video(scan_u8, f"{OUT}/rush_tri500_scan.mp4", fps=10)
    print(f"wrote scan video ({len(scan_u8)} frames)", flush=True)

    print("cairo video…", flush=True)
    cairo_u8 = []
    for i, p in enumerate(history):
        frame = cairo_frame(p, height=LIB_HEIGHT)
        cairo_u8.append(frame)
        if i % 5 == 0 or i == len(history) - 1:
            print(f"  cairo frame {i + 1}/{len(history)}", flush=True)
    write_video(cairo_u8, f"{OUT}/rush_tri500_cairo.mp4", fps=10)
    print(f"wrote cairo video ({len(cairo_u8)} frames)", flush=True)

    dia = diagram_stacked(params)
    lib_path = f"{OUT}/rush_tri500_cairo.png"
    dia.render(lib_path, height=LIB_HEIGHT)
    dia.render_svg(f"{OUT}/rush_tri500.svg", height=LIB_HEIGHT)
    write_compare_strip(final, lib_path, goal, f"{OUT}/rush_tri500_strip.png")
    to_png(start_img, f"{OUT}/rush_tri500_start.png")
    to_png(final, f"{OUT}/rush_tri500_scan.png")
    to_png(goal, f"{OUT}/rush_tri500_target.png")
    to_png(
        jnp.concatenate([goal, start_img, final], axis=1),
        f"{OUT}/rush_tri500_compare.png",
    )
    print(f"wrote {OUT}/rush_tri500_strip.png", flush=True)


if __name__ == "__main__":
    main()
