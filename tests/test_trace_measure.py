import jax
import jax.numpy as jnp
import numpy as onp

from chalk import circle, rectangle
from chalk.measure import measure_from_splits, trace_measure
from chalk.raster import rasterize, scanline_direction, scanline_origins
from chalk.style import composite, to_color
from chalk.trace import transform_trace


def test_fill_line_matches_blog():
    splits = jnp.array([3.4, 9.7, 16.9])
    mask = jnp.array([1, 1, 0])
    row = measure_from_splits(splits, mask, n_bins=100, kernel=1)
    assert abs(float(row[3]) - 0.6) < 1e-5
    assert abs(float(row[4]) - 1.0) < 1e-5
    assert float(row[9]) > 0.5
    assert float(row[10]) < 0.5


def test_trace_measure_circle_row():
    d = circle(20).translate(50, 50).fill_color("orange")
    p = scanline_origins(100, axis="x")
    α = trace_measure(d, p, scanline_direction("x"), 100, kernel=1)
    assert α.shape == (100, 100)
    assert float(α.sum()) > 10.0
    # middle scanline through the circle should be partly filled
    assert float(α[50].max()) > 0.5


def test_composite_over():
    img = jnp.ones((4, 4, 3))
    α = jnp.zeros((4, 4)).at[1:3, 1:3].set(1.0)
    out = composite(img, α, to_color("red"))
    onp.testing.assert_allclose(out[0, 0], [1, 1, 1], atol=1e-5)
    onp.testing.assert_allclose(out[2, 2], onp.asarray(to_color("red")), atol=1e-5)


def test_rasterize_smoke():
    d = circle(15).translate(32, 32).fill_color("blue")
    img = rasterize(d, 64, 64, kernel=5, paint=to_color("blue"))
    assert img.shape == (64, 64, 3)
    # background stays light; interior is bluish
    assert float(img[32, 32, 2]) > float(img[32, 32, 0])


def test_trace_measure_grad_translation():
    tr0 = rectangle(40, 40).translate(50, 50).get_trace()
    p = jnp.asarray([[0.0, 50.0, 1.0]]).reshape(1, 3, 1)
    v = jnp.asarray([1.0, 0.0, 0.0]).reshape(3, 1)

    def loss(dx):
        t = jnp.eye(3).at[0, 2].set(dx)
        α = trace_measure(transform_trace(tr0, t), p, v, 100, kernel=11)
        xs = jnp.arange(100.0)
        return (α[0] * xs).sum()

    g = jax.grad(loss)(0.0)
    assert onp.isfinite(float(g))
    assert abs(float(g)) > 1e-3
