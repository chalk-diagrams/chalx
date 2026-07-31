import jax
import jax.numpy as jnp
import numpy as onp

from chalk import circle
from chalk.measure import trace_measure
from chalk.raster import scanline_origins


def test_jit_scale_translate_trace_measure():
    unit = circle(1.0).line_width(0)
    p = scanline_origins(32, axis="x")
    v = jnp.array([[1.0], [0.0], [0.0]])

    def cover(lx, ly, rx, ry, rot):
        d = unit.scale_x(rx).scale_y(ry).rotate_rad(rot).translate(lx, ly)
        return trace_measure(d, p, v, 32, kernel=5, boundary=False)

    out = jax.jit(cover)(16.0, 16.0, 8.0, 5.0, 0.3)
    assert out.shape == (32, 32)
    assert float(out.sum()) > 1.0


def test_grad_scale_x_trace_measure():
    unit = circle(1.0).line_width(0)
    p = scanline_origins(32, axis="x")
    v = jnp.array([[1.0], [0.0], [0.0]])

    def loss(rx):
        d = unit.scale_x(rx).scale_y(5.0).translate(16.0, 16.0)
        return trace_measure(d, p, v, 32, kernel=11, boundary=False).sum()

    g = jax.jit(jax.grad(loss))(jnp.float64(8.0))
    assert onp.isfinite(float(g))


def test_scan_ellipses_grad():
    unit = circle(1.0).line_width(0)
    p = scanline_origins(24, axis="x")
    v = jnp.array([[1.0], [0.0], [0.0]])
    loc = jnp.array([[8.0, 8.0], [16.0, 12.0], [10.0, 18.0]])
    radii = jnp.array([[4.0, 3.0], [5.0, 2.0], [3.0, 4.0]])
    rots = jnp.array([[0.1], [0.8], [1.4]])

    def render(radii):
        def cover(_, xs):
            (lx, ly), (rx, ry), (rot,) = xs
            d = unit.scale_x(rx).scale_y(ry).rotate_rad(rot).translate(lx, ly)
            α = trace_measure(d, p, v, 24, kernel=5, boundary=False)
            return None, α

        _, alphas = jax.lax.scan(cover, None, (loc, radii, rots))
        return alphas.sum()

    g = jax.jit(jax.grad(render))(radii)
    assert g.shape == radii.shape
    assert bool(jnp.isfinite(g).all())
