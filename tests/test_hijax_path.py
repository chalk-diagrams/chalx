import jax
import jax.numpy as jnp
import numpy as onp
import pytest

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


def test_typeof_is_opaque_path():
    p = circle(1.0, n=16)
    ty = str(jax.typeof(p))
    assert ty.startswith("path[")
    assert "16" in ty
    assert to_points(p).shape == (16, 2)


def test_jaxpr_keeps_single_path_value():
    jaxpr = jax.jit(lambda r: translate(circle(r, n=8), 1.0, 0.0)).trace(1.0).jaxpr
    text = str(jaxpr)
    assert "path[8]" in text
    assert "call_hi_primitive" in text


def test_jit_roundtrip_points():
    pts = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)

    def fn(p):
        return to_points(translate(from_points(p, closed=True), 2.0, -1.0))

    out = jax.jit(fn)(pts)
    expected = pts + jnp.array([2.0, -1.0])
    onp.testing.assert_allclose(out, expected, rtol=1e-5)


def test_vmap_circles():
    rs = jnp.arange(1.0, 5.0)
    ps = jax.vmap(lambda r: circle(r, n=12), out_axes=PathSpec())(rs)
    assert str(jax.typeof(ps)) == "path[4;12]"
    verts = to_points(ps)
    assert verts.shape == (4, 12, 2)
    onp.testing.assert_allclose(jnp.linalg.norm(verts[0], axis=-1), 1.0, rtol=1e-5)
    onp.testing.assert_allclose(jnp.linalg.norm(verts[3], axis=-1), 4.0, rtol=1e-5)


def test_vmap_translate():
    ps = jax.vmap(lambda r: circle(r, n=8), out_axes=PathSpec())(jnp.ones(3))
    out = jax.vmap(
        lambda p: translate(p, 1.0, 2.0),
        in_axes=PathSpec(),
        out_axes=PathSpec(),
        axis_size=3,
    )(ps)
    verts = to_points(out)
    onp.testing.assert_allclose(verts.mean(axis=1), jnp.array([[1.0, 2.0]] * 3), atol=1e-5)


def test_beside_and_concat():
    d = beside(circle(1.0, n=8), square(1.0))
    assert to_points(d).shape[-2] == 8 + 4
    lo, hi = bounds(d)
    assert float(hi[0]) - float(lo[0]) > 2.0


def test_rotate_preserves_radius():
    p = rotate(circle(2.0, n=20), 35.0)
    radii = jnp.linalg.norm(to_points(p), axis=-1)
    onp.testing.assert_allclose(radii, 2.0, rtol=1e-5)


def test_grad_through_from_and_to_points():
    def loss(pts):
        return jnp.sum(to_points(from_points(pts)) ** 2)

    pts = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    g = jax.grad(loss)(pts)
    onp.testing.assert_allclose(g, 2.0 * pts, rtol=1e-5)


def test_grad_through_translate():
    def loss(pts):
        path = from_points(pts, closed=True)
        return jnp.sum(to_points(translate(path, 1.0, -2.0)) ** 2)

    pts = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=jnp.float32)
    g = jax.grad(loss)(pts)
    moved = pts + jnp.array([1.0, -2.0])
    onp.testing.assert_allclose(g, 2.0 * moved, rtol=1e-5)


def test_scan_over_batched_paths():
    ps = jax.vmap(lambda r: circle(r, n=6), out_axes=PathSpec())(
        jnp.array([1.0, 2.0, 3.0])
    )

    def step(total, path):
        lo, hi = bounds(path)
        return total + jnp.sum(hi - lo), ()

    total, _ = jax.lax.scan(step, jnp.float32(0.0), ps, length=3)
    assert total > 0


def test_attribute_access_fails_under_jit():
    p = circle(1.0, n=8)
    with pytest.raises(AttributeError):
        jax.jit(lambda path: path.vertices)(p)
