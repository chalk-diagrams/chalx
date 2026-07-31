import jax
import jax.numpy as jnp
import pytest

import chalk.transform as tx
from chalk.path import (
    Path,
    PathSpec,
    path_get_located,
    path_located_segments,
    transform_path,
)
from chalk.trail import located_points


def triangle() -> Path:
    return Path.from_points(
        [tx.P2(0.0, 0.0), tx.P2(1.0, 0.0), tx.P2(0.5, 1.0)],
        closed=True,
    )


def test_typeof_is_opaque_path():
    path = triangle()
    assert str(jax.typeof(path)) == "cpath[1]"
    assert str(jax.typeof(path_get_located(path, 0))).startswith("located[")
    assert str(jax.typeof(path_located_segments(path))) == "seg[2]"


def test_jaxpr_keeps_path_and_geometry_opaque():
    path = triangle()
    jaxpr = jax.jit(
        lambda p: transform_path(p, tx.translation(tx.V2(2.0, -1.0)))
    ).trace(path).jaxpr
    text = str(jaxpr)
    assert "cpath[1]" in text
    assert "call_hi_primitive" in text


def test_path_points_are_opaque_points():
    path = triangle()
    points = jax.jit(lambda p: located_points(path_get_located(p, 0)))(path)
    assert str(jax.typeof(points)) == "p2[2]"
    expected = jnp.array(
        [[0.0, 0.0], [1.0, 0.0]], dtype=jnp.float64
    )
    assert jnp.allclose(tx.data(points)[..., :2, 0], expected)


def test_vmap_preserves_path_type():
    path = triangle()
    paths = jax.vmap(
        lambda offset: transform_path(path, tx.translation(tx.V2(offset, 0.0))),
        out_axes=PathSpec(),
    )(jnp.arange(3.0))
    assert str(jax.typeof(paths)) == "cpath[1]"
    assert paths.shape == (3,)


def test_attribute_access_fails_under_jit():
    path = triangle()
    with pytest.raises(AttributeError):
        jax.jit(lambda p: p.loc_trails)(path)
