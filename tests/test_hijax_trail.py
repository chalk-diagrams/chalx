import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from chalk.trail import (
    Trail,
    make_located,
    trail_points,
    trail_segment,
    concat_trails,
)
from chalk.segment import segment_parts


def test_typeof_trail_and_located():
    t = Trail.circle()
    assert str(jax.typeof(t)).startswith("trail[")
    loc = t.centered()
    assert str(jax.typeof(loc)).startswith("located[")


def test_jit_trail_points():
    t = Trail.square()
    pts = jax.jit(trail_points)(t)
    assert pts.shape[-2:] == (3, 1)
    text = str(jax.jit(trail_points).trace(t).jaxpr)
    assert "call_hi_primitive" in text


def test_concat_trails():
    a, b = Trail.hrule(1), Trail.vrule(1)
    c = concat_trails(a, b)
    assert "trail[" in str(jax.typeof(c))


def test_attribute_peek_fails():
    t = Trail.hrule(1)
    with pytest.raises(AttributeError):
        jax.jit(lambda tr: tr.segments)(t)
