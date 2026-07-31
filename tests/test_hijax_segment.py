import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from chalk.segment import (
    Segment,
    concat_segments,
    make_segment,
    segment_parts,
    segment_q,
    transform_segment,
)
from chalk.transform import translation
import chalk.transform as tx


def _unit_seg():
    return Segment.make(tx.np.eye(3)[None, ...], tx.np.array([[0.0, -90.0]]))


def test_typeof_is_opaque_seg():
    s = _unit_seg()
    assert str(jax.typeof(s)).startswith("seg[")
    assert s.shape[-1] == 1 or s.shape == (1,)


def test_jaxpr_concat():
    a, b = _unit_seg(), _unit_seg()
    text = str(jax.jit(concat_segments).trace(a, b).jaxpr)
    assert "seg[" in text
    assert "call_hi_primitive" in text


def test_jit_transform():
    s = _unit_seg()
    t = translation(tx.V2(1.0, 2.0))
    out = jax.jit(lambda seg: transform_segment(seg, t))(s)
    c = onp.asarray(out.center).reshape(-1)
    onp.testing.assert_allclose(c[:2], [1.0, 2.0], rtol=1e-5, atol=1e-5)


def test_concat_changes_n():
    a, b = _unit_seg(), _unit_seg()
    c = concat_segments(a, b)
    assert str(jax.typeof(c)) == "seg[2]"


def test_attribute_access_fails_under_jit():
    s = _unit_seg()
    with pytest.raises(AttributeError):
        jax.jit(lambda seg: seg.transform)(s)


def test_jit_segment_q():
    s = _unit_seg()
    q = jax.jit(segment_q)(s)
    assert tuple(q.shape[-2:]) == (3, 1)
    onp.testing.assert_allclose(onp.asarray(s.q), onp.asarray(q), rtol=1e-5)
    text = str(jax.jit(segment_q).trace(s).jaxpr)
    assert "call_hi_primitive" in text
    t, a = jax.jit(segment_parts)(s)
    assert t.shape[-2:] == (3, 3)
    assert a.shape[-1] == 2
