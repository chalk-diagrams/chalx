import jax
import jax.numpy as jnp
import numpy as onp

from chalk.geom import Affine, Pt, Vec, data
from chalk.transform import P2, V2, ident, origin, translation, unit_x


def test_typeof_p2_v2_xf():
    assert str(jax.typeof(P2(1.0, 2.0))) == "p2[]"
    assert str(jax.typeof(V2(1.0, 0.0))) == "v2[]"
    assert str(jax.typeof(translation(V2(1.0, 2.0)))) == "xf[]"
    assert str(jax.typeof(ident)) == "xf[]"
    assert str(jax.typeof(origin)) == "p2[]"
    assert str(jax.typeof(unit_x)) == "v2[]"


def test_p2_minus_p2_is_v2():
    v = P2(3.0, 4.0) - P2(1.0, 1.0)
    assert isinstance(v, Vec)
    assert str(jax.typeof(v)) == "v2[]"
    onp.testing.assert_allclose(onp.asarray(data(v))[:2, 0], [2.0, 3.0])


def test_p2_plus_v2_is_p2():
    p = P2(1.0, 1.0) + V2(2.0, 3.0)
    assert isinstance(p, Pt)
    assert str(jax.typeof(p)).startswith("p2")
    onp.testing.assert_allclose(onp.asarray(data(p))[:2, 0], [3.0, 4.0])


def test_p2_tangent_is_v2():
    def f(t):
        return P2(jnp.cos(t), jnp.sin(t))

    p, tng = jax.jvp(f, (0.0,), (1.0,))
    assert str(jax.typeof(p)).startswith("p2")
    assert str(jax.typeof(tng)).startswith("v2")
    onp.testing.assert_allclose(onp.asarray(data(p))[:2, 0], [1.0, 0.0], atol=1e-6)
    onp.testing.assert_allclose(onp.asarray(data(tng))[:2, 0], [0.0, 1.0], atol=1e-6)


def test_jvp_p2_sub():
    from chalk.geom import p2_sub_p2

    def f(t):
        return p2_sub_p2(P2(t, 2.0 * t), P2(0.0, 0.0))

    v, dv = jax.jvp(f, (1.0,), (1.0,))
    assert str(jax.typeof(v)).startswith("v2")
    assert str(jax.typeof(dv)).startswith("v2")
    onp.testing.assert_allclose(onp.asarray(data(dv))[:2, 0], [1.0, 2.0], atol=1e-6)


def test_xf_apply_pt():
    p = translation(V2(1.0, 2.0)) @ P2(0.0, 0.0)
    assert isinstance(p, Pt)
    onp.testing.assert_allclose(onp.asarray(data(p))[:2, 0], [1.0, 2.0], atol=1e-6)


def test_batched_v2():
    v = V2(jnp.arange(3.0), jnp.ones(3))
    assert str(jax.typeof(v)) == "v2[3]"


def test_attribute_access_fails_under_jit():
    p = P2(1.0, 2.0)
    try:
        jax.jit(lambda q: q.data)(p)
    except Exception:
        return
    raise AssertionError("expected attribute access to fail under jit")
