import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from chalk import circle, unit_x
from chalk.envelope import envelope_measure, make_envelope
from chalk.trace import make_trace, trace_ray


def test_envelope_typeof_and_measure():
    env = circle(1).get_envelope()
    assert str(jax.typeof(env)).startswith("env[")
    d = envelope_measure(env, unit_x)
    assert float(onp.asarray(d).reshape(-1)[-1]) > 0


def test_jit_envelope_measure():
    env = circle(1).get_envelope()
    fn = jax.jit(lambda e: envelope_measure(e, unit_x))
    out = fn(env)
    assert out.shape == ()
    text = str(fn.trace(env).jaxpr)
    assert "call_hi_primitive" in text


def test_trace_typeof_and_ray():
    tr = circle(1).get_trace()
    assert str(jax.typeof(tr)).startswith("trace[")
    d, m = trace_ray(tr, jnp.zeros((3, 1)), unit_x)
    assert d.shape[-1] >= 1


def test_envelope_peek_fails():
    env = circle(1).get_envelope()
    with pytest.raises(AttributeError):
        jax.jit(lambda e: e.segment)(env)
