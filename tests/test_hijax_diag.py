import jax
import pytest
from colour import Color

from chalk import circle, square


def test_typeof_is_opaque_diag():
    d = circle(1).fill_color(Color("#ff9700"))
    ty = str(jax.typeof(d))
    assert ty.startswith("diag")
    assert "prim" not in ty
    assert "compose" not in ty


def test_compose_still_diag():
    d = circle(1) | square(1)
    assert str(jax.typeof(d)).startswith("diag")


def test_batched_diag():
    import jax.numpy as jnp

    d = circle(jnp.arange(1, 4))
    assert str(jax.typeof(d)) == "diag[3]"


def test_peek_root_class_not_required():
    d = circle(1) | square(1)
    # users shouldn't need the node type; typeof stays diag
    assert "diag" in str(jax.typeof(d))
