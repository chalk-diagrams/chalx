import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from colour import Color

from chalk.style import Style, StyleSpec, merge_styles, to_color


def test_typeof_is_opaque_style():
    s = Style(fill_color=to_color(Color("orange")))
    ty = str(jax.typeof(s))
    assert ty == "style[]"
    onp.testing.assert_allclose(s.fill_color_, to_color(Color("orange")), rtol=1e-5)


def test_jaxpr_keeps_single_style_value():
    a = Style(fill_color=to_color(Color("red")))
    b = Style(line_width=0.2)
    jaxpr = jax.jit(merge_styles).trace(a, b).jaxpr
    text = str(jaxpr)
    assert "style[]" in text
    assert "call_hi_primitive" in text


def test_jit_merge():
    a = Style(fill_color=to_color(Color("red")))
    b = Style(line_width=0.25)
    out = jax.jit(merge_styles)(a, b)
    onp.testing.assert_allclose(out.fill_color_, to_color(Color("red")), rtol=1e-5)
    onp.testing.assert_allclose(onp.asarray(out.line_width_).reshape(-1)[0], 0.25, rtol=1e-5)


def test_expand_dims():
    s = Style(fill_color=to_color(Color("blue")))
    e = s.expand_dims(2)
    assert str(jax.typeof(e)) == "style[1,1]"
    assert e.fill_color_.shape[:2] == (1, 1)


def test_vmap_make_style():
    from chalk.style import make_style

    colors = jnp.zeros((4, 12))
    colors = colors.at[:, 0:3].set(jnp.linspace(0, 1, 4)[:, None])
    masks = jnp.zeros((4, 12), dtype=bool).at[:, 0:3].set(True)
    styles = jax.vmap(make_style, out_axes=StyleSpec())(colors, masks)
    assert str(jax.typeof(styles)) == "style[4]"


def test_attribute_access_fails_under_jit():
    s = Style(fill_color=to_color(Color("orange")))
    with pytest.raises(AttributeError):
        jax.jit(lambda style: style.base)(s)
