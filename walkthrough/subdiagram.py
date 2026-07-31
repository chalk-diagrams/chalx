# %%
# from jaxtyping import install_import_hook
# with install_import_hook("chalk", "typeguard.typechecked"):
#     import chalk
from chalk import *
import jax.numpy as jnp
from typing import List

# %%
d = circle(2).fill_color("black").named("a") | circle(2).fill_color("blue")


def f(subs: List[Subdiagram], d: Diagram) -> Diagram:
    return d + circle(1).fill_color("red").translate_by(subs[0].get_location())


d.with_names(["a"], f)

# %%
v = circle(jnp.arange(1, 5)).get_envelope()

# %%
d = circle(jnp.arange(1, 5)).fill_color("black").named("a").hcat()


# %%
def f2(subs: List[Subdiagram], d: Diagram) -> Diagram:
    return d + circle(1).fill_color("red").translate_by(subs[0].get_location()).concat()


# %%
d = d.with_names(["a"], f2)
d.render_svg("tmp.svg")
