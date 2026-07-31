# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: chalk
#     language: python
#     name: chalk
# ---

# %% tags=["hide_inp"]
from chalk import *
import jax.numpy as jnp

# %% [markdown]
# Diagrams are set-up so that they can take
# JAX arguments support standard array broadcasting.

# %%
cs1 = circle(jnp.arange(1, 5)) + circle(jnp.arange(1, 5))


# %%
cs1 = circle(1).translate(jnp.arange(5), jnp.arange(5))
print(cs1.size())
cs1.concat()
cs1.concat().render_svg("/tmp/x.svg")

# %% [markdown]
# Style changes will broadcast over the diagram.

# %%
cs1.fill_color("orange").concat()


# %% [markdown]
# Alteratively you can apply array styles.

# %%
cs1.fill_color(jnp.ones(3) * jnp.linspace(0, 1, 5)[:, None]).concat()


# %% [markdown]
# Standard combinators also work with sized
# elements.

# %%
circle(jnp.arange(1, 4)).hcat()

# %% [markdown]
# Diagrams can have arbitrary prefix sizes.

# %%
cs2 = circle(1).translate(jnp.arange(2)[:, None], jnp.arange(2)[:, None])
print(cs2.size())
cs2.concat().concat()


# %% [markdown]
# Using broadcasting can allow for complex shapes
# such as faceting of graphs. Here's an example
# where we build a grid of clock faces.

# %%
r = circle(1) + seg(unit_x).stroke().rotate(jnp.linspace(0, 360, 12))
r.reshape((4, 3)).hcat(sep=1).vcat(sep=1)


# %% [markdown]
# We can retrieve individual elements as well.

# %%
r[4]


# %% [markdown]
# Broadcasting can also be used with the subdiagrams mechanism.
# In this example we make a grid of squares and then fill in a
# couple of them.


# %%
grid = square(jnp.ones((5, 5))).named("grid").hcat().vcat()
grid

# %%
f = jnp.array([[1, 1], [2, 3]])
sub = grid.get_subdiagram("grid")
assert sub is not None
env = sub.get_envelope()
grid + circle(0.5).fill_color("red").translate_by(env.center[f[0, :], f[1, :]]).concat()
