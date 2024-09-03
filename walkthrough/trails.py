# %% tags=["hide_inp"]
from chalk import *

# %% [markdown]
# Chalk also includes a ``Trail`` primitive which allows for creating new ``Diagram``s and more complex shapes.
# ``Trail``s are specified by position-invariant offsets and can be rendered as ``Path``s.
# See the [Koch](../examples/koch/) example for a use case.

# %% [markdown]
# ### Constructors

# %% [markdown]
# ``Trail``s are a sequence of vectors and can be constructed using the ``from_offsets`` method:

# %%
trail = Trail.from_offsets([V2(1, 0), V2(1, 1), V2(0, 1)])

# %% [markdown]
# ### Converting to ``Diagram``

# %% [markdown]
# In order to render, `Trail`s have to be turned into ``Diagram``s, which can be achieved using the `stroke` method.

# %% tags=["hide_inp"]
help(Trail.Trail.stroke)

# %%
trail.stroke()

# %% [markdown]
# ### Transformations

# %% [markdown]
# `Trail`s can be transformed using the usual geometric transformations, which are also applied to the ``Diagram`` object.
# For example, ``Trail``s can be rotated:


# %%
trail2 = trail.rotate_by(0.2)
trail2.stroke()

# %% [markdown]
# However, since `Trail`s are translation invariant, applying the `translate` method leaves the `Trail` instance unchanged.

# %% [markdown]
# ### Composition

# %% [markdown]
# ``Trail``s form a monoid with addition given by list concatenation and identity given by the empty list.
# Intuitively, adding two ``Trails`` appends the two sequences of offsets.


# %%
(trail + trail2).stroke()
