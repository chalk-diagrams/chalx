# %% tags=["hide_inp"]
from chalk import *
from chalk.transform import polar


# %% [markdown]
# Complex diagrams can be created by combining simpler diagrams
# through placement combinators. These place diagrams above, atop or
# besides other diagrams. Relative location is determined by the envelope
# and origins of the diagrams.

# %% [markdown]
# ### beside

# %% tags=["hide_inp"]
help(beside)

# %% [markdown]
#

# %%
diagram = triangle(1).beside(square(1), unit_x)
diagram

# %% [markdown]
#

# %%
triangle(1).show_beside(square(1), unit_x)

# %%
triangle(1).show_beside(triangle(1).rotate_by(1 / 6), polar(-45))

# %% [markdown]
#

# %%
triangle(1).show_beside(triangle(1).rotate_by(1 / 6), polar(-30))


# %% [markdown]
# ### above

# %% tags=["hide_inp"]
help(above)

# %% [markdown]
#

# %%
diagram = triangle(1) / square(1)
diagram

# %% [markdown]
#

# %%
diagram.show_envelope().show_origin()


# %% [markdown]
# ### atop

# %% tags=["hide_inp"]
help(atop)

# %% [markdown]
# Example 1 - Atop at origin

# %%
diagram = square(1) + triangle(1)
diagram

# %% [markdown]
#

# %%
diagram.show_envelope().show_origin()

# %% [markdown]
# Example 2 - Align then atop origin

# %%
s = square(1).align_r().align_b().show_origin()
t = triangle(1).align_l().align_t().show_origin()
s

# %% [markdown]
#

# %%
t

# %% [markdown]
#

# %%
s + t

# %%

# %% [markdown]
# ### vcat

# %% tags=["hide_inp"]
help(vcat)

# %% [markdown]
#

# %%
vcat([triangle(1), square(1), triangle(1)], 0.2)

# %% [markdown]
# ### concat

# %% tags=["hide_inp"]
help(concat)

# %% [markdown]
#

# %%
concat([triangle(1), square(1), triangle(1)])


# %% [markdown]
# ### hcat

# %% tags=["hide_inp"]
help(hcat)

# %% [markdown]
#

# %%
hcat([triangle(1), square(1), triangle(1)], 0.2)

# %% [markdown]
# ### place_on_path

# %% tags=["hide_inp"]
help(place_on_path)

# %%
place_on_path(
    [circle(0.25) for _ in range(6)],
    Trail.regular_polygon(6, 1).to_path(),
)
