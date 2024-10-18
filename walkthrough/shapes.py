# %% tags=["hide"]
from chalk import *


# %% [markdown]
# Elementary diagrams can be created using shapes:
# polygons, circle-like shapes, text and paths.

# %%
triangle(1)


# %%
square(1)



# %%
rectangle(3, 1)


# %%
hcat(
    [
        regular_polygon(5, 1),
        regular_polygon(6, 1),
        regular_polygon(7, 1),
    ],
    sep=0.5,
)



# %%
circle(1)

# %% [markdown]
# Arcs can be specified either using angles (see ``arc``) or points (see ``arc_between``).



# %%
quarter = 90
arc(1, 0, quarter)

# %%
arc(1, 0, quarter) + arc(1, 2 * quarter, 3 * quarter)



# %%
arc_between((0, 0), (1, 0), 1)




# %% [markdown]
# Note that unlike other shapes, ``text`` has an empty envelope, so we need to explicitly specify it in order to get a non-empty rendering.

# %%
text("hello", 1).with_envelope(rectangle(2.5, 1))

# %%
make_path([(0, 0), (0, 1), (1, 1), (1, 2)])
