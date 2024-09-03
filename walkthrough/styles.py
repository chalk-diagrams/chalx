# %% tags=["hide_inp"]
from chalk.core import BaseDiagram
from chalk import *


# %% [markdown]
# Diagrams can be styled using standard vector graphic style
# primitives. Colors use the Python [colour](https://github.com/vaab/colour) library.

# %%
from colour import Color

blue = Color("blue")
orange = Color("orange")

# %% [markdown]
# ### Diagram.fill_color

# %% tags=["hide_inp"]
help(BaseDiagram.fill_color)

# %% [markdown]
#

# %%
triangle(1).fill_color(blue)

# %% [markdown]
# ### Diagram.fill_opacity

# %% tags=["hide_inp"]
help(BaseDiagram.fill_opacity)

# %% [markdown]
#

# %%
triangle(1).fill_color(blue).fill_opacity(0.2)


# %% [markdown]
# ### Diagram.line_color

# %% tags=["hide_inp"]
help(BaseDiagram.line_color)

# %% [markdown]
#

# %%
triangle(1).line_color(blue)

# %% [markdown]
# ### Diagram.line_width

# %% tags=["hide_inp"]
help(BaseDiagram.line_width)

# %% [markdown]
#

# %%
triangle(1).line_width(0.05)


# %% [markdown]
# ### Diagram.dashing

# %% tags=["hide_inp"]
help(BaseDiagram.dashing)

# %% [markdown]
#

# %%
triangle(1).dashing([0.2, 0.1], 0)


# %% [markdown]
# ### Advanced Example


# %% [markdown]
# Example: Outer styles override inner styles

# %%
(triangle(1).fill_color(orange) | square(2)).fill_color(blue)
