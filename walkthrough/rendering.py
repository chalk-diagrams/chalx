# %% tags=["hide"]

from chalk.core import BaseDiagram
from chalk import circle, vcat, set_svg_height
from IPython.display import Image

# %% [markdown]
# Chalk supports three back-ends (Cairo, SVG, TikZ),
# which allow the created `Diagram`s to be rendered as PNG, SVG, PDF files, respectively.
# The three corresponding methods for rendering are: `render`, `render_svg`, `render_pdf`;
# these are documented below.
#
# ### Diagram.render

# %% tags=["hide_inp"]
help(BaseDiagram.render)

# %%
circle(1).render("circle.png")
Image("circle.png")  # type: ignore

# %% [markdown]
# ### Diagram.render_svg

# %% tags=["hide_inp"]
help(BaseDiagram.render_svg)

# %% [markdown]
# ### Diagram.render_pdf

# %% tags=["hide_inp"]
help(BaseDiagram.render_pdf)

# %% [markdown]
# ### ``Diagram``s in IPython notebooks

# %% [markdown]
# When a ``Diagram`` is used in an IPython notebook, it is automatically displayed as an SVG.
# To adjust the height of the generated image, one can use the `set_svg_height` function:

# %% tags=["hide_inp"]
help(set_svg_height)

# %% [markdown]
# This function is particularly useful for showing tall drawings:

# %%
set_svg_height(500)
vcat([circle(1) for _ in range(5)])
