""" """
# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     cell_metadata_filter: tags,title,-all
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
# ---

# %% tags=["hide_inp"]
from chalk.core import BaseDiagram
from chalk import triangle, rectangle, unit_x, arc


# %% [markdown]
"""
Each diagram has an origin and an envelope.
Manipulating the position of the diagram with respect to its origin and envelope allows for precise control of the layout.
Note that the Chalk API is immutable and always returns a new ``Diagram`` object.
"""

# %% [markdown]
"""
### Diagram.show_origin
"""

# %% tags=["hide_inp"]
help(BaseDiagram.show_origin)

# %% [markdown]
"""

"""


# %%
triangle(1).show_origin()

# %% [markdown]
"""
### Diagram.show_envelope
"""

# %% tags=["hide_inp"]
help(BaseDiagram.show_envelope)

# %% Cell 2
rectangle(1, 1).show_envelope()


# %%
triangle(1).show_envelope()


# %%
rectangle(1, 1).show_beside(triangle(1), unit_x)


# %%
(rectangle(1, 1) | triangle(1)).pad(1.4)


# %%
arc(1, 0, 90)

# %% [markdown]
"""
### Diagram.align_*
"""

# %% tags=["hide_inp"]
help(BaseDiagram.align_t)

# %% [markdown]
"""

"""

# %%
triangle(1).align_t().show_origin()

# %%

# %%
triangle(1).align_t().show_beside(rectangle(1, 1).align_b(), unit_x)


# %% tags=["hide_inp"]
help(BaseDiagram.align_r)

# %% [markdown]
"""
### Diagram.center_xy
"""

# %% tags=["hide_inp"]
help(BaseDiagram.center_xy)

# %% [markdown]
"""

"""

# %%
triangle(1).center_xy().show_envelope().show_origin()


# %% [markdown]
"""
### Diagram.pad_*
"""

# %% tags=["hide_inp"]
help(BaseDiagram.pad)


# %% [markdown]
"""

"""

# %%
triangle(1).pad(1.5).show_envelope().show_origin()


# %% [markdown]
"""
### Diagram.with_envelope
"""

# %% tags=["hide_inp"]
help(BaseDiagram.with_envelope)


# %% [markdown]
"""

"""


# %%
(rectangle(1, 1) + triangle(0.5)) | rectangle(1, 1)


# %%
(rectangle(1, 1) + triangle(0.5)).with_envelope(triangle(0.5)) | rectangle(
    1, 1
).fill_color("red")
