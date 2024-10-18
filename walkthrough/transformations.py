# %% tags=["hide_inp"]
from chalk.core import BaseDiagram
from chalk import *


# %% [markdown]
# Any Diagram (or other object in Chalk) can be transformed by affine transformation.
# These produce a new diagram in the standard manner.

# %% [markdown]
# ### scale



# %% [markdown]
#

# %%
triangle(1) | triangle(1).scale(2)

# %% [markdown]
# Transformations apply to the whole diagram.

# %%
(triangle(1) | triangle(1)).scale(2)


# %% [markdown]
# ### translate



# %% [markdown]
#

# %%
triangle(1).translate(1, 1).show_envelope().show_origin()

# %% [markdown]
#

# %%
triangle(1) + triangle(1).translate(1, 1)

# %% [markdown]
# ### shear_x



# %% [markdown]
#

# %%
square(1).shear_x(0.25).show_envelope()

# %% [markdown]
#

# %%
square(1) | square(1).shear_x(0.25)

# %% [markdown]
# ### rotate



# %% [markdown]
#

# %%
triangle(1) | triangle(1).rotate(90)

# %% [markdown]
# ### rotate_by



# %% [markdown]
#

# %%
triangle(1) | triangle(1).rotate_by(0.2)
