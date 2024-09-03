# %% tags=["hide"]

import numpy as np
from chalk import square, circle, tx, line, unit_x
from IPython.display import Image


# %%
line((0, np.arange(5, 1, -1)), (np.arange(1, 5), 0)).animate_svg("test0.svg")

# %%
c = square(5).fill_color("black") + circle(np.arange(1, 5))
c.animate("test1.gif")
c.animate_svg("test1.svg")


# %%
s = (
    square(10)
    + (
        circle(np.arange(5, 1, -1)).translate_by(
            tx.scale_vec(unit_x, np.arange(1, 6)[:, None])
        )
    ).concat()
)
print(s.shape)
s.animate("test2.gif")


# %%
s = (
    square(10)
    + (
        circle(np.arange(5, 1, -1)).translate_by(
            tx.scale_vec(unit_x, np.arange(1, 6)[:, None])
        )
    )
    .hcat()
    .vcat()
)

Image("test2.gif")
