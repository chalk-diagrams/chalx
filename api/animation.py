# + tags=["hide"]

import numpy as np
from chalk import *

# -

c = (square(1) + circle(np.arange(1, 5)))
c.animate("test1.gif")


s = square(10) + (circle(np.arange(5, 1, -1)).translate_by(tx.scale_vec(unit_x, np.arange(1, 6)[:, None]))).concat()
print(s.shape)
s.animate("test2.gif")
