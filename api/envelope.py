# from jaxtyping import install_import_hook
# with install_import_hook("chalk", "typeguard.typechecked"):
#     import chalk 
from chalk import *
import numpy as np

env = circle(1).get_envelope()
print(V2(np.arange(1, 6), np.arange(1, 6)).shape)
env(V2(np.arange(1, 6), np.arange(1, 6)))


circle(1).show_envelope()


r = rectangle(2, 4)
env = r.get_envelope()
print(env.width, env.height)
assert int(env.width) == 2
assert int(env.height) == 4



r = r | rectangle(5, 3)
env = r.get_envelope()
assert int(env.width) == 2 + 5
assert int(env.height) == 4


c = circle(np.arange(1, 5)).translate(np.arange(1,5), np.arange(1,5))
c


c.align_l()
c.concat()