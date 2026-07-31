# %%
# from jaxtyping import install_import_hook
# with install_import_hook("chalk", "typeguard.typechecked"):
#     import chalk
from chalk import *
import jax.numpy as jnp

# %%
env = circle(1).get_envelope()
print(V2(jnp.arange(1, 6), jnp.arange(1, 6)).shape)
env(V2(jnp.arange(1, 6), jnp.arange(1, 6)))


# %%
circle(1).show_envelope()


# %%
r = rectangle(2, 4)
env = r.get_envelope()
print(env.width, env.height)
assert int(env.width) == 2
assert int(env.height) == 4

# %%

# %%
r = r | rectangle(5, 3)
env = r.get_envelope()
assert int(env.width) == 2 + 5
assert int(env.height) == 4


# %%
c = circle(jnp.arange(1, 5)).translate(jnp.arange(1, 5), jnp.arange(1, 5))
c


# %%
c.align_l()
c.concat()
