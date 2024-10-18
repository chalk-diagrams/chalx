"""Code to ensure compatibility between
JAX and NumPy. This includes type compatibility and
some additional function that are not in NumPy.

These types are exported through Transform.

TODO: Currently the numpy code still requires the use of
jax for the PyTree functionality. We will move to OpTree once it
support dataclasses.
"""

import os
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, TypeVar, Union

import jax
import numpy as onp
from jaxtyping import Bool, Float, Int
from typing_extensions import Self

# TODO: This is a bit hacky, but not sure
# how to make things work with both numpy and jax
# Use an environment variable to switch between the two.
if TYPE_CHECKING or not eval(os.environ.get("CHALK_JAX", "0")):
    ops = None
    import numpy as np
    from numpy import ndarray

    JAX_MODE = False
    A = TypeVar("A")
    B = TypeVar("B", bound=np.float64)
    Array = ndarray[A, np.dtype[B]]

    # In numpy mode jit and vectorize are no-ops
    C = TypeVar("C")

    def jit(x: Callable[..., C]) -> Callable[..., C]:
        return x

    def vectorize(
        x: Any, signature: str, excluded: List[int] = []
    ) -> Callable[..., Array]:
        return x  # type: ignore

    def vmap(fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
        """Fake jax vmap for numpy as a for loop."""
        if JAX_MODE:
            return vmap(fn)

        def vmap2(x: Array) -> Array:
            if isinstance(x, tuple):
                size = x[-1].size()  # type: ignore
            elif isinstance(x, np.ndarray):  # type: ignore
                size = x.shape
            else:
                size = x.size()
            ds = []
            for k in range(size[0]):
                d = jax.tree_map(lambda x: x[k], x)
                ds.append(fn(d))
            final = jax.tree_map(lambda *x: np.stack(x, 0), *ds)

            return final  # type: ignore

        return vmap2

else:
    jit = jax.jit
    import jax.numpy as np
    from jax import config
    from jaxtyping import Array

    vectorize = np.vectorize
    vmap = jax.vmap
    JAX_MODE = True
    config.update("jax_enable_x64", True)  # type: ignore
    # config.update("jax_debug_nans", True)  # type: ignore

if TYPE_CHECKING:
    from typing import Annotated as Batched  # noqa: F401

else:
    from jaxtyping import AbstractDtype

    class Batched(AbstractDtype):
        dtypes = ["chalk"]


Batched

Scalars = Float[Array, "*#B"]
IntLike = Union[Int[Array, "*#B"], int, onp.int64]
BoolLike = Union[bool]
Ints = Int[Array, "*#B"]
Floating = Union[Scalars, IntLike, float, int, onp.int64, onp.float64]
"""A float or float array"""

Mask = Bool[Array, "*#B"]
MaskC = Bool[Array, "*#C"]
IntLikeC = Union[Int[Array, "*#C"], int, onp.int64]
ScalarsC = Float[Array, "*#C"]


@jit
@partial(vectorize, signature="()->()")
def ftos(f: Floating) -> Scalars:
    """Map a float to an array format."""
    return np.asarray(f, dtype=np.double)


tree_map = jax.tree.map


def multi_vmap(fn: Callable[[Array], Array], t: int) -> Callable[[Array], Array]:
    """Apply vmap t times"""
    for _ in range(t):
        fn = vmap(fn)
    return fn


class Batchable:
    @property
    def dtype(self) -> str:
        return "chalk"

    @property
    def shape(self) -> Tuple[int, ...]:
        assert False

    def size(self) -> Tuple[int, ...]:
        return self.shape

    def __getitem__(self, ind: int | Tuple[int, ...]) -> Self:
        shape = self.shape
        if isinstance(ind, tuple) and Ellipsis in ind:  # type: ignore
            # We only want ... to apply to the prefix args
            return jax.tree_map(
                lambda x: x[ind + (slice(None),) * (len(x.shape) - len(shape))], self
            )  # type: ignore
        else:
            return jax.tree_map(lambda x: x[ind], self)  # type: ignore


def index_update(arr: Array, index: Any, values: Any) -> Array:  # type:ignore
    """Update the array `arr` at the given `index` with `values`
    and return the updated array.
    Supports both NumPy and JAX arrays.
    """
    if isinstance(arr, onp.ndarray):  # type: ignore
        # If the array is a NumPy array
        new_arr = arr.copy()
        new_arr[index] = values
        return new_arr  # type:ignore
    else:
        # If the array is a JAX array
        arr: Array = arr.at[index].set(values)  # type: ignore
        return arr


def prefix_broadcast(x: Array, target: Tuple[int, ...], suffix_length: int) -> Array:
    return np.broadcast_to(x, target + x.shape[-suffix_length:])


__all__ = []
