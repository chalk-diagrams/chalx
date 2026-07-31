"""JAX array types and helpers used throughout Chalk."""

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, TypeVar, Union

import jax
import jax.numpy as np
from jax import config
from jaxtyping import Array, Bool, Float, Int
from typing_extensions import Self

config.update("jax_enable_x64", True)  # type: ignore

JAX_MODE = True
jit = jax.jit
vmap = jax.vmap
vectorize = np.vectorize
ops = None

if TYPE_CHECKING:
    from typing import Annotated as Batched  # noqa: F401
else:
    from jaxtyping import AbstractDtype

    class Batched(AbstractDtype):
        dtypes = ["chalk"]


Batched

Scalars = Float[Array, "*#B"]
IntLike = Union[Int[Array, "*#B"], int]
BoolLike = Union[bool]
Ints = Int[Array, "*#B"]
Floating = Union[Scalars, IntLike, float, int]
Mask = Bool[Array, "*#B"]
MaskC = Bool[Array, "*#C"]
IntLikeC = Union[Int[Array, "*#C"], int]
ScalarsC = Float[Array, "*#C"]


@jit
@partial(vectorize, signature="()->()")
def ftos(f: Floating) -> Scalars:
    return np.asarray(f, dtype=np.float64)


def tree_map(fn, tree, *rest):  # type: ignore[no-untyped-def]
    """Like ``jax.tree.map``, treating hijax values as opaque leaves."""
    from chalk.core import BaseDiagram
    from chalk.diag import map_diag_prefix
    from chalk.envelope import Envelope
    from chalk.geom import Affine, Pt, Vec, data, make_p2_from_data, make_v2_from_data, make_xf
    from chalk.path import Path
    from chalk.segment import Segment, make_segment
    from chalk.style import StyleHolder, make_style
    from chalk.trace import Trace
    from chalk.trail import Located, Trail

    opaque = (
        StyleHolder,
        Segment,
        Trail,
        Located,
        Envelope,
        Trace,
        Path,
        BaseDiagram,
        Vec,
        Pt,
        Affine,
    )

    def wrapped(x, *xs):  # type: ignore[no-untyped-def]
        if x is None:
            return None
        if isinstance(x, StyleHolder):
            if xs:
                grouped = zip(x.lo_parts(), *[s.lo_parts() for s in xs])
                return make_style(*[fn(*parts) for parts in grouped])
            return x.map_prefix(fn)
        if isinstance(x, Segment):
            if xs:
                return make_segment(
                    fn(x.transform, *[s.transform for s in xs]),
                    fn(x.angles, *[s.angles for s in xs]),
                )
            return x.map_prefix(fn)
        if isinstance(x, BaseDiagram):
            return map_diag_prefix(x, fn)
        if isinstance(x, Vec):
            return (
                make_v2_from_data(fn(data(x), *[data(o) for o in xs]))
                if xs
                else make_v2_from_data(fn(data(x)))
            )
        if isinstance(x, Pt):
            return (
                make_p2_from_data(fn(data(x), *[data(o) for o in xs]))
                if xs
                else make_p2_from_data(fn(data(x)))
            )
        if isinstance(x, Affine):
            return (
                make_xf(fn(data(x), *[data(o) for o in xs]))
                if xs
                else make_xf(fn(data(x)))
            )
        if isinstance(x, (Trail, Located, Envelope, Trace, Path)):
            return x.map_prefix(fn)
        return fn(x, *xs)

    return jax.tree.map(
        wrapped,
        tree,
        *rest,
        is_leaf=lambda x: isinstance(x, opaque),
    )


def multi_vmap(fn: Callable[[Array], Array], t: int) -> Callable[[Array], Array]:
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
            return tree_map(
                lambda x: x[ind + (slice(None),) * (len(x.shape) - len(shape))], self
            )  # type: ignore
        return tree_map(lambda x: x[ind], self)  # type: ignore


def index_update(arr: Array, index: Any, values: Any) -> Array:  # type: ignore
    return arr.at[index].set(values)  # type: ignore


def prefix_broadcast(x: Array, target: Tuple[int, ...], suffix_length: int) -> Array:
    return np.broadcast_to(x, target + x.shape[-suffix_length:])


__all__ = []
