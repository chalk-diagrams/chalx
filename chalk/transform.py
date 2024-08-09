"""
Defines the core geortric and shapes for the
chalk library. In previous versions this was a
separate library called `planar`.
"""

import functools
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Tuple, TypeVar, Union, Any

import array_api_compat.numpy as onp
import jax

from jaxtyping import Bool, Float, Int
from typing_extensions import Self, TypeAlias


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
    def jit(x: Callable[..., Array]) -> Callable[..., Array]:
        return x

    def vectorize(x: Any, signature: str, excluded: List[int] =[]) -> Callable[..., Array]:
        return x # type: ignore
    
    def vmap(fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
        "Fake jax vmap for numpy as a for loop."
        if JAX_MODE:
            return vmap(fn)

        def vmap2(x: Array) -> Array:
            if isinstance(x, tuple):
                size = x[-1].size() # type: ignore
            elif isinstance(x, np.ndarray): # type: ignore
                size = x.shape
            else:
                size = x.size()
            ds = []
            for k in range(size[0]):
                d = jax.tree_map(lambda x: x[k], x)
                ds.append(fn(d))
            final = jax.tree_map(lambda *x: np.stack(x, 0), *ds)

            return final # type: ignore

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
    config.update("jax_debug_nans", True)  # type: ignore

    

# Core shaped types used throughout
# *#B means arbitrary or no batch dimension
Affine = Float[Array, "*#B 3 3"]
Angles = Float[Array, "*#B 2"]
V2_t: TypeAlias = Float[Array, "*#B 3 1"]

P2_t = Float[Array, "*#B 3 1"]
Scalars = Float[Array, "*#B"]
IntLike = Union[Int[Array, "*#B"], int, onp.int64]
Ints = Int[Array, "*#B"]
Floating = Union[Scalars, IntLike, float, int, onp.int64, onp.float64]
Mask = Bool[Array, "*#B"]
ColorVec = Float[Array, "#*B 3"]
Property = Float[Array, "#*B"]

V2_tC = Float[Array, "*#C 3 1"]
IntLikeC = Union[Int[Array, "*#C"], int, onp.int64]
ScalarsC = Float[Array, "*#C"]


def index_update(arr: Array, index, values) -> Array:  # type:ignore
    """
    Update the array `arr` at the given `index` with `values`
    and return the updated array.
    Supports both NumPy and JAX arrays.
    """
    if isinstance(arr, onp.ndarray): # type: ignore
        # If the array is a NumPy array
        new_arr = arr.copy()
        new_arr[index] = values
        return new_arr  # type:ignore
    else:
        # If the array is a JAX array
        arr: Array = arr.at[index].set(values) # type: ignore
        return arr


def union(
    x: Tuple[Array, Array], y: Tuple[Array, Array]
) -> Tuple[Array, Array]:
    """
    Union two pairs arrays along the second axis.
    And optionally filter by mask (numpy only).
    Used by trace compose.
    """
    if isinstance(x, onp.ndarray):
        n1 = np.concatenate([x[0], y[0]], axis=1)
        m = np.concatenate([x[1], y[1]], axis=1)
        return n1, m
    else:
        n1 = np.concatenate([x[0], y[0]], axis=1)
        m = np.concatenate([x[1], y[1]], axis=1)
        return n1, m


def union_axis(
    x: Tuple[Array, Array], axis: int
) -> Tuple[Array, Array]:
    n = [
        np.squeeze(x, axis=axis)
        for x in np.split(x[0], x[0].shape[axis], axis=axis)
    ]
    m = [
        np.squeeze(x, axis=axis)
        for x in np.split(x[1], x[1].shape[axis], axis=axis)
    ]
    ret = functools.reduce(union, zip(n, m))
    return ret


# Basic geometric primitives
unit_x: V2_t = np.asarray([1.0, 0.0, 0.0]).reshape((3, 1))
unit_y: V2_t = np.asarray([0.0, 1.0, 0.0]).reshape((3, 1))
origin: P2_t = np.asarray([0.0, 0.0, 1.0]).reshape((3, 1))
ident: Affine = np.asarray(
    [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 1]]]
).reshape((3, 3))


def make_ident(shape: Tuple[int, ...]) -> Affine:
    "Create an identity affine with the given shape."
    return np.broadcast_to(ident, shape + (3, 3))


@jit
@partial(vectorize, signature="()->()")
def ftos(f: Floating) -> Scalars:
    "Map a float to an array format."
    return np.asarray(f, dtype=np.double)

# @partial(vectorize, signature="(),()->(3,1)")
def V2(x: Floating, y: Floating) -> V2_t:
    "Map (x,y) of any shape to a (batched) vector."
    if isinstance(x, float) and isinstance(y, float):
        return np.array([x, y, 0]).reshape(3, 1)

    x, y, o = np.broadcast_arrays(ftos(x), ftos(y), ftos(0.0))
    s: V2_t = np.stack([x, y, o], axis=-1)[..., None]
    return s


# @partial(vectorize, signature="(),()->(3,1)")
def P2(x: Floating, y: Floating) -> P2_t:
    "Map (x,y) of any shape to a (batched) point."
    x, y, o = np.broadcast_arrays(ftos(x), ftos(y), ftos(1.0))
    s: P2_t = np.stack([x, y, o], axis=-1)[..., None]
    return s


def norm(v: V2_t) -> V2_t:
    v = v / length(v)[..., None, None]
    return v
    


def length(v: V2_t) -> Scalars:
    "Length of a vector"
    return np.asarray(np.sqrt(length2(v)))

@partial(vectorize, signature="(3,1),()->(3,1)")
def scale_vec(v: V2_t, d: Floating) -> V2_t:
    "Scale a vector by a scalar"
    d = np.asarray(d)
    v = d[..., None, None] * v
    return v


@jit
@partial(vectorize, signature="(3,1)->()")
def length2(v: V2_t) -> Scalars:
    "Length^2 of a vector"
    return np.asarray((v * v)[..., :2, 0].sum(-1))


@jit
@partial(vectorize, signature="(3,1)->()")
def angle(v: V2_t) -> Scalars:
    "Angle of a vector in degrees"
    return np.asarray(from_rad * rad(v))


@jit
@partial(vectorize, signature="(3,1)->()")
def rad(v: P2_t) -> Scalars:
    "Angle of a vector in radians"
    return np.asarray(np.arctan2(v[..., 1, 0], v[..., 0, 0]))


@jit
@partial(vectorize, signature="(3,1)->(3,1)")
def perpendicular(v: V2_t) -> V2_t:
    "Perpendicular of a vector"
    return np.stack([-v[..., 1, :], v[..., 0, :], v[..., 2, :]], axis=-2)


@jit
@partial(vectorize, signature="(),(),(),(),(),()->(3,3)")
def make_affine(
    a: Floating,
    b: Floating,
    c: Floating,
    d: Floating,
    e: Floating,
    f: Floating,
) -> Affine:
    "Create affine array from values"
    vals = list([ftos(x) for x in [a, b, c, d, e, f, 0.0, 0.0, 1.0]])
    vals = np.broadcast_arrays(*vals) # type: ignore
    x = np.stack(vals, axis=-1)
    r : Affine = x.reshape(vals[0].shape + (3, 3))
    return r


@jit
@partial(vectorize, signature="(3,1),(3,1)->()")
def dot(v1: V2_t, v2: V2_t) -> Scalars:
    "Dot of vectors"
    return np.asarray((v1 * v2).sum(-1).sum(-1))


@jit
@partial(vectorize, signature="(3,1),(3,1)->()")
def cross(v1: V2_t, v2: V2_t) -> Scalars:
    "Cross of vectors"
    s: Scalars = np.cross(v1, v2)
    return s


def to_point(v: V2_t) -> P2_t:
    "Convert a vector to a point (allows transpose)"
    index = (Ellipsis, 2, 0)
    return index_update(v, index, 1)  # type: ignore


@jit
@partial(vectorize, signature="(3,1)->(3,1)")
def to_vec(p: P2_t) -> V2_t:
    "Convert a point to a vector (disallows transpose)"
    index = (Ellipsis, 2, 0)
    return index_update(p, index, 0)  # type: ignore


@jit
@partial(vectorize, signature="()->(3,1)")
def polar(angle: Floating) -> V2_t:
    "Angle in degress to a vector"
    rad = to_radians(angle)
    x, y = np.cos(rad), np.sin(rad)
    return V2(x, y)


@jit
@partial(vectorize, signature="(3,1)->(3,3)")
def scale(vec: V2_t) -> Affine:
    "Create a affine scale matrix"
    base = make_ident(vec.shape[:-2])
    index = (Ellipsis, np.arange(0, 2), np.arange(0, 2))
    return index_update(base, index, vec[..., :2, 0])  # type: ignore


@jit
@partial(vectorize, signature="(3,1)->(3,3)")
def translation(vec: V2_t) -> Affine:
    "Create an affine translation matrix"
    index = (Ellipsis, slice(0, 2), 2)
    base = make_ident(vec.shape[:-2])
    return index_update(base, index, vec[..., :2, 0])  # type: ignore


@jit
@partial(vectorize, signature="(3,3)->(3,1)")
def get_translation(aff: Affine) -> V2_t:
    "Get the translation of an affine matrix"
    index = (Ellipsis, slice(0, 2), 0)
    base = np.zeros((aff.shape[:-2]) + (3, 1))
    return index_update(base, index, aff[..., :2, 2])  # type: ignore


def rotation(r: Floating) -> Affine:
    "Create an affine rotation matrix in radians"
    rad = ftos(r)
    shape = rad.shape
    rad = -rad
    ca, sa = np.cos(rad), np.sin(rad)
    m = np.stack([ca, -sa, sa, ca], axis=-1).reshape(shape + (2, 2))
    index = (Ellipsis, slice(0, 2), slice(0, 2))
    base = make_ident(shape)
    return index_update(base, index, m)  # type: ignore


def rotation_angle(r: Floating) -> Affine:
    "Create an affine rotation matrix in degrees"
    return rotation(to_radians(r))


@partial(vectorize, signature="(3,3)->(3,3)")
@jit
def inv(aff: Affine) -> Affine:
    "Fast invert an affine"
    det = np.linalg.det(aff)
    if not JAX_MODE:
        assert np.all(np.abs(det) > 1e-10), "Object scaled to 0"
    idet = 1.0 / det
    sa, sb, sc = aff[..., 0, 0], aff[..., 0, 1], aff[..., 0, 2]
    sd, se, sf = aff[..., 1, 0], aff[..., 1, 1], aff[..., 1, 2]
    ra = se * idet
    rb = -sb * idet
    rd = -sd * idet
    re = sa * idet
    vals = (
        ra,
        rb,
        -sc * ra - sf * rb,
        rd,
        re,
        -sc * rd - sf * re,
        np.zeros(ra.shape),
        np.zeros(ra.shape),
        np.ones(ra.shape),
    )
    x = np.stack(vals, axis=-1)
    r : Affine = x.reshape(vals[0].shape + (3, 3))
    return r


from_rad = 180 / math.pi


@jit
@partial(vectorize, signature="()->()")
def from_radians(θ: Floating) -> Scalars:
    return np.asarray(ftos(θ) * from_rad)


@jit
@partial(vectorize, signature="()->()")
def to_radians(θ: Floating) -> Scalars:
    return np.asarray((ftos(θ) / 180) * math.pi)


@jit
@partial(vectorize, signature="(3,3)->(3,3)")
def remove_translation(aff: Affine) -> Affine:
    "Remove translation from affine"
    index = (Ellipsis, slice(0, 1), 2)
    return index_update(aff, index, 0)  # type: ignore

@jit
@partial(vectorize, signature="(3,3)->(3,3)")
def remove_scale(aff: Affine) -> Affine:
    "Remove scaling from affine"
    index = (Ellipsis, slice(0, 2), slice(0, 2))
    det = np.linalg.det(aff[index])
    return index_update(aff, index, aff[index] / np.sqrt(det[..., None, None]))  # type: ignore


@jit
@partial(vectorize, signature="(3,3)->(3,3)")
def transpose_translation(aff: Affine) -> Affine:
    index = (Ellipsis, slice(0, 2), slice(0, 2))
    swap = aff[..., :2, :2].swapaxes(-1, -2)
    return index_update(aff, index, swap)  # type: ignore


class Transformable:
    """
    Syntactic sugar to apply transformations to objects
    as methods. Creates matrices and applies them.
    """

    def apply_transform(self, t: Affine) -> Self:  # type: ignore[empty-body]
        pass

    def __rmatmul__(self, t: Affine) -> Self:
        return self._app(t)

    def __rmul__(self, t: Affine) -> Self:
        return self._app(t)

    def _app(self, t: Affine) -> Self:
        return self.apply_transform(t)

    def scale(self, α: Floating) -> Self:
        return self._app(scale(V2(α, α)))

    def scale_x(self, α: Floating) -> Self:
        return self._app(scale(V2(α, 1)))

    def scale_y(self, α: Floating) -> Self:
        return self._app(scale(V2(1, α)))

    def rotate(self, θ: Floating) -> Self:
        "Rotate by θ degrees counterclockwise"
        return self._app(rotation(to_radians(θ)))

    def rotate_rad(self, θ: Floating) -> Self:
        "Rotate by θ radians counterclockwise"
        return self._app(rotation((θ)))

    def rotate_by(self, turns: Floating) -> Self:
        "Rotate by fractions of a circle (turn)."
        θ = 2 * math.pi * turns
        return self._app(rotation((θ)))

    def reflect_x(self) -> Self:
        return self._app(scale(V2(-1, +1)))

    def reflect_y(self) -> Self:
        return self._app(scale(V2(+1, -1)))

    def shear_y(self, λ: Floating) -> Self:
        return self._app(make_affine(1.0, 0.0, 0.0, λ, 1.0, 0.0))

    def shear_x(self, λ: Floating) -> Self:
        return self._app(make_affine(1.0, λ, 0.0, 0.0, 1.0, 0.0))

    def translate(self, dx: Floating, dy: Floating) -> Self:
        return self._app(translation(V2(dx, dy)))

    def translate_by(self, vector) -> Self:  # type: ignore
        return self._app(translation(vector))


@dataclass
class Ray:
    pt: P2_t
    v: V2_t

    def point(self, len: Scalars) -> P2_t:
        p: P2_t = self.pt + len[..., None, None] * self.v
        return p


@dataclass
class BoundingBox(Transformable):
    tl: P2_t
    br: P2_t

    def apply_transform(self, t: Affine) -> Self:  # type: ignore
        # Todo: fix rotation
        tl = t @ self.tl
        br = t @ self.br
        tl2 = np.minimum(tl, br)
        br2 = np.maximum(tl, br)
        return BoundingBox(tl2, br2)  # type: ignore

    @property
    def width(self) -> Scalars:
        s:Scalars = (self.br - self.tl)[..., 0, 0]
        return s

    @property
    def height(self) -> Scalars:
        s: Scalars = (self.br - self.tl)[..., 1, 0]
        return s

    def to_rect(self) -> "Diagram": # type: ignore
        "Convert bounding box to a rectangle"
        from chalk import rectangle

        return (
            rectangle(self.width, self.height)
            .align_tl()
            .translate(-self.width / 2, -self.height / 2)
        )


@partial(vectorize, signature="(3,1),(3,1),()->(),()") # type: ignore
def ray_circle_intersection(
    anchor: P2_t, direction: V2_t, circle_radius: Floating
) -> Tuple[Scalars, Mask]:
    """Given a ray and a circle centered at the origin, return the parameter t
    where the ray meets the circle, that is:

    ray t = circle θ

    The above equation is solved as follows:

    x + t v_x = r sin θ
    y + t v_y = r cos θ

    By squaring the equations and adding them we get

    (x + t v_x)² + (y + t v_y)² = r²,

    which is equivalent to the following equation:

    (v_x² + v_y²) t² + 2 (x v_x + y v_y) t + (x² + y² - r²) = 0

    This is a quadratic equation, whose solutions are well known.

    """
    a = length2(direction)
    b = 2 * dot(anchor, direction)
    c = length2(anchor) - circle_radius**2
    Δ = b**2 - 4 * a * c
    eps = 1e-10  # rounding error tolerance

    mid = (((-eps <= Δ) & (Δ < 0)))[..., None]
    mask = (Δ < -eps)[..., None] | (mid * np.asarray([1, 0]))

    # Bump NaNs since they are going to me masked out.
    ret: Array = np.stack(
        [
            (-b - np.sqrt(Δ + 1e9 * mask[..., 0])) / (2 * a),
            (-b + np.sqrt(np.where(mid[..., 0], 0, Δ) + 1e9 * mask[..., 1]))
            / (2 * a),
        ],
        -1,
    )

    ret2 = np.where(mid, (-b / (2 * a))[..., None], ret)
    assert not isinstance(ret2, tuple)
    assert len(ret2.shape) == 3, ret2.shape
    return ret2.transpose(2, 0, 1), 1 - mask.transpose(2, 0, 1)


@partial(vectorize, excluded=[2], signature="(),()->(a,3,1)")
def arc_to_bezier(theta1: Array, theta2:Array, n:int=2) -> Array:
    """
    Returns the bezier curves for the unit circle arc from angles *theta1* to
    *theta2* (in degrees).

    *theta2* is unwrapped to produce the shortest arc within 360 degrees.
    That is, if *theta2* > *theta1* + 360, the arc will be from *theta1* to
    *theta2* - 360 and not a full circle plus some extra overlap.

    If *n* is provided, it is the number of spline segments to make.
    If *n* is not provided, the number of spline segments is
    determined based on the delta between *theta1* and *theta2*.

        Masionobe, L.  2003.  `Drawing an elliptical arc using
        polylines, quadratic or cubic Bezier curves
        <https://web.archive.org/web/20190318044212/http://www.spaceroots.org/documents/ellipse/index.html>`_.
    """
    theta1, theta2 = np.broadcast_arrays(theta1, theta2)
    extra = theta1.shape
    eta1 = theta1
    eta2 = theta2 - 360 * np.floor((theta2 - theta1) / 360)
    # Ensure 2pi range is not flattened to 0 due to floating-point errors,
    # but don't try to expand existing 0 range.
    eta2 = np.where((theta2 != theta1) & (eta2 <= eta1), eta2 + 360, eta2)
    eta1, eta2 = to_radians(eta1), to_radians(eta2)

    deta = (eta2 - eta1) / n
    t = np.tan(0.5 * deta)
    alpha = np.sin(deta) * (np.sqrt(4.0 + 3.0 * t * t) - 1) / 3.0
    alpha = alpha[..., None]

    steps = np.linspace(eta1, eta2, n + 1, axis=-1)
    cos_eta = np.cos(steps)
    sin_eta = np.sin(steps)

    xA = cos_eta[..., :-1]
    yA = sin_eta[..., :-1]
    xA_dot = -yA
    yA_dot = xA

    xB = cos_eta[..., 1:]
    yB = sin_eta[..., 1:]
    xB_dot = -yB
    yB_dot = xB

    length = n * 3

    vertices = np.ones(extra + (length, 3, 1))
    vertex_offset = 0
    end = length

    vertices = index_update(
        vertices,
        (Ellipsis, slice(vertex_offset, end, 3), 0, 0),
        xA + alpha * xA_dot,
    )
    vertices = index_update(
        vertices,
        (Ellipsis, slice(vertex_offset, end, 3), 1, 0),
        yA + alpha * yA_dot,
    )
    vertices = index_update(
        vertices,
        (Ellipsis, slice(vertex_offset + 1, end, 3), 0, 0),
        xB - alpha * xB_dot,
    )
    vertices = index_update(
        vertices,
        (Ellipsis, slice(vertex_offset + 1, end, 3), 1, 0),
        yB - alpha * yB_dot,
    )
    vertices = index_update(
        vertices, (Ellipsis, slice(vertex_offset + 2, end, 3), 0, 0), xB
    )
    vertices = index_update(
        vertices, (Ellipsis, slice(vertex_offset + 2, end, 3), 1, 0), yB
    )
    return vertices




def multi_vmap(fn: Callable[[Array], Array], t: int) -> Callable[[Array], Array]:
    "Apply vmap t times"
    for _ in range(t):
        fn = vmap(fn)
    return fn


tree_map = jax.tree.map


if TYPE_CHECKING:
    from typing import Annotated as Batched # noqa: F401
    
else:
    from jaxtyping import AbstractDtype
    class Batched(AbstractDtype):
        dtypes = ["chalk"]

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
        if isinstance(ind, tuple) and Ellipsis in ind: # type: ignore
            # We only want ... to apply to the prefix args
            return jax.tree_map(lambda x: x[ind + (slice(None),) * (len(x.shape) - len(shape))], self) # type: ignore
        else:
            return jax.tree_map(lambda x: x[ind], self) # type: ignore


def prefix_broadcast(x: Array, target: Tuple[int, ...], suffix_length: int) -> Array:
    return np.broadcast_to(x, target + x.shape[-suffix_length:])

# Explicit rexport
__all__ = ["Array", "np", "jit", "vmap", "multi_vmap", "tree_map", "Batchable", "Batched"]
