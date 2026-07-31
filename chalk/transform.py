"""Defines the core geortric and shapes for the
chalk library. In previous versions this was a
separate library called `planar`.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Any

import jax

from chalk.array_types import (
    JAX_MODE,
    Array,
    Batchable,
    Batched,
    BoolLike,
    Floating,
    IntLike,
    IntLikeC,
    Ints,
    Mask,
    MaskC,
    Scalars,
    ftos,
    index_update,
    jit,
    multi_vmap,
    np,
    prefix_broadcast,
    tree_map,
    vectorize,
    vmap,
    onp,
)
from jaxtyping import Float
from typing_extensions import Self

import chalk.geom as geom
from chalk.geom import Affine, Pt, Vec, data

# Core shaped types used throughout
# *#B means arbitrary or no batch dimension
#

AffineArr = Float[Array, "*#B 3 3"]
Angles = Float[Array, "*#B 2"]
"""An arc specified by start and angle diff."""

V2_t = Vec
P2_t = Pt
V2_tC = Float[Array, "*#C 3 1"]
P2_tC = Float[Array, "*#C 3 1"]

ColorVec = Float[Array, "#*B 3"]
"""A batch of RGB colors"""


Property = Float[Array, "#*B"]

# Homogeneous array constants (for jitted / expand code).
_unit_x_arr = np.asarray([1.0, 0.0, 0.0]).reshape((3, 1))
_unit_y_arr = np.asarray([0.0, 1.0, 0.0]).reshape((3, 1))
_origin_arr = np.asarray([0.0, 0.0, 1.0]).reshape((3, 1))
_ident_arr = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

unit_x: V2_t = geom.unit_x
unit_y: V2_t = geom.unit_y
origin: P2_t = geom.origin
ident: Affine = geom.ident


def make_ident(shape: Tuple[int, ...]) -> Affine:
    """Create an identity affine with the given shape."""
    return geom.make_ident(shape)


def V2(x: Floating, y: Floating) -> V2_t:
    """Map (x,y) of any shape to a (batched) vector."""
    x, y = np.broadcast_arrays(ftos(x), ftos(y))
    return geom.make_v2(x, y)


def P2(x: Floating, y: Floating) -> P2_t:
    """Map (x,y) of any shape to a (batched) point."""
    x, y = np.broadcast_arrays(ftos(x), ftos(y))
    return geom.make_p2(x, y)


def _as_v2(v) -> Vec:
    """Lift ``v2[]``, ``p2[]``, or a homogeneous ``(..., 3, 1)`` array to ``v2[]``."""
    if isinstance(v, Vec):
        return v
    if isinstance(v, Pt):
        return geom.to_vec(v)
    ty = jax.typeof(v)
    if isinstance(ty, geom.V2Ty):
        return v
    if isinstance(ty, geom.P2Ty):
        return geom.to_vec(v)
    arr = np.asarray(v)
    if arr.ndim >= 2 and arr.shape[-2:] == (3, 1):
        return geom.make_v2_from_data(arr.at[..., 2, 0].set(0.0))
    raise TypeError(f"expected v2[] or p2[], got {ty}")


def _as_xf(t) -> Affine:
    if isinstance(t, Affine):
        return t
    ty = jax.typeof(t)
    if isinstance(ty, geom.XfTy):
        return t
    arr = np.asarray(t)
    if arr.ndim >= 2 and arr.shape[-2:] == (3, 3):
        return geom.make_xf(arr)
    raise TypeError(f"expected xf[], got {ty}")


@jit
def to_P2(x: Float[Array, "*B 2"]) -> P2_t:
    """Map a standard vector to a point."""
    _, o = np.broadcast_arrays(x[..., :1], ftos(1.0))
    s = np.concatenate([x, o], axis=-1)[..., None]
    return geom.make_p2_from_data(s)


def norm(v: V2_t) -> V2_t:
    v = _as_v2(v)
    return geom.v2_scale(v, 1.0 / geom.length(v))


def length(v: V2_t) -> Scalars:
    """Length of a vector"""
    return geom.length(_as_v2(v))


def scale_vec(v: V2_t, d: Floating) -> V2_t:
    """Scale a vector by a scalar"""
    return geom.v2_scale(_as_v2(v), d)


def length2(v: V2_t) -> Scalars:
    """Length^2 of a vector"""
    return geom.length2(_as_v2(v))


@jit
@partial(vectorize, signature="(3,1)->()")
def _angle_arr(v) -> Scalars:
    return np.asarray(from_rad * np.arctan2(v[..., 1, 0], v[..., 0, 0]))


def angle(v: V2_t) -> Scalars:
    """Angle of a vector in degrees"""
    return _angle_arr(data(v))


@jit
@partial(vectorize, signature="(3,1)->()")
def _rad_arr(v) -> Scalars:
    return np.asarray(np.arctan2(v[..., 1, 0], v[..., 0, 0]))


def rad(v: P2_t) -> Scalars:
    """Angle of a vector in radians"""
    return _rad_arr(data(v))


@jit
@partial(vectorize, signature="(3,1)->(3,1)")
def _perpendicular_arr(v):
    return np.stack([-v[..., 1, :], v[..., 0, :], v[..., 2, :]], axis=-2)


def perpendicular(v: V2_t) -> V2_t:
    """Perpendicular of a vector"""
    return geom.make_v2_from_data(_perpendicular_arr(data(_as_v2(v))))


@jit
@partial(vectorize, signature="(),(),(),(),(),()->(3,3)")
def _make_affine_arr(a, b, c, d, e, f):
    vals = list([ftos(x) for x in [a, b, c, d, e, f, 0.0, 0.0, 1.0]])
    vals = np.broadcast_arrays(*vals)  # type: ignore
    x = np.stack(vals, axis=-1)
    return x.reshape(vals[0].shape + (3, 3))


def make_affine(
    a: Floating,
    b: Floating,
    c: Floating,
    d: Floating,
    e: Floating,
    f: Floating,
) -> Affine:
    """Create affine from values"""
    return geom.make_xf(_make_affine_arr(a, b, c, d, e, f))


@jit
@partial(vectorize, signature="(3,1),(3,1)->()")
def _dot_arr(v1, v2) -> Scalars:
    return np.asarray((v1 * v2).sum(-1).sum(-1))


def dot(v1: V2_t, v2: V2_t) -> Scalars:
    """Dot of vectors"""
    return _dot_arr(data(v1), data(v2))


@jit
@partial(vectorize, signature="(3,1),(3,1)->()")
def _cross_arr(v1, v2) -> Scalars:
    return np.cross(v1, v2)


def cross(v1: V2_t, v2: V2_t) -> Scalars:
    """Cross of vectors"""
    return _cross_arr(data(v1), data(v2))


def to_point(v: V2_t) -> P2_t:
    """Convert a vector to a point (allows transpose)."""
    if isinstance(v, Pt) or isinstance(jax.typeof(v), geom.P2Ty):
        return v
    return geom.to_point(_as_v2(v))


def to_vec(p: P2_t) -> V2_t:
    """Convert a point to a vector (disallows transpose)."""
    return _as_v2(p)


@jit
@partial(vectorize, signature="()->(3,1)")
def _polar_arr(angle: Floating):
    rad = to_radians(angle)
    x, y = np.cos(rad), np.sin(rad)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=-1)[..., None]


def polar(angle: Floating) -> V2_t:
    """Angle in degrees to a unit vector."""
    return geom.make_v2_from_data(_polar_arr(angle))


def scale(vec: V2_t) -> Affine:
    """Create an affine scale matrix."""
    return geom.xf_scale(_as_v2(vec))


def translation(vec: V2_t) -> Affine:
    """Create an affine translation matrix."""
    return geom.xf_translation(_as_v2(vec))


def get_translation(aff: Affine) -> V2_t:
    """Get the translation of an affine matrix."""
    return geom.xf_get_translation(_as_xf(aff))


def rotation(r: Floating) -> Affine:
    """Create an affine rotation matrix in radians."""
    return geom.xf_rotation(ftos(r))


def rotation_angle(r: Floating) -> Affine:
    """Create an affine rotation matrix in degrees"""
    return rotation(to_radians(r))


@partial(vectorize, signature="(3,3)->(3,3)")
@jit
def _inv_arr(aff):
    det = np.linalg.det(aff)
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
    return x.reshape(vals[0].shape + (3, 3))


def inv(aff: Affine) -> Affine:
    """Invert an affine transform."""
    return geom.make_xf(_inv_arr(data(_as_xf(aff))))


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
def _remove_translation_arr(aff):
    index = (Ellipsis, slice(0, 1), 2)
    return index_update(aff, index, 0)


def remove_translation(aff: Affine) -> Affine:
    """Remove translation from affine."""
    return geom.make_xf(_remove_translation_arr(data(_as_xf(aff))))


@jit
@partial(vectorize, signature="(3,3)->(3,3)")
def _remove_scale_arr(aff):
    index = (Ellipsis, slice(0, 2), slice(0, 2))
    det = np.linalg.det(aff[index])
    return index_update(aff, index, aff[index] / np.sqrt(det[..., None, None]))


def remove_scale(aff: Affine) -> Affine:
    """Remove scaling from affine."""
    return geom.make_xf(_remove_scale_arr(data(_as_xf(aff))))


@jit
@partial(vectorize, signature="(3,3)->(3,3)")
def _transpose_linear_arr(aff):
    index = (Ellipsis, slice(0, 2), slice(0, 2))
    swap = aff[..., :2, :2].swapaxes(-1, -2)
    return index_update(aff, index, swap)


def transpose_translation(aff: Affine) -> Affine:
    """Transpose the linear part of an affine."""
    return geom.make_xf(_transpose_linear_arr(data(_as_xf(aff))))


class Transformable:
    """Syntactic sugar to apply transformations to objects
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
        """Scale uniformly by `α`"""
        return self._app(scale(V2(α, α)))

    def scale_x(self, α: Floating) -> Self:
        """Scale horizontally by `α`"""
        return self._app(scale(V2(α, 1.0)))

    def scale_y(self, α: Floating) -> Self:
        """Scale vertically by `α`"""
        return self._app(scale(V2(1.0, α)))

    def rotate(self, deg: Floating) -> Self:
        """Rotate by `deg` degrees counterclockwise"""
        return self._app(rotation(to_radians(deg)))

    def rotate_rad(self, θ: Floating) -> Self:
        """Rotate by `θ` radians counterclockwise"""
        return self._app(rotation(θ))

    def rotate_by(self, turns: Floating) -> Self:
        """Rotate by fractions of a circle (turn)"""
        return self._app(rotation(2 * math.pi * turns))

    def reflect_x(self) -> Self:
        """Reflect across the x-axis"""
        return self._app(scale(V2(-1, +1)))

    def reflect_y(self) -> Self:
        """Reflect across the y-axis"""
        return self._app(scale(V2(+1, -1)))

    def shear_y(self, λ: Floating) -> Self:
        """Apply vertical shear by `λ`"""
        return self._app(make_affine(1.0, 0.0, 0.0, λ, 1.0, 0.0))

    def shear_x(self, λ: Floating) -> Self:
        """Apply horizontal shear by `λ`"""
        return self._app(make_affine(1.0, λ, 0.0, 0.0, 1.0, 0.0))

    def translate(self, dx: Floating, dy: Floating) -> Self:
        """Translate by `(dx, dy)`"""
        return self._app(translation(V2(dx, dy)))

    def translate_by(self, vector: V2_t) -> Self:  # type: ignore
        """Translate by `vector`"""
        return self._app(translation(vector))


@dataclass
class Ray:
    pt: P2_t
    v: V2_t

    def point(self, len: Scalars) -> P2_t:
        p = np.asarray(self.pt)
        v = np.asarray(self.v)
        return p + np.asarray(len)[..., None, None] * v


@dataclass
class BoundingBox(Transformable):
    tl: P2_t
    br: P2_t

    def apply_transform(self, t: Affine) -> Self:  # type: ignore
        # Todo: fix rotation
        t = data(t)
        tl = t @ data(self.tl)
        br = t @ data(self.br)
        tl2 = np.minimum(tl, br)
        br2 = np.maximum(tl, br)
        return BoundingBox(geom.make_p2_from_data(tl2), geom.make_p2_from_data(br2))  # type: ignore

    @property
    def width(self) -> Scalars:
        s: Scalars = (data(self.br) - data(self.tl))[..., 0, 0]
        return s

    @property
    def height(self) -> Scalars:
        s: Scalars = (data(self.br) - data(self.tl))[..., 1, 0]
        return s

    def to_rect(self) -> Any:  # type: ignore
        """Convert bounding box to a rectangle"""
        from chalk import rectangle

        return (
            rectangle(self.width, self.height)
            .align_tl()
            .translate(-self.width / 2, -self.height / 2)
        )


# @partial(vectorize, signature="(3,1),(3,1),()->(),()") # type: ignore
def ray_circle_intersection(
    anchor: P2_t, direction: V2_t, circle_radius: Floating
) -> Tuple[Scalars, Mask, Scalars, Mask]:
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
    a = (direction * direction)[..., :2, 0].sum(-1)
    b = 2 * (anchor * direction)[..., :2, 0].sum(-1)
    c = (anchor * anchor)[..., :2, 0].sum(-1) - circle_radius**2
    Δ = b**2 - 4 * a * c
    eps = 1e-10  # rounding error tolerance

    mid = (-eps <= Δ) & (Δ < 0)
    mask1 = Δ < 0
    mask2 = Δ < -eps

    ret1 = (-b - np.sqrt(Δ + 1e9 * mask1)) / (2 * a)
    ret2 = (-b + np.sqrt(np.where(mid, 0, Δ) + 1e9 * mask2)) / (2 * a)
    ret1 = np.where(mask1, -b / (2 * a), ret1)
    ret2 = np.where(mask2, -b / (2 * a), ret2)
    assert not isinstance(ret2, tuple)
    return ret1, 1 - mask1, ret2, 1 - mask2


@partial(vectorize, excluded=[2], signature="(),()->(a,3,1)")
def arc_to_bezier(theta1: Array, theta2: Array, n: int = 5) -> Array:
    """Returns the bezier curves for the unit circle arc from angles *theta1* to
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
    eta2 = theta2  # - 360 * np.floor((theta2 - theta1) / 360)
    # Ensure 2pi range is not flattened to 0 due to floating-point errors,
    # but don't try to expand existing 0 range.
    # eta2 = np.where((theta2 != theta1) & (eta2 <= eta1), eta2 + 360, eta2)
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


# TODO - move these out of transform

# Explicit rexport
__all__ = [
    "Array",
    "Floating",
    "Ints",
    "IntLike",
    "IntLikeC",
    "Mask",
    "MaskC",
    "np",
    "jit",
    "vmap",
    "multi_vmap",
    "tree_map",
    "Batchable",
    "Batched",
    "prefix_broadcast",
    "BoolLike",
    "JAX_MODE",
    "onp",
]
