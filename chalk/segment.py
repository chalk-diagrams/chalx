"""Segment is a collection of ellipse arcs with starting angle and the delta.
Every diagram in chalk is made up of these segments.
They may be either located or at the origin depending on how they are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.hijax import (
    HiType,
    MappingSpec,
    ShapedArray,
    VJPHiPrimitive,
    Zero,
    apply_derived_linearization,
    linearize_from_jvp,
    register_hitype,
    transpose_jvp,
    vjp_fwd_from_jvp,
)

import chalk.transform as tx
from chalk.monoid import reduce_associative
from chalk.transform import Affine, Angles, P2_t, V2_t

if TYPE_CHECKING:
    from jaxtyping import Array

    from chalk.trail import Trail


def _ensure_3d(x: tx.Array) -> tx.Array:
    if len(x.shape) < 3:
        return x.reshape(-1, *x.shape)
    return x


def _ensure_2d(x: tx.Array) -> tx.Array:
    if len(x.shape) < 2:
        return x.reshape(-1, *x.shape)
    return x


@dataclass(frozen=True)
class SegSpec(MappingSpec):
    pass


@dataclass(frozen=True)
class SegTy(HiType):
    """``seg[*B; N]`` — N arc segments, optional prefix batch."""

    batch_shape: Tuple[int, ...]
    n_segs: int
    dtype_name: str = "float32"

    @property
    def dtype(self):
        return jnp.dtype(self.dtype_name)

    def lo_ty(self):
        dt = jnp.dtype(self.dtype_name)
        return [
            ShapedArray(self.batch_shape + (self.n_segs, 3, 3), dt),
            ShapedArray(self.batch_shape + (self.n_segs, 2), dt),
        ]

    def lower_val(self, seg: Segment):
        return [seg.transform, seg.angles]

    def raise_val(self, transform, angles) -> Segment:
        return Segment(transform, angles)

    def to_tangent_aval(self):
        return SegTy(self.batch_shape, self.n_segs, self.dtype_name)

    def vspace_zero(self):
        dt = jnp.dtype(self.dtype_name)
        return Segment(
            jnp.zeros(self.batch_shape + (self.n_segs, 3, 3), dt),
            jnp.zeros(self.batch_shape + (self.n_segs, 2), dt),
        )

    def vspace_add(self, x: Segment, y: Segment):
        xf1, a1 = segment_parts(x)
        xf2, a2 = segment_parts(y)
        return make_segment(xf1 + xf2, a1 + a2)

    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        batch = ",".join(str(d) for d in self.batch_shape)
        prefix = f"{batch};" if batch else ""
        return f"seg[{prefix}{self.n_segs}]"

    __repr__ = str_short

    def dec_rank(self, size, spec):
        assert isinstance(spec, SegSpec)
        assert self.batch_shape and self.batch_shape[0] == size
        return SegTy(self.batch_shape[1:], self.n_segs, self.dtype_name)

    def inc_rank(self, size, spec):
        assert isinstance(spec, SegSpec)
        return SegTy((size, *self.batch_shape), self.n_segs, self.dtype_name)

    def leading_axis_spec(self):
        return SegSpec()


@dataclass(frozen=True)
class Segment:
    """Opaque hijax segment. Peek arrays only eagerly or in expand."""

    transform: Affine
    angles: Angles

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.transform.shape[:-2])

    def tuple(self) -> Tuple[Affine, Angles]:
        return self.transform, self.angles

    @staticmethod
    def empty() -> Segment:
        return make_segment(jnp.zeros((0, 3, 3)), jnp.zeros((0, 2)))

    @staticmethod
    def make(transform: Affine, angles: Angles) -> Segment:
        transform = tx.data(transform)
        assert angles.shape[-1] == 2
        angles = tx.prefix_broadcast(angles, transform.shape[:-2], 1)  # type: ignore
        return make_segment(transform, angles.astype(float))

    def promote(self) -> Segment:
        return make_segment(_ensure_3d(self.transform), _ensure_2d(self.angles))

    def map_prefix(self, fn: Callable[[Any], Any]) -> Segment:
        t, a = fn(self.transform), fn(self.angles)
        if t is None and a is None:
            return self
        return make_segment(t, a)

    def to_trail(self) -> Trail:
        from chalk.trail import make_trail

        _, angles = segment_parts(self)
        return make_trail(self, jnp.zeros(angles.shape[:-1], dtype=bool))

    def reduce(self, axis: int = 0) -> Segment:
        shape = self.shape
        return make_segment(
            self.transform.reshape(*shape[:-2], -1, 3, 3),
            self.angles.reshape(*shape[:-2], -1, 2),
        )

    def apply_transform(self, t: Affine) -> Segment:
        return transform_segment(self, t)

    def __add__(self, other: Segment) -> Segment:
        return concat_segments(self, other)

    @classmethod
    def concat(cls, elems):
        return reduce_associative(concat_segments, elems, cls.empty())

    @property
    def q(self) -> P2_t:
        return segment_q(self)

    @property
    def center(self) -> P2_t:
        return segment_center(self)

    def parts(self) -> Tuple[Affine, Angles]:
        return segment_parts(self)

    def is_in_mod_360(self, d: V2_t) -> tx.Mask:
        return _is_in_mod_360(self.angles, d)


def _is_in_mod_360(angles: Angles, d: V2_t) -> tx.Mask:
    angle0_deg = angles[..., 0]
    angle1_deg = angles.sum(-1)
    low = tx.np.minimum(angle0_deg, angle1_deg)
    high = tx.np.maximum(angle0_deg, angle1_deg)
    check = (high - low) % 360
    return tx.np.asarray(((tx.angle(d) - low) % 360) <= check)


register_hitype(
    Segment,
    lambda s: SegTy(
        tuple(s.transform.shape[:-3]) if s.transform.ndim >= 3 else (),
        int(s.transform.shape[-3]) if s.transform.ndim >= 3 else int(s.transform.shape[0]),
        jnp.asarray(s.transform).dtype.name,
    ),
)


def arc_between(p: P2_t, q: P2_t, height: tx.Scalars) -> Segment:
    p, q = tx.np.broadcast_arrays(tx.data(p), tx.data(q))
    h = abs(height)
    d = tx.length(q - p)
    # Determine the arc's angle θ and its radius r
    θ = tx.np.arccos((d**2 - 4.0 * h**2) / (d**2 + 4.0 * h**2))
    r = d / (2 * tx.np.sin(θ))

    # bend left
    bl = height > 0
    φ = tx.np.where(bl, +tx.np.pi / 2, -tx.np.pi / 2)
    dy = tx.np.where(bl, r - h, h - r)
    flip = tx.np.where(bl, 1, -1)

    diff = q - p
    angles = tx.np.stack(
        [flip * -tx.from_radians(θ), flip * 2 * tx.from_radians(θ)], -1
    )
    ret = (
        tx.translation(p)
        @ tx.rotation(-tx.rad(diff))
        @ tx.translation(tx.V2(d / 2, dy))
        @ tx.rotation(φ)
        @ tx.scale(tx.V2(r, r))
    )
    return Segment.make(ret, angles)


@tx.jit
@partial(tx.vectorize, signature="(3,3),(2),(3,1)->()")
def arc_envelope(trans: Affine, angles: Angles, d: tx.V2_tC) -> Array:
    """Compute the envelope for a batch of segments."""
    angle0_deg = angles[..., 0]
    angle1_deg = angles.sum(-1)

    is_circle = abs(angle0_deg - angle1_deg) >= 360
    v1 = tx._polar_arr(angle0_deg)
    v2 = tx._polar_arr(angle1_deg)
    d2 = (d * d)[..., :2, 0].sum(-1)
    return tx.np.where(  # type: ignore
        (is_circle | _is_in_mod_360(angles, d)),
        1 / tx.np.sqrt(d2),
        tx.np.maximum(tx._dot_arr(d, v1), tx._dot_arr(d, v2)),
    )


@partial(tx.vectorize, signature="(3,3),(2),(3,1),(3,1)->(2),(2)")
def arc_trace(
    trans: Affine, angles: Angles, p: tx.P2_tC, v: tx.V2_tC
) -> Tuple[tx.Array, tx.Array]:
    """Computes the trace for a batch of segments."""
    ray = tx.Ray(p, v)
    d1, mask1, d2, mask2 = tx.ray_circle_intersection(ray.pt, ray.v, 1)
    mask1 = mask1 & _is_in_mod_360(angles, ray.point(d1))
    mask2 = mask2 & _is_in_mod_360(angles, ray.point(d2))

    d = tx.np.stack([d1, d2], -1)
    mask = tx.np.stack([mask1, mask2], -1)
    return d, mask


def _seg_typeof(transform, angles) -> SegTy:
    t = jnp.asarray(transform)
    if t.ndim == 2:
        t = t[None, ...]
    n_segs = int(t.shape[-3])
    batch = tuple(t.shape[:-3])
    return SegTy(batch, n_segs, t.dtype.name)


class MakeSegment(VJPHiPrimitive):
    def __init__(self, t_aval, a_aval):
        self.in_avals = (t_aval, a_aval)
        t_shape = t_aval.shape
        if len(t_shape) == 2:
            batch, n = (), 1
        else:
            batch, n = tuple(t_shape[:-3]), int(t_shape[-3])
        self.out_aval = SegTy(batch, n, t_aval.dtype.name)
        self.params = {}
        super().__init__()

    def expand(self, transform, angles):
        t = jnp.asarray(transform)
        a = jnp.asarray(angles)
        if t.ndim == 2:
            t = t[None, ...]
        if a.ndim == 1:
            a = a[None, ...]
        return Segment(t, a)

    def jvp(self, primals, tangents):
        xf, ang = primals
        dxf, dang = tangents
        prim = make_segment(xf, ang)
        if isinstance(dxf, Zero):
            dxf = jnp.zeros_like(jnp.asarray(xf))
        if isinstance(dang, Zero):
            dang = jnp.zeros_like(jnp.asarray(ang))
        return prim, make_segment(dxf, dang)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        t, a = args
        dt, da = in_dims
        if dt is None and da is None:
            return make_segment(t, a), None
        return make_segment(t, a), SegSpec()


class ConcatSegments(VJPHiPrimitive):
    def __init__(self, a: SegTy, b: SegTy):
        batch = tuple(jnp.broadcast_shapes(a.batch_shape, b.batch_shape))
        self.in_avals = (a, b)
        self.out_aval = SegTy(batch, a.n_segs + b.n_segs, a.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, a: Segment, b: Segment):
        if a.transform.shape[0] == 0:
            return b
        if b.transform.shape[0] == 0:
            return a
        ta, tb = _ensure_3d(a.transform), _ensure_3d(b.transform)
        aa, ab = _ensure_2d(a.angles), _ensure_2d(b.angles)

        def broadcast_ex(x, y, axis):
            xs, ys = list(x.shape), list(y.shape)
            xs[axis] = 1
            ys[axis] = 1
            new = jnp.broadcast_shapes(xs, ys)
            xs2, ys2 = list(new), list(new)
            xs2[axis] = x.shape[axis]
            ys2[axis] = y.shape[axis]
            return jnp.broadcast_to(x, xs2), jnp.broadcast_to(y, ys2)

        trans = broadcast_ex(jnp.asarray(ta), jnp.asarray(tb), -3)
        angs = broadcast_ex(jnp.asarray(aa), jnp.asarray(ab), -2)
        return Segment(
            jnp.concatenate(trans, axis=-3), jnp.concatenate(angs, axis=-2)
        )

    def batch(self, axis_data, args, in_dims):
        a, b = args
        da, db = in_dims
        if da is None and db is None:
            return concat_segments(a, b), None
        return concat_segments(a, b), SegSpec()


class TransformSegment(VJPHiPrimitive):
    def __init__(self, seg_aval: SegTy, t_aval):
        # Affine may carry the segment axis (e.g. per-point translations).
        # Keep the segment type; broadcasting happens in expand.
        self.in_avals = (seg_aval, t_aval)
        self.out_aval = seg_aval
        self.params = {}
        super().__init__()

    def expand(self, seg: Segment, t):
        from chalk.geom import data as geom_data

        return Segment(geom_data(t) @ jnp.asarray(seg.transform), seg.angles)

    def jvp(self, primals, tangents):
        from chalk.geom import data as geom_data

        seg, t = primals
        dseg, dt = tangents
        prim = transform_segment(seg, t)
        xf, ang = jnp.asarray(seg.transform), jnp.asarray(seg.angles)
        t_arr = geom_data(t)
        dxf = jnp.zeros_like(xf) if isinstance(dseg, Zero) else jnp.asarray(dseg.transform)
        dang = jnp.zeros_like(ang) if isinstance(dseg, Zero) else jnp.asarray(dseg.angles)
        dt_arr = jnp.zeros_like(t_arr) if isinstance(dt, Zero) else geom_data(dt)
        return prim, make_segment(dt_arr @ xf + t_arr @ dxf, dang)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        seg, t = args
        ds, dt = in_dims
        if ds is None and dt is None:
            return transform_segment(seg, t), None
        return transform_segment(seg, t), SegSpec()


def make_segment(transform, angles) -> Segment:
    transform = tx.data(transform)
    angles = jnp.asarray(angles)
    return MakeSegment(jax.typeof(transform), jax.typeof(angles))(
        transform, angles
    )


def concat_segments(a, b) -> Segment:
    return ConcatSegments(jax.typeof(a), jax.typeof(b))(a, b)


def transform_segment(seg, t) -> Segment:
    t = tx.data(t)
    return TransformSegment(jax.typeof(seg), jax.typeof(t))(seg, t)


class SegmentParts(VJPHiPrimitive):
    """Unpack a segment to ``(transform, angles)`` arrays."""

    def __init__(self, seg_aval: SegTy):
        t_aval, a_aval = seg_aval.lo_ty()
        self.in_avals = (seg_aval,)
        self.out_aval = (t_aval, a_aval)
        self.params = {}
        super().__init__()

    def expand(self, seg: Segment):
        return jnp.asarray(seg.transform), jnp.asarray(seg.angles)

    def jvp(self, primals, tangents):
        (seg,), (dseg,) = primals, tangents
        xf, ang = jnp.asarray(seg.transform), jnp.asarray(seg.angles)
        if isinstance(dseg, Zero):
            return (xf, ang), (Zero(jax.typeof(xf)), Zero(jax.typeof(ang)))
        return (xf, ang), (
            jnp.asarray(dseg.transform),
            jnp.asarray(dseg.angles),
        )

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        (seg,) = args
        (d,) = in_dims
        if d is None:
            return segment_parts(seg), None
        return segment_parts(seg), (0, 0)


class SegmentQ(VJPHiPrimitive):
    """Endpoint of each arc, as homogeneous points ``[..., 3, 1]``."""

    def __init__(self, seg_aval: SegTy):
        self.in_avals = (seg_aval,)
        self.out_aval = ShapedArray(
            seg_aval.batch_shape + (seg_aval.n_segs, 3, 1),
            jnp.dtype(seg_aval.dtype_name),
        )
        self.params = {}
        super().__init__()

    def expand(self, seg: Segment):
        angles = jnp.asarray(seg.angles)
        transform = jnp.asarray(seg.transform)
        ang = angles.sum(-1)
        rad = jnp.deg2rad(ang)
        x, y = jnp.cos(rad), jnp.sin(rad)
        ones = jnp.ones_like(x)
        q = jnp.stack([x, y, ones], axis=-1)[..., None]
        return transform @ q

    def batch(self, axis_data, args, in_dims):
        (seg,) = args
        (d,) = in_dims
        if d is None:
            return segment_q(seg), None
        return segment_q(seg), 0


class SegmentCenter(VJPHiPrimitive):
    """Ellipse center of each arc, as homogeneous points."""

    def __init__(self, seg_aval: SegTy):
        self.in_avals = (seg_aval,)
        self.out_aval = ShapedArray(
            seg_aval.batch_shape + (seg_aval.n_segs, 3, 1),
            jnp.dtype(seg_aval.dtype_name),
        )
        self.params = {}
        super().__init__()

    def expand(self, seg: Segment):
        transform = jnp.asarray(seg.transform)
        origin = jnp.zeros(transform.shape[:-2] + (3, 1), dtype=transform.dtype)
        origin = origin.at[..., 2, 0].set(1.0)
        return transform @ origin

    def batch(self, axis_data, args, in_dims):
        (seg,) = args
        (d,) = in_dims
        if d is None:
            return segment_center(seg), None
        return segment_center(seg), 0


def segment_parts(seg) -> Tuple[jax.Array, jax.Array]:
    return SegmentParts(jax.typeof(seg))(seg)


def segment_q(seg) -> jax.Array:
    return SegmentQ(jax.typeof(seg))(seg)


def segment_center(seg) -> jax.Array:
    return SegmentCenter(jax.typeof(seg))(seg)


__all__ = []
