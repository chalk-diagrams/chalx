from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.hijax import (
    HiType,
    MappingSpec,
    ShapedArray,
    VJPHiPrimitive,
    register_hitype,
)

import chalk.geom as geom
import chalk.segment as arc
import chalk.transform as tx
from chalk.monoid import reduce_associative
from chalk.segment import (
    SegSpec,
    Segment,
    SegTy,
    concat_segments,
    make_segment,
    segment_parts,
    segment_q,
    transform_segment,
)
from chalk.transform import Affine, Floating, P2_t, Transformable, V2_t
from chalk.types import Diagram, TrailLike

if TYPE_CHECKING:
    from chalk.path import Path


# ---------------------------------------------------------------------------
# Hijax Trail
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrailSpec(MappingSpec):
    pass


@dataclass(frozen=True)
class TrailTy(HiType):
    seg_ty: SegTy

    def lo_ty(self):
        closed_shape = self.seg_ty.batch_shape + (self.seg_ty.n_segs,)
        return self.seg_ty.lo_ty() + [ShapedArray(closed_shape, jnp.dtype("bool"))]

    def lower_val(self, trail: Trail):
        return self.seg_ty.lower_val(trail.segments) + [trail.closed]

    def raise_val(self, transform, angles, closed) -> Trail:
        return Trail(self.seg_ty.raise_val(transform, angles), closed)

    def to_tangent_aval(self):
        return TrailTy(self.seg_ty)

    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        return f"trail[{self.seg_ty.str_short(short_dtypes, mesh_axis_types)[4:-1]}]"

    __repr__ = str_short

    def dec_rank(self, size, spec):
        assert isinstance(spec, TrailSpec)
        return TrailTy(self.seg_ty.dec_rank(size, SegSpec()))

    def inc_rank(self, size, spec):
        assert isinstance(spec, TrailSpec)
        return TrailTy(self.seg_ty.inc_rank(size, SegSpec()))

    def leading_axis_spec(self):
        return TrailSpec()


@dataclass(frozen=True)
class LocatedSpec(MappingSpec):
    pass


@dataclass(frozen=True)
class LocatedTy(HiType):
    trail_ty: TrailTy
    loc_shape: Tuple[int, ...]
    dtype_name: str = "float64"

    def lo_ty(self):
        return self.trail_ty.lo_ty() + [
            ShapedArray(self.loc_shape, jnp.dtype(self.dtype_name))
        ]

    def lower_val(self, loc: Located):
        return self.trail_ty.lower_val(loc.trail) + [loc.location]

    def raise_val(self, transform, angles, closed, location) -> Located:
        return Located(self.trail_ty.raise_val(transform, angles, closed), location)

    def to_tangent_aval(self):
        return LocatedTy(self.trail_ty, self.loc_shape, self.dtype_name)

    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        return f"located[{self.trail_ty.str_short(short_dtypes, mesh_axis_types)}]"

    __repr__ = str_short

    def dec_rank(self, size, spec):
        assert isinstance(spec, LocatedSpec)
        loc_shape = self.loc_shape[1:] if self.loc_shape else ()
        return LocatedTy(
            self.trail_ty.dec_rank(size, TrailSpec()), loc_shape, self.dtype_name
        )

    def inc_rank(self, size, spec):
        assert isinstance(spec, LocatedSpec)
        return LocatedTy(
            self.trail_ty.inc_rank(size, TrailSpec()),
            (size, *self.loc_shape),
            self.dtype_name,
        )

    def leading_axis_spec(self):
        return LocatedSpec()


@dataclass(frozen=True)
class Located(Transformable):
    """Opaque hijax located trail."""

    trail: Trail
    location: P2_t

    def map_prefix(self, fn: Callable[[Any], Any]) -> Located:
        trail = self.trail.map_prefix(fn)
        loc = fn(self.location)
        if loc is None:
            return self
        return make_located(trail, loc)

    def located_segments(self) -> Segment:
        return located_segments(self)

    def points(self) -> P2_t:
        return located_points(self)

    def _promote(self) -> Located:
        return make_located(trail_promote(self.trail), self.location)

    def stroke(self) -> Diagram:
        return self._promote().to_path().stroke()

    def apply_transform(self, t: Affine) -> Located:
        return transform_located(self, t)

    def to_path(self) -> Path:
        from chalk.path import _make_path

        return _make_path((self._promote(),))


@dataclass(frozen=True)
class Trail(Transformable, TrailLike):
    """Opaque hijax trail."""

    segments: Segment
    closed: tx.Mask

    def map_prefix(self, fn: Callable[[Any], Any]) -> Trail:
        t, a = segment_parts(self.segments)
        t2, a2, c2 = fn(t), fn(a), fn(self.closed)
        if t2 is None and a2 is None and c2 is None:
            return self
        return make_trail(make_segment(t2, a2), c2)

    @staticmethod
    def empty() -> Trail:
        seg = Segment.empty()
        t, a = segment_parts(seg)
        return make_trail(seg, jnp.zeros(a.shape[:-1], dtype=bool))

    def __add__(self, other: Trail) -> Trail:
        return concat_trails(self, other)

    @classmethod
    def concat(cls, elems):
        return reduce_associative(concat_trails, elems, cls.empty())

    def apply_transform(self, t: Affine) -> Trail:
        return transform_trail(self, t)

    def to_trail(self) -> Trail:
        return self

    def _promote(self) -> Trail:
        return trail_promote(self)

    def close(self) -> Trail:
        return trail_close(self)

    def points(self) -> P2_t:
        return trail_points(self)

    def at(self, p: P2_t) -> Located:
        return make_located(trail_promote(self), tx.to_point(p))

    def centered(self) -> Located:
        pts = trail_points(self)
        t, _ = segment_parts(self.segments)
        center = -tx.np.sum(tx.data(pts), axis=-3) / t.shape[0]
        return self.at(geom.make_v2_from_data(center))

    @staticmethod
    def from_array(offsets: V2_t, closed: bool = False) -> Trail:
        trail = seg(offsets)
        if closed:
            trail = trail.close()
        return trail

    @staticmethod
    def from_offsets(offsets: List[V2_t], closed: bool = False) -> Trail:
        data = tx.np.stack([tx.data(offset) for offset in offsets])
        return Trail.from_array(geom.make_v2_from_data(data), closed)

    @staticmethod
    def hrule(length: Floating) -> Trail:
        return seg(length * tx.unit_x)

    @staticmethod
    def vrule(length: Floating) -> Trail:
        return seg(length * tx.unit_y)

    @staticmethod
    def square() -> Trail:
        t = seg(tx.unit_x) + seg(tx.unit_y)
        return (t + t.rotate_by(0.5)).close()

    @staticmethod
    def rounded_rectangle(width: Floating, height: Floating, radius: Floating) -> Trail:
        r = radius
        edge1 = math.sqrt(2 * r * r) / 2
        edge3 = math.sqrt(r * r - edge1 * edge1)
        corner = arc_seg(tx.V2(r, r), -(r - edge3))
        b = [height - r, width - r, height - r, width - r]
        trail = Trail.concat(
            (seg(b[i] * tx.unit_y) + corner).rotate_by(i / 4) for i in range(4)
        ) + seg(0.01 * tx.unit_y)
        return trail.close()

    @staticmethod
    def circle(size: float = 1, clockwise: bool = True) -> Trail:
        sides = 4
        dangle = -90
        rotate_by = 1
        if not clockwise:
            dangle = 90
            rotate_by *= -1
        return Trail.concat(
            [
                arc_seg_angle(0, dangle).rotate_by(rotate_by * i / sides)
                for i in range(sides)
            ]
        ).close()

    @staticmethod
    def regular_polygon(sides: int, side_length: Floating) -> Trail:
        edge = Trail.hrule(1)
        return Trail.concat(edge.rotate_by(i / sides) for i in range(sides)).close()


register_hitype(Trail, lambda t: TrailTy(jax.typeof(t.segments)))
register_hitype(
    Located,
    lambda loc: LocatedTy(
        jax.typeof(loc.trail),
        tuple(jnp.asarray(loc.location).shape),
        jnp.asarray(loc.location).dtype.name,
    ),
)


class MakeTrail(VJPHiPrimitive):
    def __init__(self, seg_aval: SegTy, closed_aval):
        self.in_avals = (seg_aval, closed_aval)
        self.out_aval = TrailTy(seg_aval)
        self.params = {}
        super().__init__()

    def expand(self, segments, closed):
        return Trail(segments, jnp.asarray(closed).astype(bool))

    def batch(self, axis_data, args, in_dims):
        seg, closed = args
        if all(d is None for d in in_dims):
            return make_trail(seg, closed), None
        return make_trail(seg, closed), TrailSpec()


class ConcatTrails(VJPHiPrimitive):
    def __init__(self, a: TrailTy, b: TrailTy):
        out_seg = SegTy(
            tuple(jnp.broadcast_shapes(a.seg_ty.batch_shape, b.seg_ty.batch_shape)),
            a.seg_ty.n_segs + b.seg_ty.n_segs,
            a.seg_ty.dtype_name,
        )
        self.in_avals = (a, b)
        self.out_aval = TrailTy(out_seg)
        self.params = {}
        super().__init__()

    def expand(self, a: Trail, b: Trail):
        seg = concat_segments(a.segments, b.segments)
        _, angles = segment_parts(seg)
        return Trail(seg, jnp.zeros(angles.shape[:-1], dtype=bool))

    def batch(self, axis_data, args, in_dims):
        a, b = args
        if all(d is None for d in in_dims):
            return concat_trails(a, b), None
        return concat_trails(a, b), TrailSpec()


class TransformTrail(VJPHiPrimitive):
    def __init__(self, trail_aval: TrailTy, t_aval):
        self.in_avals = (trail_aval, t_aval)
        self.out_aval = trail_aval
        self.params = {}
        super().__init__()

    def expand(self, trail: Trail, t):
        t = tx._remove_translation_arr(tx.data(t))
        if t.ndim >= 3:
            t = t[:, None, :, :]
        return Trail(transform_segment(trail.segments, t), trail.closed)

    def batch(self, axis_data, args, in_dims):
        trail, t = args
        if all(d is None for d in in_dims):
            return transform_trail(trail, t), None
        return transform_trail(trail, t), TrailSpec()


class TrailPromote(VJPHiPrimitive):
    def __init__(self, trail_aval: TrailTy):
        seg = trail_aval.seg_ty
        # promote ensures a segment axis; n_segs unchanged if already present
        self.in_avals = (trail_aval,)
        self.out_aval = trail_aval
        self.params = {}
        super().__init__()

    def expand(self, trail: Trail):
        t, a = segment_parts(trail.segments)
        if t.ndim < 3:
            t = t.reshape(-1, *t.shape)
            a = a.reshape(-1, *a.shape) if a.ndim < 2 else a
        return Trail(make_segment(t, a), jnp.asarray(trail.closed))

    def batch(self, axis_data, args, in_dims):
        (trail,) = args
        (d,) = in_dims
        if d is None:
            return trail_promote(trail), None
        return trail_promote(trail), TrailSpec()


class TrailClose(VJPHiPrimitive):
    def __init__(self, trail_aval: TrailTy):
        self.in_avals = (trail_aval,)
        self.out_aval = trail_aval
        self.params = {}
        super().__init__()

    def expand(self, trail: Trail):
        t, a = segment_parts(trail.segments)
        if t.ndim < 3:
            t = t.reshape(-1, *t.shape)
        if a.ndim < 2:
            a = a.reshape(-1, *a.shape)
        closed = jnp.ones(a.shape[:-1], dtype=bool)
        return Trail(make_segment(t, a), closed)

    def batch(self, axis_data, args, in_dims):
        (trail,) = args
        (d,) = in_dims
        if d is None:
            return trail_close(trail), None
        return trail_close(trail), TrailSpec()


class TrailPoints(VJPHiPrimitive):
    def __init__(self, trail_aval: TrailTy):
        seg = trail_aval.seg_ty
        self.in_avals = (trail_aval,)
        self.out_aval = geom.P2Ty(
            seg.batch_shape + (seg.n_segs,), seg.dtype_name
        )
        self.params = {}
        super().__init__()

    def expand(self, trail: Trail):
        q = tx.data(segment_q(trail.segments))
        pts = jnp.cumsum(q, axis=-3) - q
        return geom.make_p2_from_data(pts.at[..., 2, 0].set(1.0))

    def batch(self, axis_data, args, in_dims):
        (trail,) = args
        (d,) = in_dims
        if d is None:
            return trail_points(trail), None
        return trail_points(trail), geom.GeomSpec()


class TrailClosed(VJPHiPrimitive):
    def __init__(self, trail_aval: TrailTy):
        seg = trail_aval.seg_ty
        self.in_avals = (trail_aval,)
        self.out_aval = ShapedArray(
            seg.batch_shape + (seg.n_segs,), jnp.dtype("bool")
        )
        self.params = {}
        super().__init__()

    def expand(self, trail: Trail):
        return jnp.asarray(trail.closed)

    def batch(self, axis_data, args, in_dims):
        (trail,) = args
        (d,) = in_dims
        if d is None:
            return trail_closed(trail), None
        return trail_closed(trail), 0


class TrailSegment(VJPHiPrimitive):
    def __init__(self, trail_aval: TrailTy):
        self.in_avals = (trail_aval,)
        self.out_aval = trail_aval.seg_ty
        self.params = {}
        super().__init__()

    def expand(self, trail: Trail):
        return trail.segments

    def batch(self, axis_data, args, in_dims):
        (trail,) = args
        (d,) = in_dims
        if d is None:
            return trail_segment(trail), None
        return trail_segment(trail), SegSpec()


class MakeLocated(VJPHiPrimitive):
    def __init__(self, trail_aval: TrailTy, loc_aval):
        self.in_avals = (trail_aval, loc_aval)
        self.out_aval = LocatedTy(
            trail_aval, tuple(loc_aval.shape), loc_aval.dtype.name
        )
        self.params = {}
        super().__init__()

    def expand(self, trail, location):
        return Located(trail, tx.data(location))

    def batch(self, axis_data, args, in_dims):
        trail, loc = args
        if all(d is None for d in in_dims):
            return make_located(trail, loc), None
        return make_located(trail, loc), LocatedSpec()


class TransformLocated(VJPHiPrimitive):
    def __init__(self, loc_aval: LocatedTy, t_aval):
        self.in_avals = (loc_aval, t_aval)
        self.out_aval = loc_aval
        self.params = {}
        super().__init__()

    def expand(self, loc: Located, t):
        t = tx.data(t)
        p = t[..., None, :, :] @ loc.location
        if p.ndim == 3:
            p = p[:, None]
        if p.ndim == 2:
            p = p[None]
        trail = transform_trail(loc.trail, tx._remove_translation_arr(t))
        return Located(trail, p)

    def batch(self, axis_data, args, in_dims):
        loc, t = args
        if all(d is None for d in in_dims):
            return transform_located(loc, t), None
        return transform_located(loc, t), LocatedSpec()


class LocatedPoints(VJPHiPrimitive):
    def __init__(self, loc_aval: LocatedTy):
        seg = loc_aval.trail_ty.seg_ty
        self.in_avals = (loc_aval,)
        self.out_aval = geom.P2Ty(
            seg.batch_shape + (seg.n_segs,), seg.dtype_name
        )
        self.params = {}
        super().__init__()

    def expand(self, loc: Located):
        pts = tx.data(trail_points(loc.trail))
        location = tx.data(loc.location)
        return geom.make_p2_from_data(pts + location[..., None, :, :])

    def batch(self, axis_data, args, in_dims):
        (loc,) = args
        (d,) = in_dims
        if d is None:
            return located_points(loc), None
        return located_points(loc), geom.GeomSpec()


class LocatedSegments(VJPHiPrimitive):
    def __init__(self, loc_aval: LocatedTy):
        self.in_avals = (loc_aval,)
        self.out_aval = loc_aval.trail_ty.seg_ty
        self.params = {}
        super().__init__()

    def expand(self, loc: Located):
        pts = located_points(loc)
        off = geom.make_v2_from_data(tx.data(pts).at[..., 2, 0].set(0.0))
        return transform_segment(loc.trail.segments, tx.translation(off))

    def batch(self, axis_data, args, in_dims):
        (loc,) = args
        (d,) = in_dims
        if d is None:
            return located_segments(loc), None
        return located_segments(loc), SegSpec()


class LocatedLocation(VJPHiPrimitive):
    def __init__(self, loc_aval: LocatedTy):
        self.in_avals = (loc_aval,)
        self.out_aval = geom.P2Ty(
            loc_aval.loc_shape[:-2], loc_aval.dtype_name
        )
        self.params = {}
        super().__init__()

    def expand(self, loc: Located):
        return geom.make_p2_from_data(tx.data(loc.location))

    def batch(self, axis_data, args, in_dims):
        (loc,) = args
        (d,) = in_dims
        if d is None:
            return located_location(loc), None
        return located_location(loc), geom.GeomSpec()


class LocatedTrail(VJPHiPrimitive):
    def __init__(self, loc_aval: LocatedTy):
        self.in_avals = (loc_aval,)
        self.out_aval = loc_aval.trail_ty
        self.params = {}
        super().__init__()

    def expand(self, loc: Located):
        return loc.trail

    def batch(self, axis_data, args, in_dims):
        (loc,) = args
        (d,) = in_dims
        if d is None:
            return located_trail(loc), None
        return located_trail(loc), TrailSpec()


def make_trail(segments, closed) -> Trail:
    closed = jnp.asarray(closed).astype(bool)
    return MakeTrail(jax.typeof(segments), jax.typeof(closed))(segments, closed)


def concat_trails(a, b) -> Trail:
    return ConcatTrails(jax.typeof(a), jax.typeof(b))(a, b)


def transform_trail(trail, t) -> Trail:
    t = tx.data(t)
    return TransformTrail(jax.typeof(trail), jax.typeof(t))(trail, t)


def trail_promote(trail) -> Trail:
    return TrailPromote(jax.typeof(trail))(trail)


def trail_close(trail) -> Trail:
    return TrailClose(jax.typeof(trail))(trail)


def trail_points(trail) -> P2_t:
    return TrailPoints(jax.typeof(trail))(trail)


def trail_closed(trail) -> jax.Array:
    return TrailClosed(jax.typeof(trail))(trail)


def trail_segment(trail) -> Segment:
    return TrailSegment(jax.typeof(trail))(trail)


def make_located(trail, location) -> Located:
    location = tx.data(location)
    return MakeLocated(jax.typeof(trail), jax.typeof(location))(trail, location)


def transform_located(loc, t) -> Located:
    t = tx.data(t)
    return TransformLocated(jax.typeof(loc), jax.typeof(t))(loc, t)


def located_points(loc) -> P2_t:
    return LocatedPoints(jax.typeof(loc))(loc)


def located_segments(loc) -> Segment:
    return LocatedSegments(jax.typeof(loc))(loc)


def located_location(loc) -> P2_t:
    return LocatedLocation(jax.typeof(loc))(loc)


def located_trail(loc) -> Trail:
    return LocatedTrail(jax.typeof(loc))(loc)


def seg(offset: V2_t) -> Trail:
    return arc_seg(offset, 1e-3)


def arc_seg(offset: V2_t, height: tx.Floating) -> Trail:
    return arc_between_trail(offset, tx.ftos(height))


def arc_seg_angle(angle: tx.Floating, dangle: tx.Floating) -> Trail:
    arc_p = tx.polar(angle)
    return Segment.make(
        tx.translation(-arc_p), tx.np.asarray([angle, dangle])
    ).to_trail()


def arc_between_trail(q: P2_t, height: tx.Scalars) -> Trail:
    return arc.arc_between(tx.P2(0, 0), q, height).to_trail()


__all__ = [
    "seg",
    "arc_seg",
    "arc_seg_angle",
    "Trail",
    "Located",
    "make_trail",
    "make_located",
    "trail_points",
    "trail_closed",
    "trail_segment",
    "located_segments",
    "located_location",
    "located_trail",
]
