from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.hijax import (
    HiType,
    MappingSpec,
    ShapedArray,
    VJPHiPrimitive,
    register_hitype,
)

import chalk.transform as tx
from chalk.monoid import Monoid
from chalk.segment import (
    SegSpec,
    Segment,
    SegTy,
    segment_parts,
    transform_segment,
    arc_trace,
)
from chalk.transform import Affine, P2_t, Transformable, V2_t
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import ApplyTransform, Primitive
    from chalk.types import Diagram


@tx.jit
def _trace(
    transform: tx.Affine, angles: tx.Angles, point: tx.P2_tC, d: tx.V2_tC
) -> Tuple[tx.Array, tx.Array]:
    point, direction = tx.np.broadcast_arrays(point, d)
    segments_shape = transform.shape[:-2]
    for _ in range(len(segments_shape)):
        point = point[..., None, :, :]
        direction = direction[..., None, :, :]

    t1 = tx.inv(transform)
    d, m = arc_trace(transform, angles, t1 @ point, t1 @ d)
    d = d.reshape(d.shape[:-2] + (-1,))
    m = m.reshape(m.shape[:-2] + (-1,))

    ad = tx.np.argsort(d + (1 - m) * 1e10, axis=-1)
    d = tx.np.take_along_axis(d, ad, axis=-1)
    m = tx.np.take_along_axis(m, ad, axis=-1)
    return (d, m)


@dataclass(frozen=True)
class TraceSpec(MappingSpec):
    pass


@dataclass(frozen=True)
class TraceTy(HiType):
    seg_ty: SegTy

    def lo_ty(self):
        return self.seg_ty.lo_ty()

    def lower_val(self, tr: Trace):
        return self.seg_ty.lower_val(tr.segment)

    def raise_val(self, transform, angles) -> Trace:
        return Trace(self.seg_ty.raise_val(transform, angles))

    def to_tangent_aval(self):
        return TraceTy(self.seg_ty)

    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        inner = self.seg_ty.str_short(short_dtypes, mesh_axis_types)[4:-1]
        return f"trace[{inner}]"

    __repr__ = str_short

    def dec_rank(self, size, spec):
        assert isinstance(spec, TraceSpec)
        return TraceTy(self.seg_ty.dec_rank(size, SegSpec()))

    def inc_rank(self, size, spec):
        assert isinstance(spec, TraceSpec)
        return TraceTy(self.seg_ty.inc_rank(size, SegSpec()))

    def leading_axis_spec(self):
        return TraceSpec()


@dataclass(frozen=True)
class Trace(Monoid, Transformable):
    """Opaque hijax trace wrapping a segment."""

    segment: Segment

    def map_prefix(self, fn):
        return make_trace(self.segment.map_prefix(fn))

    def __call__(self, point: P2_t, direction: V2_t) -> Tuple[tx.Scalars, tx.Mask]:
        return trace_ray(self, point, direction)

    def apply_transform(self, t: Affine) -> Trace:
        return transform_trace(self, t)

    def trace_v(self, p: P2_t, v: V2_t) -> Tuple[tx.V2_tC, tx.MaskC]:
        v = tx.norm(v)
        dists, m = trace_ray(self, p, v)
        d = tx.np.sort(dists + (1 - m) * 1e10, axis=-1)
        ad = tx.np.argsort(dists + (1 - m) * 1e10, axis=-1)
        m = tx.np.take_along_axis(m, ad, axis=-1)
        s = d[..., 0]
        return (tx.scale_vec(v, s), m[..., 0])

    def trace_p(self, p: P2_t, v: V2_t) -> Tuple[tx.P2_tC, tx.MaskC]:
        u, m = self.trace_v(p, v)
        return (p + u, m)

    def max_trace_v(self, p: P2_t, v: V2_t) -> Tuple[tx.V2_tC, tx.MaskC]:
        return self.trace_v(p, -v)

    def max_trace_p(self, p: P2_t, v: V2_t) -> Tuple[tx.P2_tC, tx.MaskC]:
        u, m = self.max_trace_v(p, v)
        return (p + u, m)


register_hitype(Trace, lambda tr: TraceTy(jax.typeof(tr.segment)))


class MakeTrace(VJPHiPrimitive):
    def __init__(self, seg_aval: SegTy):
        self.in_avals = (seg_aval,)
        self.out_aval = TraceTy(seg_aval)
        self.params = {}
        super().__init__()

    def expand(self, segment):
        return Trace(segment)

    def batch(self, axis_data, args, in_dims):
        (seg,) = args
        (d,) = in_dims
        if d is None:
            return make_trace(seg), None
        return make_trace(seg), TraceSpec()


class TransformTrace(VJPHiPrimitive):
    def __init__(self, tr_aval: TraceTy, t_aval):
        self.in_avals = (tr_aval, t_aval)
        self.out_aval = tr_aval
        self.params = {}
        super().__init__()

    def expand(self, tr: Trace, t):
        return Trace(transform_segment(tr.segment, t))

    def batch(self, axis_data, args, in_dims):
        tr, t = args
        if all(d is None for d in in_dims):
            return transform_trace(tr, t), None
        return transform_trace(tr, t), TraceSpec()


class TraceRay(VJPHiPrimitive):
    def __init__(self, tr_aval: TraceTy, p_aval, d_aval):
        self.in_avals = (tr_aval, p_aval, d_aval)
        # Output shapes follow _trace: dists/mask after flatten of hit axis.
        # Conservative: leave as leading-dir/point batch + trailing hits.
        seg = tr_aval.seg_ty
        p_batch = tuple(p_aval.shape[:-2])
        # _trace returns sorted hits; width is 2 * n_segs after reshape
        n_hits = max(seg.n_segs * 2, 1)
        out = ShapedArray(p_batch + (n_hits,), jnp.dtype(seg.dtype_name))
        mask = ShapedArray(p_batch + (n_hits,), jnp.dtype("bool"))
        self.out_aval = (out, mask)
        self.params = {}
        super().__init__()

    def expand(self, tr: Trace, point, direction):
        return _trace(*segment_parts(tr.segment), point, direction)

    def batch(self, axis_data, args, in_dims):
        tr, p, d = args
        if all(x is None for x in in_dims):
            return trace_ray(tr, p, d), None
        return trace_ray(tr, p, d), (0, 0)


class TraceSegment(VJPHiPrimitive):
    def __init__(self, tr_aval: TraceTy):
        self.in_avals = (tr_aval,)
        self.out_aval = tr_aval.seg_ty
        self.params = {}
        super().__init__()

    def expand(self, tr: Trace):
        return tr.segment

    def batch(self, axis_data, args, in_dims):
        (tr,) = args
        (d,) = in_dims
        if d is None:
            return trace_segment(tr), None
        return trace_segment(tr), SegSpec()


def make_trace(segment) -> Trace:
    return MakeTrace(jax.typeof(segment))(segment)


def transform_trace(tr, t) -> Trace:
    t = jnp.asarray(t)
    return TransformTrace(jax.typeof(tr), jax.typeof(t))(tr, t)


def trace_ray(tr, point, direction) -> Tuple[jax.Array, jax.Array]:
    point = jnp.asarray(point)
    direction = jnp.asarray(direction)
    return TraceRay(jax.typeof(tr), jax.typeof(point), jax.typeof(direction))(
        tr, point, direction
    )


def trace_segment(tr) -> Segment:
    return TraceSegment(jax.typeof(tr))(tr)


class _GetLocatedSegments(DiagramVisitor[Segment, Affine]):
    A_type = Segment

    def visit_primitive(self, diagram: Primitive, t: Affine) -> Segment:
        segment = diagram.prim_shape.located_segments()
        return transform_segment(segment, (t @ diagram.transform)[..., None, :, :])

    def visit_apply_transform(self, diagram: ApplyTransform, t: Affine) -> Segment:
        return diagram.diagram._accept(self, t @ diagram.transform)


def get_trace(self: Diagram) -> Trace:
    return make_trace(self._accept(_GetLocatedSegments(), tx.ident))


__all__ = [
    "Trace",
    "make_trace",
    "transform_trace",
    "trace_ray",
    "trace_segment",
]
