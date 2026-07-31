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
    Zero,
    apply_derived_linearization,
    linearize_from_jvp,
    register_hitype,
    transpose_jvp,
    vjp_fwd_from_jvp,
)

import chalk.geom as geom
import chalk.transform as tx
from chalk.segment import (
    SegSpec,
    Segment,
    SegTy,
    make_segment,
    segment_parts,
    transform_segment,
    arc_trace,
)
from chalk.transform import Affine, P2_t, Transformable, V2_t
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import ApplyTransform, Primitive
    from chalk.types import Diagram


def _trace(
    transform: tx.Affine, angles: tx.Angles, point: tx.P2_tC, d: tx.V2_tC
) -> Tuple[tx.Array, tx.Array]:
    point, direction = tx.np.broadcast_arrays(point, d)
    # Broadcast ray up to segment batch: (*ray_batch, *seg_batch, 3, 1)
    seg_batch = transform.shape[:-2]
    point = jnp.reshape(point, point.shape[:-2] + (1,) * len(seg_batch) + point.shape[-2:])
    direction = jnp.reshape(
        direction, direction.shape[:-2] + (1,) * len(seg_batch) + direction.shape[-2:]
    )
    t1 = jnp.linalg.inv(jnp.asarray(transform))
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

    @property
    def dtype(self):
        return jnp.dtype(self.seg_ty.dtype_name)

    def lo_ty(self):
        return self.seg_ty.lo_ty()

    def lower_val(self, tr: Trace):
        return self.seg_ty.lower_val(tr.segment)

    def raise_val(self, transform, angles) -> Trace:
        return Trace(self.seg_ty.raise_val(transform, angles))

    def to_tangent_aval(self):
        return TraceTy(self.seg_ty)

    def vspace_zero(self):
        return Trace(self.seg_ty.vspace_zero())

    def vspace_add(self, x: Trace, y: Trace):
        return make_trace(
            self.seg_ty.vspace_add(trace_segment(x), trace_segment(y))
        )

    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        inner = self.seg_ty.str_short(short_dtypes, mesh_axis_types)[4:-1]
        return f"trace[{inner}]"

    __repr__ = str_short

    @property
    def dtype(self):
        return self.seg_ty.dtype

    def dec_rank(self, size, spec):
        assert isinstance(spec, TraceSpec)
        return TraceTy(self.seg_ty.dec_rank(size, SegSpec()))

    def inc_rank(self, size, spec):
        assert isinstance(spec, TraceSpec)
        return TraceTy(self.seg_ty.inc_rank(size, SegSpec()))

    def leading_axis_spec(self):
        return TraceSpec()


@dataclass(frozen=True)
class Trace(Transformable):
    """Opaque hijax trace wrapping a segment."""

    segment: Segment

    def map_prefix(self, fn):
        return make_trace(self.segment.map_prefix(fn))

    def __call__(self, point: P2_t, direction: V2_t) -> Tuple[tx.Scalars, tx.Mask]:
        return trace_ray(self, point, direction)

    def apply_transform(self, t: Affine) -> Trace:
        return transform_trace(self, t)

    def trace_v(self, p: P2_t, v: V2_t) -> Tuple[tx.V2_tC, tx.MaskC]:
        vn = tx.norm(v)
        dists, m = trace_ray(self, tx.data(p), tx.data(vn))
        d = tx.np.sort(dists + (1 - m) * 1e10, axis=-1)
        ad = tx.np.argsort(dists + (1 - m) * 1e10, axis=-1)
        m = tx.np.take_along_axis(m, ad, axis=-1)
        s = d[..., 0]
        return (tx.scale_vec(vn, s), m[..., 0])

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

    def jvp(self, primals, tangents):
        (seg,), (dseg,) = primals, tangents
        prim = make_trace(seg)
        if isinstance(dseg, Zero):
            return prim, Zero(self.out_aval.to_tangent_aval())
        return prim, make_trace(dseg)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

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

    def vjp_fwd(self, nzs_in, tr, t):
        xf, ang = segment_parts(trace_segment(tr))
        t_arr = jnp.asarray(t)

        def f(xf_, ang_, t_):
            return t_ @ xf_, ang_

        (out_xf, out_ang), vjp = jax.vjp(f, xf, ang, t_arr)
        return make_trace(make_segment(out_xf, out_ang)), vjp

    def vjp_bwd_retval(self, vjp, g):
        if isinstance(g, Zero):
            dxf, dang, dt = vjp(
                (
                    jnp.zeros(self.out_aval.seg_ty.batch_shape + (self.out_aval.seg_ty.n_segs, 3, 3)),
                    jnp.zeros(self.out_aval.seg_ty.batch_shape + (self.out_aval.seg_ty.n_segs, 2)),
                )
            )
            return make_trace(make_segment(dxf, dang)), dt
        gxf, gang = segment_parts(trace_segment(g))
        dxf, dang, dt = vjp((gxf, gang))
        return make_trace(make_segment(dxf, dang)), dt

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
        mask = ShapedArray(p_batch + (n_hits,), jnp.dtype(seg.dtype_name))
        self.out_aval = (out, mask)
        self.params = {}
        super().__init__()

    def expand(self, tr: Trace, point, direction):
        xf, ang = segment_parts(trace_segment(tr))
        dist, mask = _trace(xf, ang, jnp.asarray(point), jnp.asarray(direction))
        return dist, jnp.asarray(mask, dtype=dist.dtype)

    def vjp_fwd(self, nzs_in, tr, point, direction):
        xf, ang = segment_parts(trace_segment(tr))
        point = jnp.asarray(point)
        direction = jnp.asarray(direction)
        dist, mask = _trace(xf, ang, point, direction)

        def dist_fn(xf_, ang_, p_, d_):
            return _trace(xf_, ang_, p_, d_)[0]

        _, vjp = jax.vjp(dist_fn, xf, ang, point, direction)
        return (dist, jnp.asarray(mask, dtype=dist.dtype)), vjp

    def vjp_bwd_retval(self, vjp, g):
        g_dist, _g_mask = g
        seg_ty = self.in_avals[0].seg_ty
        if isinstance(g_dist, Zero):
            dt = jnp.dtype(seg_ty.dtype_name)
            z_xf = jnp.zeros(seg_ty.batch_shape + (seg_ty.n_segs, 3, 3), dt)
            z_ang = jnp.zeros(seg_ty.batch_shape + (seg_ty.n_segs, 2), dt)
            z_p = jnp.zeros(self.in_avals[1].shape, self.in_avals[1].dtype)
            z_d = jnp.zeros(self.in_avals[2].shape, self.in_avals[2].dtype)
            return make_trace(make_segment(z_xf, z_ang)), z_p, z_d
        dxf, dang, dp, dd = vjp(jnp.asarray(g_dist))
        return make_trace(make_segment(dxf, dang)), dp, dd

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

    def jvp(self, primals, tangents):
        (tr,), (dtr,) = primals, tangents
        prim = trace_segment(tr)
        if isinstance(dtr, Zero):
            return prim, Zero(self.out_aval.to_tangent_aval())
        return prim, trace_segment(dtr)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        (tr,) = args
        (d,) = in_dims
        if d is None:
            return trace_segment(tr), None
        return trace_segment(tr), SegSpec()


def make_trace(segment) -> Trace:
    return MakeTrace(jax.typeof(segment))(segment)


def transform_trace(tr, t) -> Trace:
    return TransformTrace(jax.typeof(tr), jax.typeof(t))(tr, t)


def trace_ray(tr, point, direction) -> Tuple[jax.Array, jax.Array]:
    point = tx.data(point)
    direction = tx.data(direction)
    return TraceRay(jax.typeof(tr), jax.typeof(point), jax.typeof(direction))(
        tr, point, direction
    )


def trace_segment(tr) -> Segment:
    return TraceSegment(jax.typeof(tr))(tr)


class _GetLocatedSegments(DiagramVisitor[Segment, Affine]):
    A_type = Segment

    def visit_primitive(self, diagram: Primitive, t: Affine) -> Segment:
        segment = diagram.prim_shape.located_segments()
        transform = t @ diagram.transform
        transform = geom.make_xf(tx.data(transform)[..., None, :, :])
        return transform_segment(segment, transform)

    def visit_apply_transform(self, diagram: ApplyTransform, t: Affine) -> Segment:
        return diagram.diagram._accept(self, t @ diagram.transform)


def get_trace(self: Diagram) -> Trace:
    from jax.core import Tracer

    from chalk.diag import DiagTy

    ty = jax.typeof(self)
    if not isinstance(ty, DiagTy) or not isinstance(self, Tracer):
        return make_trace(self._accept(_GetLocatedSegments(), tx.ident))
    return _get_trace_traced(self, ty)


def _get_trace_traced(d, ty) -> Trace:
    from chalk.diag import diag_prim_trace, diag_uncons_child, diag_uncons_xf

    if ty.tag == "xf":
        child, xf = diag_uncons_xf(d)
        return transform_trace(_get_trace_traced(child, ty.child_tys[0]), xf)
    if ty.tag in ("style", "name"):
        child = diag_uncons_child(d)
        return _get_trace_traced(child, ty.child_tys[0])
    if ty.tag == "axis":
        child = diag_uncons_child(d)
        return _get_trace_traced(child, ty.child_tys[0])
    if ty.tag in ("prim", "empty", "compose"):
        return diag_prim_trace(d)
    raise ValueError(ty.tag)


def trace_measure(*args, **kwargs):
    from chalk.measure import trace_measure as _tm

    return _tm(*args, **kwargs)


__all__ = [
    "Trace",
    "make_trace",
    "transform_trace",
    "trace_ray",
    "trace_segment",
    "trace_measure",
]
