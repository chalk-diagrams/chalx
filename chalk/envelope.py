from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.hijax import (
    HiType,
    MappingSpec,
    ShapedArray,
    VJPHiPrimitive,
    register_hitype,
)
from jaxtyping import Float

import chalk.geom as geom
import chalk.transform as tx
from chalk.monoid import reduce_associative
from chalk.segment import (
    SegSpec,
    Segment,
    SegTy,
    concat_segments,
    segment_parts,
    transform_segment,
    arc_envelope,
)
from chalk.transform import (
    P2,
    V2,
    Affine,
    BoundingBox,
    P2_t,
    Scalars,
    Transformable,
    V2_t,
)
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import ApplyTransform, Compose, Primitive
    from chalk.types import Diagram


@jax.jit  # type: ignore
@partial(jnp.vectorize, signature="(3,3),(3,1)->(3,1),(3,1),(3,1),()")  # type: ignore
def pre_transform(t: Affine, v: V2_t) -> Tuple[V2_t, V2_t, V2_t, Scalars]:
    rt = tx._remove_translation_arr(t)
    inv_t = tx._inv_arr(rt)
    trans_t = tx._transpose_linear_arr(rt)
    u = jnp.zeros_like(v)
    u = u.at[..., 0, 0].set(-t[..., 0, 2]).at[..., 1, 0].set(-t[..., 1, 2])
    vi = inv_t @ v
    inp = trans_t @ v
    n2 = (inp * inp)[..., :2, 0].sum(-1, keepdims=True)[..., None]
    v_prim = inp / jnp.sqrt(n2)
    d = (v_prim * vi).sum((-2, -1))
    return v_prim, u, v, jnp.asarray(d)


@jax.jit
@partial(jnp.vectorize, signature="(3,1),(3,1),(),()->()")
def post_transform(u: V2_t, v: V2_t, d: tx.Floating, inner: tx.Floating) -> Scalars:
    after_linear = inner / d
    vv = (v * v).sum((-2, -1))
    scaled = u * (1.0 / vv)[..., None, None]
    diff = (scaled * v).sum((-2, -1))
    return jnp.asarray(after_linear - diff)


@jax.jit
def env(transform: tx.Affine, angles: tx.Angles, d: tx.V2_tC) -> tx.Array:
    batch_shape = d.shape[:-2]
    segments_shape = transform.shape[:-2]
    return_shape = batch_shape + segments_shape[:-1]
    if segments_shape[-1] == 0:
        return jnp.zeros(return_shape)
    for _ in range(len(segments_shape)):
        d = d[..., None, :, :]

    pre = pre_transform(transform, d)
    trans = arc_envelope(transform, angles, pre[0])
    v = post_transform(pre[1], pre[2], pre[3], trans).max(-1)  # type: ignore
    assert v.shape == return_shape, f"{v.shape} {return_shape}"
    return jnp.asarray(v)


ALL_DIR = jnp.stack(
    [tx._unit_x_arr, -tx._unit_x_arr, tx._unit_y_arr, -tx._unit_y_arr], axis=0
)


@dataclass(frozen=True)
class EnvSpec(MappingSpec):
    pass


@dataclass(frozen=True)
class EnvTy(HiType):
    seg_ty: SegTy

    def lo_ty(self):
        return self.seg_ty.lo_ty()

    def lower_val(self, envelope: Envelope):
        return self.seg_ty.lower_val(envelope.segment)

    def raise_val(self, transform, angles) -> Envelope:
        return Envelope(self.seg_ty.raise_val(transform, angles))

    def to_tangent_aval(self):
        return EnvTy(self.seg_ty)

    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        inner = self.seg_ty.str_short(short_dtypes, mesh_axis_types)[4:-1]
        return f"env[{inner}]"

    __repr__ = str_short

    def dec_rank(self, size, spec):
        assert isinstance(spec, EnvSpec)
        return EnvTy(self.seg_ty.dec_rank(size, SegSpec()))

    def inc_rank(self, size, spec):
        assert isinstance(spec, EnvSpec)
        return EnvTy(self.seg_ty.inc_rank(size, SegSpec()))

    def leading_axis_spec(self):
        return EnvSpec()


@dataclass(frozen=True)
class Envelope(Transformable):
    """Opaque hijax envelope wrapping a segment."""

    segment: Segment

    def map_prefix(self, fn):
        return make_envelope(self.segment.map_prefix(fn))

    def __call__(self, direction: tx.V2_tC) -> Float[tx.Array, "..."]:
        return envelope_measure(self, tx.data(direction))

    def __add__(self, other: Envelope) -> Envelope:
        return concat_envelopes(self, other)

    @classmethod
    def concat(cls, elems):
        from chalk.segment import Segment

        return reduce_associative(concat_envelopes, elems, make_envelope(Segment.empty()))

    @property
    def center(self) -> P2_t:
        d = envelope_measure(self, ALL_DIR)
        return P2((-d[1] + d[0]) / 2, (-d[3] + d[2]) / 2)

    @property
    def width(self) -> Scalars:
        d1 = envelope_measure(self, ALL_DIR[:2])
        return jnp.asarray(d1[0] + d1[1])

    @property
    def height(self) -> Scalars:
        d1 = envelope_measure(self, ALL_DIR[2:])
        return jnp.asarray(d1[0] + d1[1])

    def size(self) -> Tuple[Scalars, Scalars]:
        d = envelope_measure(self, ALL_DIR)
        return jnp.asarray(d[0] + d[1]), jnp.asarray(d[2] + d[3])

    def envelope_v(self, v: V2_t) -> V2_t:
        v = tx.norm(v)
        d = envelope_measure(self, v)
        return tx.scale_vec(v, d)

    @staticmethod
    def from_bounding_box(box: BoundingBox, d: V2_t) -> Scalars:
        v = box.rotate_rad(tx.rad(d)).br[:, 0, 0]
        return v / tx.length(d)

    def to_bounding_box(self) -> BoundingBox:
        d = envelope_measure(self, ALL_DIR)
        return tx.BoundingBox(V2(-d[1], -d[3]), V2(d[0], d[2]))

    def to_path(self, angle: int = 45) -> Iterable[P2_t]:
        pts = []
        for i in range(0, 361, angle):
            v = tx.polar(i)
            pts.append(tx.scale_vec(v, envelope_measure(self, v)))
        return pts

    def to_segments(self, angle: int = 45) -> V2_t:
        v = tx.polar(jnp.arange(0, 361, angle) * 1.0)
        return tx.scale_vec(v, envelope_measure(self, v))

    def apply_transform(self, t: Affine) -> Envelope:
        return transform_envelope(self, t)

    all_dir = ALL_DIR


register_hitype(Envelope, lambda e: EnvTy(jax.typeof(e.segment)))


class MakeEnvelope(VJPHiPrimitive):
    def __init__(self, seg_aval: SegTy):
        self.in_avals = (seg_aval,)
        self.out_aval = EnvTy(seg_aval)
        self.params = {}
        super().__init__()

    def expand(self, segment):
        return Envelope(segment)

    def batch(self, axis_data, args, in_dims):
        (seg,) = args
        (d,) = in_dims
        if d is None:
            return make_envelope(seg), None
        return make_envelope(seg), EnvSpec()


class ConcatEnvelopes(VJPHiPrimitive):
    def __init__(self, a: EnvTy, b: EnvTy):
        out_seg = SegTy(
            tuple(jnp.broadcast_shapes(a.seg_ty.batch_shape, b.seg_ty.batch_shape)),
            a.seg_ty.n_segs + b.seg_ty.n_segs,
            a.seg_ty.dtype_name,
        )
        self.in_avals = (a, b)
        self.out_aval = EnvTy(out_seg)
        self.params = {}
        super().__init__()

    def expand(self, a: Envelope, b: Envelope):
        return Envelope(concat_segments(a.segment, b.segment))

    def batch(self, axis_data, args, in_dims):
        a, b = args
        if all(d is None for d in in_dims):
            return concat_envelopes(a, b), None
        return concat_envelopes(a, b), EnvSpec()


class TransformEnvelope(VJPHiPrimitive):
    def __init__(self, env_aval: EnvTy, t_aval):
        self.in_avals = (env_aval, t_aval)
        self.out_aval = env_aval
        self.params = {}
        super().__init__()

    def expand(self, envelope: Envelope, t):
        return Envelope(transform_segment(envelope.segment, t[..., None, :, :]))

    def batch(self, axis_data, args, in_dims):
        env_, t = args
        if all(d is None for d in in_dims):
            return transform_envelope(env_, t), None
        return transform_envelope(env_, t), EnvSpec()


class EnvelopeMeasure(VJPHiPrimitive):
    def __init__(self, env_aval: EnvTy, dir_aval):
        self.in_avals = (env_aval, dir_aval)
        dir_batch = tuple(dir_aval.shape[:-2])
        out_shape = dir_batch + env_aval.seg_ty.batch_shape
        self.out_aval = ShapedArray(out_shape, jnp.dtype(env_aval.seg_ty.dtype_name))
        self.params = {}
        super().__init__()

    def expand(self, envelope: Envelope, direction):
        return env(*segment_parts(envelope.segment), direction)

    def batch(self, axis_data, args, in_dims):
        env_, d = args
        de, dd = in_dims
        if de is None and dd is None:
            return envelope_measure(env_, d), None
        out_dim = 0 if de is not None else (0 if dd is not None else None)
        return envelope_measure(env_, d), out_dim


class EnvelopeSegment(VJPHiPrimitive):
    def __init__(self, env_aval: EnvTy):
        self.in_avals = (env_aval,)
        self.out_aval = env_aval.seg_ty
        self.params = {}
        super().__init__()

    def expand(self, envelope: Envelope):
        return envelope.segment

    def batch(self, axis_data, args, in_dims):
        (env_,) = args
        (d,) = in_dims
        if d is None:
            return envelope_segment(env_), None
        return envelope_segment(env_), SegSpec()


def make_envelope(segment) -> Envelope:
    return MakeEnvelope(jax.typeof(segment))(segment)


def concat_envelopes(a, b) -> Envelope:
    return ConcatEnvelopes(jax.typeof(a), jax.typeof(b))(a, b)


def transform_envelope(envelope, t) -> Envelope:
    t = tx.data(t)
    return TransformEnvelope(jax.typeof(envelope), jax.typeof(t))(envelope, t)


def envelope_measure(envelope, direction) -> jax.Array:
    direction = tx.data(direction)
    return EnvelopeMeasure(jax.typeof(envelope), jax.typeof(direction))(
        envelope, direction
    )


def envelope_segment(envelope) -> Segment:
    return EnvelopeSegment(jax.typeof(envelope))(envelope)


class GetLocatedSegments(DiagramVisitor[Segment, Affine]):
    A_type = Segment

    def visit_primitive(self, diagram: Primitive, t: Affine) -> Segment:
        segment = diagram.prim_shape.located_segments()
        t = t @ diagram.transform
        if jax.typeof(t).batch:
            t = geom.make_xf(tx.data(t)[..., None, :, :])
        return transform_segment(segment, t)

    def visit_compose(self, diagram: Compose, t: Affine) -> Segment:
        if diagram.envelope is not None:
            return diagram.envelope._accept(self, t)
        return self.A_type.concat([d._accept(self, t) for d in diagram.diagrams])

    def visit_apply_transform(self, diagram: ApplyTransform, t: Affine) -> Segment:
        return diagram.diagram._accept(self, t @ diagram.transform)


def get_envelope(self: Diagram, t: Optional[Affine] = None) -> Envelope:
    if t is None:
        t = tx.ident
    segment = self._accept(GetLocatedSegments(), t)
    transform, _ = segment_parts(segment)
    seg_shape = transform.shape[:-2]
    assert (
        seg_shape[: len(self.shape)] == self.shape
    ), f"{transform.shape} {self.shape}"
    return make_envelope(segment)


__all__ = [
    "Envelope",
    "make_envelope",
    "concat_envelopes",
    "transform_envelope",
    "envelope_measure",
    "envelope_segment",
]
