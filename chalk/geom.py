"""Opaque hijax geometry: vectors, points, and affine transforms.

``p2[]`` / ``v2[]`` / ``xf[]`` appear as single values in jaxprs.
The tangent type of a point is a vector (``P2`` → ``V2``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple, Union

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

_DT = jnp.dtype("float64")


def _accum(primals_or_accums, cotangents):
    from jax.experimental.hijax import GradAccum

    for slot, ct in zip(primals_or_accums, cotangents):
        if isinstance(slot, GradAccum) and ct is not None:
            slot.accum(ct)
    return None


def _batch_of_hom(arr) -> Tuple[int, ...]:
    a = jnp.asarray(arr)
    return tuple(a.shape[:-2])


def _batch_of_xf(arr) -> Tuple[int, ...]:
    a = jnp.asarray(arr)
    return tuple(a.shape[:-2])


def _dtype_name(arr) -> str:
    return jnp.asarray(arr).dtype.name


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Vec:
    """Homogeneous vector ``(x, y, 0)``. Opaque hijax ``v2[*B]``."""

    data: jax.Array

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(jnp.asarray(self.data).shape)

    def __add__(self, other):
        return v2_add(self, as_vec(other))

    def __radd__(self, other):
        return v2_add(as_vec(other), self)

    def __sub__(self, other):
        return v2_sub(self, as_vec(other))

    def __rsub__(self, other):
        return v2_sub(as_vec(other), self)

    def __neg__(self):
        return v2_neg(self)

    def __mul__(self, scalar):
        if isinstance(scalar, (Vec, Pt, Affine)):
            raise TypeError("use dot/cross for vector–vector products")
        return v2_scale(self, scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return v2_scale(self, 1.0 / scalar)


@dataclass(frozen=True)
class Pt:
    """Homogeneous point ``(x, y, 1)``. Opaque hijax ``p2[*B]``."""

    data: jax.Array

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(jnp.asarray(self.data).shape)

    def __add__(self, other):
        return p2_add_v2(self, as_vec(other))

    def __radd__(self, other):
        return p2_add_v2(self, as_vec(other))

    def __sub__(self, other):
        other = _coerce_pv(other)
        if isinstance(other, Pt):
            return p2_sub_p2(self, other)
        return p2_sub_v2(self, as_vec(other))

    def __neg__(self):
        raise TypeError("points have no additive inverse; subtract from another point")


@dataclass(frozen=True)
class Affine:
    """Homogeneous 3×3 affine transform. Opaque hijax ``xf[*B]``."""

    data: jax.Array

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(jnp.asarray(self.data).shape)

    def __matmul__(self, other):
        other = _coerce_any(other)
        if isinstance(other, Affine):
            return xf_compose(self, other)
        if isinstance(other, Pt):
            return xf_apply_pt(self, other)
        if isinstance(other, Vec):
            return xf_apply_vec(self, other)
        arr = jnp.asarray(other)
        if arr.shape[-2:] == (3, 3):
            return xf_compose(self, make_xf(arr))
        if arr.shape[-2:] == (3, 1):
            return xf_apply_hom(self, arr)
        raise TypeError(f"unsupported @ operand shape {arr.shape}")

    def __rmatmul__(self, other):
        other = _coerce_any(other)
        if isinstance(other, Affine):
            return xf_compose(other, self)
        arr = jnp.asarray(other)
        if arr.shape[-2:] == (3, 3):
            return xf_compose(make_xf(arr), self)
        raise TypeError(f"unsupported rmatmul operand {type(other)}")


Geom = Union[Vec, Pt, Affine]


def data(x: Any) -> jax.Array:
    """Lower geom values to arrays (eager, JIT, and AD)."""
    if isinstance(x, (Vec, Pt, Affine)):
        return jnp.asarray(x.data)
    try:
        ty = jax.typeof(x)
    except Exception:
        return jnp.asarray(x)
    if isinstance(ty, V2Ty):
        return v2_to_array(x)
    if isinstance(ty, P2Ty):
        return p2_to_array(x)
    if isinstance(ty, XfTy):
        return xf_to_array(x)
    return jnp.asarray(x)


def as_vec(x: Any) -> Vec:
    if isinstance(x, Vec):
        return x
    if isinstance(x, Pt):
        return to_vec(x)
    return make_v2_from_data(data(x).at[..., 2, 0].set(0.0))


def _coerce_pv(x: Any):
    if isinstance(x, (Pt, Vec)):
        return x
    arr = jnp.asarray(x)
    w = arr[..., 2, 0]
    # Heuristic for eager array interop only.
    if jnp.all(jnp.abs(w - 1.0) < 1e-6):
        return make_p2_from_data(arr)
    return make_v2_from_data(arr.at[..., 2, 0].set(0.0))


def _coerce_any(x: Any):
    if isinstance(x, (Pt, Vec, Affine)):
        return x
    return x


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeomSpec(MappingSpec):
    pass


@dataclass(frozen=True)
class V2Ty(HiType):
    batch: Tuple[int, ...] = ()
    dtype_name: str = "float64"

    def lo_ty(self):
        return [ShapedArray(self.batch + (3, 1), jnp.dtype(self.dtype_name))]

    def lower_val(self, v: Vec):
        return [v.data]

    def raise_val(self, arr) -> Vec:
        return Vec(arr)

    def to_tangent_aval(self):
        return V2Ty(self.batch, self.dtype_name)

    def vspace_zero(self):
        return Vec(jnp.zeros(self.batch + (3, 1), dtype=jnp.dtype(self.dtype_name)))

    def vspace_add(self, x: Vec, y: Vec):
        return Vec(x.data + y.data)

    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        b = ",".join(str(d) for d in self.batch)
        return f"v2[{b}]" if b else "v2[]"

    __repr__ = str_short

    @property
    def dtype(self):
        return jnp.dtype(self.dtype_name)

    def dec_rank(self, size, spec):
        assert isinstance(spec, GeomSpec) and self.batch and self.batch[0] == size
        return V2Ty(self.batch[1:], self.dtype_name)

    def inc_rank(self, size, spec):
        assert isinstance(spec, GeomSpec)
        return V2Ty((size, *self.batch), self.dtype_name)

    def leading_axis_spec(self):
        return GeomSpec()


@dataclass(frozen=True)
class P2Ty(HiType):
    batch: Tuple[int, ...] = ()
    dtype_name: str = "float64"

    def lo_ty(self):
        return [ShapedArray(self.batch + (3, 1), jnp.dtype(self.dtype_name))]

    def lower_val(self, p: Pt):
        return [p.data]

    def raise_val(self, arr) -> Pt:
        return Pt(arr)

    def to_tangent_aval(self):
        return V2Ty(self.batch, self.dtype_name)

    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        b = ",".join(str(d) for d in self.batch)
        return f"p2[{b}]" if b else "p2[]"

    __repr__ = str_short

    @property
    def dtype(self):
        return jnp.dtype(self.dtype_name)

    def dec_rank(self, size, spec):
        assert isinstance(spec, GeomSpec) and self.batch and self.batch[0] == size
        return P2Ty(self.batch[1:], self.dtype_name)

    def inc_rank(self, size, spec):
        assert isinstance(spec, GeomSpec)
        return P2Ty((size, *self.batch), self.dtype_name)

    def leading_axis_spec(self):
        return GeomSpec()


@dataclass(frozen=True)
class XfTy(HiType):
    batch: Tuple[int, ...] = ()
    dtype_name: str = "float64"

    def lo_ty(self):
        return [ShapedArray(self.batch + (3, 3), jnp.dtype(self.dtype_name))]

    def lower_val(self, xf: Affine):
        return [xf.data]

    def raise_val(self, arr) -> Affine:
        return Affine(arr)

    def to_tangent_aval(self):
        return XfTy(self.batch, self.dtype_name)

    def vspace_zero(self):
        return Affine(jnp.zeros(self.batch + (3, 3), dtype=jnp.dtype(self.dtype_name)))

    def vspace_add(self, x: Affine, y: Affine):
        return Affine(x.data + y.data)

    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        b = ",".join(str(d) for d in self.batch)
        return f"xf[{b}]" if b else "xf[]"

    __repr__ = str_short

    @property
    def dtype(self):
        return jnp.dtype(self.dtype_name)

    def dec_rank(self, size, spec):
        assert isinstance(spec, GeomSpec) and self.batch and self.batch[0] == size
        return XfTy(self.batch[1:], self.dtype_name)

    def inc_rank(self, size, spec):
        assert isinstance(spec, GeomSpec)
        return XfTy((size, *self.batch), self.dtype_name)

    def leading_axis_spec(self):
        return GeomSpec()


def _typeof_vec(v: Vec) -> V2Ty:
    d = jnp.asarray(v.data)
    return V2Ty(_batch_of_hom(d), d.dtype.name)


def _typeof_pt(p: Pt) -> P2Ty:
    d = jnp.asarray(p.data)
    return P2Ty(_batch_of_hom(d), d.dtype.name)


def _typeof_xf(xf: Affine) -> XfTy:
    d = jnp.asarray(xf.data)
    return XfTy(_batch_of_xf(d), d.dtype.name)


register_hitype(Vec, _typeof_vec)
register_hitype(Pt, _typeof_pt)
register_hitype(Affine, _typeof_xf)


def _out_dim(in_dims):
    return None if all(d is None for d in in_dims) else GeomSpec()


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


class MakeV2(VJPHiPrimitive):
    def __init__(self, x_aval, y_aval):
        self.in_avals = (x_aval, y_aval)
        batch = tuple(jnp.broadcast_shapes(x_aval.shape, y_aval.shape))
        self.out_aval = V2Ty(batch, jnp.dtype(x_aval.dtype).name)
        self.params = {}
        super().__init__()

    def expand(self, x, y):
        x, y, z = jnp.broadcast_arrays(
            jnp.asarray(x, dtype=_DT),
            jnp.asarray(y, dtype=_DT),
            jnp.asarray(0.0, dtype=_DT),
        )
        return Vec(jnp.stack([x, y, z], axis=-1)[..., None])

    def jvp(self, primals, tangents):
        x, y = primals
        dx, dy = tangents
        prim_out = make_v2(x, y)
        if isinstance(dx, Zero):
            dx = jnp.zeros_like(jnp.asarray(x, dtype=_DT))
        if isinstance(dy, Zero):
            dy = jnp.zeros_like(jnp.asarray(y, dtype=_DT))
        return prim_out, make_v2(dx, dy)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, x, y):
        from jax._src.ad_util import Zero as AdZero

        v_ct = cts
        if isinstance(v_ct, (Zero, AdZero)):
            return None
        arr = v2_to_array(v_ct)
        return _accum((x, y), (arr[..., 0, 0], arr[..., 1, 0]))

    def batch(self, axis_data, args, in_dims):
        return make_v2(*args), _out_dim(in_dims)


def make_v2(x, y) -> Vec:
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return MakeV2(jax.typeof(x), jax.typeof(y))(x, y)


class MakeP2(VJPHiPrimitive):
    def __init__(self, x_aval, y_aval):
        self.in_avals = (x_aval, y_aval)
        batch = tuple(jnp.broadcast_shapes(x_aval.shape, y_aval.shape))
        self.out_aval = P2Ty(batch, jnp.dtype(x_aval.dtype).name)
        self.params = {}
        super().__init__()

    def expand(self, x, y):
        x, y, w = jnp.broadcast_arrays(
            jnp.asarray(x, dtype=_DT),
            jnp.asarray(y, dtype=_DT),
            jnp.asarray(1.0, dtype=_DT),
        )
        return Pt(jnp.stack([x, y, w], axis=-1)[..., None])

    def jvp(self, primals, tangents):
        x, y = primals
        dx, dy = tangents
        prim_out = make_p2(x, y)
        if isinstance(dx, Zero):
            dx = jnp.zeros_like(jnp.asarray(x, dtype=_DT))
        if isinstance(dy, Zero):
            dy = jnp.zeros_like(jnp.asarray(y, dtype=_DT))
        return prim_out, make_v2(dx, dy)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        return make_p2(*args), _out_dim(in_dims)


def make_p2(x, y) -> Pt:
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    return MakeP2(jax.typeof(x), jax.typeof(y))(x, y)


class MakeFromData(VJPHiPrimitive):
    def __init__(self, arr_aval, kind: str):
        self.in_avals = (arr_aval,)
        batch = tuple(arr_aval.shape[:-2])
        dt = jnp.dtype(arr_aval.dtype).name
        if kind == "v2":
            out = V2Ty(batch, dt)
        elif kind == "p2":
            out = P2Ty(batch, dt)
        else:
            out = XfTy(batch, dt)
        self.out_aval = out
        self.params = dict(kind=kind)
        super().__init__()

    def expand(self, arr):
        arr = jnp.asarray(arr)
        if self.kind == "v2":
            return Vec(arr)
        if self.kind == "p2":
            return Pt(arr)
        return Affine(arr)

    def jvp(self, primals, tangents):
        (arr,) = primals
        (darr,) = tangents
        prim_out = MakeFromData(jax.typeof(arr), self.kind)(arr)
        if self.kind == "p2":
            if isinstance(darr, Zero):
                z = jnp.zeros_like(jnp.asarray(arr)).at[..., 2, 0].set(0.0)
                return prim_out, make_v2_from_data(z)
            d = jnp.asarray(darr).at[..., 2, 0].set(0.0)
            return prim_out, make_v2_from_data(d)
        if isinstance(darr, Zero):
            darr = jnp.zeros_like(jnp.asarray(arr))
        return prim_out, MakeFromData(jax.typeof(jnp.asarray(darr)), self.kind)(darr)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, arr):
        from jax._src.ad_util import Zero as AdZero

        if isinstance(cts, (Zero, AdZero)):
            return None
        if self.kind == "xf":
            ct = xf_to_array(cts)
        else:
            ct = v2_to_array(cts)
        return _accum((arr,), (ct,))

    def batch(self, axis_data, args, in_dims):
        (arr,) = args
        if self.kind == "v2":
            out = make_v2_from_data(arr)
        elif self.kind == "p2":
            out = make_p2_from_data(arr)
        else:
            out = make_xf(arr)
        return out, _out_dim(in_dims)


def make_v2_from_data(arr) -> Vec:
    arr = jnp.asarray(arr)
    return MakeFromData(jax.typeof(arr), "v2")(arr)


def make_p2_from_data(arr) -> Pt:
    arr = jnp.asarray(arr)
    return MakeFromData(jax.typeof(arr), "p2")(arr)


def make_xf(arr) -> Affine:
    arr = jnp.asarray(arr)
    return MakeFromData(jax.typeof(arr), "xf")(arr)


class GeomToArray(VJPHiPrimitive):
    """Lower opaque geom to its array payload."""

    def __init__(self, aval, kind: str):
        self.in_avals = (aval,)
        if kind == "xf":
            self.out_aval = ShapedArray(aval.batch + (3, 3), jnp.dtype(aval.dtype_name))
        else:
            self.out_aval = ShapedArray(aval.batch + (3, 1), jnp.dtype(aval.dtype_name))
        self.params = dict(kind=kind)
        super().__init__()

    def expand(self, val):
        return jnp.asarray(val.data)

    def jvp(self, primals, tangents):
        (val,), (dval,) = primals, tangents
        prim = GeomToArray(jax.typeof(val), self.kind)(val)
        if isinstance(dval, Zero):
            return prim, jnp.zeros(self.out_aval.shape, dtype=self.out_aval.dtype)
        dkind = "v2" if self.kind == "p2" else self.kind
        return prim, GeomToArray(jax.typeof(dval), dkind)(dval)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, val):
        from jax._src.ad_util import Zero as AdZero

        arr_ct = cts
        if isinstance(arr_ct, (Zero, AdZero)):
            return None
        if self.kind == "xf":
            val_ct = make_xf(arr_ct)
        elif self.kind == "p2":
            val_ct = make_v2_from_data(jnp.asarray(arr_ct).at[..., 2, 0].set(0.0))
        else:
            val_ct = make_v2_from_data(arr_ct)
        return _accum((val,), (val_ct,))

    def batch(self, axis_data, args, in_dims):
        (val,) = args
        if self.kind == "xf":
            out = xf_to_array(val)
        elif self.kind == "p2":
            out = p2_to_array(val)
        else:
            out = v2_to_array(val)
        return out, (None if all(d is None for d in in_dims) else 0)


def v2_to_array(v) -> jax.Array:
    return GeomToArray(jax.typeof(v), "v2")(v)


def p2_to_array(p) -> jax.Array:
    return GeomToArray(jax.typeof(p), "p2")(p)


def xf_to_array(t) -> jax.Array:
    return GeomToArray(jax.typeof(t), "xf")(t)


# ---------------------------------------------------------------------------
# Binary ops
# ---------------------------------------------------------------------------


class V2Bin(VJPHiPrimitive):
    def __init__(self, a: V2Ty, b: V2Ty, op: str):
        batch = tuple(jnp.broadcast_shapes(a.batch, b.batch))
        self.in_avals = (a, b)
        self.out_aval = V2Ty(batch, a.dtype_name)
        self.params = dict(op=op)
        super().__init__()

    def expand(self, a: Vec, b: Vec):
        if self.op == "add":
            return Vec(a.data + b.data)
        return Vec(a.data - b.data)

    def jvp(self, primals, tangents):
        a, b = primals
        da, db = tangents
        prim_out = v2_add(a, b) if self.op == "add" else v2_sub(a, b)
        if isinstance(da, Zero):
            da = V2Ty(jax.typeof(a).batch, jax.typeof(a).dtype_name).vspace_zero()
        if isinstance(db, Zero):
            db = V2Ty(jax.typeof(b).batch, jax.typeof(b).dtype_name).vspace_zero()
        tan_out = v2_add(da, db) if self.op == "add" else v2_sub(da, db)
        return prim_out, tan_out

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        a, b = args
        out = v2_add(a, b) if self.op == "add" else v2_sub(a, b)
        return out, _out_dim(in_dims)


def v2_add(a: Vec, b: Vec) -> Vec:
    return V2Bin(jax.typeof(a), jax.typeof(b), "add")(a, b)


def v2_sub(a: Vec, b: Vec) -> Vec:
    return V2Bin(jax.typeof(a), jax.typeof(b), "sub")(a, b)


class V2Neg(VJPHiPrimitive):
    def __init__(self, a: V2Ty):
        self.in_avals = (a,)
        self.out_aval = a
        self.params = {}
        super().__init__()

    def expand(self, a: Vec):
        return Vec(-a.data)

    def jvp(self, primals, tangents):
        (a,), (da,) = primals, tangents
        if isinstance(da, Zero):
            da = jax.typeof(a).vspace_zero()
        return v2_neg(a), v2_neg(da)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        return v2_neg(args[0]), _out_dim(in_dims)


def v2_neg(a: Vec) -> Vec:
    return V2Neg(jax.typeof(a))(a)


class V2Scale(VJPHiPrimitive):
    def __init__(self, v: V2Ty, s_aval):
        batch = tuple(jnp.broadcast_shapes(v.batch, tuple(s_aval.shape)))
        self.in_avals = (v, s_aval)
        self.out_aval = V2Ty(batch, v.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, v: Vec, s):
        s = jnp.asarray(s)[..., None, None]
        return Vec(v.data * s)

    def jvp(self, primals, tangents):
        v, s = primals
        dv, ds = tangents
        prim_out = v2_scale(v, s)
        if isinstance(dv, Zero):
            dv = jax.typeof(v).vspace_zero()
        term_v = v2_scale(dv, s)
        if isinstance(ds, Zero):
            return prim_out, term_v
        return prim_out, v2_add(term_v, v2_scale(v, ds))

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp

    def vjp_bwd_retval(self, res, g):
        raise NotImplementedError

    def batch(self, axis_data, args, in_dims):
        return v2_scale(*args), _out_dim(in_dims)


def v2_scale(v: Vec, s) -> Vec:
    s = jnp.asarray(s)
    return V2Scale(jax.typeof(v), jax.typeof(s))(v, s)


class P2AddV2(VJPHiPrimitive):
    def __init__(self, p: P2Ty, v: V2Ty):
        batch = tuple(jnp.broadcast_shapes(p.batch, v.batch))
        self.in_avals = (p, v)
        self.out_aval = P2Ty(batch, p.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, p: Pt, v: Vec):
        return Pt(p.data + v.data)

    def jvp(self, primals, tangents):
        p, v = primals
        dp, dv = tangents
        prim_out = p2_add_v2(p, v)
        if isinstance(dp, Zero):
            dp = jax.typeof(p).to_tangent_aval().vspace_zero()
        if isinstance(dv, Zero):
            dv = jax.typeof(v).vspace_zero()
        return prim_out, v2_add(dp, dv)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        return p2_add_v2(*args), _out_dim(in_dims)


def p2_add_v2(p: Pt, v: Vec) -> Pt:
    return P2AddV2(jax.typeof(p), jax.typeof(v))(p, v)


class P2SubP2(VJPHiPrimitive):
    def __init__(self, p: P2Ty, q: P2Ty):
        batch = tuple(jnp.broadcast_shapes(p.batch, q.batch))
        self.in_avals = (p, q)
        self.out_aval = V2Ty(batch, p.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, p: Pt, q: Pt):
        d = p.data - q.data
        return Vec(d.at[..., 2, 0].set(0.0))

    def jvp(self, primals, tangents):
        p, q = primals
        dp, dq = tangents
        prim_out = p2_sub_p2(p, q)
        if isinstance(dp, Zero):
            dp = jax.typeof(p).to_tangent_aval().vspace_zero()
        if isinstance(dq, Zero):
            dq = jax.typeof(q).to_tangent_aval().vspace_zero()
        return prim_out, v2_sub(dp, dq)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        return p2_sub_p2(*args), _out_dim(in_dims)


def p2_sub_p2(p: Pt, q: Pt) -> Vec:
    return P2SubP2(jax.typeof(p), jax.typeof(q))(p, q)


class P2SubV2(VJPHiPrimitive):
    def __init__(self, p: P2Ty, v: V2Ty):
        batch = tuple(jnp.broadcast_shapes(p.batch, v.batch))
        self.in_avals = (p, v)
        self.out_aval = P2Ty(batch, p.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, p: Pt, v: Vec):
        return Pt(p.data - v.data)

    def jvp(self, primals, tangents):
        p, v = primals
        dp, dv = tangents
        prim_out = p2_sub_v2(p, v)
        if isinstance(dp, Zero):
            dp = jax.typeof(p).to_tangent_aval().vspace_zero()
        if isinstance(dv, Zero):
            dv = jax.typeof(v).vspace_zero()
        return prim_out, v2_sub(dp, dv)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        return p2_sub_v2(*args), _out_dim(in_dims)


def p2_sub_v2(p: Pt, v: Vec) -> Pt:
    return P2SubV2(jax.typeof(p), jax.typeof(v))(p, v)


class ToVec(VJPHiPrimitive):
    def __init__(self, p: P2Ty):
        self.in_avals = (p,)
        self.out_aval = V2Ty(p.batch, p.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, p: Pt):
        return Vec(p.data.at[..., 2, 0].set(0.0))

    def jvp(self, primals, tangents):
        (p,), (dp,) = primals, tangents
        if isinstance(dp, Zero):
            dp = jax.typeof(p).to_tangent_aval().vspace_zero()
        return to_vec(p), dp

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        return to_vec(args[0]), _out_dim(in_dims)


def to_vec(p: Pt) -> Vec:
    return ToVec(jax.typeof(p))(p)


class ToPoint(VJPHiPrimitive):
    def __init__(self, v: V2Ty):
        self.in_avals = (v,)
        self.out_aval = P2Ty(v.batch, v.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, v: Vec):
        return Pt(v.data.at[..., 2, 0].set(1.0))

    def jvp(self, primals, tangents):
        (v,), (dv,) = primals, tangents
        if isinstance(dv, Zero):
            dv = jax.typeof(v).vspace_zero()
        return to_point(v), dv

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        return to_point(args[0]), _out_dim(in_dims)


def to_point(v: Vec) -> Pt:
    return ToPoint(jax.typeof(v))(v)


# ---------------------------------------------------------------------------
# Affine ops
# ---------------------------------------------------------------------------


class XfCompose(VJPHiPrimitive):
    def __init__(self, a: XfTy, b: XfTy):
        batch = tuple(jnp.broadcast_shapes(a.batch, b.batch))
        self.in_avals = (a, b)
        self.out_aval = XfTy(batch, a.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, a: Affine, b: Affine):
        return Affine(a.data @ b.data)

    def jvp(self, primals, tangents):
        a, b = primals
        da, db = tangents
        prim_out = xf_compose(a, b)
        terms = []
        if not isinstance(da, Zero):
            terms.append(xf_compose(da, b))
        if not isinstance(db, Zero):
            terms.append(xf_compose(a, db))
        if not terms:
            tan = jax.typeof(prim_out).to_tangent_aval().vspace_zero()
        else:
            tan = terms[0]
            for t in terms[1:]:
                tan = Affine(tan.data + t.data)
        return prim_out, tan

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, a, b):
        from jax._src.ad_util import Zero as AdZero

        c_ct = cts
        if isinstance(c_ct, (Zero, AdZero)):
            return None
        # d(A@B)=dA@B + A@dB ⇒ A_ct = C_ct @ B^T , B_ct = A^T @ C_ct (on 3x3)
        ca = xf_to_array(c_ct)
        aa = xf_to_array(a)
        ba = xf_to_array(b)
        a_ct = make_xf(ca @ jnp.swapaxes(ba, -1, -2))
        b_ct = make_xf(jnp.swapaxes(aa, -1, -2) @ ca)
        return _accum((a, b), (a_ct, b_ct))

    def batch(self, axis_data, args, in_dims):
        return xf_compose(*args), _out_dim(in_dims)


def xf_compose(a: Affine, b: Affine) -> Affine:
    return XfCompose(jax.typeof(a), jax.typeof(b))(a, b)


class XfApplyPt(VJPHiPrimitive):
    def __init__(self, t: XfTy, p: P2Ty):
        batch = tuple(jnp.broadcast_shapes(t.batch, p.batch))
        self.in_avals = (t, p)
        self.out_aval = P2Ty(batch, p.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, t: Affine, p: Pt):
        return Pt(t.data @ p.data)

    def jvp(self, primals, tangents):
        t, p = primals
        dt, dp = tangents
        prim_out = xf_apply_pt(t, p)
        terms = []
        if not isinstance(dt, Zero):
            terms.append(make_v2_from_data((dt.data @ p.data).at[..., 2, 0].set(0.0)))
        if not isinstance(dp, Zero):
            terms.append(xf_apply_vec(t, dp))
        if not terms:
            tan = jax.typeof(prim_out).to_tangent_aval().vspace_zero()
        else:
            tan = terms[0]
            for x in terms[1:]:
                tan = v2_add(tan, x)
        return prim_out, tan

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp

    def batch(self, axis_data, args, in_dims):
        return xf_apply_pt(*args), _out_dim(in_dims)


def xf_apply_pt(t: Affine, p: Pt) -> Pt:
    return XfApplyPt(jax.typeof(t), jax.typeof(p))(t, p)


class XfApplyVec(VJPHiPrimitive):
    def __init__(self, t: XfTy, v: V2Ty):
        batch = tuple(jnp.broadcast_shapes(t.batch, v.batch))
        self.in_avals = (t, v)
        self.out_aval = V2Ty(batch, v.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, t: Affine, v: Vec):
        out = t.data @ v.data
        return Vec(out.at[..., 2, 0].set(0.0))

    def jvp(self, primals, tangents):
        t, v = primals
        dt, dv = tangents
        prim_out = xf_apply_vec(t, v)
        terms = []
        if not isinstance(dt, Zero):
            terms.append(make_v2_from_data((dt.data @ v.data).at[..., 2, 0].set(0.0)))
        if not isinstance(dv, Zero):
            terms.append(xf_apply_vec(t, dv))
        if not terms:
            tan = jax.typeof(prim_out).vspace_zero()
        else:
            tan = terms[0]
            for x in terms[1:]:
                tan = v2_add(tan, x)
        return prim_out, tan

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp

    def batch(self, axis_data, args, in_dims):
        return xf_apply_vec(*args), _out_dim(in_dims)


def xf_apply_vec(t: Affine, v: Vec) -> Vec:
    return XfApplyVec(jax.typeof(t), jax.typeof(v))(t, v)


class XfApplyHom(VJPHiPrimitive):
    """Apply affine to a raw homogeneous ``(*B, 3, 1)`` array → same shape array."""

    def __init__(self, t: XfTy, h_aval):
        batch = tuple(jnp.broadcast_shapes(t.batch, tuple(h_aval.shape[:-2])))
        self.in_avals = (t, h_aval)
        self.out_aval = ShapedArray(batch + tuple(h_aval.shape[-2:]), h_aval.dtype)
        self.params = {}
        super().__init__()

    def expand(self, t: Affine, h):
        return t.data @ jnp.asarray(h)

    def batch(self, axis_data, args, in_dims):
        out = xf_apply_hom(*args)
        return out, (None if all(d is None for d in in_dims) else 0)


def xf_apply_hom(t: Affine, h):
    h = jnp.asarray(h)
    return XfApplyHom(jax.typeof(t), jax.typeof(h))(t, h)


class XfFromTranslation(VJPHiPrimitive):
    def __init__(self, v: V2Ty):
        self.in_avals = (v,)
        self.out_aval = XfTy(v.batch, v.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, v: Vec):
        batch = v.data.shape[:-2]
        eye = jnp.broadcast_to(jnp.eye(3, dtype=v.data.dtype), batch + (3, 3))
        return Affine(eye.at[..., 0, 2].set(v.data[..., 0, 0]).at[..., 1, 2].set(v.data[..., 1, 0]))

    def jvp(self, primals, tangents):
        (v,), (dv,) = primals, tangents
        prim_out = xf_translation(v)
        if isinstance(dv, Zero):
            return prim_out, jax.typeof(prim_out).vspace_zero()
        return prim_out, xf_translation_tangent(dv)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, v):
        from jax._src.ad_util import Zero as AdZero

        xf_ct = cts
        if isinstance(xf_ct, (Zero, AdZero)):
            return None
        arr = xf_to_array(xf_ct)
        v_ct = make_v2(arr[..., 0, 2], arr[..., 1, 2])
        return _accum((v,), (v_ct,))

    def batch(self, axis_data, args, in_dims):
        return xf_translation(args[0]), _out_dim(in_dims)


def xf_translation(v: Vec) -> Affine:
    return XfFromTranslation(jax.typeof(v))(v)


class XfTranslationTangent(VJPHiPrimitive):
    def __init__(self, v: V2Ty):
        self.in_avals = (v,)
        self.out_aval = XfTy(v.batch, v.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, v: Vec):
        z = jnp.zeros(v.data.shape[:-2] + (3, 3), dtype=v.data.dtype)
        return Affine(
            z.at[..., 0, 2].set(v.data[..., 0, 0]).at[..., 1, 2].set(v.data[..., 1, 0])
        )

    def jvp(self, primals, tangents):
        (v,), (dv,) = primals, tangents
        prim = xf_translation_tangent(v)
        if isinstance(dv, Zero):
            return prim, jax.typeof(prim).vspace_zero()
        return prim, xf_translation_tangent(dv)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, v):
        from jax._src.ad_util import Zero as AdZero

        xf_ct = cts
        if isinstance(xf_ct, (Zero, AdZero)):
            return None
        arr = xf_to_array(xf_ct)
        v_ct = make_v2(arr[..., 0, 2], arr[..., 1, 2])
        return _accum((v,), (v_ct,))

    def batch(self, axis_data, args, in_dims):
        return xf_translation_tangent(args[0]), _out_dim(in_dims)


def xf_translation_tangent(v) -> Affine:
    return XfTranslationTangent(jax.typeof(v))(v)


class XfFromScale(VJPHiPrimitive):
    def __init__(self, v: V2Ty):
        self.in_avals = (v,)
        self.out_aval = XfTy(v.batch, v.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, v: Vec):
        batch = v.data.shape[:-2]
        eye = jnp.broadcast_to(jnp.eye(3, dtype=v.data.dtype), batch + (3, 3))
        return Affine(
            eye.at[..., 0, 0]
            .set(v.data[..., 0, 0])
            .at[..., 1, 1]
            .set(v.data[..., 1, 0])
        )

    def jvp(self, primals, tangents):
        (v,), (dv,) = primals, tangents
        prim_out = xf_scale(v)
        if isinstance(dv, Zero):
            return prim_out, jax.typeof(prim_out).vspace_zero()
        return prim_out, xf_scale_tangent(dv)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, v):
        from jax._src.ad_util import Zero as AdZero

        xf_ct = cts
        if isinstance(xf_ct, (Zero, AdZero)):
            return None
        arr = xf_to_array(xf_ct)
        v_ct = make_v2(arr[..., 0, 0], arr[..., 1, 1])
        return _accum((v,), (v_ct,))

    def batch(self, axis_data, args, in_dims):
        return xf_scale(args[0]), _out_dim(in_dims)


def xf_scale(v: Vec) -> Affine:
    return XfFromScale(jax.typeof(v))(v)


class XfScaleTangent(VJPHiPrimitive):
    def __init__(self, v: V2Ty):
        self.in_avals = (v,)
        self.out_aval = XfTy(v.batch, v.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, v: Vec):
        z = jnp.zeros(v.data.shape[:-2] + (3, 3), dtype=v.data.dtype)
        return Affine(
            z.at[..., 0, 0].set(v.data[..., 0, 0]).at[..., 1, 1].set(v.data[..., 1, 0])
        )

    def jvp(self, primals, tangents):
        (v,), (dv,) = primals, tangents
        prim = xf_scale_tangent(v)
        if isinstance(dv, Zero):
            return prim, jax.typeof(prim).vspace_zero()
        return prim, xf_scale_tangent(dv)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, v):
        from jax._src.ad_util import Zero as AdZero

        xf_ct = cts
        if isinstance(xf_ct, (Zero, AdZero)):
            return None
        arr = xf_to_array(xf_ct)
        v_ct = make_v2(arr[..., 0, 0], arr[..., 1, 1])
        return _accum((v,), (v_ct,))

    def batch(self, axis_data, args, in_dims):
        return xf_scale_tangent(args[0]), _out_dim(in_dims)


def xf_scale_tangent(v) -> Affine:
    return XfScaleTangent(jax.typeof(v))(v)


class XfFromRotation(VJPHiPrimitive):
    """Rotation by ``θ`` radians, matching ``transform.rotation`` (y-up chalk)."""

    def __init__(self, r_aval):
        self.in_avals = (r_aval,)
        batch = tuple(r_aval.shape)
        self.out_aval = XfTy(batch, jnp.dtype(r_aval.dtype).name)
        self.params = {}
        super().__init__()

    def expand(self, r):
        rad = -jnp.asarray(r, dtype=_DT)
        ca, sa = jnp.cos(rad), jnp.sin(rad)
        batch = rad.shape
        eye = jnp.broadcast_to(jnp.eye(3, dtype=_DT), batch + (3, 3))
        return Affine(
            eye.at[..., 0, 0]
            .set(ca)
            .at[..., 0, 1]
            .set(-sa)
            .at[..., 1, 0]
            .set(sa)
            .at[..., 1, 1]
            .set(ca)
        )

    def jvp(self, primals, tangents):
        (r,), (dr,) = primals, tangents
        prim_out = xf_rotation(r)
        if isinstance(dr, Zero):
            return prim_out, jax.typeof(prim_out).vspace_zero()
        r = jnp.asarray(r, dtype=_DT)
        dr = jnp.asarray(dr, dtype=_DT)
        s, c = jnp.sin(r), jnp.cos(r)
        z = jnp.zeros(jnp.broadcast_shapes(r.shape, dr.shape) + (3, 3), dtype=_DT)
        tan = (
            z.at[..., 0, 0]
            .set(-s * dr)
            .at[..., 0, 1]
            .set(c * dr)
            .at[..., 1, 0]
            .set(-c * dr)
            .at[..., 1, 1]
            .set(-s * dr)
        )
        return prim_out, make_xf(tan)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, r):
        from jax._src.ad_util import Zero as AdZero

        xf_ct = cts
        if isinstance(xf_ct, (Zero, AdZero)):
            return None
        arr = xf_to_array(xf_ct)
        # M = [[c, s], [-s, c]] with c=cos(r), s=sin(r); dM/dr = [[-s, c], [-c, -s]]
        # <dM, dM/dr> = -s M00 + c M01 - c M10 - s M11
        s, c = jnp.sin(r), jnp.cos(r)
        dr_ct = (
            -s * arr[..., 0, 0]
            + c * arr[..., 0, 1]
            - c * arr[..., 1, 0]
            - s * arr[..., 1, 1]
        )
        return _accum((r,), (dr_ct,))

    def batch(self, axis_data, args, in_dims):
        return xf_rotation(args[0]), _out_dim(in_dims)


def xf_rotation(r) -> Affine:
    r = jnp.asarray(r)
    return XfFromRotation(jax.typeof(r))(r)


class XfRotationTangent(VJPHiPrimitive):
    def __init__(self, r_aval, dr_aval):
        self.in_avals = (r_aval, dr_aval)
        batch = tuple(jnp.broadcast_shapes(r_aval.shape, dr_aval.shape))
        self.out_aval = XfTy(batch, jnp.dtype(r_aval.dtype).name)
        self.params = {}
        super().__init__()

    def expand(self, r, dr):
        r = jnp.asarray(r, dtype=_DT)
        dr = jnp.asarray(dr, dtype=_DT)
        # d/dθ of rotation(θ) with internal rad=-θ.
        s, c = jnp.sin(r), jnp.cos(r)
        z = jnp.zeros(jnp.broadcast_shapes(r.shape, dr.shape) + (3, 3), dtype=_DT)
        return Affine(
            z.at[..., 0, 0]
            .set(-s * dr)
            .at[..., 0, 1]
            .set(c * dr)
            .at[..., 1, 0]
            .set(-c * dr)
            .at[..., 1, 1]
            .set(-s * dr)
        )

    def vjp_fwd(self, nzs_in, r, dr):
        return xf_rotation_tangent(r, dr), (jnp.asarray(r), jnp.asarray(dr))

    def vjp_bwd_retval(self, res, g):
        r, dr = res
        if isinstance(g, Zero):
            z = jnp.zeros_like(r)
            return z, z
        arr = xf_to_array(g)
        s, c = jnp.sin(r), jnp.cos(r)
        m00, m01 = arr[..., 0, 0], arr[..., 0, 1]
        m10, m11 = arr[..., 1, 0], arr[..., 1, 1]
        ddr_ct = -s * m00 + c * m01 - c * m10 - s * m11
        r_ct = dr * (-c * m00 - s * m01 + s * m10 - c * m11)
        return r_ct, ddr_ct

    def batch(self, axis_data, args, in_dims):
        return xf_rotation_tangent(*args), _out_dim(in_dims)


def xf_rotation_tangent(r, dr) -> Affine:
    r = jnp.asarray(r)
    dr = jnp.asarray(dr)
    return XfRotationTangent(jax.typeof(r), jax.typeof(dr))(r, dr)


class GetTranslation(VJPHiPrimitive):
    def __init__(self, t: XfTy):
        self.in_avals = (t,)
        self.out_aval = V2Ty(t.batch, t.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, t: Affine):
        z = jnp.zeros(t.data.shape[:-2] + (3, 1), dtype=t.data.dtype)
        return Vec(z.at[..., 0, 0].set(t.data[..., 0, 2]).at[..., 1, 0].set(t.data[..., 1, 2]))

    def batch(self, axis_data, args, in_dims):
        return xf_get_translation(args[0]), _out_dim(in_dims)


def xf_get_translation(t: Affine) -> Vec:
    return GetTranslation(jax.typeof(t))(t)


class Length2(VJPHiPrimitive):
    def __init__(self, v: V2Ty):
        self.in_avals = (v,)
        self.out_aval = ShapedArray(v.batch, jnp.dtype(v.dtype_name))
        self.params = {}
        super().__init__()

    def expand(self, v: Vec):
        return (v.data * v.data)[..., :2, 0].sum(-1)

    def jvp(self, primals, tangents):
        (v,), (dv,) = primals, tangents
        prim_out = length2(v)
        if isinstance(dv, Zero):
            return prim_out, jnp.zeros(
                jax.typeof(v).batch, dtype=jnp.dtype(jax.typeof(v).dtype_name)
            )
        va, da = v2_to_array(v), v2_to_array(dv)
        return prim_out, 2 * ((va * da)[..., :2, 0].sum(-1))

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        out = length2(args[0])
        return out, (None if in_dims[0] is None else 0)


def length2(v: Vec):
    return Length2(jax.typeof(v))(v)


class Length(VJPHiPrimitive):
    def __init__(self, v: V2Ty):
        self.in_avals = (v,)
        self.out_aval = ShapedArray(v.batch, jnp.dtype(v.dtype_name))
        self.params = {}
        super().__init__()

    def expand(self, v: Vec):
        return jnp.sqrt(length2(v))

    def batch(self, axis_data, args, in_dims):
        out = length(args[0])
        return out, (None if in_dims[0] is None else 0)


def length(v: Vec):
    return Length(jax.typeof(v))(v)


# Clean up leftover junk in MakeV2 first version - already deleted.

# Constant unit values as hijax (eager).
unit_x = Vec(jnp.asarray([1.0, 0.0, 0.0], dtype=_DT).reshape(3, 1))
unit_y = Vec(jnp.asarray([0.0, 1.0, 0.0], dtype=_DT).reshape(3, 1))
origin = Pt(jnp.asarray([0.0, 0.0, 1.0], dtype=_DT).reshape(3, 1))
ident = Affine(jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=_DT))


def make_ident(shape: Tuple[int, ...]) -> Affine:
    return make_xf(jnp.broadcast_to(ident.data, shape + (3, 3)))
