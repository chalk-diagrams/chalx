"""Opaque hijax Diagram type.

Users only see ``diag[]`` / ``diag[B]``. Node kinds (prim, compose, …) are
internal to ``DiagTy`` so ``lo_ty`` works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.hijax import (
    HiType,
    MappingSpec,
    ShapedArray,
    VJPHiPrimitive,
    Zero,
    apply_derived_linearization,
    aval_method,
    linearize_from_jvp,
    register_hitype,
    transpose_jvp,
    vjp_fwd_from_jvp,
)

import chalk.core as C
from chalk.path import Path, PathTy
from chalk.style import StyleHolder, StyleTy
from chalk.subdiagram import Name
from chalk.transform import Affine

_DT = jnp.dtype("float64")


@dataclass(frozen=True)
class DiagSpec(MappingSpec):
    pass


@dataclass(frozen=True)
class DiagTy(HiType):
    """Internal diagram type. Prints as ``diag`` for users."""

    tag: str  # empty|prim|xf|style|name|compose|axis
    batch: Tuple[int, ...] = ()
    path_ty: Optional[PathTy] = None
    style_ty: Optional[StyleTy] = None
    xf_shape: Tuple[int, ...] = ()
    has_order: bool = False
    child_tys: Tuple["DiagTy", ...] = ()
    env_ty: Optional["DiagTy"] = None
    name: Optional[Tuple[Any, ...]] = None

    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        batch = ",".join(str(d) for d in self.batch)
        return f"diag[{batch}]" if batch else "diag[]"

    __repr__ = str_short

    @property
    def dtype(self):
        return _DT

    def to_tangent_aval(self):
        return self

    def vspace_zero(self):
        return self.raise_val(
            *[jnp.zeros(a.shape, dtype=a.dtype) for a in self.lo_ty()]
        )

    def vspace_add(self, x, y):
        return diag_vspace_add(x, y)

    def lo_ty(self):
        if self.tag == "empty":
            return [ShapedArray((), _DT)]  # dummy unit
        if self.tag == "prim":
            assert self.path_ty is not None
            los = list(self.path_ty.lo_ty())
            los.append(ShapedArray(self.xf_shape + (3, 3), _DT))
            if self.style_ty is not None:
                los.extend(self.style_ty.lo_ty())
            if self.has_order:
                los.append(ShapedArray(self.xf_shape, jnp.dtype("int32")))
            return los
        if self.tag == "xf":
            los = [ShapedArray(self.xf_shape + (3, 3), _DT)]
            los.extend(self.child_tys[0].lo_ty())
            return los
        if self.tag == "style":
            assert self.style_ty is not None
            los = list(self.style_ty.lo_ty())
            los.extend(self.child_tys[0].lo_ty())
            return los
        if self.tag == "name":
            return list(self.child_tys[0].lo_ty())
        if self.tag == "compose":
            los = []
            if self.env_ty is not None:
                los.extend(self.env_ty.lo_ty())
            for c in self.child_tys:
                los.extend(c.lo_ty())
            return los
        if self.tag == "axis":
            return list(self.child_tys[0].lo_ty())
        raise ValueError(self.tag)

    def lower_val(self, d):
        if self.tag == "empty":
            return [jnp.asarray(0.0)]
        if self.tag == "prim":
            assert self.path_ty is not None
            los = list(self.path_ty.lower_val(d.prim_shape))
            los.append(jnp.asarray(d.transform, dtype=_DT))
            if self.style_ty is not None:
                los.extend(self.style_ty.lower_val(d.style))
            if self.has_order:
                los.append(jnp.asarray(d.order, dtype=jnp.int32))
            return los
        if self.tag == "xf":
            return [jnp.asarray(d.transform, dtype=_DT)] + list(
                self.child_tys[0].lower_val(d.diagram)
            )
        if self.tag == "style":
            assert self.style_ty is not None
            return list(self.style_ty.lower_val(d.style)) + list(
                self.child_tys[0].lower_val(d.diagram)
            )
        if self.tag == "name":
            return list(self.child_tys[0].lower_val(d.diagram))
        if self.tag == "compose":
            los = []
            if self.env_ty is not None:
                los.extend(self.env_ty.lower_val(d.envelope))
            for ty, ch in zip(self.child_tys, d.diagrams):
                los.extend(ty.lower_val(ch))
            return los
        if self.tag == "axis":
            return list(self.child_tys[0].lower_val(d.diagrams))
        raise ValueError(self.tag)

    def raise_val(self, *los):
        it = iter(los)
        return _raise(self, it)

    def dec_rank(self, size, spec):
        assert isinstance(spec, DiagSpec)
        assert self.batch and self.batch[0] == size
        return _map_batch(self, self.batch[1:])

    def inc_rank(self, size, spec):
        assert isinstance(spec, DiagSpec)
        return _map_batch(self, (size, *self.batch))

    def leading_axis_spec(self):
        return DiagSpec()


def _map_batch(ty: DiagTy, batch: Tuple[int, ...]) -> DiagTy:
    children = tuple(_map_batch(c, batch) for c in ty.child_tys)
    env = _map_batch(ty.env_ty, batch) if ty.env_ty is not None else None
    path_ty = ty.path_ty
    style_ty = ty.style_ty
    if style_ty is not None:
        style_ty = type(style_ty)(batch, style_ty.dtype_name)
    return DiagTy(
        ty.tag,
        batch,
        path_ty,
        style_ty,
        batch,
        ty.has_order,
        children,
        env,
        ty.name,
    )


def _take(it, n):
    return [next(it) for _ in range(n)]


def _raise(ty: DiagTy, it):
    if ty.tag == "empty":
        next(it)
        return C.Empty()
    if ty.tag == "prim":
        assert ty.path_ty is not None
        path = ty.path_ty.raise_val(*_take(it, len(ty.path_ty.lo_ty())))
        xf = next(it)
        style = None
        if ty.style_ty is not None:
            style = ty.style_ty.raise_val(*_take(it, len(ty.style_ty.lo_ty())))
        order = next(it) if ty.has_order else None
        return C.Primitive(path, style, xf, order)
    if ty.tag == "xf":
        xf = next(it)
        child = _raise(ty.child_tys[0], it)
        return C.ApplyTransform(xf, child)
    if ty.tag == "style":
        assert ty.style_ty is not None
        style = ty.style_ty.raise_val(*_take(it, len(ty.style_ty.lo_ty())))
        child = _raise(ty.child_tys[0], it)
        return C.ApplyStyle(style, child)
    if ty.tag == "name":
        child = _raise(ty.child_tys[0], it)
        return C.ApplyName(Name(ty.name or ()), child)
    if ty.tag == "compose":
        env = _raise(ty.env_ty, it) if ty.env_ty is not None else None
        children = tuple(_raise(c, it) for c in ty.child_tys)
        return C.Compose(env, children)
    if ty.tag == "axis":
        child = _raise(ty.child_tys[0], it)
        return C.ComposeAxis(child)
    raise ValueError(ty.tag)


def typeof_diagram(d) -> DiagTy:
    if isinstance(d, C.Empty):
        return DiagTy("empty")
    if isinstance(d, C.Primitive):
        xf = tuple(jnp.asarray(d.transform).shape[:-2])
        return DiagTy(
            "prim",
            xf,
            jax.typeof(d.prim_shape),
            jax.typeof(d.style) if d.style is not None else None,
            xf,
            d.order is not None,
            (),
            None,
            None,
        )
    if isinstance(d, C.ApplyTransform):
        xf = tuple(jnp.asarray(d.transform).shape[:-2])
        child = typeof_diagram(d.diagram)
        return DiagTy("xf", xf, None, None, xf, False, (child,), None, None)
    if isinstance(d, C.ApplyStyle):
        child = typeof_diagram(d.diagram)
        sty = jax.typeof(d.style)
        batch = sty.batch_shape
        return DiagTy("style", batch, None, sty, (), False, (child,), None, None)
    if isinstance(d, C.ApplyName):
        child = typeof_diagram(d.diagram)
        return DiagTy(
            "name",
            child.batch,
            None,
            None,
            (),
            False,
            (child,),
            None,
            d.dname.atomic_names,
        )
    if isinstance(d, C.Compose):
        children = tuple(typeof_diagram(c) for c in d.diagrams)
        env = typeof_diagram(d.envelope) if d.envelope is not None else None
        batch = children[0].batch if children else ()
        return DiagTy("compose", batch, None, None, (), False, children, env, None)
    if isinstance(d, C.ComposeAxis):
        child = typeof_diagram(d.diagrams)
        batch = child.batch[:-1] if child.batch else ()
        return DiagTy("axis", batch, None, None, (), False, (child,), None, None)
    raise TypeError(type(d))


def _register():
    for cls in (
        C.Empty,
        C.Primitive,
        C.ApplyTransform,
        C.ApplyStyle,
        C.ApplyName,
        C.Compose,
        C.ComposeAxis,
    ):
        register_hitype(cls, typeof_diagram)


_register()


# ---------------------------------------------------------------------------
# Constructor primitives (opaque build API)
# ---------------------------------------------------------------------------


class DiagEmpty(VJPHiPrimitive):
    def __init__(self):
        self.in_avals = ()
        self.out_aval = DiagTy("empty")
        self.params = {}
        super().__init__()

    def expand(self):
        return C.Empty()

    def batch(self, axis_data, args, in_dims):
        return diag_empty(), None


def diag_empty():
    return DiagEmpty()()


class DiagPrim(VJPHiPrimitive):
    def __init__(self, path_aval, xf_aval, style_aval=None, order_aval=None):
        ins = [path_aval, xf_aval]
        if style_aval is not None:
            ins.append(style_aval)
        if order_aval is not None:
            ins.append(order_aval)
        xf_shape = tuple(xf_aval.shape[:-2])
        self.in_avals = tuple(ins)
        self.out_aval = DiagTy(
            "prim",
            xf_shape,
            path_aval,
            style_aval,
            xf_shape,
            order_aval is not None,
        )
        self.params = dict(
            has_style=style_aval is not None, has_order=order_aval is not None
        )
        super().__init__()

    def expand(self, path, xf, *rest):
        style = rest[0] if self.has_style else None
        order = rest[-1] if self.has_order else None
        return C.Primitive(path, style, xf, order)

    def batch(self, axis_data, args, in_dims):
        path, xf, *rest = args
        style = rest[0] if self.has_style else None
        order = rest[-1] if self.has_order else None
        out = diag_prim(path, xf, style, order)
        return out, (None if all(d is None for d in in_dims) else DiagSpec())


def diag_prim(path, xf, style=None, order=None):
    from chalk.geom import data as geom_data

    xf = geom_data(xf)
    args = [path, xf]
    sav = jax.typeof(style) if style is not None else None
    oav = jax.typeof(order) if order is not None else None
    if style is not None:
        args.append(style)
    if order is not None:
        args.append(order)
    return DiagPrim(jax.typeof(path), jax.typeof(xf), sav, oav)(*args)


class DiagXf(VJPHiPrimitive):
    def __init__(self, child_aval: DiagTy, xf_aval):
        self.in_avals = (child_aval, xf_aval)
        xf_shape = tuple(xf_aval.shape[:-2])
        self.out_aval = DiagTy("xf", xf_shape, None, None, xf_shape, False, (child_aval,))
        self.params = {}
        super().__init__()

    def expand(self, child, xf):
        return C.ApplyTransform(xf, child)

    def jvp(self, primals, tangents):
        child, xf = primals
        dchild, dxf = tangents
        prim = diag_xf(child, xf)
        if isinstance(dxf, Zero):
            dxf = jnp.zeros_like(xf)
        if isinstance(dchild, Zero):
            dchild = jax.typeof(child).vspace_zero()
        return prim, diag_xf(dchild, dxf)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, child, xf):
        from jax.experimental.hijax import GradAccum
        from jax._src.ad_util import Zero as AdZero

        d_ct = cts
        if isinstance(d_ct, (Zero, AdZero)):
            return None
        child_ct, xf_ct = diag_uncons_xf(d_ct)
        if isinstance(child, GradAccum):
            child.accum(child_ct)
        if isinstance(xf, GradAccum):
            xf.accum(xf_ct)
        return None

    def batch(self, axis_data, args, in_dims):
        child, xf = args
        out = diag_xf(child, xf)
        return out, (None if all(d is None for d in in_dims) else DiagSpec())


def diag_xf(diagram, xf):
    from chalk.geom import data as geom_data

    xf = geom_data(xf)
    return DiagXf(jax.typeof(diagram), jax.typeof(xf))(diagram, xf)


class DiagStyle(VJPHiPrimitive):
    def __init__(self, child_aval: DiagTy, style_aval: StyleTy):
        self.in_avals = (child_aval, style_aval)
        self.out_aval = DiagTy(
            "style", style_aval.batch_shape, None, style_aval, (), False, (child_aval,)
        )
        self.params = {}
        super().__init__()

    def expand(self, child, style):
        return C.ApplyStyle(style, child)

    def jvp(self, primals, tangents):
        child, style = primals
        dchild, dstyle = tangents
        prim = diag_style(child, style)
        if isinstance(dchild, Zero):
            dchild = jax.typeof(child).vspace_zero()
        if isinstance(dstyle, Zero):
            dstyle = jax.typeof(style).vspace_zero()
        return prim, diag_style(dchild, dstyle)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, child, style):
        from jax.experimental.hijax import GradAccum
        from jax._src.ad_util import Zero as AdZero

        d_ct = cts
        if isinstance(d_ct, (Zero, AdZero)):
            return None
        child_ct = diag_uncons_child(d_ct)
        if isinstance(child, GradAccum):
            child.accum(child_ct)
        if isinstance(style, GradAccum):
            style.accum(self.in_avals[1].vspace_zero())
        return None

    def batch(self, axis_data, args, in_dims):
        child, style = args
        out = diag_style(child, style)
        return out, (None if all(d is None for d in in_dims) else DiagSpec())


def diag_style(diagram, style):
    return DiagStyle(jax.typeof(diagram), jax.typeof(style))(diagram, style)


class DiagName(VJPHiPrimitive):
    def __init__(self, child_aval: DiagTy, name: Tuple[Any, ...]):
        self.in_avals = (child_aval,)
        self.out_aval = DiagTy(
            "name", child_aval.batch, None, None, (), False, (child_aval,), None, name
        )
        self.params = dict(name=name)
        super().__init__()

    def expand(self, child):
        return C.ApplyName(Name(self.name), child)

    def batch(self, axis_data, args, in_dims):
        (child,) = args
        out = diag_name(child, Name(self.name))
        return out, (None if in_dims[0] is None else DiagSpec())


def diag_name(diagram, name: Name):
    return DiagName(jax.typeof(diagram), name.atomic_names)(diagram)


class DiagCompose(VJPHiPrimitive):
    def __init__(self, child_avals: Tuple[DiagTy, ...], env_aval: Optional[DiagTy]):
        ins = list(child_avals) if env_aval is None else [env_aval, *child_avals]
        batch = child_avals[0].batch if child_avals else ()
        self.in_avals = tuple(ins)
        self.out_aval = DiagTy(
            "compose", batch, None, None, (), False, child_avals, env_aval
        )
        self.params = dict(n=len(child_avals), has_env=env_aval is not None)
        super().__init__()

    def expand(self, *args):
        if self.has_env:
            env, *children = args
        else:
            env, children = None, args
        return C.Compose(env, tuple(children))

    def batch(self, axis_data, args, in_dims):
        if self.has_env:
            env, *children = args
        else:
            env, children = None, args
        out = diag_compose(tuple(children), env)
        return out, (None if all(d is None for d in in_dims) else DiagSpec())


def diag_compose(children, envelope=None):
    children = tuple(children)
    env_av = jax.typeof(envelope) if envelope is not None else None
    args = (envelope, *children) if envelope is not None else children
    return DiagCompose(tuple(jax.typeof(c) for c in children), env_av)(*args)


class DiagAxis(VJPHiPrimitive):
    def __init__(self, child_aval: DiagTy):
        self.in_avals = (child_aval,)
        batch = child_aval.batch[:-1] if child_aval.batch else ()
        self.out_aval = DiagTy("axis", batch, None, None, (), False, (child_aval,))
        self.params = {}
        super().__init__()

    def expand(self, child):
        return C.ComposeAxis(child)

    def batch(self, axis_data, args, in_dims):
        (child,) = args
        out = diag_axis(child)
        return out, (None if in_dims[0] is None else DiagSpec())


def diag_axis(diagram):
    return DiagAxis(jax.typeof(diagram))(diagram)


def map_diag_prefix(d, fn: Callable):
    if isinstance(d, C.Empty):
        return d
    if isinstance(d, C.Primitive):
        path = d.prim_shape.map_prefix(fn)
        style = d.style.map_prefix(fn) if d.style is not None else None
        xf = fn(d.transform)
        order = fn(d.order) if d.order is not None else None
        if xf is None:
            return d
        return diag_prim(path, xf, style, order)
    if isinstance(d, C.ApplyTransform):
        xf = fn(d.transform)
        child = map_diag_prefix(d.diagram, fn)
        if xf is None:
            return diag_xf(child, d.transform)
        return diag_xf(child, xf)
    if isinstance(d, C.ApplyStyle):
        return diag_style(map_diag_prefix(d.diagram, fn), d.style.map_prefix(fn))
    if isinstance(d, C.ApplyName):
        return diag_name(map_diag_prefix(d.diagram, fn), d.dname)
    if isinstance(d, C.Compose):
        env = map_diag_prefix(d.envelope, fn) if d.envelope is not None else None
        return diag_compose(tuple(map_diag_prefix(c, fn) for c in d.diagrams), env)
    if isinstance(d, C.ComposeAxis):
        return diag_axis(map_diag_prefix(d.diagrams, fn))
    return d


class DiagVspaceAdd(VJPHiPrimitive):
    def __init__(self, dty: DiagTy):
        self.in_avals = (dty, dty)
        self.out_aval = dty
        self.params = dict(dty=dty)
        super().__init__()

    def expand(self, x, y):
        ty = self.dty
        los = [a + b for a, b in zip(ty.lower_val(x), ty.lower_val(y))]
        return ty.raise_val(*los)

    def jvp(self, primals, tangents):
        x, y = primals
        dx, dy = tangents
        prim = diag_vspace_add(x, y)
        if isinstance(dx, Zero):
            dx = jax.typeof(x).vspace_zero()
        if isinstance(dy, Zero):
            dy = jax.typeof(y).vspace_zero()
        return prim, diag_vspace_add(dx, dy)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def batch(self, axis_data, args, in_dims):
        return diag_vspace_add(*args), (
            None if all(d is None for d in in_dims) else DiagSpec()
        )


def diag_vspace_add(x, y):
    return DiagVspaceAdd(jax.typeof(x))(x, y)


class DiagUnconsXf(VJPHiPrimitive):
    def __init__(self, dty: DiagTy):
        assert dty.tag == "xf"
        self.in_avals = (dty,)
        self.out_aval = (
            dty.child_tys[0],
            ShapedArray(dty.xf_shape + (3, 3), _DT),
        )
        self.params = dict(dty=dty)
        super().__init__()

    def expand(self, d):
        return d.diagram, jnp.asarray(d.transform, dtype=_DT)

    def jvp(self, primals, tangents):
        (d,), (dd,) = primals, tangents
        child, xf = diag_uncons_xf(d)
        if isinstance(dd, Zero):
            return (child, xf), (
                jax.typeof(child).vspace_zero(),
                jnp.zeros_like(xf),
            )
        dchild, dxf = diag_uncons_xf(dd)
        return (child, xf), (dchild, dxf)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, d):
        from jax.experimental.hijax import GradAccum
        from jax._src.ad_util import Zero as AdZero

        child_ct, xf_ct = cts
        if isinstance(child_ct, (Zero, AdZero)):
            child_ct = self.out_aval[0].vspace_zero()
        if isinstance(xf_ct, (Zero, AdZero)):
            xf_ct = jnp.zeros(self.out_aval[1].shape, dtype=self.out_aval[1].dtype)
        d_ct = diag_xf(child_ct, xf_ct)
        if isinstance(d, GradAccum):
            d.accum(d_ct)
        return None

    def batch(self, axis_data, args, in_dims):
        (d,) = args
        out = diag_uncons_xf(d)
        return out, (None if in_dims[0] is None else (DiagSpec(), 0))


def diag_uncons_xf(d):
    return DiagUnconsXf(jax.typeof(d))(d)


class DiagUnconsChild(VJPHiPrimitive):
    def __init__(self, dty: DiagTy):
        self.in_avals = (dty,)
        self.out_aval = dty.child_tys[0]
        self.params = dict(dty=dty)
        super().__init__()

    def expand(self, d):
        return d.diagram

    def jvp(self, primals, tangents):
        (d,), (dd,) = primals, tangents
        child = diag_uncons_child(d)
        if isinstance(dd, Zero):
            return child, jax.typeof(child).vspace_zero()
        return child, diag_uncons_child(dd)

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, d):
        from jax.experimental.hijax import GradAccum
        from jax._src.ad_util import Zero as AdZero

        child_ct = cts
        if isinstance(child_ct, (Zero, AdZero)):
            child_ct = self.out_aval.vspace_zero()
        # Re-wrap with zero style/name payload via vspace on full type is hard;
        # style/name don't affect trace, cotangent on child only.
        d_ct = child_ct
        if isinstance(d, GradAccum):
            # Cotangent of a style wrapper is a style-diag with child cotangent.
            dty = self.dty
            if dty.tag == "style" and dty.style_ty is not None:
                d_ct = diag_style(child_ct, dty.style_ty.vspace_zero())
            elif dty.tag == "name":
                from chalk.subdiagram import Name

                d_ct = diag_name(child_ct, Name(dty.name or ()))
            elif dty.tag == "axis":
                d_ct = diag_axis(child_ct)
            d.accum(d_ct)
        return None

    def batch(self, axis_data, args, in_dims):
        out = diag_uncons_child(args[0])
        return out, (None if in_dims[0] is None else DiagSpec())


def diag_uncons_child(d):
    return DiagUnconsChild(jax.typeof(d))(d)


class DiagPrimTrace(VJPHiPrimitive):
    def __init__(self, dty: DiagTy):
        from chalk.trace import TraceTy

        self.in_avals = (dty,)
        self.out_aval = _trace_ty_from_diag(dty)
        self.params = dict(dty=dty)
        super().__init__()

    def expand(self, d):
        from chalk.trace import _GetLocatedSegments, make_trace
        import chalk.transform as tx

        return make_trace(d._accept(_GetLocatedSegments(), tx._ident_arr))

    def jvp(self, primals, tangents):
        (d,), _dd = primals, tangents
        prim = diag_prim_trace(d)
        return prim, jax.typeof(prim).vspace_zero()

    lin = linearize_from_jvp
    linearized = apply_derived_linearization
    vjp_fwd = vjp_fwd_from_jvp
    vjp_bwd_retval = transpose_jvp

    def transpose(self, cts, d):
        from jax.experimental.hijax import GradAccum

        if isinstance(d, GradAccum):
            d.accum(self.in_avals[0].vspace_zero())
        return None

    def batch(self, axis_data, args, in_dims):
        from chalk.trace import TraceSpec

        out = diag_prim_trace(args[0])
        return out, (None if in_dims[0] is None else TraceSpec())


def diag_prim_trace(d):
    return DiagPrimTrace(jax.typeof(d))(d)


def _trace_ty_from_diag(dty: DiagTy):
    from chalk.segment import SegSpec, SegTy
    from chalk.trace import TraceTy

    if dty.tag in ("xf", "style", "name", "axis"):
        tr = _trace_ty_from_diag(dty.child_tys[0])
        if dty.tag == "xf" and dty.batch != tr.seg_ty.batch_shape:
            st = tr.seg_ty
            out = st
            for size in reversed(dty.batch[len(st.batch_shape) :]):
                out = out.inc_rank(size, SegSpec())
            return TraceTy(out)
        return tr
    if dty.tag == "prim":
        assert dty.path_ty is not None
        n_segs = 0
        dtype = "float64"
        for lt in dty.path_ty.loc_tys:
            n_segs += lt.trail_ty.seg_ty.n_segs
            dtype = lt.trail_ty.seg_ty.dtype_name
        return TraceTy(SegTy(dty.batch, max(n_segs, 1), dtype))
    if dty.tag == "empty":
        return TraceTy(SegTy((), 1, "float64"))
    if dty.tag == "compose":
        parts = [_trace_ty_from_diag(c) for c in dty.child_tys]
        n = sum(p.seg_ty.n_segs for p in parts) if parts else 1
        batch = parts[0].seg_ty.batch_shape if parts else ()
        dtype = parts[0].seg_ty.dtype_name if parts else "float64"
        return TraceTy(SegTy(batch, max(n, 1), dtype))
    raise ValueError(dty.tag)


def _aval_apply_transform(d, t):
    from chalk.geom import data as geom_data

    return diag_xf(d, geom_data(t))


def _aval_apply_style(d, style):
    return diag_style(d, style)


def _aval_get_trace(d):
    from chalk.trace import get_trace

    return get_trace(d)


def _install_diag_aval_methods():
    from chalk.style import Stylable
    from chalk.transform import Transformable

    DiagTy.apply_transform = aval_method(_aval_apply_transform)
    DiagTy._app = aval_method(_aval_apply_transform)
    DiagTy.apply_style = aval_method(_aval_apply_style)
    DiagTy.get_trace = aval_method(_aval_get_trace)
    DiagTy.scale = aval_method(Transformable.scale)
    DiagTy.scale_x = aval_method(Transformable.scale_x)
    DiagTy.scale_y = aval_method(Transformable.scale_y)
    DiagTy.rotate = aval_method(Transformable.rotate)
    DiagTy.rotate_rad = aval_method(Transformable.rotate_rad)
    DiagTy.rotate_by = aval_method(Transformable.rotate_by)
    DiagTy.translate = aval_method(Transformable.translate)
    DiagTy.translate_by = aval_method(Transformable.translate_by)
    DiagTy.fill_color = aval_method(Stylable.fill_color)
    DiagTy.line_width = aval_method(Stylable.line_width)
    DiagTy.line_color = aval_method(Stylable.line_color)
    DiagTy.fill_opacity = aval_method(Stylable.fill_opacity)


_install_diag_aval_methods()
