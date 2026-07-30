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
    register_hitype,
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

    def to_tangent_aval(self):
        return self

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
        return C.Primitive(path, style, xf, order)
    if isinstance(d, C.ApplyTransform):
        xf = fn(d.transform)
        child = map_diag_prefix(d.diagram, fn)
        if xf is None:
            return C.ApplyTransform(d.transform, child)
        return C.ApplyTransform(xf, child)
    if isinstance(d, C.ApplyStyle):
        return C.ApplyStyle(d.style.map_prefix(fn), map_diag_prefix(d.diagram, fn))
    if isinstance(d, C.ApplyName):
        return C.ApplyName(d.dname, map_diag_prefix(d.diagram, fn))
    if isinstance(d, C.Compose):
        env = map_diag_prefix(d.envelope, fn) if d.envelope is not None else None
        return C.Compose(env, tuple(map_diag_prefix(c, fn) for c in d.diagrams))
    if isinstance(d, C.ComposeAxis):
        return C.ComposeAxis(map_diag_prefix(d.diagrams, fn))
    return d
