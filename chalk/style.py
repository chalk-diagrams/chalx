from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as onp
from colour import Color
from jax.experimental.hijax import (
    HiType,
    MappingSpec,
    ShapedArray,
    VJPHiPrimitive,
    register_hitype,
)
from typing_extensions import Self

import chalk.transform as tx
from chalk.transform import ColorVec, Property, Scalars

PropLike = Union[Property, float]
ColorLike = Union[str, Color, ColorVec]

# flags[..., i]
_F_FILL_COLOR = 0
_F_LINE_COLOR = 1
_F_FILL_OPACITY = 2
_F_LINE_OPACITY = 3
_F_LINE_WIDTH = 4
_N_FLAGS = 5


def to_color(c: ColorLike) -> ColorVec:
    """Convert colour name / Color / RGB array to an RGB vector."""
    if isinstance(c, str):
        return tx.np.asarray(Color(c).rgb)
    if isinstance(c, Color):
        return tx.np.asarray(c.rgb)
    return c


_DEFAULT_FILL = jnp.asarray(Color("white").rgb, dtype=jnp.float32)
_DEFAULT_LINE = jnp.asarray(Color("black").rgb, dtype=jnp.float32)
_DEFAULT_FILL_OPACITY = jnp.float32(1.0)
_DEFAULT_LINE_OPACITY = jnp.float32(1.0)
_DEFAULT_LINE_WIDTH = jnp.float32(0.1)


class Stylable:
    def line_width(self, width: float) -> Self:
        return self.apply_style(Style(line_width=width))

    def line_color(self, color: ColorLike) -> Self:
        return self.apply_style(Style(line_color=to_color(color)))

    def fill_color(self, color: ColorLike) -> Self:
        return self.apply_style(Style(fill_color=to_color(color)))

    def fill_opacity(self, opacity: float) -> Self:
        return self.apply_style(Style(fill_opacity=opacity))

    def dashing(self, dashing_strokes: List[float], offset: float) -> Self:
        return self.apply_style(Style())

    def apply_style(self: Self, style: StyleHolder) -> Self:
        raise NotImplementedError("Abstract")


class WidthType(Enum):
    LOCAL = auto()
    NORMALIZED = auto()


def _batch_of(value: Any, is_color: bool) -> Tuple[int, ...]:
    x = jnp.asarray(value)
    if is_color:
        return tuple(x.shape[:-1]) if x.ndim >= 1 else ()
    return tuple(x.shape)


def _broadcast_color(value: Any, batch: Tuple[int, ...]) -> jax.Array:
    x = jnp.asarray(value, dtype=jnp.float32)
    if x.shape == (3,):
        x = jnp.broadcast_to(x, batch + (3,))
    elif x.shape == batch + (3,):
        pass
    elif x.ndim == 1 and batch == ():
        x = jnp.asarray(x, dtype=jnp.float32)
    else:
        x = jnp.broadcast_to(x, batch + (3,))
    return x


def _broadcast_scalar(value: Any, batch: Tuple[int, ...]) -> jax.Array:
    x = jnp.asarray(value, dtype=jnp.float32)
    return jnp.broadcast_to(x, batch)


def Style(
    line_width: Optional[PropLike] = None,
    line_color: Optional[ColorLike] = None,
    line_opacity: Optional[PropLike] = None,
    fill_color: Optional[ColorLike] = None,
    fill_opacity: Optional[PropLike] = None,
) -> StyleHolder:
    """Build a style from named properties (batched if any arg is)."""
    batches = [()]
    if fill_color is not None:
        batches.append(_batch_of(to_color(fill_color), True))
    if line_color is not None:
        batches.append(_batch_of(to_color(line_color), True))
    if fill_opacity is not None:
        batches.append(_batch_of(fill_opacity, False))
    if line_opacity is not None:
        batches.append(_batch_of(line_opacity, False))
    if line_width is not None:
        batches.append(_batch_of(line_width, False))
    batch: Tuple[int, ...] = tuple(jnp.broadcast_shapes(*batches))

    fc = jnp.broadcast_to(_DEFAULT_FILL, batch + (3,))
    lc = jnp.broadcast_to(_DEFAULT_LINE, batch + (3,))
    fo = jnp.broadcast_to(_DEFAULT_FILL_OPACITY, batch)
    lo = jnp.broadcast_to(_DEFAULT_LINE_OPACITY, batch)
    lw = jnp.broadcast_to(_DEFAULT_LINE_WIDTH, batch)
    flags = jnp.zeros(batch + (_N_FLAGS,), dtype=bool)

    if fill_color is not None:
        fc = _broadcast_color(to_color(fill_color), batch)
        flags = flags.at[..., _F_FILL_COLOR].set(True)
    if line_color is not None:
        lc = _broadcast_color(to_color(line_color), batch)
        flags = flags.at[..., _F_LINE_COLOR].set(True)
    if fill_opacity is not None:
        fo = _broadcast_scalar(fill_opacity, batch)
        flags = flags.at[..., _F_FILL_OPACITY].set(True)
    if line_opacity is not None:
        lo = _broadcast_scalar(line_opacity, batch)
        flags = flags.at[..., _F_LINE_OPACITY].set(True)
    if line_width is not None:
        lw = _broadcast_scalar(line_width, batch)
        flags = flags.at[..., _F_LINE_WIDTH].set(True)

    return make_style(fc, lc, fo, lo, lw, flags)


# ---------------------------------------------------------------------------
# Hijax style — separate color / scalar payloads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StyleSpec(MappingSpec):
    """vmap spec: styles batch on a leading axis."""


@dataclass(frozen=True)
class StyleTy(HiType):
    batch_shape: Tuple[int, ...]
    dtype_name: str = "float32"

    def lo_ty(self):
        b = self.batch_shape
        dt = jnp.dtype(self.dtype_name)
        return [
            ShapedArray(b + (3,), dt),  # fill_color RGB
            ShapedArray(b + (3,), dt),  # line_color RGB
            ShapedArray(b, dt),  # fill_opacity
            ShapedArray(b, dt),  # line_opacity
            ShapedArray(b, dt),  # line_width
            ShapedArray(b + (_N_FLAGS,), jnp.dtype("bool")),
        ]

    def lower_val(self, style: StyleHolder):
        return [
            style.fill_rgb,
            style.line_rgb,
            style.fill_alpha,
            style.line_alpha,
            style.stroke_width,
            style.set_flags,
        ]

    def raise_val(self, fill_rgb, line_rgb, fill_alpha, line_alpha, stroke_width, set_flags):
        return StyleHolder(
            fill_rgb, line_rgb, fill_alpha, line_alpha, stroke_width, set_flags
        )

    def to_tangent_aval(self):
        return StyleTy(self.batch_shape, self.dtype_name)

    def vspace_zero(self):
        return StyleHolder.empty_with_batch(self.batch_shape, self.dtype_name)

    def vspace_add(self, x, y):
        return merge_styles(x, y)

    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        batch = ",".join(str(d) for d in self.batch_shape)
        return f"style[{batch}]" if batch else "style[]"

    __repr__ = str_short

    def dec_rank(self, size, spec):
        assert isinstance(spec, StyleSpec)
        assert self.batch_shape and self.batch_shape[0] == size
        return StyleTy(self.batch_shape[1:], self.dtype_name)

    def inc_rank(self, size, spec):
        assert isinstance(spec, StyleSpec)
        return StyleTy((size, *self.batch_shape), self.dtype_name)

    def leading_axis_spec(self):
        return StyleSpec()


@dataclass(frozen=True)
class StyleHolder(Stylable):
    """Opaque hijax style with separate color and stroke fields."""

    fill_rgb: Scalars
    line_rgb: Scalars
    fill_alpha: Scalars
    line_alpha: Scalars
    stroke_width: Scalars
    set_flags: Any

    def lo_parts(self) -> Tuple[Any, ...]:
        return (
            self.fill_rgb,
            self.line_rgb,
            self.fill_alpha,
            self.line_alpha,
            self.stroke_width,
            self.set_flags,
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(jnp.asarray(self.stroke_width).shape)

    def size(self) -> Tuple[int, ...]:
        return self.shape

    def expand_dims(self, n: int = 1) -> StyleHolder:
        return expand_style(self, n)

    def map_prefix(self, fn: Callable[[Any], Any]) -> StyleHolder:
        parts = [fn(p) for p in self.lo_parts()]
        if all(p is None for p in parts):
            return self
        return make_style(*parts)

    def _flag(self, i: int) -> Any:
        return onp.asarray(self.set_flags)[..., i]

    @property
    def fill_color_(self) -> ColorVec:
        val = onp.asarray(self.fill_rgb)
        flag = self._flag(_F_FILL_COLOR)
        default = onp.broadcast_to(onp.asarray(_DEFAULT_FILL), val.shape)
        return onp.where(flag[..., None], val, default)

    @property
    def line_color_(self) -> ColorVec:
        val = onp.asarray(self.line_rgb)
        flag = self._flag(_F_LINE_COLOR)
        default = onp.broadcast_to(onp.asarray(_DEFAULT_LINE), val.shape)
        return onp.where(flag[..., None], val, default)

    @property
    def fill_opacity_(self) -> Property:
        val = onp.asarray(self.fill_alpha)
        flag = self._flag(_F_FILL_OPACITY)
        default = onp.broadcast_to(onp.asarray(_DEFAULT_FILL_OPACITY), val.shape)
        return onp.where(flag, val, default)

    @property
    def line_opacity_(self) -> Property:
        val = onp.asarray(self.line_alpha)
        flag = self._flag(_F_LINE_OPACITY)
        default = onp.broadcast_to(onp.asarray(_DEFAULT_LINE_OPACITY), val.shape)
        return onp.where(flag, val, default)

    @property
    def line_width_(self) -> Property:
        val = onp.asarray(self.stroke_width)
        flag = self._flag(_F_LINE_WIDTH)
        default = onp.broadcast_to(onp.asarray(_DEFAULT_LINE_WIDTH), val.shape)
        return onp.where(flag, val, default)

    @property
    def output_size(self) -> Property:
        return onp.asarray(200.0)

    @property
    def dashing_(self) -> None:
        return None

    @classmethod
    def empty(cls) -> StyleHolder:
        return cls.empty_with_batch(())

    @classmethod
    def empty_with_batch(cls, batch: Tuple[int, ...], dtype_name: str = "float32") -> StyleHolder:
        dt = jnp.dtype(dtype_name)
        return make_style(
            jnp.broadcast_to(_DEFAULT_FILL.astype(dt), batch + (3,)),
            jnp.broadcast_to(_DEFAULT_LINE.astype(dt), batch + (3,)),
            jnp.broadcast_to(jnp.asarray(_DEFAULT_FILL_OPACITY, dt), batch),
            jnp.broadcast_to(jnp.asarray(_DEFAULT_LINE_OPACITY, dt), batch),
            jnp.broadcast_to(jnp.asarray(_DEFAULT_LINE_WIDTH, dt), batch),
            jnp.zeros(batch + (_N_FLAGS,), dtype=bool),
        )

    @classmethod
    def root(cls, output_size: float) -> StyleHolder:
        return Style()

    def apply_style(self, other: StyleHolder) -> StyleHolder:
        return self.merge(other)

    def merge(self, other: StyleHolder) -> StyleHolder:
        return merge_styles(self, other)

    def to_mpl(self) -> Dict[str, Any]:
        lw = onp.asarray(self.line_width_)
        alpha = onp.asarray(self.fill_opacity_)
        return {
            "facecolor": onp.asarray(self.fill_color_),
            "edgecolor": onp.asarray(self.line_color_),
            "linewidth": lw,
            "alpha": alpha,
        }


register_hitype(
    StyleHolder,
    lambda s: StyleTy(
        tuple(jnp.asarray(s.stroke_width).shape),
        jnp.asarray(s.fill_rgb).dtype.name,
    ),
)


class MakeStyle(VJPHiPrimitive):
    def __init__(self, *avals):
        fc, lc, fo, lo, lw, flags = avals
        batch = tuple(lw.shape)
        self.in_avals = avals
        self.out_aval = StyleTy(batch, fc.dtype.name)
        self.params = {}
        super().__init__()

    def expand(self, fill_color, line_color, fill_opacity, line_opacity, line_width, flags):
        return StyleHolder(
            jnp.asarray(fill_color),
            jnp.asarray(line_color),
            jnp.asarray(fill_opacity),
            jnp.asarray(line_opacity),
            jnp.asarray(line_width),
            jnp.asarray(flags).astype(bool),
        )

    def batch(self, axis_data, args, in_dims):
        if all(d is None for d in in_dims):
            return make_style(*args), None
        moved = []
        for a, d in zip(args, in_dims):
            if d is not None and d != 0:
                a = jnp.moveaxis(a, d, 0)
            moved.append(a)
        return make_style(*moved), StyleSpec()


class MergeStyles(VJPHiPrimitive):
    def __init__(self, a: StyleTy, b: StyleTy):
        batch = tuple(jnp.broadcast_shapes(a.batch_shape, b.batch_shape))
        self.in_avals = (a, b)
        self.out_aval = StyleTy(batch, a.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, a: StyleHolder, b: StyleHolder):
        af, bf = a.set_flags, b.set_flags

        def overlay_color(av, bv, fi):
            return jnp.where(bf[..., fi, None], bv, av)

        def overlay_scalar(av, bv, fi):
            return jnp.where(bf[..., fi], bv, av)

        return StyleHolder(
            overlay_color(a.fill_rgb, b.fill_rgb, _F_FILL_COLOR),
            overlay_color(a.line_rgb, b.line_rgb, _F_LINE_COLOR),
            overlay_scalar(a.fill_alpha, b.fill_alpha, _F_FILL_OPACITY),
            overlay_scalar(a.line_alpha, b.line_alpha, _F_LINE_OPACITY),
            overlay_scalar(a.stroke_width, b.stroke_width, _F_LINE_WIDTH),
            af | bf,
        )

    def batch(self, axis_data, args, in_dims):
        a, b = args
        if all(d is None for d in in_dims):
            return merge_styles(a, b), None
        return merge_styles(a, b), StyleSpec()


class ExpandStyle(VJPHiPrimitive):
    def __init__(self, style_aval: StyleTy, n: int):
        self.in_avals = (style_aval,)
        self.out_aval = StyleTy(
            style_aval.batch_shape + (1,) * int(n), style_aval.dtype_name
        )
        self.params = dict(n=int(n))
        super().__init__()

    def expand(self, style: StyleHolder):
        parts = list(style.lo_parts())
        for _ in range(self.n):
            parts = [p[..., None, :] if i in (0, 1, 5) else p[..., None] for i, p in enumerate(parts)]
        return StyleHolder(*parts)

    def batch(self, axis_data, args, in_dims):
        (style,) = args
        (d,) = in_dims
        if d is None:
            return expand_style(style, self.n), None
        return expand_style(style, self.n), StyleSpec()


def make_style(
    fill_color, line_color, fill_opacity, line_opacity, line_width, flags
) -> StyleHolder:
    parts = [
        jnp.asarray(fill_color),
        jnp.asarray(line_color),
        jnp.asarray(fill_opacity),
        jnp.asarray(line_opacity),
        jnp.asarray(line_width),
        jnp.asarray(flags).astype(bool),
    ]
    return MakeStyle(*[jax.typeof(p) for p in parts])(*parts)


def merge_styles(a, b) -> StyleHolder:
    return MergeStyles(jax.typeof(a), jax.typeof(b))(a, b)


def expand_style(style, n: int = 1) -> StyleHolder:
    if n == 0:
        return style
    return ExpandStyle(jax.typeof(style), n)(style)


__all__ = ["Style", "to_color", "StyleHolder", "StyleTy", "StyleSpec", "make_style"]
