from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
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

    @property
    def dtype(self):
        return jnp.dtype(self.dtype_name)

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

    def map_prefix(self, fn: Callable[[Any], Any]) -> StyleHolder:
        parts = [fn(p) for p in self.lo_parts()]
        if all(p is None for p in parts):
            return self
        return make_style(*parts)

    def _flag(self, i: int) -> Any:
        return self.set_flags[..., i]

    @property
    def fill_color_(self) -> ColorVec:
        val = jnp.asarray(self.fill_rgb)
        flag = self._flag(_F_FILL_COLOR)
        default = jnp.broadcast_to(_DEFAULT_FILL, val.shape)
        return jnp.where(flag[..., None], val, default)

    @property
    def line_color_(self) -> ColorVec:
        val = jnp.asarray(self.line_rgb)
        flag = self._flag(_F_LINE_COLOR)
        default = jnp.broadcast_to(_DEFAULT_LINE, val.shape)
        return jnp.where(flag[..., None], val, default)

    @property
    def fill_opacity_(self) -> Property:
        val = jnp.asarray(self.fill_alpha)
        flag = self._flag(_F_FILL_OPACITY)
        default = jnp.broadcast_to(_DEFAULT_FILL_OPACITY, val.shape)
        return jnp.where(flag, val, default)

    @property
    def line_opacity_(self) -> Property:
        val = jnp.asarray(self.line_alpha)
        flag = self._flag(_F_LINE_OPACITY)
        default = jnp.broadcast_to(_DEFAULT_LINE_OPACITY, val.shape)
        return jnp.where(flag, val, default)

    @property
    def line_width_(self) -> Property:
        val = jnp.asarray(self.stroke_width)
        flag = self._flag(_F_LINE_WIDTH)
        default = jnp.broadcast_to(_DEFAULT_LINE_WIDTH, val.shape)
        return jnp.where(flag, val, default)

    @property
    def output_size(self) -> Property:
        return jnp.asarray(200.0)

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
        return style_to_mpl(self)


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


class StyleToMpl(VJPHiPrimitive):
    def __init__(self, style_aval: StyleTy):
        b = style_aval.batch_shape
        dt = jnp.dtype(style_aval.dtype_name)
        self.in_avals = (style_aval,)
        self.out_aval = {
            "facecolor": ShapedArray(b + (3,), dt),
            "edgecolor": ShapedArray(b + (3,), dt),
            "linewidth": ShapedArray(b, dt),
            "alpha": ShapedArray(b, dt),
        }
        self.params = {}
        super().__init__()

    def expand(self, style: StyleHolder):
        flags = jnp.asarray(style.set_flags)
        fc = jnp.where(
            flags[..., _F_FILL_COLOR, None],
            style.fill_rgb,
            jnp.broadcast_to(_DEFAULT_FILL, style.fill_rgb.shape),
        )
        lc = jnp.where(
            flags[..., _F_LINE_COLOR, None],
            style.line_rgb,
            jnp.broadcast_to(_DEFAULT_LINE, style.line_rgb.shape),
        )
        fo = jnp.where(
            flags[..., _F_FILL_OPACITY],
            style.fill_alpha,
            jnp.broadcast_to(_DEFAULT_FILL_OPACITY, style.fill_alpha.shape),
        )
        lw = jnp.where(
            flags[..., _F_LINE_WIDTH],
            style.stroke_width,
            jnp.broadcast_to(_DEFAULT_LINE_WIDTH, style.stroke_width.shape),
        )
        return {
            "facecolor": fc,
            "edgecolor": lc,
            "linewidth": lw,
            "alpha": fo,
        }

    def batch(self, axis_data, args, in_dims):
        (style,) = args
        (d,) = in_dims
        if d is None:
            return style_to_mpl(style), None
        return style_to_mpl(style), {k: 0 for k in ("facecolor", "edgecolor", "linewidth", "alpha")}


def style_to_mpl(style) -> Dict[str, Any]:
    return StyleToMpl(jax.typeof(style))(style)


def _paint_rgba(paint):
    """Unpremultiplied ``(rgb [..., 3], opacity [...])`` from a paint value."""
    if isinstance(paint, StyleHolder):
        mpl = style_to_mpl(paint)
        return jnp.asarray(mpl["facecolor"]), jnp.asarray(mpl["alpha"])
    if isinstance(paint, (tuple, list)) and len(paint) == 2 and not isinstance(
        paint[0], (str, bytes)
    ):
        return jnp.asarray(paint[0]), jnp.asarray(paint[1])
    if isinstance(paint, str):
        return jnp.asarray(to_color(paint)), jnp.asarray(1.0)
    paint = jnp.asarray(paint)
    if paint.shape[-1] == 4:
        return paint[..., :3], paint[..., 3]
    return paint, jnp.asarray(1.0)


def composite(img, coverage, paint):
    """Porter-Duff over: geometric coverage × style opacity.

    Matches Cairo ``set_source_rgba(r, g, b, a)`` + fill and SVG
    ``fill-opacity``. ``coverage`` comes from ``trace_measure``; ``paint``
    may be a ``StyleHolder``, RGB, RGBA, colour name, or ``(rgb, opacity)``.
    """
    img = jnp.asarray(img)
    coverage = jnp.asarray(coverage)
    rgb, opacity = _paint_rgba(paint)
    extra_a = coverage.ndim - opacity.ndim
    if extra_a > 0:
        opacity = opacity.reshape(opacity.shape + (1,) * extra_a)
    extra_c = coverage.ndim - (rgb.ndim - 1)
    if extra_c > 0:
        rgb = rgb.reshape(rgb.shape[:-1] + (1,) * extra_c + rgb.shape[-1:])
    alpha = coverage * opacity
    return (1.0 - alpha[..., None]) * img + alpha[..., None] * rgb


def soft_perm_ascending(scores, temperature: float = 0.25):
    """Differentiable permutation: ``P[i, j] ≈ 1`` iff ``scores[j]`` is *i*-th smallest.

    NeuralSort (Grover et al.) on ``-scores`` so small scores paint first
    (behind). ``temperature → 0`` recovers a hard permutation matrix.
    """
    s = jnp.asarray(scores).reshape(-1)
    n = s.shape[0]
    tau = jnp.asarray(temperature, dtype=s.dtype)
    inv = -s
    abs_diff = jnp.abs(inv[:, None] - inv[None, :])
    b = abs_diff.sum(axis=0)
    scaling = (n + 1 - 2 * (jnp.arange(n, dtype=s.dtype) + 1))
    logits = (scaling[:, None] * inv[None, :] - b[None, :]) / jnp.maximum(tau, 1e-6)
    return jax.nn.softmax(logits, axis=-1)


def modulate_opacity(opacity, z, *, floor: float = 0.2, sharpness: float = 1.5):
    """Scale fill opacity by frontness of ``z`` (high z → nearer 1)."""
    gate = floor + (1.0 - floor) * jax.nn.sigmoid(sharpness * jnp.asarray(z))
    return jnp.asarray(opacity) * gate


def composite_by_z(img, coverages, paints, z, *, temperature: float = 0.25, hard: bool = True):
    """Porter-Duff over after sorting layers by ``z`` (low = behind).

    ``coverages`` is ``[N, H, W]``, ``paints`` RGB ``[N, 3]`` or a pair
    ``(rgb[N,3], opacity[N])``, ``z`` is ``[N]``.

    Opacity is also gated by ``z`` so Cairo (hard argsort + the same gate)
    matches the scanline. Soft NeuralSort mixing is opt-in via ``hard=False``.
    """
    img = jnp.asarray(img)
    coverages = jnp.asarray(coverages)
    z = jnp.asarray(z).reshape(-1)
    if isinstance(paints, (tuple, list)) and len(paints) == 2:
        rgb, opacity = jnp.asarray(paints[0]), jnp.asarray(paints[1])
    else:
        rgb, opacity = _paint_rgba(paints)
        rgb = jnp.asarray(rgb)
        opacity = jnp.broadcast_to(jnp.asarray(opacity), z.shape)
    opacity = modulate_opacity(opacity, z)

    if hard:
        order = jnp.argsort(z)
        cov_o, rgb_o, op_o = coverages[order], rgb[order], opacity[order]
    else:
        perm = soft_perm_ascending(z, temperature)
        cov_o = jnp.einsum("ij,jhw->ihw", perm, coverages)
        rgb_o = jnp.einsum("ij,jc->ic", perm, rgb)
        op_o = jnp.einsum("ij,j->i", perm, opacity)

    def paint_over(dst, xs):
        coverage, color, a = xs
        return composite(dst, coverage, (color, a)), None

    out, _ = jax.lax.scan(paint_over, img, (cov_o, rgb_o, op_o))
    return out


__all__ = [
    "Style",
    "to_color",
    "StyleHolder",
    "StyleTy",
    "StyleSpec",
    "make_style",
    "style_to_mpl",
    "composite",
    "soft_perm_ascending",
    "modulate_opacity",
    "composite_by_z",
]
