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
from chalk.transform import ColorVec, Mask, Property, Scalars

PropLike = Union[Property, float]
ColorLike = Union[str, Color, ColorVec]


def to_color(c: ColorLike) -> ColorVec:
    """Convert various color representations to a ColorVec.

    Args:
        c: Color representation, can be:
           - str: A color name or hex code
           - `Color`: A colour.Color object
           - ColorVec: Already in the correct format

    Returns:
        ColorVec: A numpy array representing RGB values

    """
    if isinstance(c, str):
        return tx.np.asarray(Color(c).rgb)
    elif isinstance(c, Color):
        return tx.np.asarray(c.rgb)
    return c


FC = Color("white")
LC = Color("black")
LW = 0.1

STYLE_LOCATIONS = {
    "fill_color": (0, 3),
    "fill_opacity": (3, 4),
    "line_color": (4, 7),
    "line_opacity": (7, 8),
    "line_width": (8, 9),
    "output_size": (9, 10),
    "dashing": (10, 12),
}

DEFAULTS = {
    "fill_color": to_color(FC),
    "fill_opacity": tx.np.asarray([1.0]),
    "line_color": to_color(LC),
    "line_opacity": tx.np.asarray([1.0]),
    "line_width": tx.np.asarray([LW]),
    "output_size": tx.np.asarray(200.0),
    "dashing": tx.np.asarray(0),
}
STYLE_SIZE = 12


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
        """TODO: implement this function."""
        return self.apply_style(Style())

    def apply_style(self: Self, style: StyleHolder) -> Self:
        raise NotImplementedError("Abstract")


def m(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    return a if a is not None else b


class WidthType(Enum):
    LOCAL = auto()
    NORMALIZED = auto()


@tx.jit
def Style(
    line_width: Optional[PropLike] = None,
    line_color: Optional[ColorLike] = None,
    line_opacity: Optional[PropLike] = None,
    fill_color: Optional[ColorLike] = None,
    fill_opacity: Optional[PropLike] = None,
) -> StyleHolder:
    """Create a StyleHolder with specified style properties.

    Args:
        line_width: Width of the line. Can be a float or a `Property`.
            Shape: Scalar or broadcastable to the shape of the diagram.
        line_color: Color of the line. Can be a string, `Color` object, or `ColorVec`.
            Shape: RGB tuple or broadcastable to (3,) for each point.
        line_opacity: Opacity of the line. Can be a float or a `Property`.
            Shape: Scalar or broadcastable to the shape of the diagram.
        fill_color: Color of the fill. Can be a string, `Color` object, or `ColorVec`.
            Shape: RGB tuple or broadcastable to (3,) for each point.
        fill_opacity: Opacity of the fill. Can be a float or a `Property`.
            Shape: Scalar or broadcastable to the shape of the diagram.

    Returns:
        A `StyleHolder` object with the specified style properties.

    """
    b = (
        tx.np.zeros(STYLE_SIZE),
        tx.np.zeros(STYLE_SIZE, dtype=bool),
    )

    def update(
        b: Tuple[tx.Array, tx.Array], key: str, value: Any
    ) -> Tuple[tx.Array, tx.Array]:  # type: ignore
        base, mask = b
        index = (Ellipsis, slice(*STYLE_LOCATIONS[key]))
        if value is not None:
            value = tx.np.asarray(value)
            if len(value.shape) != len(base.shape) - 1:
                n = tx.np.zeros(
                    value.shape[: len(value.shape) - len(DEFAULTS[key].shape)]
                    + (STYLE_SIZE,)
                )
                base, _ = tx.np.broadcast_arrays(base, n)
                mask, _ = tx.np.broadcast_arrays(mask, n)
            base = tx.index_update(base, index, value)  # type: ignore
            mask = tx.index_update(mask, index, True)  # type: ignore
        return base, mask

    if line_width is not None:
        b = update(b, "line_width", tx.np.asarray(line_width)[..., None])
    b = update(b, "line_color", line_color)
    if line_opacity is not None:
        b = update(b, "line_opacity", tx.np.asarray(line_opacity)[..., None])
    b = update(b, "fill_color", fill_color)
    if fill_opacity is not None:
        b = update(b, "fill_opacity", tx.np.asarray(fill_opacity)[..., None])
    return make_style(*b)


# ---------------------------------------------------------------------------
# Hijax StyleHolder — opaque style[*B]
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StyleSpec(MappingSpec):
    """vmap spec: styles batch on a leading axis."""


@dataclass(frozen=True)
class StyleTy(HiType):
    batch_shape: Tuple[int, ...]
    dtype_name: str = "float32"

    def lo_ty(self):
        shape = self.batch_shape + (STYLE_SIZE,)
        return [
            ShapedArray(shape, jnp.dtype(self.dtype_name)),
            ShapedArray(shape, jnp.dtype("bool")),
        ]

    def lower_val(self, style: StyleHolder):
        return [style.base, style.mask]

    def raise_val(self, base, mask) -> StyleHolder:
        return StyleHolder(base, mask)

    def to_tangent_aval(self):
        return StyleTy(self.batch_shape, self.dtype_name)

    def vspace_zero(self):
        z = jnp.zeros(
            self.batch_shape + (STYLE_SIZE,), dtype=jnp.dtype(self.dtype_name)
        )
        return StyleHolder(z, jnp.zeros_like(z, dtype=bool))

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
    """Opaque hijax style. Peek ``base``/``mask`` only eagerly or in expand."""

    base: Scalars
    mask: Mask

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.base.shape[:-1])

    def size(self) -> Tuple[int, ...]:
        return self.shape

    def expand_dims(self, n: int = 1) -> StyleHolder:
        return expand_style(self, n)

    def map_prefix(self, fn: Callable[[Any], Any]) -> StyleHolder:
        """Apply an array fn to prefix-batched lojax components (eager)."""
        return make_style(fn(self.base), fn(self.mask))

    def get(self, key: str) -> tx.Scalars:
        import numpy as onp

        base = onp.asarray(self.base)
        mask = onp.asarray(self.mask)
        v = base[..., slice(*STYLE_LOCATIONS[key])]
        return onp.where(
            mask[..., slice(*STYLE_LOCATIONS[key])], v, onp.asarray(DEFAULTS[key])
        )

    @property
    def line_width_(self) -> Property:
        return self.get("line_width")

    @property
    def line_color_(self) -> ColorVec:
        return self.get("line_color")

    @property
    def line_opacity_(self) -> Property:
        return self.get("line_opacity")

    @property
    def fill_color_(self) -> ColorVec:
        return self.get("fill_color")

    @property
    def fill_opacity_(self) -> Property:
        return self.get("fill_opacity")

    @property
    def output_size(self) -> Property:
        return self.get("output_size")

    @property
    def dashing_(self) -> None:
        return None

    @classmethod
    def empty(cls) -> StyleHolder:
        return make_style(
            tx.np.zeros((STYLE_SIZE,)),
            tx.np.zeros((STYLE_SIZE,), dtype=bool),
        )

    @classmethod
    def root(cls, output_size: float) -> StyleHolder:
        return Style()

    def apply_style(self, other: StyleHolder) -> StyleHolder:
        return self.merge(other)

    def merge(self, other: StyleHolder) -> StyleHolder:
        return merge_styles(self, other)

    def to_mpl(self) -> Dict[str, Any]:
        style = {}
        f = self.fill_color_
        style["facecolor"] = f
        lc = self.line_color_
        style["edgecolor"] = lc
        lw = self.line_width_
        style["linewidth"] = lw[..., 0]
        style["alpha"] = self.fill_opacity_[..., 0]
        return style


register_hitype(
    StyleHolder,
    lambda s: StyleTy(
        tuple(s.base.shape[:-1]),
        jnp.asarray(s.base).dtype.name,
    ),
)


class MakeStyle(VJPHiPrimitive):
    def __init__(self, base_aval, mask_aval):
        if tuple(base_aval.shape) != tuple(mask_aval.shape):
            raise TypeError(f"style base/mask shape mismatch: {base_aval} {mask_aval}")
        if not base_aval.shape or base_aval.shape[-1] != STYLE_SIZE:
            raise TypeError(f"style feature dim must be {STYLE_SIZE}, got {base_aval}")
        self.in_avals = (base_aval, mask_aval)
        self.out_aval = StyleTy(
            tuple(base_aval.shape[:-1]), base_aval.dtype.name
        )
        self.params = {}
        super().__init__()

    def expand(self, base, mask):
        return StyleHolder(jnp.asarray(base), jnp.asarray(mask).astype(bool))

    def batch(self, axis_data, args, in_dims):
        base, mask = args
        db, dm = in_dims
        if db is None and dm is None:
            return make_style(base, mask), None
        if db is not None and db != 0:
            base = jnp.moveaxis(base, db, 0)
        if dm is not None and dm != 0:
            mask = jnp.moveaxis(mask, dm, 0)
        return make_style(base, mask), StyleSpec()


class MergeStyles(VJPHiPrimitive):
    def __init__(self, a: StyleTy, b: StyleTy):
        batch = tuple(jnp.broadcast_shapes(a.batch_shape, b.batch_shape))
        self.in_avals = (a, b)
        self.out_aval = StyleTy(batch, a.dtype_name)
        self.params = {}
        super().__init__()

    def expand(self, a: StyleHolder, b: StyleHolder):
        mask = a.mask | b.mask
        base = jnp.where(b.mask, b.base, a.base)
        return StyleHolder(base, mask)

    def batch(self, axis_data, args, in_dims):
        a, b = args
        da, db = in_dims
        if da is None and db is None:
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
        base, mask = style.base, style.mask
        for _ in range(self.n):
            base = base[..., None, :]
            mask = mask[..., None, :]
        return StyleHolder(base, mask)

    def batch(self, axis_data, args, in_dims):
        (style,) = args
        (d,) = in_dims
        if d is None:
            return expand_style(style, self.n), None
        return expand_style(style, self.n), StyleSpec()


def make_style(base, mask) -> StyleHolder:
    base = jnp.asarray(base)
    mask = jnp.asarray(mask).astype(bool)
    return MakeStyle(jax.typeof(base), jax.typeof(mask))(base, mask)


def merge_styles(a, b) -> StyleHolder:
    return MergeStyles(jax.typeof(a), jax.typeof(b))(a, b)


def expand_style(style, n: int = 1) -> StyleHolder:
    if n == 0:
        return style
    return ExpandStyle(jax.typeof(style), n)(style)


__all__ = ["Style", "to_color", "StyleHolder", "StyleTy", "StyleSpec"]
