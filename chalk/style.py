from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

from colour import Color
from typing_extensions import Self

import chalk.transform as tx
from chalk.transform import ColorVec, Mask, Property, Scalars

PropLike = Union[Property, float]
ColorLike = Union[str, Color, ColorVec]


def to_color(c: ColorLike) -> ColorVec:
    """Convert various color representations to a ColorVec.

    Args:
    ----
        c: Color representation, can be:
           - str: A color name or hex code
           - `Color`: A colour.Color object
           - ColorVec: Already in the correct format

    Returns:
    -------
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
    ----
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
    -------
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
    return StyleHolder(*b)


@dataclass(frozen=True)
class StyleHolder(Stylable, tx.Batchable):
    base: Scalars
    mask: Mask

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.base.shape[:-1]

    def get(self, key: str) -> tx.Scalars:
        v = self.base[..., slice(*STYLE_LOCATIONS[key])]
        return tx.np.where(
            self.mask[..., slice(*STYLE_LOCATIONS[key])], v, DEFAULTS[key]
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
        return cls(
            tx.np.zeros((STYLE_SIZE)),
            tx.np.zeros((STYLE_SIZE), dtype=bool),
        )

    @classmethod
    def root(cls, output_size: float) -> StyleHolder:
        return Style()

    def apply_style(self, other: StyleHolder) -> StyleHolder:
        return self.merge(other)

    def merge(self, other: StyleHolder) -> StyleHolder:
        mask = self.mask | other.mask
        base = tx.np.where(other.mask, other.base, self.base)
        return StyleHolder(base, mask)

    def to_mpl(self) -> Dict[str, Any]:
        style = {}
        f = self.fill_color_
        style["facecolor"] = f  # (f[0], f[1], f[2])
        # style += f"fill: rgb({f[0]} {f[1]} {f[2]});"
        lc = self.line_color_
        style["edgecolor"] = lc  # (lc[0], lc[1], lc[2])

        # Set by observation
        lw = self.line_width_
        style["linewidth"] = lw[..., 0]
        style["alpha"] = self.fill_opacity_[..., 0]
        return style


BatchStyle = tx.Batched[StyleHolder, "#*B"]

__all__ = ["Style", "to_color"]
