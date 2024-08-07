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
        return self.apply_style(Style(line_width_=width))

    def line_color(self, color: ColorLike) -> Self:
        return self.apply_style(Style(line_color_=to_color(color)))

    def fill_color(self, color: ColorLike) -> Self:
        return self.apply_style(Style(fill_color_=to_color(color)))

    def fill_opacity(self, opacity: float) -> Self:
        return self.apply_style(Style(fill_opacity_=opacity))

    def dashing(self, dashing_strokes: List[float], offset: float) -> Self:
        "TODO: implement this function."
        return self.apply_style(Style())

    def apply_style(self: Self, style: StyleHolder) -> Self:
        raise NotImplementedError("Abstract")


def m(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    return a if a is not None else b


class WidthType(Enum):
    LOCAL = auto()
    NORMALIZED = auto()


def Style(
    line_width_: Optional[PropLike] = None,
    line_color_: Optional[ColorLike] = None,
    line_opacity_: Optional[PropLike] = None,
    fill_color_: Optional[ColorLike] = None,
    fill_opacity_: Optional[PropLike] = None,
    dashing_: Optional[PropLike] = None,
    output_size: Optional[PropLike] = None,
) -> StyleHolder:
    b = (
        tx.np.zeros(STYLE_SIZE),
        tx.np.zeros(STYLE_SIZE, dtype=bool),
    )

    def update(b, key: str, value):  # type: ignore
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

    if line_width_ is not None:
        b = update(b, "line_width", tx.np.asarray(line_width_)[..., None])
    b = update(b, "line_color", line_color_)
    if line_opacity_ is not None:
        b = update(b, "line_opacity", tx.np.asarray(line_opacity_)[..., None])
    b = update(b, "fill_color", fill_color_)
    if fill_opacity_ is not None:
        b = update(b, "fill_opacity", tx.np.asarray(fill_opacity_)[..., None])
    b = update(b, "output_size", output_size)
    return StyleHolder(*b)


@dataclass(frozen=True)
class StyleHolder(Stylable, tx.Batchable):
    base: Scalars
    mask: Mask

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.base.shape[:-1]

    def get(self, key: str) -> tx.Scalars:
        v = self.base[slice(*STYLE_LOCATIONS[key])]
        return tx.np.where(
            self.mask[slice(*STYLE_LOCATIONS[key])], v, DEFAULTS[key]
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
        if self.fill_color_ is not None:
            f = self.fill_color_
            style["facecolor"] = f  # (f[0], f[1], f[2])
            # style += f"fill: rgb({f[0]} {f[1]} {f[2]});"
        if self.line_color_ is not None:
            lc = self.line_color_
            style["edgecolor"] = lc  # (lc[0], lc[1], lc[2])
        else:
            style["edgecolor"] = "black"

        # Set by observation
        if self.line_width_ is not None:
            lw = self.line_width_
            style["linewidth"] = lw.reshape(-1)[0]
        if self.fill_opacity_ is not None:
            style["alpha"] = self.fill_opacity_[0]
        return style

BatchStyle = tx.Batched[StyleHolder, "#*B"]