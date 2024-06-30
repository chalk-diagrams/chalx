from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import chalk.transform as tx
from chalk.envelope import Envelope
from chalk.trace import TraceDistances
from chalk.trail import Trail
from chalk.transform import BoundingBox
from chalk.types import Diagram
from chalk.visitor import C, ShapeVisitor


@dataclass(unsafe_hash=True, frozen=True)
class Shape:
    """Shape class."""

    def get_bounding_box(self) -> BoundingBox:
        raise NotImplementedError

    def envelope(self, t: tx.V2_t) -> tx.Scalars:
        return Envelope.from_bounding_box(self.get_bounding_box(), t)

    def split(self, i: int) -> Self:
        return self

    def get_trace(self, t: tx.Ray) -> TraceDistances:
        box = self.get_bounding_box()
        return (
            Trail.rectangle(box.width, box.height)
            .stroke()
            .center_xy()
            .get_trace()(t.pt, t.v)
        )

    def accept(self, visitor: ShapeVisitor[C], **kwargs: Any) -> C:
        raise NotImplementedError

    def stroke(self) -> Diagram:
        """Returns a primitive (shape) with strokes

        Returns:
            Diagram: A diagram.
        """
        from chalk.core import Primitive

        return Primitive.from_shape(self)


@dataclass(unsafe_hash=True, frozen=True)
class Spacer(Shape):
    """Spacer class."""

    width: tx.Scalars
    height: tx.Scalars

    def get_bounding_box(self) -> BoundingBox:
        left = -self.width / 2
        top = -self.height / 2
        tl = tx.P2(left, top)
        br = tx.P2(left + self.width, top + self.height)
        return BoundingBox(tl, br)

    def accept(self, visitor: ShapeVisitor[C], **kwargs: Any) -> C:
        return visitor.visit_spacer(self, **kwargs)
