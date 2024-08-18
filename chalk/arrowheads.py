from colour import Color

import chalk.transform as tx
from chalk.path import Path, from_list_of_tuples
from chalk.types import Diagram

black = Color("black")


def tri() -> Diagram:
    from chalk.core import Empty

    return (
        from_list_of_tuples(
            [(1.0, 0), (0.0, -1.0), (-1.0, 0), (1.0, 0)], closed=True
        )
        # .remove_scale()
        .stroke()
        .rotate_by(-0.25)
        .fill_color(Color("black"))
        .center_xy()
        .align_r()
        .line_width(0)
        .with_envelope(Empty())
    )


def dart(cut: float = 0.2) -> Diagram:
    from chalk.core import Empty

    pts = tx.np.stack(
        [
            tx.P2(0, -cut),
            tx.P2(1.0, cut),
            tx.P2(0.0, -1.0 - cut),
            tx.P2(-1.0, +cut),
            tx.P2(0, -cut),
        ]
    )
    pts = (
        tx.rotation_angle(-90)
        @ tx.translation(tx.V2(1.5 * cut, 1 + 3 * cut))
        @ pts
    )

    return (
        Path.from_array(
            pts,
            closed=True,
        )
        .remove_scale()
        .stroke()
        .fill_color(Color("black"))
        # .rotate_by(-0.25)
        # .center_xy()
        # .align_r()
        .line_width(0)
        .with_envelope(Empty())
    )


# @dataclass(unsafe_hash=True, frozen=True)
# class ArrowHead(Shape):
#     """Arrow Head."""

#     arrow_shape: Diagram

#     def get_bounding_box(self) -> BoundingBox:
#         # Arrow head don't have a bounding box since we can't accurately know
#         # the size until rendering
#         eps = 1e-4
#         self.bb = BoundingBox(tx.origin, tx.origin + P2(eps, eps))
#         return self.bb

#     def accept(self, visitor: ShapeVisitor[C], **kwargs: Any) -> C:
#         return visitor.visit_arrowhead(self, **kwargs)
