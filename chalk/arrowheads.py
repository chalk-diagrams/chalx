from colour import Color

import chalk.transform as tx
from chalk.path import Path
from chalk.types import Diagram

black = Color("black")


def tri() -> Diagram:
    """Triangle arrowhead"""
    from chalk.core import Empty

    return (
        Path.from_list_of_tuples(
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
    """Dart arrowhead"""
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
    pts = tx.rotation_angle(-90) @ tx.translation(tx.V2(1.5 * cut, 1 + 3 * cut)) @ pts

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


__all__ = ["dart", "tri"]
