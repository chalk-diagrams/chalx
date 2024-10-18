"""A set of helper functions that help users debug diagram properties."""

import chalk.transform as tx
from chalk.combinators import concat
from chalk.path import Path
from chalk.shapes import circle, text
from chalk.trail import seg
from chalk.transform import V2_t
from chalk.types import Diagram


def show_origin(self: Diagram) -> Diagram:
    envelope = self.get_envelope()
    origin_size = tx.np.maximum(
        0.1, tx.np.minimum(envelope.height, envelope.width) / 50
    )
    origin = circle(origin_size).line_color("red")
    return self + origin


def show_envelope(self: Diagram, phantom: bool = False, angle: int = 45) -> Diagram:
    self.show_origin()
    envelope = self.get_envelope()
    outer: Diagram = (
        Path.from_points(list(envelope.to_path(angle)))
        .stroke()
        .fill_opacity(0)
        .line_color("red")
    )
    segments = envelope.to_segments(angle)

    outer = outer + (
        concat([seg(segments[i][None]).stroke() for i in range(segments.shape[0])])
        .line_color("blue")
        .dashing([0.01, 0.01], 0)
    )

    new = self + outer
    if phantom:
        new = new.with_envelope(self)
    return new


def show_beside(self: Diagram, other: Diagram, direction: V2_t) -> Diagram:
    envelope1 = self.get_envelope()
    envelope2 = other.get_envelope()
    v1 = envelope1.envelope_v(direction)
    one: Diagram = (
        Path.from_points([tx.origin, v1])
        .stroke()
        .line_color("red")
        .dashing([0.01, 0.01], 0)
        .line_width(0.01)
    )
    v2 = envelope2.envelope_v(-direction)
    two: Diagram = (
        Path.from_points([tx.origin, v2])
        .stroke()
        .line_color("red")
        .dashing([0.01, 0.01], 0)
        .line_width(0.01)
    )
    split: Diagram = (
        Path.from_points(
            [
                v1 + tx.perpendicular(direction),
                v1 - tx.perpendicular(direction),
            ]
        )
        .stroke()
        .line_color("blue")
        .line_width(0.02)
    )
    one = (self.show_origin() + one + split).with_envelope(self)
    two = (other.show_origin() + two).with_envelope(other)
    return one.beside(two, direction)


def show_labels(self: Diagram, font_size: tx.Floating = 1) -> Diagram:
    for name, subs in self.get_sub_map(tx.ident).items():
        for sub in subs:
            n = str(name)
            p = sub.get_location()
            self = self + text(n, font_size).fill_color("red").translate_by(p)
    return self


__all__ = []
