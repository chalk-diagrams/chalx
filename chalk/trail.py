from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import chalk.transform as tx

# from chalk.envelope import Envelope
from chalk.monoid import Monoid
from chalk.segment import Segment
import chalk.segment as arc

# from chalk.trace import Trace, TraceDistances
from chalk.transform import Affine, Floating, P2_t, Transformable, V2_t
from chalk.types import Diagram, TrailLike

if TYPE_CHECKING:
    from chalk.path import Path


@dataclass(frozen=True, unsafe_hash=True)
class Located(Transformable):
    """
    A trail with a location for the origin. 
    """

    trail: Trail
    location: P2_t

    def split(self, i: int) -> Located:
        return Located(self.trail.split(i), self.location[i])

    def located_segments(self) -> Segment:
        pts = self.points()
        return self.trail.segments.apply_transform(tx.translation(pts))

    def points(self) -> P2_t:
        return self.trail.points() + self.location[..., None, :, :]

    def _promote(self) -> Located:
        return Located(self.trail._promote(), self.location)

    def stroke(self) -> Diagram:
        return self._promote().to_path().stroke()

    def apply_transform(self, t: Affine) -> Located:
        p = t @ self.location
        if len(p.shape) == 3:
            p = p[:, None]
        if len(p.shape) == 2:
            p = p[None]

        return Located(self.trail.apply_transform(tx.remove_translation(t)), p)

    def to_path(self) -> Path:
        from chalk.path import Path

        return Path(tuple([Located(self.trail._promote(), self.location)]))


@dataclass(frozen=True, unsafe_hash=True)
class Trail(Monoid, Transformable, TrailLike):
    segments: Segment

    closed: tx.Mask = field(default_factory=lambda: tx.np.asarray(False))

    def split(self, i: int) -> Trail:
        return Trail(self.segments.split(i), self.closed[i])

    # Monoid
    @staticmethod
    def empty() -> Trail:
        return Trail(
            Segment(tx.np.zeros((0, 3, 3)), tx.np.zeros((0, 2))),
            tx.np.asarray(False),
        )

    def __add__(self, other: Trail) -> Trail:
        # assert not (self.closed or other.closed), "Cannot add closed trails"
        return Trail(self.segments + other.segments, tx.np.asarray(False))

    # Transformable
    def apply_transform(self, t: Affine) -> Trail:
        t = tx.remove_translation(t)
        return Trail(self.segments.apply_transform(t), self.closed)

    # Trail-like
    def to_trail(self) -> Trail:
        return self

    def _promote(self) -> Trail:
        return Trail(self.segments.promote(), self.closed)

    def close(self) -> Trail:
        return Trail(self.segments, tx.np.asarray(True))._promote()

    def points(self) -> P2_t:
        q = self.segments.q
        return tx.to_point(tx.np.cumsum(q, axis=-3) - q)

    def at(self, p: P2_t) -> Located:
        return Located(self._promote(), tx.to_point(p))

    # def reverse(self) -> Trail:
    #     return Trail(
    #         [seg.reverse() for seg in reversed(self.segments)],
    #         reversed(segment_angles),
    #         self.closed,
    #     )

    def centered(self) -> Located:
        return self.at(
            -tx.np.sum(self.points(), axis=-3) / self.segments.t.shape[0]
        )

    # Misc. Constructors
    # Todo: Move out of this class?
    @staticmethod
    def from_offsets(offsets: List[V2_t], closed: bool = False) -> Trail:
        trail = Trail.concat([arc.seg(off) for off in offsets])
        if closed:
            trail = trail.close()
        return trail

    @staticmethod
    def hrule(length: Floating) -> Trail:
        return arc.seg(length * tx.unit_x)

    @staticmethod
    def vrule(length: Floating) -> Trail:
        return arc.seg(length * tx.unit_y)

    @staticmethod
    def rectangle(width: Floating, height: Floating) -> Trail:
        t = arc.seg(tx.unit_x) + arc.seg(tx.unit_y)
        return (t + t.rotate_by(0.5)).close().scale_x(width).scale_y(height)

    @staticmethod
    def rounded_rectangle(
        width: Floating, height: Floating, radius: Floating
    ) -> Trail:
        r = radius
        edge1 = math.sqrt(2 * r * r) / 2
        edge3 = math.sqrt(r * r - edge1 * edge1)
        corner = arc.arc_seg(tx.V2(r, r), -(r - edge3))
        b = [height - r, width - r, height - r, width - r]
        trail = Trail.concat(
            (arc.seg(b[i] * tx.unit_y) + corner).rotate_by(i / 4)
            for i in range(4)
        ) + arc.seg(0.01 * tx.unit_y)
        return trail.close()

    @staticmethod
    def circle(radius: Floating = 1.0, clockwise: bool = True) -> Trail:
        sides = 4
        dangle = -90
        rotate_by = 1
        if not clockwise:
            dangle = 90
            rotate_by *= -1
        return (
            Trail.concat(
                [
                    arc.arc_seg_angle(0, dangle).rotate_by(
                        rotate_by * i / sides
                    )
                    for i in range(sides)
                ]
            )
            .close()
            .scale(radius)
        )

    @staticmethod
    def regular_polygon(sides: int, side_length: Floating) -> Trail:
        edge = Trail.hrule(1)
        return (
            Trail.concat(edge.rotate_by(i / sides) for i in range(sides))
            .close()
            .scale(side_length)
        )
