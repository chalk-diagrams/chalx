from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import chalk.segment as arc
import chalk.transform as tx

# from chalk.envelope import Envelope
from chalk.monoid import Monoid
from chalk.segment import Segment

# from chalk.trace import Trace, TraceDistances
from chalk.transform import Affine, Floating, P2_t, Transformable, V2_t
from chalk.types import Diagram, TrailLike

if TYPE_CHECKING:
    from chalk.path import Path


@dataclass(frozen=True, unsafe_hash=True)
class Located(Transformable):
    """A trail with a location for the origin."""

    trail: Trail
    location: P2_t

    # def split(self, i: int) -> Located:
    #     return Located(self.trail.split(i), self.location[i])

    def located_segments(self) -> Segment:
        pts = self.points()
        return self.trail.segments.apply_transform(tx.translation(pts))

    def points(self) -> P2_t:
        r: P2_t = self.trail.points() + self.location[..., None, :, :]
        return r

    def _promote(self) -> Located:
        return Located(self.trail._promote(), self.location)

    def stroke(self) -> Diagram:
        return self._promote().to_path().stroke()

    def apply_transform(self, t: Affine) -> Located:
        p = t[..., None, :, :] @ self.location
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
    closed: tx.Mask

    # Monoid
    @staticmethod
    def empty() -> Trail:
        """Empty trail for monoid"""
        seg = Segment(tx.np.zeros((0, 3, 3)), tx.np.zeros((0, 2)))
        return Trail(seg, tx.np.full(seg.angles.shape[:-1], False))

    def __add__(self, other: Trail) -> Trail:
        # assert not (self.closed or other.closed), "Cannot add closed trails"
        seg = self.segments + other.segments
        return Trail(seg, tx.np.full(seg.angles.shape[:-1], False))

    # Transformable
    def apply_transform(self, t: Affine) -> Trail:
        """Apply affine transformation to the trail."""
        t = tx.remove_translation(t)
        if len(t.shape) >= 3:
            t = t[:, None, :, :]
        return Trail(self.segments.apply_transform(t), self.closed)

    # Trail-like
    def to_trail(self) -> Trail:
        """Convert to a Trail."""
        return self

    def _promote(self) -> Trail:
        return Trail(self.segments.promote(), self.closed)

    def close(self) -> Trail:
        """Close the trail."""
        return Trail(self.segments, tx.np.ones(self.segments.shape[:-1]))._promote()

    def points(self) -> P2_t:
        """Get points along the trail."""
        q = self.segments.q
        return tx.to_point(tx.np.cumsum(q, axis=-3) - q)

    def at(self, p: P2_t) -> Located:
        """Place the trail at a specific point."""
        return Located(self._promote(), tx.to_point(p))

    # def reverse(self) -> Trail:
    #     return Trail(
    #         [seg.reverse() for seg in reversed(self.segments)],
    #         reversed(segment_angles),
    #         self.closed,
    #     )

    def centered(self) -> Located:
        """Center the trail around the origin."""
        return self.at(
            -tx.np.sum(self.points(), axis=-3) / self.segments.transform.shape[0]
        )

    # Misc. Constructors
    # Todo: Move out of this class?
    @staticmethod
    def from_array(offsets: V2_t, closed: bool = False) -> Trail:
        """Create a `Trail` from an array of offsets."""
        trail = seg(offsets)
        if closed:
            trail = trail.close()
        return trail

    # Misc. Constructors

    @staticmethod
    def from_offsets(offsets: List[V2_t], closed: bool = False) -> Trail:
        """Create a `Trail` from a list of offsets."""
        return Trail.from_array(tx.np.stack(offsets), closed)

    @staticmethod
    def hrule(length: Floating) -> Trail:
        """Create a horizontal rule `Trail` of given length."""
        return seg(length * tx.unit_x)

    @staticmethod
    def vrule(length: Floating) -> Trail:
        """Create a vertical rule `Trail` of given length."""
        return seg(length * tx.unit_y)

    @staticmethod
    def square() -> Trail:
        """Create a square `Trail`."""
        t = seg(tx.unit_x) + seg(tx.unit_y)
        return (t + t.rotate_by(0.5)).close()

    @staticmethod
    def rounded_rectangle(width: Floating, height: Floating, radius: Floating) -> Trail:
        """Create a rounded rectangle `Trail` with given dimensions and corner radius."""
        r = radius
        edge1 = math.sqrt(2 * r * r) / 2
        edge3 = math.sqrt(r * r - edge1 * edge1)
        corner = arc_seg(tx.V2(r, r), -(r - edge3))
        b = [height - r, width - r, height - r, width - r]
        trail = Trail.concat(
            (seg(b[i] * tx.unit_y) + corner).rotate_by(i / 4) for i in range(4)
        ) + seg(0.01 * tx.unit_y)
        return trail.close()

    @staticmethod
    def circle(size: float = 1, clockwise: bool = True) -> Trail:
        """Create a circular `Trail` in the specified direction."""
        sides = 4
        dangle = -90
        rotate_by = 1
        if not clockwise:
            dangle = 90
            rotate_by *= -1
        return Trail.concat(
            [
                arc_seg_angle(0, dangle).rotate_by(rotate_by * i / sides)
                for i in range(sides)
            ]
        ).close()

    @staticmethod
    def regular_polygon(sides: int, side_length: Floating) -> Trail:
        """Create a regular polygon `Trail` with given number of sides and side length."""
        edge = Trail.hrule(1)
        return Trail.concat(edge.rotate_by(i / sides) for i in range(sides)).close()


def seg(offset: V2_t) -> Trail:
    """Draw a straight `Trail` from the origin to `offset` vector."""
    return arc_seg(offset, 1e-3)


def arc_seg(offset: V2_t, height: tx.Floating) -> Trail:
    """Draw curved  `trail` from the origin to `offset` curved to `height`."""
    return arc_between_trail(offset, tx.ftos(height))


def arc_seg_angle(angle: tx.Floating, dangle: tx.Floating) -> Trail:
    """Draw semi-circle from angle to angle+dangle centered at the origin."""
    arc_p = tx.to_point(tx.polar(angle))
    return Segment.make(
        tx.translation(-arc_p), tx.np.asarray([angle, dangle])
    ).to_trail()


def arc_between_trail(q: P2_t, height: tx.Scalars) -> Trail:
    return arc.arc_between(tx.P2(0, 0), q, height).to_trail()


__all__ = ["seg", "arc_seg", "arc_seg_angle", "Trail"]
