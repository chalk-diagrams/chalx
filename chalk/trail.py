from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Tuple, Union

from chalk.envelope import Envelope
from chalk.monoid import Monoid
import chalk.shapes.arc as arc
from chalk.shapes.arc import Segment
from chalk.trace import Trace
import chalk.transform as tx
from chalk.transform import (
    P2_t,
    V2_t,
    Affine,
    Transformable,
    Floating
)
import jax
import jax.numpy as np
from chalk.types import Diagram, Enveloped, Traceable, TrailLike


if TYPE_CHECKING:
    from chalk.shapes.path import Path
    from jaxtyping import Float, Bool, Array

@dataclass
class Located(Enveloped, Traceable, Transformable):
    trail: Trail
    location: P2_t

    def located_segments(self) -> Segment:
        pts = self.points()
        return self.trail.segments.apply_transform(tx.translation(pts))
    
    def points(self) -> Float[Array, "#B 3 1"]:
        return self.trail.points() + self.location 

    def get_envelope(self) -> Envelope:
        s = self.located_segments()
        env = arc.arc_envelope(s.angles)
        t = s.transform
        rt = tx.remove_translation(t)
        inv_t = tx.inv(rt)
        trans_t = tx.transpose_translation(rt)
        u: V2_t = -tx.get_translation(t)
        def wrapped(v: V2_t):
            # Linear
            vi = inv_t @ v
            v_prim = tx.norm(trans_t @ v)
            inner = env(v_prim)
            d = tx.dot(v_prim, vi)
            after_linear = inner / d


            # Translation
            diff = tx.dot((u / tx.dot(v, v)), v)
            out = after_linear - diff
            return tx.np.max(out, axis=0)

        return Envelope(wrapped)


    # def get_trace(self) -> Trace:
    #     return Trace.concat(
    #         segment.get_trace().translate_by(location)
    #         for segment, location in self.located_segments()
    #     )

    def stroke(self) -> Diagram:
        return self.to_path().stroke()

    def apply_transform(self, t: Affine) -> Located:
        return Located(
            self.trail.apply_transform(t), 
            t @ self.location
        )

    def to_path(self) -> Path:
        from chalk.shapes.path import Path

        return Path([self]) 

@dataclass
class Trail(Monoid, Transformable, TrailLike):
    segments: Segment
    
    closed: bool = False

    # Monoid
    @staticmethod
    def empty() -> Trail:
        return Trail(Segment(np.array([]), np.array([])), False)

    def __add__(self, other: Trail) -> Trail:
        assert not (self.closed or other.closed), "Cannot add closed trails"
        return Trail(self.segments + other.segments, False)

    # Transformable
    def apply_transform(self, t: Affine) -> Trail:
        t = tx.remove_translation(t)
        return Trail(
            self.segments.apply_transform(t),
            self.closed
        )

    # Trail-like
    def to_trail(self) -> Trail:
        return self

    def close(self) -> Trail:
        return Trail(self.segments, True)

    def points(self) -> Float[Array, "B 3"]:
        q = self.segments.q
        return (np.cumsum(q, axis=0) - q).at[..., 2, 0].set(1)

    def at(self, p: P2_t) -> Located:
        return Located(self, p)

    # def reverse(self) -> Trail:
    #     return Trail(
    #         [seg.reverse() for seg in reversed(self.segments)],
    #         reversed(segment_angles), 
    #         self.closed,
    #     )

    def centered(self) -> Located:
        return self.at(-sum(self.points(), tx.P2(0, 0)) / self.segments.t.shape[0])

    # # Misc. Constructor
    # @staticmethod
    # def from_offsets(offsets: List[V2_t], closed: bool = False) -> Trail:
    #     return Trail([Segment(off) for off in offsets], closed)


    @staticmethod
    def hrule(length: Floating) -> Trail:
        return arc.seg(length * tx.unit_x)

    @staticmethod
    def vrule(length: Floating) -> Trail:
        return arc.seg(length * tx.unit_y)

    _rectangle = None
    @staticmethod
    def rectangle(width: Floating, height: Floating) -> Trail:
        if Trail._rectangle is None:
            t = arc.seg(tx.unit_x) + arc.seg(tx.unit_y)
            Trail._rectangle = (t + t.rotate_by(0.5)).close()
        return Trail._rectangle.scale_x(width).scale_y(height)

    @staticmethod
    def rounded_rectangle(width: Floating, height: Floating, radius: Floating) -> Trail:
        r = radius
        edge1 = math.sqrt(2 * r * r) / 2
        edge3 = math.sqrt(r * r - edge1 * edge1)
        corner = arc.arc_seg(tx.V2(r, r), -(r - edge3))
        b = [height - r, width - r, height - r, width - r]
        trail = Trail.concat(
            (arc.seg(b[i] * tx.unit_y) + corner).rotate_by(i / 4) for i in range(4)
        ) + arc.seg(0.01 * tx.unit_y)
        return trail.close()

    _circle = {}
    @staticmethod
    def circle(radius: Floating = 1.0, clockwise: bool = True) -> Trail:
        if clockwise in Trail._circle:
            return Trail._circle[clockwise]
        else:
            sides = 4
            dangle = -90
            rotate_by = 1
            if not clockwise:
                dangle = 90
                rotate_by *= -1
            Trail._circle[clockwise] = Trail.concat(
                [
                    arc.arc_seg_angle(0, dangle).rotate_by(rotate_by * i / sides)
                    for i in range(sides)
                ]
            ).close()
        return (
            Trail._circle[clockwise]
            .scale(radius)
        )

    _polygon = {}
    @staticmethod
    def regular_polygon(sides: int, side_length: Floating) -> Trail:
        if sides not in Trail._polygon:
            edge = Trail.hrule(1)
            Trail._polygon[sides] = Trail.concat(
                edge.rotate_by(i / sides) for i in range(sides)
            ).close()
        return Trail._polygon[sides].scale(side_length)
