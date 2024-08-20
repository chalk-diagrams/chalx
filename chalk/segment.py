"""
Segment is a collection of ellipse arcs with starting angle and the delta.
Every diagram in chalk is made up of these segments.
They may be either located or at the origin depending on how they are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Tuple

import chalk.transform as tx
from chalk.monoid import Monoid
from chalk.transform import Affine, Angles, Batchable, Batched, P2_t, V2_t

if TYPE_CHECKING:
    from jaxtyping import Array

    from chalk.trail import Trail


def ensure_3d(x: tx.Array) -> tx.Array:
    if len(x.shape) < 3:
        return x.reshape(-1, *x.shape)
    return x


def ensure_2d(x: tx.Array) -> tx.Array:
    if len(x.shape) < 2:
        return x.reshape(-1, *x.shape)
    return x


@dataclass(frozen=True)
class Segment(Monoid, Batchable):
    """
    A batch of ellipse arcs with starting angle and the delta.
    The monoid operation is the concat along the batch dimension.
    """

    transform: Affine
    angles: Angles

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.transform.shape[:-2]

    @property
    def dtype(self) -> str:
        return "seg"

    def tuple(self) -> Tuple[Affine, Angles]:
        return self.transform, self.angles

    @staticmethod
    def empty() -> Batched[Segment, "0"]:
        return Segment(
            tx.np.empty((0, 3, 3)),
            tx.np.empty((0, 2)),
        )

    @staticmethod
    def make(transform: Affine, angles: Angles) -> Segment_t:
        assert angles.shape[-1] == 2
        angles = tx.prefix_broadcast(angles, transform.shape[:-2], 1)  # type: ignore
        return Segment(transform, angles.astype(float))

    def promote(self) -> Segment:
        "Ensures that there is a batch axis"
        return Segment(ensure_3d(self.transform), ensure_2d(self.angles))

    def to_trail(self) -> Trail:
        from chalk.trail import Trail

        return Trail(self, tx.np.full(self.angles.shape[:-1], False))

    def reduce(self, axis: int = 0) -> Segment:
        shape = self.shape
        return Segment(
            self.transform.reshape(*shape[:-2],-1, 3, 3), 
            self.angles.reshape(*shape[:-2], -1, 2)
        )

    # Transformable
    def apply_transform(self, t: Affine) -> Segment_t:
        return Segment.make(t @ self.transform, self.angles)

    def __add__(self, other: Segment) -> Segment:
        def broadcast_ex(a, b, axis): # type: ignore
            a_s, b_s = list(a.shape), list(b.shape)
            a_s[axis] = 1
            b_s[axis] = 1
            new = tx.np.broadcast_shapes(a_s, b_s)
            a_s1, b_s2 = list(new), list(new)
            a_s1[axis] = a.shape[axis]
            b_s2[axis] = b.shape[axis]
            return tx.np.broadcast_to(a, a_s1), tx.np.broadcast_to(b, b_s2)

        if self.transform.shape[0] == 0:
            return other
        self, other = self.promote(), other.promote()
        trans = broadcast_ex(self.transform, other.transform, -3)
        angles = broadcast_ex(self.angles, other.angles, -2)
        return Segment.make(
            tx.np.concatenate(trans, axis=-3),
            tx.np.concatenate(angles, axis=-2),
        )

    @property
    def q(self: Segment_t) -> P2_t:
        "Target point"
        q: P2_t = tx.to_point(tx.polar(self.angles.sum(-1)))
        q = self.transform @ q
        return q

    @property
    def center(self) -> P2_t:
        center: P2_t = self.transform @ tx.P2(0, 0)
        return center

    def is_in_mod_360(self, d: V2_t) -> tx.Mask:
        angle0_deg = self.angles[..., 0]
        angle1_deg = self.angles.sum(-1)

        low = tx.np.minimum(angle0_deg, angle1_deg)
        high = tx.np.maximum(angle0_deg, angle1_deg)
        check = (high - low) % 360
        return tx.np.asarray(((tx.angle(d) - low) % 360) <= check)


def arc_between(p: P2_t, q: P2_t, height: tx.Scalars) -> Segment_t:
    p, q = tx.np.broadcast_arrays(p, q)
    h = abs(height)
    d = tx.length(q - p)
    # Determine the arc's angle θ and its radius r
    θ = tx.np.arccos((d**2 - 4.0 * h**2) / (d**2 + 4.0 * h**2))
    r = d / (2 * tx.np.sin(θ))

    # bend left
    bl = height > 0
    φ = tx.np.where(bl, +tx.np.pi / 2, -tx.np.pi / 2)
    dy = tx.np.where(bl, r - h, h - r)
    flip = tx.np.where(bl, 1, -1)

    diff = q - p
    angles = tx.np.stack(
        [flip * -tx.from_radians(θ), flip * 2 * tx.from_radians(θ)], -1
    )
    ret = (
        tx.translation(p)
        @ tx.rotation(-tx.rad(diff))
        @ tx.translation(tx.V2(d / 2, dy))
        @ tx.rotation(φ)
        @ tx.scale(tx.V2(r, r))
    )
    return Segment.make(ret, angles)


@tx.jit
@partial(tx.vectorize, signature="(3,3),(2),(3,1)->()")
def arc_envelope(trans: Affine, angles: Angles, d: tx.V2_tC) -> Array:
    """
    Compute the envelope for a batch of segments.
    """
    angle0_deg = angles[..., 0]
    angle1_deg = angles.sum(-1)

    is_circle = abs(angle0_deg - angle1_deg) >= 360
    v1 = tx.polar(angle0_deg)
    v2 = tx.polar(angle1_deg)

    return tx.np.where(  # type: ignore
        (is_circle | Segment(trans, angles).is_in_mod_360(d)),
        # Case 1: P2 at arc
        1 / tx.length(d),
        # Case 2: P2 outside of arc
        tx.np.maximum(tx.dot(d, v1), tx.dot(d, v2)),
    )


@tx.jit
@partial(tx.vectorize, signature="(3,3),(2),(3,1),(3,1)->(2),(2)")
def arc_trace(
    trans: Affine, angles: Angles, p: tx.P2_tC, v: tx.V2_tC
) -> Tuple[tx.Array, tx.Array]:
    """
    Computes the trace for a batch of segments.
    """
    ray = tx.Ray(p, v)
    segment = Segment(trans, angles)
    d1, mask1, d2, mask2 = tx.ray_circle_intersection(ray.pt, ray.v, 1)

    # Mask out traces that are not in the angle range.
    mask1 = mask1 & segment.is_in_mod_360(ray.point(d1))
    mask2 = mask2 & segment.is_in_mod_360(ray.point(d2))

    d = tx.np.stack([d1, d2], -1)
    mask = tx.np.stack([mask1, mask2], -1)
    return d, mask


BatchSegment = Batched[Segment, "*#B"]
Segment_t = BatchSegment
