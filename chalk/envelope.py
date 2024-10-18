from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Iterable, Optional, Tuple

from jaxtyping import Float

import chalk.transform as tx
from chalk.monoid import Monoid
from chalk.segment import BatchSegment, Segment, arc_envelope
from chalk.transform import (
    P2,
    V2,
    Affine,
    BoundingBox,
    P2_t,
    Scalars,
    Transformable,
    V2_t,
)
from chalk.transform import Batchable, Batched
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import ApplyTransform, Compose, Primitive
    from chalk.types import Diagram


@tx.jit  # type: ignore
@partial(tx.vectorize, signature="(3,3),(3,1)->(3,1),(3,1),(3,1),()")  # type: ignore
def pre_transform(t: Affine, v: V2_t) -> Tuple[V2_t, V2_t, V2_t, Scalars]:
    """Reshapes the input vector `v` to compute the correct envelope for the transformation."""
    rt = tx.remove_translation(t)
    inv_t = tx.inv(rt)
    trans_t = tx.transpose_translation(rt)
    u: V2_t = -tx.get_translation(t)
    vi = inv_t @ v
    inp = trans_t @ v
    v_prim = tx.norm(inp)
    d = tx.dot(v_prim, vi)
    return v_prim, u, v, tx.np.asarray(d)


@tx.jit
@partial(tx.vectorize, signature="(3,1),(3,1),(),()->()")
def post_transform(u: V2_t, v: V2_t, d: tx.Floating, inner: tx.Floating) -> Scalars:
    """Adjusts the envelope to take the affine transformation into account."""
    after_linear = inner / d
    diff = tx.dot(tx.scale_vec(u, 1 / tx.dot(v, v)), v)
    return tx.np.asarray(after_linear - diff)


@tx.jit
def env(transform: tx.Affine, angles: tx.Angles, d: tx.V2_tC) -> tx.Array:
    # Push the user batch dimensions to the left.
    batch_shape = d.shape[:-2]
    segments_shape = transform.shape[:-2]
    return_shape = batch_shape + segments_shape[:-1]
    if segments_shape[-1] == 0:
        return tx.np.zeros(return_shape)
    for _ in range(len(segments_shape)):
        d = d[..., None, :, :]

    pre = pre_transform(transform, d)
    trans = arc_envelope(transform, angles, pre[0])
    v = post_transform(pre[1], pre[2], pre[3], trans).max(-1)  # type: ignore
    assert v.shape == return_shape, f"{v.shape} {return_shape}"
    return tx.np.asarray(v)


@dataclass
class Envelope(Transformable, Monoid, Batchable):
    segment: BatchSegment

    def __call__(self: BatchEnvelope, direction: tx.V2_tC) -> Float[tx.Array, "..."]:
        """Compute the shortest distance from the origin to the envelope boundary in the given
        direction.

        This function returns the distance to the envelope boundary for each batch of the envelope
        and each batch of the input direction. The output shape will be "*C *B" where B represents
        the batch dimensions of the envelope and C represents the batch dimensions of the direction.

        Args:
            direction: The direction vector to measure the distance.

        Returns:
            An array of distances with shape "*C *B".

        """
        return env(*self.segment.tuple(), direction)
        return env(*self.segment.tuple(), direction)

    def __add__(self: BatchEnvelope, other: BatchEnvelope) -> BatchEnvelope:
        return Envelope(self.segment + other.segment)

    all_dir = tx.np.stack([tx.unit_x, -tx.unit_x, tx.unit_y, -tx.unit_y], axis=0)

    @property
    def center(self: BatchEnvelope) -> P2_t:
        """Calculate the center point based on left, right, top, and bottom distances from origin."""
        d = self(Envelope.all_dir)
        return P2(
            (-d[1] + d[0]) / 2,
            (-d[3] + d[2]) / 2,
        )

    @property
    def width(self: BatchEnvelope) -> Scalars:
        """Calculate the width based on left and right distances from origin."""
        d1 = self(Envelope.all_dir[:2])
        return tx.np.asarray(d1[0] + d1[1])

    @property
    def height(self: BatchEnvelope) -> Scalars:
        """Calculate the height based on top and bottom distances from origin."""
        d1 = self(Envelope.all_dir[2:])
        return tx.np.asarray(d1[0] + d1[1])

    def size(self: BatchEnvelope) -> Tuple[Scalars, Scalars]:
        """Calculate width and height based on left, right, top, and bottom distances from origin."""
        d = self(Envelope.all_dir)
        width = tx.np.asarray(d[0] + d[1])
        height = tx.np.asarray(d[2] + d[3])
        return width, height

    def envelope_v(self: BatchEnvelope, v: V2_t) -> V2_t:
        """Calculate the envelope vector in a given direction from origin."""
        v = tx.norm(v)
        d = self(v)
        return tx.scale_vec(v, d)

    @staticmethod
    def from_bounding_box(box: BoundingBox, d: V2_t) -> Scalars:
        """Calculate envelope scalar from bounding box in a given direction from origin."""
        v = box.rotate_rad(tx.rad(d)).br[:, 0, 0]
        v = v / tx.length(d)
        return v

    def to_bounding_box(self: Envelope) -> BoundingBox:
        """Convert envelope to bounding box."""
        d = self(Envelope.all_dir)
        return tx.BoundingBox(V2(-d[1], -d[3]), V2(d[0], d[2]))

    def to_path(self, angle: int = 45) -> Iterable[P2_t]:
        """Draws an envelope by sampling every 10 degrees."""
        pts = []
        for i in range(0, 361, angle):
            v = tx.polar(i)
            pts.append(tx.scale_vec(v, self(v)))
        return pts

    def to_segments(self, angle: int = 45) -> V2_t:
        """Draws an envelope by sampling every 10 degrees."""
        v = tx.polar(tx.np.arange(0, 361, angle) * 1.0)
        return tx.scale_vec(v, self(v))

    def apply_transform(self, t: Affine) -> Envelope:
        """Apply affine transformation to the envelope."""
        return Envelope(self.segment.apply_transform(t[..., None, :, :]))


class GetLocatedSegments(DiagramVisitor[Segment, Affine]):
    """Collapse a diagram to its underlying segments."""

    A_type = Segment

    def visit_primitive(self, diagram: Primitive, t: Affine) -> Segment:
        segment = diagram.prim_shape.located_segments()
        t = t @ diagram.transform
        if len(t.shape) >= 3:
            t = t[..., None, :, :]
        segment = segment.apply_transform(t)
        return segment

    def visit_compose(self, diagram: Compose, t: Affine) -> Segment:
        # Compose nodes can override the envelope.
        if diagram.envelope is not None:
            return diagram.envelope._accept(self, t)
        return self.A_type.concat([d._accept(self, t) for d in diagram.diagrams])

    def visit_apply_transform(self, diagram: ApplyTransform, t: Affine) -> Segment:
        return diagram.diagram._accept(self, t @ diagram.transform)


@tx.jit
def get_envelope(self: Diagram, t: Optional[Affine] = None) -> Envelope:
    # assert self.size() == ()
    if t is None:
        t = tx.ident
    segment = self._accept(GetLocatedSegments(), t)
    assert (
        segment.shape[: len(self.shape)] == self.shape
    ), f"{segment.transform.shape} {self.shape}"
    return Envelope(segment)


BatchEnvelope = Batched[Envelope, "*#B"]

__all__ = ["BatchEnvelope", "Envelope"]
