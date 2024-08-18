from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import chalk.transform as tx
from chalk.monoid import Monoid
from chalk.segment import Segment, arc_trace
from chalk.transform import Affine, P2_t, Transformable, V2_t
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import ApplyTransform, Primitive
    from chalk.types import Diagram

TraceDistances = Tuple[tx.Scalars, tx.Mask]


@tx.jit
def trace(
    transform: tx.Affine, angles: tx.Angles, point: tx.P2_tC, d: tx.V2_tC
):
    point, direction = tx.np.broadcast_arrays(point, d)

    # Push the __call__ batch dimensions to the left.
    # batch_shape = point.shape[:-2]

    segments_shape = transform.shape[:-2]
    for _ in range(len(segments_shape)):
        point = point[..., None, :, :]
        direction = direction[..., None, :, :]

    t1 = tx.inv(transform)
    d, m = arc_trace(transform, angles, t1 @ point, t1 @ d)
    d = d.reshape(d.shape[:-2] + (-1,))
    m = m.reshape(m.shape[:-2] + (-1,))

    ad = tx.np.argsort(d + (1 - m) * 1e10, axis=-1)
    d = tx.np.take_along_axis(d, ad, axis=-1)
    m = tx.np.take_along_axis(m, ad, axis=-1)
    return (d, m)


@dataclass
class Trace(Monoid, Transformable):
    """
    A trace is used to compute the distance
    to a segment along a given ray.

    In practice, this object just stores the
    batched segment for a diagram.
    """

    segment: Segment

    def __call__(self, point: P2_t, direction: V2_t) -> TraceDistances:
        return trace(*self.segment.tuple(), point, direction)

    def apply_transform(self, t: Affine) -> Trace:
        return Trace(self.segment.apply_transform(t))

    def trace_v(self, p: P2_t, v: V2_t) -> TraceDistances:
        v = tx.norm(v)
        dists, m = self(p, v)

        d = tx.np.sort(dists + (1 - m) * 1e10, axis=1)
        ad = tx.np.argsort(dists + (1 - m) * 1e10, axis=1)
        m = tx.np.take_along_axis(m, ad, axis=1)
        s = d[:, 0]
        return (s[..., None] * v, m[:, 0])

    def trace_p(self, p: P2_t, v: V2_t) -> TraceDistances:
        u, m = self.trace_v(p, v)
        return (p + u, m)

    def max_trace_v(self, p: P2_t, v: V2_t) -> TraceDistances:
        return self.trace_v(p, -v)

    def max_trace_p(self, p: P2_t, v: V2_t) -> TraceDistances:
        u, m = self.max_trace_v(p, v)
        return (p + u, m)


class GetLocatedSegments(DiagramVisitor[Segment, Affine]):
    "Collapses a diagram into a batch of its segments."
    A_type = Segment

    def visit_primitive(self, diagram: Primitive, t: Affine) -> Segment:
        segment = diagram.prim_shape.located_segments()
        return segment.apply_transform(
            (t @ diagram.transform)[..., None, :, :]
        )

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine
    ) -> Segment:
        "Defaults to pass over"
        return diagram.diagram.accept(self, t @ diagram.transform)


def get_trace(self: Diagram) -> Trace:
    return Trace(self.accept(GetLocatedSegments(), tx.ident))
