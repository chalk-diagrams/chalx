from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import chalk.transform as tx
from chalk.monoid import Monoid
from chalk.segment import Segment, arc_trace
from chalk.transform import Affine, P2_t, Transformable, V2_t
from chalk.visitor import DiagramVisitor
from chalk.array_types import Batched

if TYPE_CHECKING:
    from chalk.core import ApplyTransform, Primitive
    from chalk.types import Diagram


@tx.jit
def _trace(
    transform: tx.Affine, angles: tx.Angles, point: tx.P2_tC, d: tx.V2_tC
) -> Tuple[tx.Array, tx.Array]:
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
    """A trace is used to compute the distance
    to a segment along a given ray.

    In practice, this object just stores the
    batched segment for a diagram.
    """

    segment: Segment

    def __call__(self, point: P2_t, direction: V2_t) -> Tuple[tx.Scalars, tx.Mask]:
        """Compute the trace for given point and direction.

        Args:
        ----
            point: The starting point for the trace.
            direction: The direction vector for the trace.

        Returns:
            A tuple containing:
            - Distances to intersections
            - Mask indicating valid intersections

        """
        return _trace(*self.segment.tuple(), point, direction)

    def apply_transform(self, t: Affine) -> Trace:
        """Apply an affine transformation to this Trace."""
        return Trace(self.segment.apply_transform(t))

    def trace_v(self, p: P2_t, v: V2_t) -> Tuple[tx.V2_tC, tx.MaskC]:
        """Compute the vectors to intersection from `p` along `v`

        Args:
        ----
            p: The starting point for the trace.
            v: The direction vector for the trace.

        Returns:
            A tuple containing:
            - The vector to intersection
            - Mask indicating valid intersections

        """
        v = tx.norm(v)
        dists, m = self(p, v)

        d = tx.np.sort(dists + (1 - m) * 1e10, axis=-1)
        ad = tx.np.argsort(dists + (1 - m) * 1e10, axis=-1)
        m = tx.np.take_along_axis(m, ad, axis=-1)
        s = d[..., 0]
        return (tx.scale_vec(v, s), m[..., 0])

    def trace_p(self, p: P2_t, v: V2_t) -> Tuple[tx.P2_tC, tx.MaskC]:
        """Compute the intersection point from `p` along `v`"""
        u, m = self.trace_v(p, v)
        return (p + u, m)

    def max_trace_v(self, p: P2_t, v: V2_t) -> Tuple[tx.V2_tC, tx.MaskC]:
        """Compute the maximum trace vector from `p` in direction `-v`"""
        return self.trace_v(p, -v)

    def max_trace_p(self, p: P2_t, v: V2_t) -> Tuple[tx.P2_tC, tx.MaskC]:
        """Compute the maximum trace point from `p` in direction `-v`"""
        u, m = self.max_trace_v(p, v)
        return (p + u, m)


class _GetLocatedSegments(DiagramVisitor[Segment, Affine]):
    A_type = Segment

    def visit_primitive(self, diagram: Primitive, t: Affine) -> Segment:
        segment = diagram.prim_shape.located_segments()
        return segment.apply_transform((t @ diagram.transform)[..., None, :, :])

    def visit_apply_transform(self, diagram: ApplyTransform, t: Affine) -> Segment:
        # Defaults to pass over
        return diagram.diagram._accept(self, t @ diagram.transform)


def get_trace(self: Diagram) -> Trace:
    return Trace(self._accept(_GetLocatedSegments(), tx.ident))


BatchTrace = Batched[Trace, "*#B"]

__all__ = ["BatchTrace", "Trace"]
