from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Self, Tuple

import chalk.transform as tx
from chalk.monoid import Monoid
from chalk.segment import Segment, arc_trace
from chalk.transform import Affine, P2_t, Ray, Transformable, V2_t
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import ApplyTransform, Primitive
    from chalk.types import Diagram


@dataclass
class TraceDistances(Monoid):
    distance: tx.Scalars
    mask: tx.Mask

    def __iter__(self):  # type: ignore
        yield self.distance
        yield self.mask

    def tuple(self) -> Tuple[tx.Scalars, tx.Mask]:
        return self.distance, self.mask

    def __getitem__(self, i: int):  # type: ignore
        if i == 0:
            return self.distance
        if i == 1:
            return self.mask

    def __add__(self, other: TraceDistances) -> TraceDistances:  # type: ignore
        return TraceDistances(*tx.union(self.tuple(), other.tuple()))

    @staticmethod
    def empty() -> TraceDistances:
        return TraceDistances(tx.np.asarray([]), tx.np.asarray([]))

    def reduce(self, axis: int = 0) -> TraceDistances:
        return TraceDistances(
            *tx.union_axis((self.distance, self.mask), axis=axis)
        )


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
        if len(point.shape) == 2:
            point = point.reshape(1, 3, 1)
        if len(direction.shape) == 2:
            direction = direction.reshape(1, 3, 1)
        assert point[..., -1, 0] == 1.0, point
        d, m = Trace.general_transform(
            self.segment.transform,
            lambda x: arc_trace(self.segment, x),
            Ray(point, direction),
        )

        ad = tx.np.argsort(d + (1 - m) * 1e10, axis=1)
        d = tx.np.take_along_axis(d, ad, axis=1)
        m = tx.np.take_along_axis(m, ad, axis=1)
        return TraceDistances(d, m)

    @staticmethod
    def general_transform(
        t: Affine, fn: Callable[[tx.Ray], Tuple[tx.Array, tx.Array]], r: tx.Ray
    ) -> TraceDistances:  # type: ignore
        t1 = tx.inv(t)

        def wrapped(
            ray: Ray,
        ) -> TraceDistances:
            td = TraceDistances(
                *fn(
                    Ray(
                        t1 @ ray.pt[..., None, :, :],
                        t1 @ ray.v[..., None, :, :],
                    )
                )
            )
            return td.reduce(axis=-1)

        return wrapped(r)

    def apply_transform(self, t: Affine) -> Trace:
        return Trace(self.segment.apply_transform(t))

    def trace_v(self, p: P2_t, v: V2_t) -> TraceDistances:
        v = tx.norm(v)
        dists, m = self(p, v)

        d = tx.np.sort(dists + (1 - m) * 1e10, axis=1)
        ad = tx.np.argsort(dists + (1 - m) * 1e10, axis=1)
        m = tx.np.take_along_axis(m, ad, axis=1)
        s = d[:, 0]
        return TraceDistances(s[..., None] * v, m[:, 0])

    def trace_p(self, p: P2_t, v: V2_t) -> TraceDistances:
        u, m = self.trace_v(p, v)
        return TraceDistances(p + u, m)

    def max_trace_v(self, p: P2_t, v: V2_t) -> TraceDistances:
        return self.trace_v(p, -v)

    def max_trace_p(self, p: P2_t, v: V2_t) -> TraceDistances:
        u, m = self.max_trace_v(p, v)
        return TraceDistances(p + u, m)

    @staticmethod
    def combine(p1: TraceDistances, p2: TraceDistances) -> TraceDistances:
        ps, m = p1
        ps2, m2 = p2
        ps = tx.np.concatenate([ps, ps2], axis=1)
        m = tx.np.concatenate([m, m2], axis=1)
        ad = tx.np.argsort(ps + (1 - m) * 1e10, axis=1)
        ps = tx.np.take_along_axis(ps, ad, axis=1)
        m = tx.np.take_along_axis(m, ad, axis=1)
        return TraceDistances(ps, m)


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
