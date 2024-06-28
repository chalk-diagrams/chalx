from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Tuple
from typing import NamedTuple
import chalk.transform as tx
from chalk.monoid import Monoid
from dataclasses import dataclass
from chalk.transform import (
    Affine,
    P2_t,
    Ray,
    Transformable,
    V2_t,
)

from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:

    from chalk.core import ApplyTransform, Primitive, Compose, ComposeAxis
    from chalk.types import Diagram


@dataclass
class TraceDistances(Monoid):
    distance: tx.Scalars
    mask: tx.Mask

    def __iter__(self): # type: ignore
        yield self.distance
        yield self.mask

    def tuple(self) -> Tuple[tx.Scalars, tx.Mask]: 
        return self.distance, self.mask

    def __getitem__(self, i: int): # type: ignore
        if i == 0: return self.distance
        if i == 1: return self.mask
    
    def __add__(self, other: Self) -> Self: # type: ignore
        return TraceDistances(*tx.X.union(self.tuple(), other))

    @staticmethod
    def empty() -> TraceDistances:
        return TraceDistances(tx.X.np.asarray([]), tx.X.np.asarray([]))
    
    def reduce(self, axis: int=0) -> TraceDistances:
        return TraceDistances(
                *tx.X.union_axis((self.distance, self.mask), axis=axis)
            )

@dataclass
class Trace(Monoid, Transformable):
    diagram: Diagram
    affine: Affine

    # def __init__(self, d f: Callable[[Ray], TraceDistances]) -> None:
    #     self.f = f

    def __call__(self, point: P2_t, direction: V2_t) -> TraceDistances:
        # def apply(x):  # type: ignore
        #     return self.diagram.accept(ApplyTrace(), x[..., 0, :, :]).d[..., None]
        if len(point.shape) == 2:
            point = point.reshape(1, 3, 1)
        if len(direction.shape) == 2:
            direction = direction.reshape(1, 3, 1)
        d, m = Trace.transform(lambda x: self.diagram.accept(ApplyTrace(), x), self.affine,
            Ray(point, direction))
        # d, m = self.f(Ray(point, direction))
        ad = tx.X.np.argsort(d + (1 - m) * 1e10, axis=1)
        d = tx.X.np.take_along_axis(d, ad, axis=1)
        m = tx.X.np.take_along_axis(m, ad, axis=1)
        return TraceDistances(d, m)

    # # Monoid
    # @classmethod
    # def empty(cls) -> Trace:
    #     return cls(lambda _: (tx.X.np.asarray([]), tx.X.np.asarray([])))

    # def __add__(self, other: Trace) -> Trace:
    #     return Trace(lambda ray: tx.X.union(self.f(ray), other.f(ray)))

    @staticmethod
    def general_transform(t: Affine, fn: Callable[[tx.Ray], TraceDistances],
                           r: tx.Ray) -> TraceDistances:  # type: ignore
        t1 = tx.inv(t)

        def wrapped(
            ray: Ray,
        ) -> TraceDistances:
            td = fn(
                Ray(
                    t1 @ ray.pt[..., None, :, :],
                    t1 @ ray.v[..., None, :, :],
                )
            )
            return td.reduce(axis=-1)

        return wrapped(r)

    def apply_transform(self, t: Affine) -> Trace:
        return Trace(self.diagram, t @ self.affine)

    # Transformable
    @staticmethod
    def transform(fn: Callable[[tx.Ray], TraceDistances], 
                  t: Affine, r: Ray) -> TraceDistances:
        def apply(ray: Ray):  # type: ignore
            t, m = fn(Ray(ray.pt[..., 0, :, :], ray.v[..., 0, :, :]))
            return TraceDistances(t[..., None], m[..., None])

        return Trace.general_transform(t, apply, r)

    def trace_v(self, p: P2_t, v: V2_t) -> TraceDistances:
        v = tx.norm(v)
        dists, m = self(p, v)

        d = tx.X.np.sort(dists + (1 - m) * 1e10, axis=1)
        ad = tx.X.np.argsort(dists + (1 - m) * 1e10, axis=1)
        m = tx.X.np.take_along_axis(m, ad, axis=1)
        s = d[:, 0:1]
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
        ps = tx.X.np.concatenate([ps, ps2], axis=1)
        m = tx.X.np.concatenate([m, m2], axis=1)
        ad = tx.X.np.argsort(ps + (1 - m) * 1e10, axis=1)
        ps = tx.X.np.take_along_axis(ps, ad, axis=1)
        m = tx.X.np.take_along_axis(m, ad, axis=1)
        return TraceDistances(ps, m)

class ApplyTrace(DiagramVisitor[TraceDistances, Ray]):
    A_type = TraceDistances
    def visit_primitive(self, diagram: Primitive, ray: Ray) -> TraceDistances:
        return Trace.transform(lambda x: diagram.shape.get_trace(x), 
                               diagram.transform, ray)


    def visit_apply_transform(self, diagram: ApplyTransform, ray: Ray) -> TraceDistances:
        return Trace.transform(lambda x: diagram.diagram.accept(self, x), 
                                diagram.transform, ray)


class GetTrace(DiagramVisitor[Trace, Affine]):
    A_type = Trace

    # def visit_primitive(self, diagram: Primitive, t: Affine) -> Trace:
    #     new_transform = t @ diagram.transform

    #     if diagram.is_multi():
    #         # MultiPrimitive only work in jax mode.
    #         import jax

    #         def trace(ray: Ray) -> TraceDistances:
    #             def inner(shape : Traceable, transform: Affine) -> TraceDistances: 
    #                 trace = shape.get_trace().apply_transform(transform)
    #                 return trace(ray.pt, ray.v)

    #             r = jax.vmap(inner)(diagram.shape, diagram.transform)
    #             return tx.X.union_axis(r, axis=0)

    #         return Trace(trace)
    #     else:
    #         return diagram.shape.get_trace().apply_transform(new_transform)

    def visit_primitive(self, diagram: Primitive, t: Affine) -> Trace:
        return Trace(diagram, t)

    def visit_compose(self, diagram: Compose, t: Affine) -> Trace:
        return Trace(diagram, t)

    def visit_compose_axis(self, diagram: ComposeAxis, t: Affine) -> Trace:
        return Trace(diagram, t)

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine
    ) -> Trace:
        return diagram.diagram.accept(self, t @ diagram.transform)


def get_trace(self: Diagram) -> Trace:
    return self.accept(GetTrace(), tx.X.ident)
