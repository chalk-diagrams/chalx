from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Self, Tuple

import chalk.transform as tx
from chalk.monoid import Monoid
from chalk.segment import Segment, arc_envelope
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
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    from chalk.core import ApplyTransform, Compose, Primitive
    from chalk.types import Diagram


@dataclass
class EnvDistance(Monoid):
    d: Scalars

    def __add__(self, other: Self) -> EnvDistance:
        return EnvDistance(tx.np.maximum(self.d, other.d))

    @staticmethod
    def empty() -> EnvDistance:
        return EnvDistance(tx.np.asarray(-1e5))

    def reduce(self, axis: int = 0) -> EnvDistance:
        return EnvDistance(tx.np.max(self.d, axis=axis))


@tx.jit
@partial(tx.vectorize, signature="(3,3),(3,1)->(3,1),(3,1),(3,1),()")
def pre_transform(t: Affine, v: V2_t) -> Tuple[V2_t, V2_t, V2_t, Scalars]:
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
def post_transform(
    u: V2_t, v: V2_t, d: tx.Floating, inner: tx.Floating
) -> Scalars:
    after_linear = inner / d

    # Translation
    diff = tx.dot(tx.scale_vec(u, 1 / tx.dot(v, v)), v)
    return tx.np.asarray(after_linear - diff)
    # return tx.np.max(out, axis=-1)


@dataclass
class Envelope(Transformable, Monoid):
    segment: Segment

    def __call__(self, direction: V2_t) -> Scalars:
        def run(d):
            if self.segment.angles.shape[0] == 0:
                return 0
            @partial(tx.np.vectorize, signature="(a,3,3),(a,2)->()")
            def env(t, ang):
                v = Envelope.general_transform(
                    t, lambda x: arc_envelope(t, ang, x), d
                ).max()
                return v
            return env(self.segment.t, self.segment.angles)

        run = tx.multi_vmap(run, len(direction.shape) - 2)  # type: ignore
        return run(direction)  # type: ignore

    def __add__(self, other):
        return Envelope(self.segment + other.segment)

    all_dir = tx.np.stack(
        [tx.unit_x, -tx.unit_x, tx.unit_y, -tx.unit_y], axis=0
    )

    @property
    def center(self) -> P2_t:
        d = [
            self(Envelope.all_dir[d]) for d in range(Envelope.all_dir.shape[0])
        ]
        return P2(
            (-d[1] + d[0]) / 2,
            (-d[3] + d[2]) / 2,
        )

    @property
    def width(self) -> Scalars:
        d1 = self(Envelope.all_dir[:2])
        return tx.np.asarray(d1[0] + d1[1])

    @property
    def height(self) -> Scalars:
        d1 = self(Envelope.all_dir[2:])
        return tx.np.asarray(d1[0] + d1[1])

    def envelope_v(self, v: V2_t) -> V2_t:
        # if self.is_empty:
        #     return V2(0, 0)
        v = tx.norm(v)
        d = self(v)
        return tx.scale_vec(v, d)

    @staticmethod
    def from_bounding_box(box: BoundingBox, d: V2_t) -> Scalars:
        v = box.rotate_rad(tx.rad(d)).br[:, 0, 0]
        return v / tx.length(d)

    def to_bounding_box(self: Envelope) -> BoundingBox:
        d = [
            self(Envelope.all_dir[d]) for d in range(Envelope.all_dir.shape[0])
        ]
        # d = self(Envelope.all_dir)
        return tx.BoundingBox(V2(-d[1], -d[3]), V2(d[0], d[2]))

    def to_path(self, angle: int = 45) -> Iterable[P2_t]:
        "Draws an envelope by sampling every 10 degrees."
        pts = []
        for i in range(0, 361, angle):
            v = tx.polar(i)
            pts.append(tx.scale_vec(v, self(v)))
        return pts

    def to_segments(self, angle: int = 45) -> V2_t:
        "Draws an envelope by sampling every 10 degrees."
        v = tx.polar(tx.np.arange(0, 361, angle) * 1.0)
        return tx.scale_vec(v, self(v))

    @staticmethod
    def general_transform(
        t: Affine, fn: Callable[[V2_t], Scalars], d: V2_t
    ) -> tx.ScalarsC:
        pre = pre_transform(t, d)
        return post_transform(pre[1], pre[2], pre[3], fn(pre[0]))

    def apply_transform(self, t: Affine) -> Envelope:
        return Envelope(self.segment.apply_transform(t[..., None, :, :]))


class GetLocatedSegments(DiagramVisitor[Segment, Affine]):
    """
    Collapse a diagram to its underlying segments.
    """

    A_type = Segment

    def visit_primitive(self, diagram: Primitive, t: Affine) -> Segment:
        segment = diagram.prim_shape.located_segments()
        t = (t @ diagram.transform)
        if len(t.shape) >= 3:
            t = t[..., None, :, :]
        segment =  segment.apply_transform(t)
        return segment

    def visit_compose(self, diagram: Compose, t: Affine) -> Segment:
        # Compose nodes can override the envelope.
        if diagram.envelope is not None:
            return diagram.envelope.accept(self, t)
        return self.A_type.concat(
            [d.accept(self, t) for d in diagram.diagrams]
        )

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine
    ) -> Segment:
        "Defaults to pass over"
        return diagram.diagram.accept(self, t @ diagram.transform)


def get_envelope(self: Diagram, t: Optional[Affine] = None) -> Envelope:
    # assert self.size() == ()
    if t is None:
        t = tx.ident
    segment = self.accept(GetLocatedSegments(), t)
    return Envelope(segment)
