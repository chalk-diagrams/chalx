from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Self, Tuple

import chalk.transform as tx
from chalk.monoid import Monoid
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
    from chalk.core import (
        ApplyTransform,
        Compose,
        ComposeAxis,
        Empty,
        Primitive,
    )
    from chalk.types import Diagram, Enveloped

# quantize = tx.np.linspace(-100, 100, 1000)
# mult = tx.np.array([1000, 1, 0])[None]


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
@partial(tx.np.vectorize, signature="(3,3),(3,1)->(3,1),(3,1),(3,1),()")
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
@partial(tx.np.vectorize, signature="(3,1),(3,1),(),()->()")
def post_transform(u: V2_t, v: V2_t, d: tx.Floating, inner: tx.Floating) -> Scalars:
    after_linear = inner / d

    # Translation
    diff = tx.dot(tx.scale_vec(u, 1 / tx.dot(v, v)), v)
    return tx.np.asarray(after_linear - diff)
    # return tx.np.max(out, axis=-1)


@dataclass
class Envelope(Transformable, Monoid):
    diagram: Diagram
    affine: Affine


    def __call__(self, direction: V2_t) -> Scalars:
        def get_env(diagram):
            assert diagram.size() == ()
            @partial(tx.np.vectorize, signature="(3, 1)->()")
            def run(d):
                return Envelope.general_transform(
                    self.affine,
                    lambda x: diagram.accept(ApplyEnvelope(), x).d,
                    d,
                )
            return run(direction)

        size = self.diagram.size()
        if size == ():
            return get_env(self.diagram)
        else:
            for _ in range(len(size)):
                get_env = tx.vmap(get_env) # type: ignore
            return get_env(self.diagram)

    
    
    # # Monoid
    @staticmethod
    def empty() -> Envelope:
        from chalk.core import Empty

        return Envelope(Empty(), tx.ident)

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
        d1 = self(Envelope.all_dir[0])
        d2 = self(Envelope.all_dir[1])
        return tx.np.asarray(d1 + d2)

    @property
    def height(self) -> Scalars:
        d1 = self(Envelope.all_dir[2])
        d2 = self(Envelope.all_dir[3])
        return tx.np.asarray(d1 + d2)

    @staticmethod
    def general_transform(
        t: Affine, fn: Callable[[V2_t], Scalars], d: V2_t
    ) -> tx.ScalarsC:
        pre = pre_transform(t, d)
        return post_transform(pre[1], pre[2], pre[3],
                                fn(pre[0]))

    # tx.np.max(out, axis=-1)
    # rt = tx.remove_translation(t)
    # inv_t = tx.inv(rt)
    # trans_t = tx.transpose_translation(rt)
    # u: V2_t = -tx.get_translation(t)

    # def wrapped(v: V2_t) -> tx.Scalars:
    #     # Linear
    #     v = v[..., None, :, :]

    #     vi = inv_t @ v
    #     inp = trans_t @ v
    #     v_prim = tx.norm(inp)
    #     inner = fn(v_prim)
    #     d = tx.dot(v_prim, vi)
    #     after_linear = inner / d

    #     # Translation
    #     diff = tx.dot((u / tx.dot(v, v)[..., None, None]), v)
    #     out = after_linear - diff
    #     return tx.np.max(out, axis=-1)

    # return wrapped(d)

    def apply_transform(self, t: Affine) -> Envelope:
        return Envelope(self.diagram, t @ self.affine)

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



class ApplyEnvelope(DiagramVisitor[EnvDistance, V2_t]):
    A_type = EnvDistance

    def visit_primitive(self, diagram: Primitive, t: V2_t) -> EnvDistance:
        pe = EnvDistance(Envelope.general_transform(
            diagram.transform, diagram.prim_shape.envelope, t))
        return EnvDistance(tx.np.max(pe.d, axis=-1))

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: V2_t
    ) -> EnvDistance:
        return EnvDistance(
            Envelope.general_transform(
                diagram.transform,
                lambda x: diagram.diagram.accept(self, x).d,
                t,
            )
        )

    def visit_compose(self, diagram: Compose, arg):
        "Compose defaults to monoid over children"
        if diagram.envelope is not None:
            return diagram.envelope.diagram.accept(self, arg)
        return self.A_type.concat(
            [d.accept(self, arg) for d in diagram.diagrams]
        )


class GetEnvelope(DiagramVisitor[Envelope, Affine]):
    A_type = Envelope

    def visit_primitive(self, diagram: Primitive, t: Affine) -> Envelope:

        new_transform = t @ diagram.transform
        return Envelope(diagram, t)

    def visit_compose(self, diagram: Compose, t: Affine) -> Envelope:
        if diagram.envelope is not None:
            return Envelope(diagram.envelope.diagram, t)
        return Envelope(diagram, t)

    def visit_compose_axis(self, diagram: ComposeAxis, t: Affine) -> Envelope:
        return Envelope(diagram, t)

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine
    ) -> Envelope:
        n = t @ diagram.transform
        return diagram.diagram.accept(self, n)


def get_envelope(self: Diagram, t: Optional[Affine] = None) -> Envelope:
    # assert self.size() == ()
    if t is None:
        t = tx.ident
    return self.accept(GetEnvelope(), t)
