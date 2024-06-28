from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Self



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
    from chalk.core import ApplyTransform, Compose, Primitive, Empty, ComposeAxis
    from chalk.types import Diagram
    from chalk.types import Enveloped

# quantize = tx.np.linspace(-100, 100, 1000)
# mult = tx.np.array([1000, 1, 0])[None]

@dataclass
class EnvDistance(Monoid):
    d: Scalars

    def __add__(self, other: Self) -> EnvDistance:
        return EnvDistance(tx.X.np.maximum(self.d, other.d))

    @staticmethod
    def empty() -> EnvDistance:
        return EnvDistance(tx.X.np.asarray(-1e5))
    
    def reduce(self, axis: int=0) -> EnvDistance:
        return EnvDistance(tx.X.np.max(self.d, axis=axis))

import jax
@jax.jit
def pre_transform(t: Affine, v: V2_t):
    print(t.shape, v.shape)
    rt = tx.remove_translation(t)
    inv_t = tx.inv(rt)
    trans_t = tx.transpose_translation(rt)
    u: V2_t = -tx.get_translation(t)
    v = v[..., None, :, :]

    vi = inv_t @ v
    inp = trans_t @ v
    v_prim = tx.norm(inp)
    return u, v, v_prim, vi

@jax.jit
def post_transform(u, v, v_prim, vi, inner):
    d = tx.dot(v_prim, vi)
    after_linear = inner / d

    # Translation
    diff = tx.dot((u / tx.dot(v, v)[..., None, None]), v)
    out = after_linear - diff
    return tx.X.np.max(out, axis=-1)

@dataclass
class Envelope(Transformable, Monoid):
    diagram: Diagram
    affine: Affine

    def __call__(self, direction: V2_t) -> Scalars:
        def get_env(diagram):
            return Envelope.transform(self.affine, 
                                     lambda x: diagram.accept(ApplyEnvelope(), x).d, 
                                     direction)
        size = self.diagram.size()
        if size == ():
            return get_env(self.diagram)
        else:
            import jax
            for _ in range(len(size)):
                get_env = jax.vmap(get_env)
            return get_env(self.diagram)
        

    # # Monoid
    @staticmethod
    def empty() -> Envelope:
        from chalk.core  import Empty
        return Envelope(Empty(), tx.X.ident)

    all_dir = tx.X.np.concatenate(
        [tx.X.unit_x, -tx.X.unit_x, tx.X.unit_y, -tx.X.unit_y], axis=0
    )

    @property
    def center(self) -> P2_t:
        # Get all the directions
        d = self(Envelope.all_dir)
        # d = list([self(Envelope.all_dir[i][None])         
        #     for i in range(4)])
        return P2(
            (-d[1] + d[0]) / 2,
            (-d[3] + d[2]) / 2,
        )

    @property
    def width(self) -> Scalars:
        #assert not self.is_empty
        # d = self(Envelope.all_dir[0:1]) + self(Envelope.all_dir[1:2])
        d = self(Envelope.all_dir[:2])
        return d.sum(-1)

    @property
    def height(self) -> Scalars:
        #assert not self.is_empty
        d = self(Envelope.all_dir[2:])
        return d.sum(-1)


    @staticmethod
    def general_transform(t: Affine, fn: Callable[[V2_t], Scalars], 
                          d: V2_t) -> Scalars: 
        pre = pre_transform(t, d)
        inner = fn(pre[2])
        return post_transform(*pre, inner)
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
        #     return tx.X.np.max(out, axis=-1)

        # return wrapped(d)

    def apply_transform(self, t: Affine) -> Envelope:
        return Envelope(self.diagram, t @ self.affine)

    @staticmethod
    def transform(t: Affine, fn: Callable[[V2_t], Scalars], d: V2_t) -> Scalars:
        def apply(x):  # type: ignore
            return fn(x[..., 0, :, :])[..., None]

        return Envelope.general_transform(t, apply, d)

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
        d = self(Envelope.all_dir)
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
        v = tx.polar(tx.X.np.arange(0, 361, angle) * 1.0)
        return tx.scale_vec(v, self(v))

import jax
from functools import partial

@jax.jit
def path_envelope(trans, path, v):
    return EnvDistance(
            Envelope.transform(trans,
                               path.envelope,
                               v))


class ApplyEnvelope(DiagramVisitor[EnvDistance, V2_t]):
    A_type = EnvDistance

    def visit_primitive(self, diagram: Primitive, t: V2_t) -> EnvDistance:
        return path_envelope(diagram.transform, diagram.shape, t)
        # return EnvDistance(
        #     Envelope.transform(diagram.transform, 
        #                        diagram.shape.envelope,
        #                        t))


    def visit_apply_transform(self, diagram: ApplyTransform, t: V2_t) -> EnvDistance:
        return EnvDistance(Envelope.transform(
            diagram.transform,
            lambda x: diagram.diagram.accept(self, x).d,
            t))



class GetEnvelope(DiagramVisitor[Envelope, Affine]):
    A_type = Envelope

    def visit_primitive(self, diagram: Primitive, t: Affine) -> Envelope:

        new_transform = t @ diagram.transform
        # if diagram.is_multi():
        #     # MultiPrimitive only work in jax mode.
        #     import jax

        #     def env(v: V2_t) -> Scalars:
        #         def inner(shape: Enveloped, transform: Affine) -> Scalars:
        #             env = shape.get_envelope().apply_transform(transform)
        #             return env(v)

        #         r = jax.vmap(inner)(diagram.shape, diagram.transform)
        #         return r.max(0)

        #     return Envelope(env)
        # else:
        return Envelope(diagram, t) #.get_envelope().apply_transform(new_transform)

    def visit_compose(self, diagram: Compose, t: Affine) -> Envelope:
        if diagram.envelope is not None:
            print("skipped")
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
    #assert self.size() == ()
    if t is None:
        t = tx.X.ident
    return self.accept(GetEnvelope(), t)
