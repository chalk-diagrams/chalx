"""Opaque batched Path via JAX hijax types.

Chalk currently batch-broadcasts ``Path`` by storing prefix batch axes on
every internal array and treating the dataclass as a pytree (see
``chalk.path.Path`` + ``Batchable``). That makes Paths *look* like JAX
arrays, but under ``jit``/``vmap`` they explode into independent leaves
and callers can freely break invariants.

Hijax types are the better model: a Path is one value of one type in a
jaxpr, produced and consumed only through primitives.

This module is a standalone mock of that design. It does not replace
``chalk.path`` yet.

Public API (all go through hijax primitives)::

    from chalk.hijax_path import circle, from_points, translate, concat, to_points

    p = circle(1.0)
    ps = jax.vmap(circle, out_axes=PathSpec())(jnp.arange(1.0, 5.0))
    verts = to_points(translate(p, 1.0, 0.0))

Internally a path is a polyline:

* ``vertices`` — ``float32[*batch, n_pts, 2]``
* ``closed``   — ``bool[*batch]``

Do not construct ``HiPath`` in traced code; use primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.experimental.hijax import (
    HiType,
    MappingSpec,
    ShapedArray,
    VJPHiPrimitive,
    register_hitype,
)

# ---------------------------------------------------------------------------
# Value + type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HiPath:
    """Concrete path payload. Opaque outside primitive ``expand`` methods."""

    vertices: jax.Array  # f32[*B, N, 2]
    closed: jax.Array  # bool[*B]


@dataclass(frozen=True)
class PathSpec(MappingSpec):
    """vmap mapping spec: paths batch only along a new leading axis."""


@dataclass(frozen=True)
class PathTy(HiType):
    """Abstract type of an opaque path: ``path[*B; N]``."""

    batch_shape: Tuple[int, ...]
    n_pts: int

    def lo_ty(self):
        return [
            ShapedArray(self.batch_shape + (self.n_pts, 2), jnp.dtype("float32")),
            ShapedArray(self.batch_shape, jnp.dtype("bool")),
        ]

    def lower_val(self, path: HiPath):
        return [path.vertices, path.closed]

    def raise_val(self, vertices, closed) -> HiPath:
        return HiPath(vertices, closed)

    def to_tangent_aval(self):
        # Vertex displacements; not another Path.
        return ShapedArray(
            self.batch_shape + (self.n_pts, 2), jnp.dtype("float32")
        )

    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        batch = ",".join(str(d) for d in self.batch_shape)
        prefix = f"{batch};" if batch else ""
        return f"path[{prefix}{self.n_pts}]"

    __repr__ = str_short

    def dec_rank(self, size, spec):
        assert isinstance(spec, PathSpec)
        assert self.batch_shape and self.batch_shape[0] == size
        return PathTy(self.batch_shape[1:], self.n_pts)

    def inc_rank(self, size, spec):
        assert isinstance(spec, PathSpec)
        return PathTy((size, *self.batch_shape), self.n_pts)

    def leading_axis_spec(self):
        return PathSpec()


register_hitype(
    HiPath,
    lambda p: PathTy(tuple(p.vertices.shape[:-2]), int(p.vertices.shape[-2])),
)


def _as_closed_array(closed: bool | jax.Array, batch_shape: Tuple[int, ...]):
    if isinstance(closed, jax.Array):
        return closed.astype(bool)
    return jnp.broadcast_to(jnp.asarray(closed, dtype=bool), batch_shape)


def _identity_affine() -> jax.Array:
    return jnp.eye(3, dtype=jnp.float32)


def _apply_affine(vertices: jax.Array, affine: jax.Array) -> jax.Array:
    """Apply a (batched) 3x3 affine to vertices ``[..., N, 2]``."""
    v_batch = vertices.shape[:-2]
    a_batch = affine.shape[:-2]
    batch = tuple(jnp.broadcast_shapes(v_batch, a_batch))
    vertices = jnp.broadcast_to(vertices, batch + vertices.shape[-2:])
    affine = jnp.broadcast_to(affine, batch + (3, 3))
    ones = jnp.ones(vertices.shape[:-1] + (1,), dtype=vertices.dtype)
    homo = jnp.concatenate([vertices, ones], axis=-1)
    moved = jnp.einsum("...ij,...nj->...ni", affine, homo)
    return moved[..., :2]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class FromPoints(VJPHiPrimitive):
    def __init__(self, pts_aval, closed: bool):
        if pts_aval.dtype != jnp.dtype("float32"):
            raise TypeError(pts_aval.dtype)
        if pts_aval.ndim < 2 or pts_aval.shape[-1] != 2:
            raise TypeError(f"points must be [..., N, 2], got {pts_aval}")
        self.in_avals = (pts_aval,)
        self.out_aval = PathTy(tuple(pts_aval.shape[:-2]), int(pts_aval.shape[-2]))
        self.params = dict(closed=bool(closed))
        super().__init__()

    def expand(self, pts):
        closed = _as_closed_array(self.closed, tuple(pts.shape[:-2]))
        return HiPath(pts.astype(jnp.float32), closed)

    def batch(self, axis_data, args, in_dims):
        (pts,) = args
        (d,) = in_dims
        if d is None:
            return from_points(pts, closed=self.closed), None
        pts = jnp.moveaxis(pts, d, 0)
        return from_points(pts, closed=self.closed), PathSpec()

    def vjp_fwd(self, nzs_in, pts):
        return self(pts), None

    def vjp_bwd_retval(self, _res, g):
        return (g,)


class ToPoints(VJPHiPrimitive):
    def __init__(self, path_aval: PathTy):
        self.in_avals = (path_aval,)
        self.out_aval = ShapedArray(
            path_aval.batch_shape + (path_aval.n_pts, 2), jnp.dtype("float32")
        )
        self.params = {}
        super().__init__()

    def expand(self, path: HiPath):
        return path.vertices

    def batch(self, axis_data, args, in_dims):
        (path,) = args
        (d,) = in_dims
        if d is None:
            return to_points(path), None
        assert isinstance(d, PathSpec)
        return to_points(path), 0

    def vjp_fwd(self, nzs_in, path):
        return self(path), None

    def vjp_bwd_retval(self, _res, g):
        return (g,)


class Transform(VJPHiPrimitive):
    def __init__(self, path_aval: PathTy, affine_aval):
        if affine_aval.shape[-2:] != (3, 3):
            raise TypeError(f"affine must end in 3x3, got {affine_aval}")
        batch = tuple(
            jnp.broadcast_shapes(path_aval.batch_shape, tuple(affine_aval.shape[:-2]))
        )
        self.in_avals = (path_aval, affine_aval)
        self.out_aval = PathTy(batch, path_aval.n_pts)
        self.params = {}
        super().__init__()

    def expand(self, path: HiPath, affine):
        verts = _apply_affine(path.vertices, affine)
        closed = jnp.broadcast_to(path.closed, verts.shape[:-2])
        return HiPath(verts, closed)

    def batch(self, axis_data, args, in_dims):
        path, affine = args
        d_path, d_aff = in_dims
        if d_path is None and d_aff is None:
            return transform(path, affine), None
        if d_aff is not None and d_aff != 0:
            affine = jnp.moveaxis(affine, d_aff, 0)
        return transform(path, affine), PathSpec()

    def vjp_fwd(self, nzs_in, path, affine):
        return self(path, affine), affine

    def vjp_bwd_retval(self, affine, g):
        # Path tangents are vertex arrays; apply R^T to push back.
        r_t = jnp.swapaxes(affine[..., :2, :2], -1, -2)
        batch = tuple(jnp.broadcast_shapes(r_t.shape[:-2], g.shape[:-2]))
        r_t = jnp.broadcast_to(r_t, batch + (2, 2))
        g = jnp.broadcast_to(g, batch + g.shape[-2:])
        g_path = jnp.einsum("...ij,...nj->...ni", r_t, g)
        return (g_path, jnp.zeros_like(affine))


class Concat(VJPHiPrimitive):
    def __init__(self, a: PathTy, b: PathTy):
        batch = tuple(jnp.broadcast_shapes(a.batch_shape, b.batch_shape))
        self.in_avals = (a, b)
        self.out_aval = PathTy(batch, a.n_pts + b.n_pts)
        self.params = {}
        super().__init__()

    def expand(self, p: HiPath, q: HiPath):
        batch = tuple(
            jnp.broadcast_shapes(p.vertices.shape[:-2], q.vertices.shape[:-2])
        )
        pv = jnp.broadcast_to(p.vertices, batch + p.vertices.shape[-2:])
        qv = jnp.broadcast_to(q.vertices, batch + q.vertices.shape[-2:])
        pc = jnp.broadcast_to(p.closed, batch)
        qc = jnp.broadcast_to(q.closed, batch)
        return HiPath(jnp.concatenate([pv, qv], axis=-2), pc & qc)

    def batch(self, axis_data, args, in_dims):
        p, q = args
        dp, dq = in_dims
        if dp is None and dq is None:
            return concat(p, q), None
        return concat(p, q), PathSpec()


class Bounds(VJPHiPrimitive):
    """Axis-aligned bounding box: returns ``(min_xy, max_xy)`` each ``[..., 2]``."""

    def __init__(self, path_aval: PathTy):
        box = ShapedArray(path_aval.batch_shape + (2,), jnp.dtype("float32"))
        self.in_avals = (path_aval,)
        self.out_aval = (box, box)
        self.params = {}
        super().__init__()

    def expand(self, path: HiPath):
        return path.vertices.min(axis=-2), path.vertices.max(axis=-2)

    def batch(self, axis_data, args, in_dims):
        (path,) = args
        (d,) = in_dims
        if d is None:
            return bounds(path), None
        assert isinstance(d, PathSpec)
        return bounds(path), (0, 0)


# ---------------------------------------------------------------------------
# Public constructors / ops
# ---------------------------------------------------------------------------


def from_points(pts, closed: bool = False) -> HiPath:
    pts = jnp.asarray(pts, dtype=jnp.float32)
    return FromPoints(jax.typeof(pts), closed)(pts)


def to_points(path) -> jax.Array:
    return ToPoints(jax.typeof(path))(path)


def transform(path, affine) -> HiPath:
    affine = jnp.asarray(affine, dtype=jnp.float32)
    return Transform(jax.typeof(path), jax.typeof(affine))(path, affine)


def concat(p, q) -> HiPath:
    return Concat(jax.typeof(p), jax.typeof(q))(p, q)


def bounds(path):
    return Bounds(jax.typeof(path))(path)


def translate(path, dx, dy=0.0) -> HiPath:
    dx = jnp.asarray(dx, dtype=jnp.float32)
    dy = jnp.asarray(dy, dtype=jnp.float32)
    # Build a (possibly batched) translation matrix without peeking at path.
    affine = jnp.zeros(jnp.broadcast_shapes(dx.shape, dy.shape) + (3, 3), jnp.float32)
    affine = affine.at[..., 0, 0].set(1.0)
    affine = affine.at[..., 1, 1].set(1.0)
    affine = affine.at[..., 2, 2].set(1.0)
    affine = affine.at[..., 0, 2].set(dx)
    affine = affine.at[..., 1, 2].set(dy)
    return transform(path, affine)


def rotate(path, degrees) -> HiPath:
    rad = jnp.deg2rad(jnp.float32(degrees))
    c, s = jnp.cos(rad), jnp.sin(rad)
    affine = jnp.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32
    )
    return transform(path, affine)


def scale(path, sx, sy=None) -> HiPath:
    if sy is None:
        sy = sx
    affine = jnp.array(
        [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float32
    )
    return transform(path, affine)


def circle(radius=1.0, n: int = 32) -> HiPath:
    """Unit-circle polyline scaled by ``radius`` (batched if ``radius`` is)."""
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False, dtype=jnp.float32)
    unit = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    r = jnp.asarray(radius, dtype=jnp.float32)
    pts = unit * r[..., None, None]
    return from_points(pts, closed=True)


def square(size=1.0) -> HiPath:
    half = jnp.asarray(size, dtype=jnp.float32) / 2.0
    # Broadcast-friendly corners: [*B, 4, 2]
    z = jnp.zeros((), dtype=jnp.float32)
    o = half + z
    corners = jnp.stack(
        [
            jnp.stack([-o, -o], axis=-1),
            jnp.stack([o, -o], axis=-1),
            jnp.stack([o, o], axis=-1),
            jnp.stack([-o, o], axis=-1),
        ],
        axis=-2,
    )
    return from_points(corners, closed=True)


def beside(p, q, sep: float = 0.1) -> HiPath:
    """Place ``q`` to the right of ``p``. Ordinary traced code over primitives."""
    _, max_p = bounds(p)
    min_q, _ = bounds(q)
    dx = max_p[..., 0] - min_q[..., 0] + jnp.float32(sep)
    return concat(p, translate(q, dx, 0.0))


__all__ = [
    "HiPath",
    "PathSpec",
    "PathTy",
    "beside",
    "bounds",
    "circle",
    "concat",
    "from_points",
    "rotate",
    "scale",
    "square",
    "to_points",
    "transform",
    "translate",
]
