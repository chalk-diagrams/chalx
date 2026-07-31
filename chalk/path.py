from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.experimental.hijax import (
    HiType,
    MappingSpec,
    ShapedArray,
    VJPHiPrimitive,
    register_hitype,
)

import chalk.geom as geom
from chalk import transform as tx
from chalk.segment import Segment, concat_segments
from chalk.trail import (
    Located,
    LocatedSpec,
    LocatedTy,
    Trail,
    located_segments,
    make_located,
    transform_located,
)
from chalk.transform import P2_t, Transformable
from chalk.types import Diagram


@dataclass(frozen=True)
class Text:
    text: tx.Array

    def to_str(self) -> str:
        return bytes(jax.device_get(self.text)).decode("utf-8")


@dataclass(frozen=True)
class PathSpec(MappingSpec):
    pass


@dataclass(frozen=True)
class PathTy(HiType):
    loc_tys: Tuple[LocatedTy, ...]
    text_len: int = 0
    has_scale_inv: bool = False

    @property
    def n_locs(self) -> int:
        return len(self.loc_tys)

    def lo_ty(self):
        los = []
        for lt in self.loc_tys:
            los.extend(lt.lo_ty())
        if self.text_len:
            los.append(ShapedArray((self.text_len,), jnp.dtype("uint8")))
        if self.has_scale_inv:
            los.append(ShapedArray((), jnp.dtype("bool")))
        return los

    def lower_val(self, path: Path):
        los = []
        for loc, lt in zip(path.loc_trails, self.loc_tys):
            los.extend(lt.lower_val(loc))
        if self.text_len:
            assert path.text is not None
            los.append(jnp.asarray(path.text.text, dtype=jnp.uint8))
        if self.has_scale_inv:
            los.append(jnp.asarray(True))
        return los

    def raise_val(self, *los) -> Path:
        it = iter(los)
        locs = []
        for lt in self.loc_tys:
            parts = [next(it) for _ in lt.lo_ty()]
            locs.append(lt.raise_val(*parts))
        text = None
        scale_inv = None
        if self.text_len:
            text = Text(next(it))
        if self.has_scale_inv:
            next(it)
            scale_inv = jnp.asarray(True)
        return Path(tuple(locs), text, scale_inv)

    def to_tangent_aval(self):
        return PathTy(self.loc_tys, self.text_len, self.has_scale_inv)

    def str_short(self, short_dtypes=False, mesh_axis_types=False):
        n = self.n_locs
        extra = []
        if self.text_len:
            extra.append(f"txt{self.text_len}")
        if self.has_scale_inv:
            extra.append("si")
        suffix = (";" + ",".join(extra)) if extra else ""
        return f"cpath[{n}{suffix}]"

    __repr__ = str_short

    def dec_rank(self, size, spec):
        assert isinstance(spec, PathSpec)
        return PathTy(
            tuple(lt.dec_rank(size, LocatedSpec()) for lt in self.loc_tys),
            self.text_len,
            self.has_scale_inv,
        )

    def inc_rank(self, size, spec):
        assert isinstance(spec, PathSpec)
        return PathTy(
            tuple(lt.inc_rank(size, LocatedSpec()) for lt in self.loc_tys),
            self.text_len,
            self.has_scale_inv,
        )

    def leading_axis_spec(self):
        return PathSpec()


@dataclass(frozen=True)
class Path(Transformable):
    """Opaque hijax path: zero or more Located contours (+ optional text)."""

    loc_trails: Tuple[Located, ...]
    text: Optional[Text] = None
    scale_invariant: Optional[tx.Mask] = None

    def map_prefix(self, fn):
        locs = tuple(loc.map_prefix(fn) for loc in self.loc_trails)
        return _make_path(locs, self.text, self.scale_invariant)

    @property
    def shape(self) -> Tuple[int, ...]:
        return path_shape(self)

    def remove_scale(self) -> Path:
        return path_remove_scale(self)

    def located_segments(self) -> Segment:
        return path_located_segments(self)

    @staticmethod
    def empty() -> Path:
        return empty_path()

    def __add__(self, other: Path) -> Path:
        return concat_paths(self, other)

    @classmethod
    def concat(cls, elems):
        from chalk.monoid import reduce_associative

        return reduce_associative(concat_paths, elems, cls.empty())

    def apply_transform(self, t: tx.Affine) -> Path:
        return transform_path(self, t)

    def points(self) -> Iterable[P2_t]:
        from chalk.trail import located_points

        for i in range(jax.typeof(self).n_locs):
            yield located_points(path_get_located(self, i))

    def stroke(self) -> Diagram:
        from chalk.core import Primitive

        return Primitive.from_path(self)

    @staticmethod
    def from_array(points: P2_t, closed: bool = False) -> Path:
        point_data = tx.data(points)
        l = point_data.shape[-3]
        if l == 0:
            return Path.empty()
        offsets = (
            point_data[..., tx.np.arange(1, l), :, :]
            - point_data[..., tx.np.arange(0, l - 1), :, :]
        )
        trail = Trail.from_array(geom.make_v2_from_data(offsets), closed)
        start = geom.make_p2_from_data(point_data[..., 0, :, :])
        return _make_path((trail.at(start),))

    @staticmethod
    def from_points(points: List[P2_t], closed: bool = False) -> Path:
        ls_points = tx.np.broadcast_arrays(*[tx.data(p) for p in points])
        return Path.from_array(tx.np.stack(ls_points, axis=-3), closed)

    @staticmethod
    def from_point(point: P2_t) -> Path:
        return Path.from_points([point])

    @staticmethod
    def from_text(s: str) -> Path:
        return _make_path(
            (),
            Text(jnp.frombuffer(bytes(s, "utf-8"), dtype=jnp.uint8)),
            None,
        )

    @staticmethod
    def from_pairs(segs: List[Tuple[P2_t, P2_t]], closed: bool = False) -> Path:
        if not segs:
            return Path.empty()
        ls = [segs[0][0]]
        for seg in segs:
            ls.append(seg[1])
        return Path.from_points(ls, closed)

    @staticmethod
    def from_list_of_tuples(
        coords: List[Tuple[tx.Floating, tx.Floating]], closed: bool = False
    ) -> Path:
        points = list([tx.P2(x, y) for x, y in coords])
        return Path.from_points(points, closed)


def _path_typeof(path: Path) -> PathTy:
    loc_tys = tuple(jax.typeof(loc) for loc in path.loc_trails)
    text_len = int(path.text.text.size) if path.text is not None else 0
    return PathTy(loc_tys, text_len, path.scale_invariant is not None)


register_hitype(Path, _path_typeof)


class MakePath(VJPHiPrimitive):
    def __init__(self, loc_avals: Tuple[LocatedTy, ...], text_aval=None, has_si=False):
        ins: list = list(loc_avals)
        text_len = 0
        if text_aval is not None:
            ins.append(text_aval)
            text_len = int(text_aval.shape[0])
        if has_si:
            ins.append(ShapedArray((), jnp.dtype("bool")))
        self.in_avals = tuple(ins)
        self.out_aval = PathTy(loc_avals, text_len, has_si)
        self.params = dict(n_locs=len(loc_avals), has_si=has_si, text_len=text_len)
        super().__init__()

    def expand(self, *args):
        n = self.n_locs
        locs = tuple(args[:n])
        rest = args[n:]
        text = None
        scale_inv = None
        i = 0
        if self.text_len:
            text = Text(rest[i])
            i += 1
        if self.has_si:
            scale_inv = rest[i]
        return Path(locs, text, scale_inv)

    def batch(self, axis_data, args, in_dims):
        n = self.n_locs
        locs = args[:n]
        rest = list(args[n:])
        text = Text(rest[0]) if self.text_len else None
        si = rest[-1] if self.has_si else None
        out = _make_path(locs, text, si)
        if all(d is None for d in in_dims):
            return out, None
        return out, PathSpec()


def _make_path(
    locs: Sequence[Located],
    text: Optional[Text] = None,
    scale_invariant=None,
) -> Path:
    locs = tuple(locs)
    args: list = list(locs)
    text_aval = None
    has_si = scale_invariant is not None
    if text is not None:
        buf = jnp.asarray(text.text, dtype=jnp.uint8)
        args.append(buf)
        text_aval = jax.typeof(buf)
    if has_si:
        args.append(jnp.asarray(True))
    loc_avals = tuple(jax.typeof(l) for l in locs)
    return MakePath(loc_avals, text_aval, has_si)(*args)


def empty_path() -> Path:
    return _make_path(())


class ConcatPaths(VJPHiPrimitive):
    def __init__(self, a: PathTy, b: PathTy):
        self.in_avals = (a, b)
        self.out_aval = PathTy(
            a.loc_tys + b.loc_tys,
            # text only kept if exclusively on one side and empty other locs? keep first non-zero
            a.text_len or b.text_len,
            a.has_scale_inv or b.has_scale_inv,
        )
        self.params = {}
        super().__init__()

    def expand(self, a: Path, b: Path):
        return Path(
            a.loc_trails + b.loc_trails,
            a.text or b.text,
            a.scale_invariant if a.scale_invariant is not None else b.scale_invariant,
        )

    def batch(self, axis_data, args, in_dims):
        a, b = args
        if all(d is None for d in in_dims):
            return concat_paths(a, b), None
        return concat_paths(a, b), PathSpec()


def concat_paths(a, b) -> Path:
    return ConcatPaths(jax.typeof(a), jax.typeof(b))(a, b)


class TransformPath(VJPHiPrimitive):
    def __init__(self, path_aval: PathTy, t_aval):
        self.in_avals = (path_aval, t_aval)
        self.out_aval = path_aval
        self.params = {}
        super().__init__()

    def expand(self, path: Path, t):
        locs = tuple(transform_located(loc, t) for loc in path.loc_trails)
        return Path(locs, path.text, path.scale_invariant)

    def batch(self, axis_data, args, in_dims):
        path, t = args
        if all(d is None for d in in_dims):
            return transform_path(path, t), None
        return transform_path(path, t), PathSpec()


def transform_path(path, t) -> Path:
    t = tx.data(t)
    return TransformPath(jax.typeof(path), jax.typeof(t))(path, t)


class PathLocatedSegments(VJPHiPrimitive):
    def __init__(self, path_aval: PathTy):
        self.in_avals = (path_aval,)
        if path_aval.loc_tys:
            # concat all located segments: n_segs sums
            n = sum(lt.trail_ty.seg_ty.n_segs for lt in path_aval.loc_tys)
            batch = path_aval.loc_tys[0].trail_ty.seg_ty.batch_shape
            dtype = path_aval.loc_tys[0].trail_ty.seg_ty.dtype_name
            from chalk.segment import SegTy

            self.out_aval = SegTy(batch, n, dtype)
        else:
            from chalk.segment import SegTy

            self.out_aval = SegTy((), 0, "float64")
        self.params = {}
        super().__init__()

    def expand(self, path: Path):
        ls = Segment.empty()
        for loc in path.loc_trails:
            ls = concat_segments(ls, located_segments(loc))
        return ls

    def batch(self, axis_data, args, in_dims):
        (path,) = args
        (d,) = in_dims
        if d is None:
            return path_located_segments(path), None
        from chalk.segment import SegSpec

        return path_located_segments(path), SegSpec()


def path_located_segments(path) -> Segment:
    return PathLocatedSegments(jax.typeof(path))(path)


class PathGetLocated(VJPHiPrimitive):
    def __init__(self, path_aval: PathTy, idx: int):
        self.in_avals = (path_aval,)
        self.out_aval = path_aval.loc_tys[idx]
        self.params = dict(idx=int(idx))
        super().__init__()

    def expand(self, path: Path):
        return path.loc_trails[self.idx]

    def batch(self, axis_data, args, in_dims):
        (path,) = args
        (d,) = in_dims
        if d is None:
            return path_get_located(path, self.idx), None
        return path_get_located(path, self.idx), LocatedSpec()


def path_get_located(path, idx: int) -> Located:
    return PathGetLocated(jax.typeof(path), idx)(path)


class PathShape(VJPHiPrimitive):
    def __init__(self, path_aval: PathTy):
        self.in_avals = (path_aval,)
        # return a small int vector via host; use empty marker array of batch rank
        batch = ()
        if path_aval.loc_tys:
            batch = path_aval.loc_tys[0].trail_ty.seg_ty.batch_shape
        self.out_aval = ShapedArray((len(batch),), jnp.dtype("int32"))
        self.params = dict(batch=batch)
        super().__init__()

    def expand(self, path: Path):
        return jnp.asarray(self.batch, dtype=jnp.int32)

    def batch(self, axis_data, args, in_dims):
        (path,) = args
        return path_shape_vec(path), 0


def path_shape_vec(path):
    return PathShape(jax.typeof(path))(path)


def path_shape(path) -> Tuple[int, ...]:
    ty = jax.typeof(path)
    if not ty.loc_tys:
        return ()
    return ty.loc_tys[0].trail_ty.seg_ty.batch_shape


class PathRemoveScale(VJPHiPrimitive):
    def __init__(self, path_aval: PathTy):
        self.in_avals = (path_aval,)
        self.out_aval = PathTy(path_aval.loc_tys, path_aval.text_len, True)
        self.params = {}
        super().__init__()

    def expand(self, path: Path):
        return Path(path.loc_trails, path.text, jnp.asarray(True))

    def batch(self, axis_data, args, in_dims):
        (path,) = args
        (d,) = in_dims
        if d is None:
            return path_remove_scale(path), None
        return path_remove_scale(path), PathSpec()


def path_remove_scale(path) -> Path:
    return PathRemoveScale(jax.typeof(path))(path)


class PathScaleInv(VJPHiPrimitive):
    def __init__(self, path_aval: PathTy):
        self.in_avals = (path_aval,)
        self.out_aval = ShapedArray((), jnp.dtype("bool"))
        self.params = {}
        super().__init__()

    def expand(self, path: Path):
        return jnp.asarray(path.scale_invariant is not None)

    def batch(self, axis_data, args, in_dims):
        (path,) = args
        return path_is_scale_invariant(path), None


def path_is_scale_invariant(path):
    return PathScaleInv(jax.typeof(path))(path)


class PathTextBytes(VJPHiPrimitive):
    def __init__(self, path_aval: PathTy):
        self.in_avals = (path_aval,)
        self.out_aval = ShapedArray((path_aval.text_len,), jnp.dtype("uint8"))
        self.params = {}
        super().__init__()

    def expand(self, path: Path):
        if path.text is None:
            return jnp.zeros((0,), dtype=jnp.uint8)
        return jnp.asarray(path.text.text, dtype=jnp.uint8)

    def batch(self, axis_data, args, in_dims):
        (path,) = args
        return path_text_bytes(path), None


def path_text_bytes(path):
    return PathTextBytes(jax.typeof(path))(path)


__all__ = [
    "Path",
    "Text",
    "empty_path",
    "concat_paths",
    "transform_path",
    "path_located_segments",
    "path_get_located",
    "path_is_scale_invariant",
    "path_text_bytes",
    "PathSpec",
]
