from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from chalk import transform as tx
from chalk.segment import Segment
from chalk.trail import Located, Trail
from chalk.transform import Batched, P2_t, Transformable
from chalk.types import BatchDiagram


@dataclass(frozen=True)
class Text:
    text: tx.Array

    def to_str(self) -> str:
        return tx.np.ndarray.tobytes(self.text).decode("utf-8")  # type: ignore


@dataclass(unsafe_hash=True)
class Path(Transformable, tx.Batchable):
    """Path class."""

    loc_trails: Tuple[Located, ...]
    text: Optional[Text] = None
    scale_invariant: Optional[tx.Mask] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        if not self.loc_trails:
            return ()
        return self.loc_trails[0].trail.segments.angles.shape[:-3]

    def remove_scale(self) -> Path:
        return Path(self.loc_trails, self.text, tx.np.array(True))

    def located_segments(self) -> Segment:
        ls = Segment.empty()
        for loc_trail in self.loc_trails:
            if ls is None:  # type: ignore
                ls = loc_trail.located_segments()
            else:
                ls += loc_trail.located_segments()
        return ls

    # Monoid - compose
    @staticmethod
    def empty() -> Path:
        return Path(())

    def __add__(self: BatchPath, other: BatchPath) -> BatchPath:
        return Path(self.loc_trails + other.loc_trails)

    def apply_transform(self: BatchPath, t: tx.Affine) -> BatchPath:
        return Path(
            tuple(
                [loc_trail.apply_transform(t) for loc_trail in self.loc_trails]
            )
        )

    def points(self) -> Iterable[P2_t]:
        for loc_trails in self.loc_trails:
            for pt in loc_trails.points():
                yield pt

    def stroke(self: BatchPath) -> BatchDiagram:
        "Returns a primitive diagram from a path"

        from chalk.core import Primitive

        return Primitive.from_path(self)

    # Constructors
    @staticmethod
    def from_array(points: P2_t, closed: bool = False) -> Path:
        l = points.shape[0]
        if l == 0:
            return Path.empty()
        offsets = points[tx.np.arange(1, l)] - points[tx.np.arange(0, l - 1)]
        trail = Trail.from_array(offsets, closed)
        return Path(tuple([trail.at(points[0])]))

    # Constructors


def from_points(points: List[P2_t], closed: bool = False) -> Path:
    return Path.from_array(tx.np.stack(points))


def from_point(point: P2_t) -> Path:
    return from_points([point])


def from_text(s: str) -> Path:
    return Path((), Text(tx.np.frombuffer(bytes(s.format(123456), 'utf-8'), dtype=tx.np.uint8)))


def from_pairs(segs: List[Tuple[P2_t, P2_t]], closed: bool = False) -> Path:
    if not segs:
        return Path.empty()
    ls = [segs[0][0]]
    for seg in segs:
        assert seg[0] == ls[-1]
        ls.append(seg[1])
    return from_points(ls, closed)


def from_list_of_tuples(
    coords: List[Tuple[tx.Floating, tx.Floating]], closed: bool = False
) -> Path:
    points = list([tx.P2(x, y) for x, y in coords])
    return from_points(points, closed)


BatchPath = Batched[Path, "*#B"]
