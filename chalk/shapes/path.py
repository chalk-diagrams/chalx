from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from chalk import transform as tx

# from chalk.envelope import Envelope
from chalk.envelope import EnvDistance
from chalk.trace import TraceDistances
from chalk.trail import Located, Trail
from chalk.transform import P2_t, Transformable
from chalk.types import Diagram


def make_path(
    segments: List[Tuple[float, float]], closed: bool = False
) -> Diagram:
    p = Path.from_list_of_tuples(segments, closed).stroke()
    return p

@dataclass(frozen=True)
class Text:
    text: tx.Array

    def to_str(self) -> str:
        return self.text.tostring().decode("utf-8")

@dataclass(unsafe_hash=True)
class Path(Transformable):
    """Path class."""

    loc_trails: Tuple[Located, ...]
    text: Optional[Text] = None

    def split(self, i: int) -> Path:
        return Path(tuple([loc.split(i) for loc in self.loc_trails]))

    def located_segments(self) -> Segment:
        ls = None
        for loc_trail in self.loc_trails:
            if ls is None:
                ls = loc_trail.located_segments()
            else:
                ls += loc_trail.located_segments() 
        return ls
        


    # Monoid - compose
    @staticmethod
    def empty() -> Path:
        return Path(())

    def __add__(self, other: Path) -> Path:
        return Path(self.loc_trails + other.loc_trails)

    def apply_transform(self, t: tx.Affine) -> Path:
        return Path(
            tuple(
                [loc_trail.apply_transform(t) for loc_trail in self.loc_trails]
            )
        )

    def points(self) -> Iterable[P2_t]:
        for loc_trails in self.loc_trails:
            for pt in loc_trails.points():
                yield pt

    def envelope(self, t: tx.V2_t) -> tx.Scalars:
        return EnvDistance.concat(
            (EnvDistance(loc.envelope(t)) for loc in self.loc_trails)
        ).d

    def get_trace(self, t: tx.Ray) -> TraceDistances:
        return TraceDistances.concat((loc.trace(t) for loc in self.loc_trails))

    def stroke(self) -> Diagram:
        """Returns a primitive (shape) with strokes

        Returns:
            Diagram: A diagram.
        """
        from chalk.core import Primitive

        return Primitive.from_path(self)

    # Constructors
    @staticmethod
    def from_points(points: List[P2_t], closed: bool = False) -> Path:
        if not points:
            return Path.empty()
        start = points[0]
        trail = Trail.from_offsets(
            [pt2 - pt1 for pt1, pt2 in zip(points, points[1:])], closed
        )
        return Path(tuple([trail.at(start)]))

    @staticmethod
    def from_point(point: P2_t) -> Path:
        return Path.from_points([point])

    @staticmethod
    def from_text(s: str) -> Path:
        return Path((), Text(tx.np.array(list(s), dtype='S1')))


    @staticmethod
    def from_pairs(
        segs: List[Tuple[P2_t, P2_t]], closed: bool = False
    ) -> Path:
        if not segs:
            return Path.empty()
        ls = [segs[0][0]]
        for seg in segs:
            assert seg[0] == ls[-1]
            ls.append(seg[1])
        return Path.from_points(ls, closed)

    @staticmethod
    def from_list_of_tuples(
        coords: List[Tuple[float, float]], closed: bool = False
    ) -> Path:
        points = list([tx.P2(x, y) for x, y in coords])
        return Path.from_points(points, closed)
