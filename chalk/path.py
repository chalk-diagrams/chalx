from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from chalk import transform as tx
from chalk.segment import Segment
from chalk.trail import Located, Trail
from chalk.transform import P2_t, Transformable
from chalk.types import Diagram


@dataclass(frozen=True)
class Text:
    text: tx.Array

    def to_str(self) -> str:
        return tx.np.ndarray.tobytes(self.text).decode("utf-8")  # type: ignore


@dataclass(unsafe_hash=True)
class Path(Transformable):
    """Geometric path. Batching belongs on Diagrams / hijax Paths, not here."""

    loc_trails: Tuple[Located, ...]
    text: Optional[Text] = None
    scale_invariant: Optional[tx.Mask] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        """Prefix shape of stored geometry (usually empty)."""
        if not self.loc_trails:
            return ()
        from chalk.segment import segment_parts
        from chalk.trail import located_trail, trail_segment

        t, _ = segment_parts(trail_segment(located_trail(self.loc_trails[0])))
        return tuple(t.shape[:-3]) if t.ndim >= 3 else ()

    def remove_scale(self) -> Path:
        """Remove scale from the path."""
        return Path(self.loc_trails, self.text, tx.np.array(True))

    def located_segments(self) -> Segment:
        """Get located segments of the path."""
        from chalk.segment import concat_segments

        ls = Segment.empty()
        for loc_trail in self.loc_trails:
            nxt = loc_trail.located_segments()
            ls = concat_segments(ls, nxt)
        return ls

    # Monoid - compose
    @staticmethod
    def empty() -> Path:
        """Create an empty path."""
        return Path(())

    def __add__(self, other: Path) -> Path:
        """Add two paths."""
        return Path(self.loc_trails + other.loc_trails)

    def apply_transform(self, t: tx.Affine) -> Path:
        """Apply an affine transformation to the path."""
        return Path(
            tuple([loc_trail.apply_transform(t) for loc_trail in self.loc_trails])
        )

    def points(self) -> Iterable[P2_t]:
        """Iterate over points in the path."""
        for loc_trails in self.loc_trails:
            for pt in loc_trails.points():
                yield pt

    def stroke(self) -> Diagram:
        """Returns a primitive diagram from a path"""
        from chalk.core import Primitive

        return Primitive.from_path(self)

    # Constructors
    @staticmethod
    def from_array(points: P2_t, closed: bool = False) -> Path:
        """Create a path from an array of points."""
        l = points.shape[-3]
        if l == 0:
            return Path.empty()

        offsets = (
            points[..., tx.np.arange(1, l), :, :]
            - points[..., tx.np.arange(0, l - 1), :, :]
        )
        trail = Trail.from_array(offsets, closed)
        return Path(tuple([trail.at(points[..., 0, :, :])]))

    # Constructors

    @staticmethod
    def from_points(points: List[P2_t], closed: bool = False) -> Path:
        """Create a path from a list of points."""
        ls_points = tx.np.broadcast_arrays(*points)
        return Path.from_array(tx.np.stack(ls_points, axis=-3), closed)

    @staticmethod
    def from_point(point: P2_t) -> Path:
        """Create a path from a single point."""
        return Path.from_points([point])

    @staticmethod
    def from_text(s: str) -> Path:
        """Create a path from text."""
        return Path(
            (),
            Text(tx.np.frombuffer(bytes(s, "utf-8"), dtype=tx.np.uint8)),
        )

    @staticmethod
    def from_pairs(segs: List[Tuple[P2_t, P2_t]], closed: bool = False) -> Path:
        """Create a path from a list of point pairs."""
        if not segs:
            return Path.empty()
        ls = [segs[0][0]]
        for seg in segs:
            # assert seg[0] == ls[-1]
            ls.append(seg[1])
        return Path.from_points(ls, closed)

    @staticmethod
    def from_list_of_tuples(
        coords: List[Tuple[tx.Floating, tx.Floating]], closed: bool = False
    ) -> Path:
        """Create a `Path` from a list of coordinate tuples."""
        points = list([tx.P2(x, y) for x, y in coords])
        return Path.from_points(points, closed)


__all__ = ["Path"]
