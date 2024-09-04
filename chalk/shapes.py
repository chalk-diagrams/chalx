from typing import List, Optional, Tuple, Union

# Todo: fix this export
from chalk.trail import Trail  # noqa: N812
import chalk.transform as tx
from chalk.path import from_list_of_tuples, Path, from_text  # noqa: F401
from chalk.trail import arc_seg, arc_seg_angle  # noqa: F401
from chalk.transform import P2, P2_t
from chalk.types import BatchDiagram, Diagram

# Functions mirroring Diagrams.2d.Shapes


def text(s: str, size: tx.Floating) -> Diagram:
    """Create a text diagram with given string and size."""
    return from_text(s).stroke().scale(size).scale_y(-1)


def hrule(length: tx.Floating) -> BatchDiagram:
    """Create a horizontal rule of specified length."""
    return Trail.hrule(length).stroke().center_xy()


def vrule(length: tx.Floating) -> Diagram:
    """Create a vertical rule of specified length."""
    return Trail.vrule(length).stroke().center_xy()


# def polygon(sides: int, radius: float, rotation: float = 0) -> Diagram:
#     """
#     Draw a polygon.

#     Args:
#        sides (int): Number of sides.
#        radius (float): Internal radius.
#        rotation: (int): Rotation in degrees

#     Returns:
#        Diagram
#     """
#     return Trail.polygon(sides, radius, to_radians(rotation)).stroke()


def regular_polygon(sides: int, side_length: tx.Floating) -> Diagram:
    """Draws a regular polygon with given number of sides and given side
    length. The polygon is oriented with one edge parallel to the x-axis.
    """
    return Trail.regular_polygon(sides, side_length).centered().stroke()


def triangle(width: tx.Floating) -> Diagram:
    """Draws an equilateral triangle with the side length specified by
    the ``width`` argument. The origin is the traingle's centroid.
    """
    return regular_polygon(3, width)


def line(
    from_: Tuple[tx.Floating, tx.Floating], to: Tuple[tx.Floating, tx.Floating]
) -> Diagram:
    """Create a line from one point to another."""
    return make_path([from_, to])


def make_path(
    segments: Union[List[Tuple[tx.Floating, tx.Floating]], tx.P2_t],
    closed: bool = False,
) -> Diagram:
    """Construct a path from a list of segments or points."""
    if isinstance(segments, (list, tuple)):
        p = from_list_of_tuples(segments, closed).stroke()
    else:
        p = Path.from_array(segments, closed).stroke()
    return p


def rectangle(
    width: tx.Floating, height: tx.Floating, radius: Optional[float] = None
) -> BatchDiagram:
    """Draws a rectangle.

    Args:
    ----
        width (float): Width
        height (float): Height
        radius (Optional[float]): Radius for rounded corners.

    Returns:
    -------
        Diagrams

    """
    if radius is None:
        return (
            Trail.square()
            .stroke()
            .scale_x(width)
            .scale_y(height)
            .translate(-width / 2, -height / 2)
        )
    else:
        return (
            Trail.rounded_rectangle(width, height, radius)
            .stroke()
            .translate(-width / 2, -height / 2)
        )


def square(side: tx.Floating) -> BatchDiagram:
    """Draws a square with the specified side length. The origin is the
    center of the square.
    """
    return rectangle(side, side)


def circle(radius: tx.Floating) -> BatchDiagram:
    """Draws a circle with the specified ``radius``."""
    return Trail.circle().stroke().translate(radius, 0).scale(radius).center_xy()


def arc(radius: tx.Floating, angle0: tx.Floating, angle1: tx.Floating) -> Diagram:
    """Draws an arc.

    Args:
    ----
      radius (float): Circle radius.
      angle0 (float): Starting cutoff in degrees.
      angle1 (float): Finishing cutoff in degrees.

    Returns:
    -------
      Diagram

    """
    return (
        arc_seg_angle(tx.ftos(angle0), tx.ftos(angle1 - angle0))
        .at(tx.polar(angle0))
        .stroke()
        .scale(radius)
        .fill_opacity(0)
    )


def arc_between(
    point1: Union[P2_t, Tuple[float, float]],
    point2: Union[P2_t, Tuple[float, float]],
    height: float,
) -> Diagram:
    """Makes an arc starting at point1 and ending at point2, with the midpoint
    at a distance of abs(height) away from the straight line from point1 to
    point2. A positive value of height results in an arc to the left of the
    line from point1 to point2; a negative value yields one to the right.
    The implementation is based on the the function arcBetween from Haskell's
    diagrams:
    https://hackage.haskell.org/package/diagrams-lib-1.4.5.1/docs/src/Diagrams.TwoD.Arc.html#arcBetween
    """
    p = point1 if not isinstance(point1, tuple) else P2(*point1)
    q = point2 if not isinstance(point2, tuple) else P2(*point2)
    return arc_seg(q - p, height).at(p).stroke().fill_opacity(0)


def Spacer(width: tx.Floating, height: tx.Floating) -> Diagram:
    return (
        rectangle(tx.np.maximum(width, 1e-5), tx.np.maximum(height, 1e-5))
        .fill_opacity(0)
        .line_width(0)
    )


def hstrut(width: tx.Floating) -> Diagram:
    """Create a horizontal strut with given width."""
    return Spacer(tx.ftos(width), tx.ftos(0))


def strut(width: tx.Floating, height: tx.Floating) -> Diagram:
    """Create a strut with given width and height."""
    return Spacer(tx.ftos(width), tx.ftos(height))


def vstrut(height: tx.Floating) -> Diagram:
    """Create a vertical strut with given height."""
    return Spacer(tx.ftos(0), tx.ftos(height))


__all__ = [
    "text",
    "hrule",
    "vrule",
    "regular_polygon",
    "triangle",
    "line",
    "make_path",
    "rectangle",
    "square",
    "circle",
    "arc",
    "arc_between",
    "hstrut",
    "strut",
    "vstrut",
]
