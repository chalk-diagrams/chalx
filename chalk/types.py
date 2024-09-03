"""Define the core types for chalk.
Designed to be the top of the import
hierarchy.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)

import chalk.transform as tx
from chalk.monoid import Monoid
from chalk.style import Stylable, StyleHolder
from chalk.transform import P2_t, V2_t

if TYPE_CHECKING:
    from chalk.arrow import ArrowOpts
    from chalk.backend.patch import Patch
    from chalk.envelope import BatchEnvelope
    from chalk.path import Path
    from chalk.subdiagram import Name, Subdiagram
    from chalk.trace import BatchTrace
    from chalk.trail import Located, Trail
    from chalk.visitor import A, DiagramVisitor


class TrailLike(Protocol):
    def to_trail(self) -> Trail: ...

    def to_path(self, location: P2_t = tx.P2(0.0, 0.0)) -> Path:
        return self.at(location).to_path()

    def at(self, location: P2_t) -> Located:
        return self.to_trail().at(location)

    def stroke(self) -> Diagram:
        return self.at(tx.P2(0, 0)).stroke()


class Diagram(Stylable, tx.Transformable, Monoid, tx.Batchable):
    # Standard diagram types
    def apply_transform(self: BatchDiagram, t: tx.Affine) -> BroadDiagram:
        """Apply an affine transformation to a batched diagram.

        This method takes a batched `Diagram` and a batched `Affine` transformation
        and applies the transformation to the diagram. The result is a broadcasted
        `Diagram`.

        Args:
        ----
            self: A batched `Diagram` to be transformed.
            t: A batched `Affine` transformation to apply.

        Returns:
        -------
            A broadcasted `Diagram` with the transformation applied.

        """
        ...

    def __add__(self: BatchDiagram, other: BatchDiagram) -> BroadDiagram: ...

    def __or__(self: BatchDiagram, d: BatchDiagram) -> BroadDiagram: ...

    def __truediv__(self: BatchDiagram, other: BatchDiagram) -> BroadDiagram: ...

    def __floordiv__(self: BatchDiagram, other: BatchDiagram) -> BroadDiagram: ...

    def pad(self, extra: tx.Floating) -> Diagram:
        """Converts envelope to bounding box and grows by `extra` factor"""
        ...

    def scale_uniform_to_x(self, x: tx.Floating) -> Diagram:
        """Scale diagram uniformly to fit given width"""
        ...

    def scale_uniform_to_y(self, y: tx.Floating) -> Diagram:
        """Scale diagram uniformly to fit given height"""
        ...

    def align(self: Diagram, v: V2_t) -> Diagram:
        """Align diagram in direction `v`."""
        ...

    def align_t(self: Diagram) -> Diagram:
        """Align diagram to top."""
        ...

    def align_b(self: Diagram) -> Diagram:
        """Align diagram to bottom."""
        ...

    def align_l(self: Diagram) -> Diagram:
        """Align diagram to left."""
        ...

    def align_r(self: Diagram) -> Diagram:
        """Align diagram to right."""
        ...

    def align_tl(self: Diagram) -> Diagram:
        """Align diagram to top-left corner."""
        ...

    def align_tr(self: Diagram) -> Diagram:
        """Align diagram to top-right corner."""
        ...

    def align_bl(self: Diagram) -> Diagram:
        """Align diagram to bottom-left corner."""
        ...

    def align_br(self: Diagram) -> Diagram:
        """Align diagram to bottom-right corner."""
        ...

    def snug(self: Diagram, v: V2_t) -> Diagram:
        """Version of `align` based on `Trace`"""
        ...

    def center_xy(self: Diagram) -> Diagram:
        """Center the diagram in both x and y directions."""
        ...

    def get_subdiagram(self, name: Any) -> Optional[Subdiagram]:
        """Retrieve a subdiagram by name if it exists."""
        ...

    def get_sub_map(self, t: tx.Affine) -> Dict[Name, List[Subdiagram]]:
        """Get a mapping of names to lists of subdiagrams with an applied transformation."""
        ...

    def with_names(
        self,
        names: List[Any],
        f: Callable[[List[Subdiagram], Diagram], Diagram],
    ) -> Diagram:
        """Apply a function to named `Subdiagram`s of this diagram.

        Args:
        ----
            names: List of names to match against subdiagrams.
            f: Function to apply to matched subdiagrams and the current diagram.
               Takes a list of matched `Subdiagram`s and the current `Diagram` as input.
               Should return a new `Diagram`.

        Returns:
        -------
            A new `Diagram` with the function applied to matched subdiagrams.

        """
        ...

    def get_envelope(self: BatchDiagram) -> BatchEnvelope:
        """Get the envelope of the diagram."""
        ...

    def get_trace(self: BatchDiagram) -> BatchTrace:
        """Get the trace of the diagram."""
        ...

    def with_envelope(self, other: Diagram) -> Diagram:
        """Create a new diagram with the envelope of another."""
        ...

    def show_origin(self) -> Diagram:
        """Show the origin of the diagram."""
        ...

    def show_labels(self: Diagram, font_size: tx.Floating = 1) -> Diagram:
        """Show labels on the diagram."""
        ...

    def show_envelope(self, phantom: bool = False, angle: int = 45) -> Diagram:
        """Show the envelope of the diagram."""
        ...

    def show_beside(self: Diagram, other: Diagram, direction: V2_t) -> Diagram:
        """Show this diagram beside another in the given direction."""
        ...

    def size(self) -> Tuple[int, ...]:
        """Get the shape of the diagram."""
        ...

    def named(self: Diagram, name: Any) -> Diagram:
        """Name this diagram."""
        ...

    def qualify(self: Diagram, name: Any) -> Diagram:
        """Qualify the name of this diagram."""
        ...

    def hcat(self: ExtraDiagram, sep: Optional[tx.Floating] = None) -> BatchDiagram:
        """Concatenate diagrams horizontally.

        This method operates on a batched `Diagram` with shape "B ...".
        The return is a `Diagram` with shape "...".
        It works on a single batched input.

        Args:
        ----
            sep: Optional separation distance between diagrams.

        Returns:
        -------
            A new `Diagram` with the input diagrams concatenated horizontally.

        """
        ...

    def vcat(self: ExtraDiagram, sep: Optional[tx.Floating] = None) -> BatchDiagram:
        """Concatenate diagrams vertically.

        This method operates on a batched `Diagram` with shape "B ...".
        The return is a `Diagram` with shape "...".
        It works on a single batched input.

        Args:
        ----
            sep: Optional separation distance between diagrams.

        Returns:
        -------
            A new `Diagram` with the input diagrams concatenated vertically.

        """
        ...

    def concat(  # type: ignore[empty-body, override]
        self: ExtraDiagram,
    ) -> BatchDiagram:
        """Concatenate diagrams.

        This method operates on a batched `Diagram` with shape "B ...".
        The return is a `Diagram` with shape "...".
        It works on a single batched input.

        Returns
        -------
            A new `Diagram` with the input diagrams concatenated.

        """
        ...

    def juxtapose_snug(
        self: BatchDiagram, other: BatchDiagram, direction: V2_t
    ) -> BroadDiagram: ...

    def beside_snug(
        self: BatchDiagram, other: BatchDiagram, direction: V2_t
    ) -> BroadDiagram: ...

    def juxtapose(
        self: BatchDiagram, other: BatchDiagram, direction: V2_t
    ) -> BroadDiagram:
        """Place `other` diagram to touch `self` along the given direction.

        Args:
        ----
            self: A batched `Diagram` object.
            other: Another batched `Diagram` object to juxtapose.
            direction: Normalized vector indicating the direction of juxtaposition.

        Returns:
        -------
            A new broadcasted `Diagram` with `other` juxtaposed to `self`.

        """
        ...

    def atop(self: BatchDiagram, other: BatchDiagram) -> BroadDiagram:
        """Place this diagram atop another."""
        ...

    def above(self: BatchDiagram, other: BatchDiagram) -> BatchDiagram:
        """Place this diagram above another."""
        ...

    def beside(
        self: BatchDiagram, other: BatchDiagram, direction: V2_t
    ) -> BroadDiagram:
        """Place this diagram beside another in the given direction."""
        ...

    def connect(
        self: BatchDiagram,
        name1: Any,
        name2: Any,
        style: Optional[ArrowOpts] = None,
    ) -> BatchDiagram:
        """Connect two named subdiagrams with an arrow.

        Args:
        ----
            name1: Name of the first subdiagram.
            name2: Name of the second subdiagram.
            style: Optional arrow style options.

        Returns:
        -------
            A new `Diagram` with the connection added.
        """
        ...

    def connect_outside(
        self: BatchDiagram,
        name1: Any,
        name2: Any,
        style: Optional[ArrowOpts] = None,
    ) -> BatchDiagram:
        """Connect two named subdiagrams with an arrow outside their envelopes.

        Args:
        ----
            name1: Name of the first subdiagram.
            name2: Name of the second subdiagram.
            style: Optional arrow style options.

        Returns:
        -------
            A new `Diagram` with the outside connection added.
        """
        ...

    def connect_perim(
        self: BatchDiagram,
        name1: Any,
        name2: Any,
        v1: V2_t,
        v2: V2_t,
        style: Optional[ArrowOpts] = None,
    ) -> BatchDiagram:
        """Connect two named subdiagrams with an arrow between specified perimeter points.

        Args:
        ----
            name1: Name of the first subdiagram.
            name2: Name of the second subdiagram.
            v1: Vector specifying the connection point on the first subdiagram's perimeter.
            v2: Vector specifying the connection point on the second subdiagram's perimeter.
            style: Optional arrow style options.

        Returns:
        -------
            A new `Diagram` with the perimeter connection added.
        """
        ...

    def add_axis(self, size: int) -> Diagram:
        """Add a new axis to the diagram."""
        ...

    def repeat_axis(self, size: int, axis: int) -> Diagram:
        """Repeat the diagram along the specified axis."""
        ...

    def swapaxes(self, a: int, b: int) -> Diagram:
        """Swap the specified prefix axes of the diagram."""
        ...

    def broadcast_diagrams(
        self: BatchDiagram, other: BatchDiagram
    ) -> Tuple[BroadDiagram, BroadDiagram]:
        """Broadcast two diagrams to compatible shapes."""
        ...

    def reshape(self, shape: Tuple[int, ...]) -> Diagram:
        """Reshape the prefix of the diagram to the specified shape."""
        ...

    def animate(
        self: OneDDiagram,
        path: str,
        height: int = 128,
        width: Optional[int] = None,
        draw_height: Optional[int] = None,
    ) -> None:
        """Animate the diagram to a GIF and save to the specified path."""
        ...

    def animate_svg(
        self: OneDDiagram,
        path: str,
        height: int = 128,
        width: Optional[int] = None,
        draw_height: Optional[int] = None,
    ) -> None:
        """Animate the diagram as SVG and save to the specified path."""
        ...

    def render(
        self: SingleDiagram,
        path: str,
        height: int = 128,
        width: Optional[int] = None,
        draw_height: Optional[int] = None,
    ) -> None:
        """Render the diagram and save to the specified path."""
        ...

    def render_svg(
        self: SingleDiagram,
        path: str,
        height: int = 128,
        width: Optional[int] = None,
        draw_height: Optional[int] = None,
    ) -> None:
        """Render the diagram as SVG and save to the specified path."""
        ...

    def render_mpl(
        self: SingleDiagram,
        path: str,
        height: int = 128,
        width: Optional[int] = None,
        draw_height: Optional[int] = None,
    ) -> None:
        """Render the diagram using Matplotlib and save to the specified path."""
        ...

    # Private methods
    def _layout(
        self,
        height: tx.IntLike,
        width: Optional[tx.IntLike] = None,
        draw_height: Optional[tx.IntLike] = None,
    ) -> Tuple[List[Patch], tx.IntLike, tx.IntLike]:
        raise NotImplementedError()

    def _compose_axis(self) -> Diagram:
        raise NotImplementedError()

    def _style(self, style: StyleHolder) -> Diagram:
        raise NotImplementedError()

    def _normalize(self) -> Diagram:
        raise NotImplementedError()

    def _compose(
        self, envelope: Optional[Diagram], other: Optional[Diagram] = None
    ) -> Diagram:
        raise NotImplementedError()

    def _accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        raise NotImplementedError()


# Diagram with shape
BatchDiagram = tx.Batched[Diagram, "*#B"]

# Broadcasted diagram
BroadDiagram = tx.Batched[Diagram, "*B"]

# Diagram before composition
ExtraDiagram = tx.Batched[Diagram, "*#B A"]
EmptyDiagram = tx.Batched[Diagram, ""]
SingleDiagram = tx.Batched[Diagram, ""]
OneDDiagram = tx.Batched[Diagram, "T"]
