from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Optional, Tuple, TypeVar

import chalk.align
import chalk.arrow
import chalk.backend.cairo
import chalk.backend.matplotlib
import chalk.backend.tikz
import chalk.backend.svg
import chalk.broadcast
import chalk.combinators
import chalk.envelope
import chalk.layout
import chalk.model
import chalk.monoid
import chalk.subdiagram
import chalk.trace
import chalk.transform as tx
import chalk.types
from chalk.broadcast import broadcast_diagrams
from chalk.path import Path
from chalk.style import BatchStyle, StyleHolder
from chalk.transform import Affine, Batched
from chalk.types import BatchDiagram, BroadDiagram, Diagram
from chalk.visitor import DiagramVisitor

Trail = Any
A = TypeVar("A", bound=chalk.monoid.Monoid)

Svg_Height = 200
Svg_Draw_Height = None


def set_svg_height(height: int) -> None:
    """Globally set the svg preview height for notebooks."""
    global Svg_Height
    Svg_Height = height


def set_svg_draw_height(height: int) -> None:
    """Globally set the svg preview height for notebooks."""
    global Svg_Draw_Height
    Svg_Draw_Height = height


@dataclass(frozen=True)
class BaseDiagram(chalk.types.Diagram):
    """Diagram class."""

    # Monoid
    __add__ = chalk.combinators.atop

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.size()

    def __rmatmul__(self: BatchDiagram, t: Affine) -> BroadDiagram:  # type: ignore
        return self.apply_transform(t)  # type: ignore

    def _accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        raise NotImplementedError

    @classmethod
    def empty(cls) -> EmptyDia:  # type: ignore
        return Empty()

    # Tranformable
    def apply_transform(self: B1, t: Affine) -> B:  # type: ignore
        new_diagram = ApplyTransform(t, Empty())
        new, other = broadcast_diagrams(new_diagram, self)
        assert isinstance(new, ApplyTransform)
        return ApplyTransform(new.transform, other)

    def _compose_axis(self: BatchedDia) -> ReducedDia:  # type: ignore
        return ComposeAxis(self)

    # Stylable
    def apply_style(self: B1, style: StyleHolder) -> B:  # type: ignore
        new_diagram = ApplyStyle(style, Empty())
        new_diagram, self = broadcast_diagrams(new_diagram, self)
        return ApplyStyle(new_diagram.style, self)

    def __repr__(self) -> str:
        return f"Diagram[{self.shape}]"

    def __tree_pp__(self, **kwargs):  # type: ignore
        import jax._src.pretty_printer as pp

        return pp.text(f"Diagram[{self.shape}]")

    def _compose(
        self,
        envelope: Optional[BatchDiagram] = None,
        other: Optional[BatchDiagram] = None,
    ) -> BroadDiagram:
        if other is None and isinstance(self, Compose):
            return Compose(envelope, tuple(self.diagrams))
        if other is None and not isinstance(self, Compose):
            return Compose(envelope, (self,))

        # Broadcast
        self, other = broadcast_diagrams(self, other)  # type: ignore
        assert other is not None
        if isinstance(self, Empty):
            return other
        elif isinstance(self, Compose) and isinstance(other, Compose):
            if self.envelope is None and other.envelope is None:
                return Compose(envelope, self.diagrams + other.diagrams)
            else:
                return Compose(envelope, (self, other))
        elif isinstance(other, Empty) and not isinstance(self, Compose):
            return Compose(envelope, (self,))

        elif isinstance(other, Compose) and other.envelope is None:
            return Compose(envelope, (self,) + other.diagrams)
        else:
            return Compose(envelope, (self, other))

    # Layout
    _layout = chalk.layout.layout
    animate = chalk.backend.cairo.animate
    animate_svg = chalk.backend.svg.animate

    # Getters
    get_envelope = chalk.envelope.get_envelope
    get_trace = chalk.trace.get_trace

    # Names

    get_subdiagram = chalk.subdiagram.get_subdiagram
    get_sub_map = chalk.subdiagram.get_sub_map
    with_names = chalk.subdiagram.with_names
    qualify = chalk.subdiagram.qualify
    named = chalk.subdiagram.named

    # Broadcast
    add_axis = chalk.broadcast.add_axis
    reshape = chalk.broadcast.reshape
    swapaxes = chalk.broadcast.swapaxes
    repeat_axis = chalk.broadcast.repeat_axis
    broadcast_diagrams = chalk.broadcast.broadcast_diagrams
    size = chalk.broadcast.size

    # Combinators
    with_envelope = chalk.combinators.with_envelope
    juxtapose = chalk.combinators.juxtapose
    juxtapose_snug = chalk.combinators.juxtapose_snug
    beside_snug = chalk.combinators.beside_snug
    above = chalk.combinators.above
    atop = chalk.combinators.atop
    beside = chalk.combinators.beside
    above = chalk.combinators.above

    # Align
    align = chalk.align.align_to
    align_t = chalk.align.align_t
    align_b = chalk.align.align_b
    align_l = chalk.align.align_l
    align_r = chalk.align.align_r
    align_tr = chalk.align.align_tr
    align_tl = chalk.align.align_tl
    align_bl = chalk.align.align_bl
    align_br = chalk.align.align_br
    center_xy = chalk.align.center_xy
    center = chalk.align.center
    scale_uniform_to_y = chalk.align.scale_uniform_to_y
    scale_uniform_to_x = chalk.align.scale_uniform_to_x
    snug = chalk.align.snug

    # Arrows
    connect = chalk.arrow.connect
    connect_outside = chalk.arrow.connect_outside
    connect_perim = chalk.arrow.connect_perim

    # Model
    show_origin = chalk.model.show_origin
    show_envelope = chalk.model.show_envelope
    show_beside = chalk.model.show_beside
    show_labels = chalk.model.show_labels

    # Combinators
    pad = chalk.combinators.pad
    hcat = chalk.combinators.batch_hcat
    vcat = chalk.combinators.batch_vcat
    concat = chalk.combinators.batch_concat

    # Infix
    def __or__(self, d: Diagram) -> Diagram:
        return chalk.combinators.beside(self, d, tx.unit_x)

    __truediv__ = chalk.combinators.above

    # Rendering
    render = chalk.backend.cairo.render
    render_png = chalk.backend.cairo.render
    render_svg = chalk.backend.svg.render
    render_mpl = chalk.backend.matplotlib.render

    def render_pdf(self, *args, **kwargs) -> None:  # type: ignore
        print("Currently PDF rendering is disabled")

    # Flatten
    def _normalize(self) -> Diagram:
        if not isinstance(self, (Primitive, ApplyTransform)):
            return self.scale(1.0)
        return self

    def _repr_svg_(self) -> str:
        global Svg_Height, Svg_Draw_Height
        f = tempfile.NamedTemporaryFile(delete=False)
        self.render_svg(f.name, height=Svg_Height, draw_height=Svg_Draw_Height)
        f.close()
        svg = open(f.name).read()
        os.unlink(f.name)
        return svg

    def _repr_png_(self):
        global Svg_Height, Svg_Draw_Height
        f = tempfile.NamedTemporaryFile(delete=False)
        self.render(f.name, height=Svg_Height, draw_height=Svg_Draw_Height)
        f.close()
        png = open(f.name, "rb").read()
        os.unlink(f.name)
        return png

    def _repr_html_(self) -> str | tuple[str, Any]:
        """Returns a rich HTML representation of an object."""
        return self._repr_svg_()


@dataclass(frozen=True)
class Primitive(BaseDiagram):
    """Primitive class.

    This is derived from a ``chalk.core.Diagram`` class.
    """

    prim_shape: Path
    style: Optional[StyleHolder]
    transform: Affine
    order: Optional[tx.Ints] = None

    def set_order(self: BatchPrimitive, order: tx.Ints) -> BatchPrimitive:
        return Primitive(self.prim_shape, self.style, self.transform, order)

    @classmethod
    def from_path(cls, shape: Path) -> BatchPrimitive:
        # assert shape.size() == (), f"Shape size: {shape.size()}"
        return cls(shape, None, tx.make_ident(shape.size()))

    def apply_transform(self: BatchPrimitive, t: Affine) -> BatchPrimitive:
        chalk.broadcast.check(t.shape[:-2], self.shape, str(type(self)), "Transform")
        new_transform = t @ self.transform
        new_diagram = ApplyTransform(new_transform, Empty())
        new_diagram, self = broadcast_diagrams(new_diagram, self)
        return Primitive(self.prim_shape, self.style, new_diagram.transform)

    def apply_style(self: BatchPrimitive, other_style: BatchStyle) -> BatchPrimitive:
        new_diagram = ApplyStyle(other_style, Empty())
        new_diagram, self = broadcast_diagrams(new_diagram, self)
        return Primitive(
            self.prim_shape,
            (
                self.style.merge(new_diagram.style)
                if self.style is not None
                else new_diagram.style
            ),
            self.transform,
            self.order,
        )

    def _accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_primitive(self, args)


BatchPrimitive = Batched[Primitive, "#*B"]


@dataclass(unsafe_hash=True, frozen=True)
class Empty(BaseDiagram):
    def _accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_empty(self, args)

    def apply_transform(self, t: Affine) -> Empty:
        return Empty()

    def apply_style(self, style: StyleHolder) -> Empty:
        return Empty()


@dataclass(unsafe_hash=True, frozen=True)
class Compose(BaseDiagram):
    envelope: Optional[Diagram]
    diagrams: tuple[Diagram, ...]

    def _accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_compose(self, args)


@dataclass(frozen=True)
class ComposeAxis(BaseDiagram):
    diagrams: Diagram

    def _accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_compose_axis(self, args)


@dataclass(unsafe_hash=True, frozen=True)
class ApplyTransform(BaseDiagram):
    transform: Affine
    diagram: Diagram

    def _accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_apply_transform(self, args)

    def apply_transform(self, t: Affine) -> ApplyTransform:
        new_diagram = ApplyTransform(t @ self.transform, Empty())
        new, other = broadcast_diagrams(new_diagram, self.diagram)
        return ApplyTransform(new.transform, other)


@dataclass(frozen=True)
class ApplyStyle(BaseDiagram):
    style: StyleHolder
    diagram: Diagram

    def _accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_apply_style(self, args)

    def apply_style(self, style: BatchStyle) -> ApplyStyle:
        new_style = ApplyStyle(style, Empty())
        new_style, self = broadcast_diagrams(new_style, self)
        app_style = new_style.style.merge(self.style)
        return ApplyStyle(app_style, self.diagram)


@dataclass(frozen=True)
class ApplyName(BaseDiagram):
    """ApplyName class."""

    dname: chalk.subdiagram.Name
    diagram: Diagram

    def _accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_apply_name(self, args)
