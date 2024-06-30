from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Sequence, Tuple, TypeVar

import chalk.align
import chalk.arrow
import chalk.backend.cairo
import chalk.backend.matplotlib
import chalk.backend.svg
import chalk.backend.tikz
import chalk.combinators
import chalk.model
import chalk.subdiagram
import chalk.trace
import chalk.transform as tx
import chalk.types
from chalk.envelope import Envelope
from chalk.monoid import MList, Monoid
from chalk.shapes.path import Path
from chalk.style import StyleHolder
from chalk.subdiagram import Name
from chalk.transform import Affine
from chalk.types import Diagram, Shape
from chalk.utils import imgen
from chalk.visitor import DiagramVisitor

Trail = Any
A = TypeVar("A", bound=chalk.monoid.Monoid)

SVG_HEIGHT = 200
SVG_DRAW_HEIGHT = None


def set_svg_height(height: int) -> None:
    "Globally set the svg preview height for notebooks."
    global SVG_HEIGHT
    SVG_HEIGHT = height


def set_svg_draw_height(height: int) -> None:
    "Globally set the svg preview height for notebooks."
    global SVG_DRAW_HEIGHT
    SVG_DRAW_HEIGHT = height


@dataclass(unsafe_hash=True, frozen=True)
class BaseDiagram(chalk.types.Diagram):
    """Diagram class."""

    # Monoid
    __add__ = chalk.combinators.atop

    @property
    def dtype(self) -> str:
        return "batched_diagram"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.size()

    def __getitem__(self, ind: Union[int, Tuple[int]]) -> Diagram:
        import jax

        return jax.tree.map(lambda x: x[ind], self)

    def __rmatmul__(self, t: Affine) -> BaseDiagram:  # type: ignore
        return self.apply_transform(t)  # type: ignore

    @classmethod
    def empty(cls) -> Diagram:  # type: ignore
        return Empty()

    # Tranformable
    def apply_transform(self, t: Affine) -> Diagram:  # type: ignore
        new_diagram = ApplyTransform(t, Empty())
        new, other = new_diagram.broadcast_diagrams(self)  # type: ignore
        return ApplyTransform(new.transform, other)

    def compose_axis(self) -> Diagram:  # type: ignore
        return ComposeAxis(self)

    # Stylable
    def apply_style(self, style: StyleHolder) -> Diagram:  # type: ignore
        new_diagram = ApplyStyle(style, None)
        new_diagram, self = new_diagram.broadcast_diagrams(self)  # type: ignore
        return ApplyStyle(new_diagram.style, self)

    def _style(self, style: StyleHolder) -> Diagram:
        return self.apply_style(style)

    def add_axis(self, size: int) -> Diagram:
        import jax

        return jax.tree.map(
            lambda x: tx.np.repeat(x[None], size, axis=0), self
        )

    def repeat_axis(self, size: int, axis) -> Diagram:
        import jax

        return jax.tree.map(lambda x: tx.np.repeat(x, size, axis=axis), self)

    def broadcast_diagrams(self, other: Diagram) -> Tuple[Diagram, Diagram]:
        size = self.size()
        other_size = other.size()
        if size == other_size:
            return self, other
        ml = max(len(size), len(other_size))
        for i in range(ml):
            off = -1 - i
            if i > len(other_size) - 1:
                other = other.add_axis(size[off])
            elif i > len(size) - 1:
                self = self.add_axis(other_size[off])
            elif size[off] == 1 and other_size[off] != 1:
                self = self.repeat_axis(other_size[off], len(size) + off)
            elif size[off] != 1 and other_size[off] == 1:
                other = other.repeat_axis(size[off], len(other_size) + off)
        assert (
            self.size() == other.size()
        ), f"{size} {other_size} {self.size()} {other.size()}"
        return self, other

    def compose(
        self,
        envelope: Optional[Envelope] = None,
        other: Optional[Diagram] = None,
    ) -> Diagram:
        if other is None and isinstance(self, Compose):
            return Compose(envelope, tuple(self.diagrams))
        if other is None and isinstance(self, Compose):
            return Compose(envelope, (self,))

        other = other if other is not None else Empty()
        # Broadcast
        self, other = self.broadcast_diagrams(other)  # type: ignore

        if isinstance(self, Empty):
            return other
        elif isinstance(self, Compose) and isinstance(other, Compose):
            return Compose(envelope, self.diagrams + other.diagrams)
        elif isinstance(other, Empty) and not isinstance(self, Compose):
            return Compose(envelope, (self,))

        elif isinstance(other, Compose):
            return Compose(envelope, (self,) + other.diagrams)
        else:
            return Compose(envelope, (self, other))

    def named(self, name: Name) -> Diagram:
        """Add a name (or a sequence of names) to a diagram."""
        return ApplyName(name, self)

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

    # Flatten
    def _normalize(self) -> Diagram:
        if not isinstance(self, (Primitive, ApplyTransform)):
            return self.scale(1.0)
        return self

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
    __floordiv__ = chalk.combinators.above2

    def display(
        self, height: int = 256, verbose: bool = True, **kwargs: Any
    ) -> None:
        """Display the diagram using the default renderer.

        Note: see ``chalk.utils.imgen`` for details on the keyword arguments.
        """
        # update kwargs with defaults and user-specified values
        kwargs.update({"height": height})
        kwargs.update({"verbose": verbose})
        kwargs.update({"dirpath": None})
        kwargs.update({"wait": kwargs.get("wait", 1)})
        # render and display the diagram
        imgen(self, **kwargs)

    # Rendering
    render = chalk.backend.cairo.render
    render_png = chalk.backend.cairo.render
    render_svg = chalk.backend.svg.render
    render_mpl = chalk.backend.matplotlib.render
    plot = chalk.backend.matplotlib.plot

    def render_pdf(self, *args, **kwargs) -> None:  # type: ignore
        print("Currently PDF rendering is disabled")

    def _repr_svg_(self) -> str:
        global SVG_HEIGHT
        f = tempfile.NamedTemporaryFile(delete=False)
        self.render_svg(f.name, height=SVG_HEIGHT, draw_height=SVG_DRAW_HEIGHT)
        f.close()
        svg = open(f.name).read()
        os.unlink(f.name)
        return svg

    def _repr_html_(self) -> str | tuple[str, Any]:
        """Returns a rich HTML representation of an object."""
        return self._repr_svg_()

    # Getters
    get_envelope = chalk.envelope.get_envelope
    get_trace = chalk.trace.get_trace
    get_subdiagram = chalk.subdiagram.get_subdiagram
    get_sub_map = chalk.subdiagram.get_sub_map

    with_names = chalk.subdiagram.with_names

    def qualify(self, name: Name) -> Diagram:
        """Prefix names in the diagram by a given name or sequence of names."""
        return self.accept(Qualify(name), None)

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        raise NotImplementedError

    def get_primitives(self) -> List[Primitive]:
        return self.accept(ToListOrder(), tx.ident).ls

    def size(self) -> Tuple[int, ...]:
        return self.accept(ToSize(), Size.empty()).d

    def layout(
        self, height: tx.IntLike = 128, width: Optional[tx.IntLike] = None
    ) -> Tuple[List[Primitive], tx.IntLike, tx.IntLike]:
        envelope = self.get_envelope()
        assert envelope is not None

        pad = 0.05

        # infer width to preserve aspect ratio
        if width is None:
            width = tx.np.round(
                height * envelope.width / envelope.height
            ).astype(int)
        else:
            width = width
        assert width is not None
        # determine scale to fit the largest axis in the target frame size
        α = tx.np.where(
            envelope.width - width <= envelope.height - height,
            height / ((1 + pad) * envelope.height),
            width / ((1 + pad) * envelope.width),
        )

        print("inner")
        s = self.scale(α).center_xy().pad(1 + pad)
        e = s.get_envelope()
        assert e is not None
        s = s.translate(e(-tx.unit_x), e(-tx.unit_y))

        style = StyleHolder.root(tx.np.maximum(width, height))
        s = s._style(style)
        print("outer")
        return s.get_primitives(), height, width


@dataclass(unsafe_hash=True, frozen=True)
class Primitive(BaseDiagram):
    """Primitive class.

    This is derived from a ``chalk.core.Diagram`` class.

    [TODO]: explain what Primitive class is for.
    """

    prim_shape: Shape
    style: Optional[StyleHolder]
    transform: Affine
    order: Optional[tx.IntLike] = None

    def is_multi(self) -> bool:
        return self.size() != ()

    def set_order(self, order: tx.IntLike) -> Primitive:
        return Primitive(self.prim_shape, self.style, self.transform, order)

    def split(self, ind: int) -> Primitive:
        return Primitive(
            self.prim_shape.split(ind),
            self.style.split(ind) if self.style is not None else None,
            self.transform[ind],
        )

    @classmethod
    def from_shape(cls, shape: Shape) -> Primitive:
        """Creates a primitive from a shape using the default style (only line
        stroke, no fill) and the identity transformation.

        Args:
            shape (Shape): A shape object.

        Returns:
            Primitive: A diagram object.
        """
        return cls(shape, None, tx.ident)

    def apply_transform(self, t: Affine) -> Primitive:
        if hasattr(self.transform, "shape"):
            new_transform = t @ self.transform
        else:
            new_transform = t
        new_diagram = ApplyTransform(new_transform, Empty())
        new_diagram, self = new_diagram.broadcast_diagrams(self)  # type: ignore
        return Primitive(self.prim_shape, self.style, new_diagram.transform)

    def apply_style(self, other_style: StyleHolder) -> Primitive:
        """Applies a style and returns a primitive.

        Args:
            other_style (Style): A style object.

        Returns:
            Primitive
        """
        if other_style is None:
            return Primitive(self.prim_shape, None, self.transform, self.order)
        new_diagram = ApplyStyle(other_style, None)
        new_diagram, self = new_diagram.broadcast_diagrams(self)  # type: ignore

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

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_primitive(self, args)


@dataclass(unsafe_hash=True, frozen=True)
class Empty(BaseDiagram):
    """An Empty diagram class."""

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_empty(self, args)

    def apply_transform(self, t: Affine) -> Empty:
        return Empty()

    def apply_style(self, style: StyleHolder) -> Empty:
        return Empty()


@dataclass(unsafe_hash=True, frozen=True)
class Compose(BaseDiagram):
    """Compose class."""

    envelope: Optional[Envelope]
    diagrams: tuple[Diagram, ...]

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_compose(self, args)


@dataclass(unsafe_hash=True, frozen=True)
class ComposeAxis(BaseDiagram):
    """ComposeAxis class."""

    diagrams: Diagram

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_compose_axis(self, args)


@dataclass(unsafe_hash=True, frozen=True)
class ApplyTransform(BaseDiagram):
    """ApplyTransform class."""

    transform: Affine
    diagram: Diagram

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_apply_transform(self, args)

    def apply_transform(self, t: Affine) -> ApplyTransform:
        new_diagram = ApplyTransform(t @ self.transform, Empty())
        new, other = new_diagram.broadcast_diagrams(self.diagram)  # type: ignore
        return ApplyTransform(new.transform, other)


@dataclass(unsafe_hash=True, frozen=True)
class ApplyStyle(BaseDiagram):
    """ApplyStyle class."""

    style: StyleHolder
    diagram: Diagram

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_apply_style(self, args)

    def apply_style(self, style: Optional[StyleHolder]) -> ApplyStyle:
        if style is None:
            return ApplyStyle(None, self.diagram)

        new_style = ApplyStyle(style, None)
        new_style, self = new_style.broadcast_diagrams(self)  # type: ignore

        app_style = new_style.style.merge(self.style)
        return ApplyStyle(app_style, self.diagram)


@dataclass(unsafe_hash=True, frozen=True)
class ApplyName(BaseDiagram):
    """ApplyName class."""

    dname: Name
    diagram: Diagram

    def accept(self, visitor: DiagramVisitor[A, Any], args: Any) -> A:
        return visitor.visit_apply_name(self, args)


@dataclass(unsafe_hash=True, frozen=True)
class Qualify(DiagramVisitor[Diagram, None]):
    A_type = Diagram

    def __init__(self, name: Name):
        self.name = name

    def visit_primitive(self, diagram: Primitive, args: None) -> Diagram:
        return diagram

    def visit_compose(self, diagram: Compose, args: None) -> Diagram:
        return Compose(
            diagram.envelope,
            tuple([d.accept(self, None) for d in diagram.diagrams]),
        )

    def visit_apply_transform(
        self, diagram: ApplyTransform, args: None
    ) -> Diagram:
        return ApplyTransform(
            diagram.transform,
            diagram.diagram.accept(self, None),
        )

    def visit_apply_style(self, diagram: ApplyStyle, args: None) -> Diagram:
        return ApplyStyle(
            diagram.style,
            diagram.diagram.accept(self, None),
        )

    def visit_apply_name(self, diagram: ApplyName, args: None) -> Diagram:
        return ApplyName(
            self.name + diagram.dname, diagram.diagram.accept(self, None)
        )


@dataclass
class Size(Monoid):
    d: Tuple[int, ...]

    @staticmethod
    def empty() -> Size:
        return Size(())

    def __add__(self, other: Size) -> Size:
        return Size(tx.np.broadcast_shapes(self.d, other.d))

    def remove_axis(self, axis: int) -> Size:
        return Size(self.d[:-1])


class ToSize(DiagramVisitor[Size, Size]):
    A_type = Size

    def visit_primitive(self, diagram: Primitive, t: Size) -> Size:
        return Size(diagram.transform.shape[:-2])

    def visit_apply_transform(self, diagram: ApplyTransform, t: Size) -> Size:
        return Size(diagram.transform.shape[:-2])

    def visit_apply_style(self, diagram: ApplyStyle, t: Size) -> Size:
        if diagram.style is None:
            return diagram.diagram.accept(self, t)
        return Size(diagram.style.size())

    def visit_compose_axis(self, diagram: ComposeAxis, t: Size) -> Size:
        return diagram.diagrams.accept(self, t).remove_axis(0)


@dataclass
class OrderList(Monoid):
    ls: List[Primitive]
    counter: tx.IntLike

    @staticmethod
    def empty() -> OrderList:
        return OrderList([], tx.np.asarray(0))

    def __add__(self, other: OrderList) -> OrderList:
        sc = self.counter
        # sc = add_dim(sc, len(other.counter.shape) - len(self.counter.shape))
        return OrderList(
            self.ls
            + [
                prim.set_order(
                    prim.order
                    + add_dim(sc, len(prim.order.shape) - len(sc.shape))
                )  # type:ignore
                for prim in other.ls
            ],
            (sc + other.counter),
        )


class ToListOrder(DiagramVisitor[OrderList, Affine]):
    """Compiles a `Diagram` to a list of `Primitive`s. The transformation `t`
    is accumulated upwards, from the tree's leaves.
    """

    A_type = OrderList

    def visit_primitive(self, diagram: Primitive, t: Affine) -> OrderList:
        size = diagram.size()
        return OrderList(
            [diagram.apply_transform(t).set_order(tx.np.zeros(size))],
            tx.np.ones(size),
        )

    def visit_apply_transform(
        self, diagram: ApplyTransform, t: Affine
    ) -> OrderList:
        t_new = t @ diagram.transform
        return diagram.diagram.accept(self, t_new)

    def visit_apply_style(self, diagram: ApplyStyle, t: Affine) -> OrderList:
        a = diagram.diagram.accept(self, t)
        return OrderList(
            [
                prim.apply_style(
                    add_dim(
                        diagram.style, len(prim.size()) - len(diagram.size())
                    )
                )
                for prim in a.ls
            ],
            a.counter,
        )

    def visit_compose_axis(self, diagram: ComposeAxis, t: Affine) -> OrderList:
        s = diagram.diagrams.size()
        stride = s[-1]
        internal = diagram.diagrams.accept(self, t[..., None, :, :])
        update = tx.np.arange(stride)

        last_counter = tx.np.where(
            tx.np.arange(stride) == 0,
            0,
            tx.np.roll(
                tx.np.cumsum(internal.counter, axis=-1), 1, axis=-1
            ),
        )

        # ls = [prim.set_order(tx.np.cumsum(prim.order, len(s)- 1))
        #       for prim in internal.ls]
        ls = [
            prim.set_order(
                prim.order  # type: ignore
                + add_dim(last_counter, len(prim.size()) - len(s))
            )
            for prim in internal.ls
        ]

        counter = tx.np.sum(internal.counter, axis=-1)
        assert counter.shape == diagram.size()
        return OrderList(ls, counter)


def add_dim(m: tx.Array, size: int) -> tx.Array:
    for s in range(size):
        m = m[..., None]
    return m
