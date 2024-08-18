from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, TypeVar

import chalk.transform as tx
from chalk.monoid import Monoid
from chalk.types import Diagram
from chalk.visitor import DiagramVisitor

if TYPE_CHECKING:
    import chalk.core as core


def add_axis(self: Diagram, size: int) -> Diagram:
    return tx.tree_map(  # type: ignore
        lambda x: tx.np.repeat(x[None], size, axis=0), self
    )


def size(self: Diagram) -> Tuple[int, ...]:
    "Get the size of a batch diagram."
    return self.accept(ToSize(), Size.empty()).d


def reshape(self: Diagram, shape: Tuple[int, ...]) -> Diagram:
    old_shape = len(self.size())
    return tx.tree_map(  # type: ignore
        lambda x: x.reshape(shape + x.shape[old_shape:]), self
    )


def repeat_axis(self: Diagram, size: int, axis: int) -> Diagram:
    return tx.tree_map(lambda x: tx.np.repeat(x, size, axis=axis), self)  # type: ignore


def check(a: Tuple[int, ...], b: Tuple[int, ...], s1: str, s2: str) -> None:
    try:
        tx.np.broadcast_shapes(a, b)
    except ValueError:
        assert False, f"Broadcast error: {s1} Shape: {a} {s2} Shape: {b}"


def check_consistent(self: Diagram) -> None:
    shape = self.shape

    def check(x: tx.Array) -> None:
        assert x.shape[: len(shape)] == shape

    tx.tree_map(check, self)


V1 = TypeVar("V1", bound=Diagram)
V2 = TypeVar("V2", bound=Diagram)


def broadcast_diagrams(self: V1, other: V2) -> Tuple[V1, V2]:
    """
    Returns a version of diagram A and B
    that have the same shape.
    """
    size = self.size()
    other_size = other.size()
    if size == other_size:
        return self, other
    check(size, other_size, str(type(self)), str(type(other)))
    ml = max(len(size), len(other_size))
    for i in range(ml):
        off = -1 - i
        if i > len(other_size) - 1:
            other = other.add_axis(size[off])  # type: ignore
        elif i > len(size) - 1:
            self = self.add_axis(other_size[off])  # type: ignore
        elif size[off] == 1 and other_size[off] != 1:
            self = self.repeat_axis(other_size[off], len(size) + off)  # type: ignore
        elif size[off] != 1 and other_size[off] == 1:
            other = other.repeat_axis(size[off], len(other_size) + off)  # type: ignore
    check_consistent(self)
    check_consistent(other)
    assert (
        self.size() == other.size()
    ), f"{size} {other_size} {self.size()} {other.size()}"
    return self, other


@dataclass
class Size(Monoid):
    d: Tuple[int, ...]

    @classmethod
    def empty(cls) -> Size:
        return Size(())

    def __add__(self, other: Size) -> Size:
        return Size(tx.np.broadcast_shapes(self.d, other.d))

    def remove_axis(self, axis: int) -> Size:
        return Size(self.d[:-1])


class ToSize(DiagramVisitor[Size, Size]):
    """
    Get the size of the diagram. Walks up
    the tree until it his a batched element.
    """

    A_type = Size

    def visit_primitive(self, diagram: core.Primitive, t: Size) -> Size:
        return Size(diagram.transform.shape[:-2])

    def visit_apply_transform(
        self, diagram: core.ApplyTransform, t: Size
    ) -> Size:
        return Size(diagram.transform.shape[:-2])

    def visit_apply_name(self, diagram: core.ApplyName, t: Size) -> Size:
        return diagram.diagram.accept(self, t)

    def visit_apply_style(self, diagram: core.ApplyStyle, t: Size) -> Size:
        if diagram.style is None:  # type: ignore
            return diagram.diagram.accept(self, t)
        return Size(diagram.style.size())

    def visit_compose_axis(self, diagram: core.ComposeAxis, t: Size) -> Size:
        return diagram.diagrams.accept(self, t).remove_axis(0)
