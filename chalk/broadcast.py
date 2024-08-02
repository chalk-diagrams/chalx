from __future__ import annotations

from typing import TypeVar, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from chalk.monoid import Monoid
import chalk.transform as tx
from chalk.visitor import DiagramVisitor
from chalk.types import Diagram

if TYPE_CHECKING:
    from chalk.core import ApplyName, ApplyTransform, Compose, Primitive, ApplyStyle, ComposeAxis


def add_axis(self: Diagram, size: int) -> Diagram:
    return tx.tree_map(lambda x: tx.np.repeat(x[None], size, axis=0), self) # type: ignore


def size(self: Diagram) -> Tuple[int, ...]:
    return self.accept(ToSize(), Size.empty()).d


def repeat_axis(self: Diagram, size: int, axis: int) -> Diagram:
    return tx.tree_map(lambda x: tx.np.repeat(x, size, axis=axis), self) # type: ignore


def getitem(self: Diagram, ind: Union[None, int, Tuple[int, ...]]) -> Diagram:
    return tx.tree_map(lambda x: x[ind], self) # type: ignore

A = TypeVar("A", bound=Diagram)
B = TypeVar("B", bound=Diagram)

def broadcast_diagrams(self: A, other: B) -> Tuple[A, B]:
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


    def visit_apply_name(self, diagram: ApplyName, t: Size) -> Size:
        return diagram.diagram.accept(self, t)

    def visit_apply_style(self, diagram: ApplyStyle, t: Size) -> Size:
        if diagram.style is None:
            return diagram.diagram.accept(self, t)
        return Size(diagram.style.size())

    def visit_compose_axis(self, diagram: ComposeAxis, t: Size) -> Size:
        return diagram.diagrams.accept(self, t).remove_axis(0)
