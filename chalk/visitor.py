from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

import chalk.transform as tx

if TYPE_CHECKING:
    from chalk.core import (
        ApplyName,
        ApplyStyle,
        ApplyTransform,
        Compose,
        ComposeAxis,
        Empty,
        Primitive,
    )
    A = TypeVar("A")
else:
    A = TypeVar("A")

B = TypeVar("B")


class DiagramVisitor(Generic[A, B]):
    """Class for traversing the diagram tree.
    Can be thought of as a tree fold.
    Type B is passed up the tree.
    Type A is accumulated down the tree.
    Type A needs ``empty()`` and ``__add__`` (monoid ops).
    """

    A_type: type[A]

    def visit_primitive(self, diagram: Primitive, arg: B) -> A:
        # Default primitive defaults to empty monoid
        return self.A_type.empty()

    def visit_empty(self, diagram: Empty, arg: B) -> A:
        # Default for empty to empty monoid
        return self.A_type.empty()

    def visit_compose(self, diagram: Compose, arg: B) -> A:
        # Compose defaults to monoid over children
        from chalk.monoid import reduce_associative

        elems = [d._accept(self, arg) for d in diagram.diagrams]
        return reduce_associative(
            lambda a, b: a + b, elems, self.A_type.empty()
        )

    def visit_compose_axis(self, diagram: ComposeAxis, t: B) -> A:
        from functools import partial

        size = diagram.diagrams.size()
        axis = len(diagram.diagrams.size()) - 1
        fn = diagram.diagrams._accept.__func__  # type: ignore
        fn = partial(fn, visitor=self, args=t)
        ds = []
        for k in range(int(size[-1])):
            d = tx.tree_map(lambda x: x.take(k, axis), diagram.diagrams)
            ds.append(fn(d))
        ed = tx.tree_map(lambda *x: tx.np.stack(x, axis), *ds)
        return self.A_type.reduce(ed, axis)

    def visit_apply_transform(self, diagram: ApplyTransform, arg: B) -> A:
        # Defaults to passing over transform
        return diagram.diagram._accept(self, arg)

    def visit_apply_style(self, diagram: ApplyStyle, arg: B) -> A:
        # Defaults to passing over style
        return diagram.diagram._accept(self, arg)

    def visit_apply_name(self, diagram: ApplyName, arg: B) -> A:
        # Defaults to passing over name
        return diagram.diagram._accept(self, arg)


__all__ = []
