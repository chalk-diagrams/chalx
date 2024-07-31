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
    from chalk.monoid import Monoid

    A = TypeVar("A", bound=Monoid)
else:
    A = TypeVar("A")

B = TypeVar("B")


class DiagramVisitor(Generic[A, B]):
    A_type: type[A]

    def visit_primitive(self, diagram: Primitive, arg: B) -> A:
        "Primitive defaults to empty"
        return self.A_type.empty()

    def visit_empty(self, diagram: Empty, arg: B) -> A:
        "Empty defaults to empty"
        return self.A_type.empty()

    def visit_compose(self, diagram: Compose, arg: B) -> A:
        "Compose defaults to monoid over children"
        return self.A_type.concat(
            [d.accept(self, arg) for d in diagram.diagrams]
        )

    def visit_compose_axis(self, diagram: ComposeAxis, t: B) -> A:
        from functools import partial

        size = diagram.diagrams.size()
        axis = len(diagram.diagrams.size()) - 1
        fn = diagram.diagrams.accept.__func__  # type: ignore
        fn = partial(fn, visitor=self, args=t)
        # tx.np.vectorize(partial(fn, visitor=self, args=t),
        if not tx.JAX_MODE:
            ds = []
            for k in range(size[-1]):
                d = tx.tree_map(lambda x: x.take(k, axis), diagram.diagrams)
                ds.append(fn(d))
            ed = tx.tree_map(lambda *x: tx.np.stack(x, axis), *ds)
            # assert ed.size() == size
        else:
            import jax
            ed: A = jax.vmap(fn, in_axes=axis, out_axes=axis)(diagram.diagrams)
        return self.A_type.reduce(ed, axis)

    def visit_apply_transform(self, diagram: ApplyTransform, arg: B) -> A:
        "Defaults to pass over"
        return diagram.diagram.accept(self, arg)

    def visit_apply_style(self, diagram: ApplyStyle, arg: B) -> A:
        "Defaults to pass over"
        return diagram.diagram.accept(self, arg)

    def visit_apply_name(self, diagram: ApplyName, arg: B) -> A:
        "Defaults to pass over"
        return diagram.diagram.accept(self, arg)
