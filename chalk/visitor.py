from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

import chalk.transform as tx

if TYPE_CHECKING:
    from chalk.ArrowHead import ArrowHead
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
    from chalk.Path import Path
    from chalk.shapes import Image, Latex, Spacer, Text

    A = TypeVar("A", bound=Monoid)
else:
    A = TypeVar("A")

B = TypeVar("B")

# from itertools import product

# def to_size(index, shape):
#     if len(index) > len(shape):
#         index = index[len(index)-len(shape):]
#     return tuple(i if s > 1 else 0 for s, i in zip(shape, index))

# def remove(index, axis):
#     return tuple(i for j, i in enumerate(index) if j != axis)

# def rproduct(rtops):
#     return product(*map(range, rtops))


class DiagramVisitor(Generic[A, B]):
    A_type: type[A]

    # def visit_primitive_array(self, diagram: Primitive, arg: B) -> A:
    #     size = diagram.size()
    #     return {key: self.visit_primitive(diagram[key], arg[key])
    #             for key in rproduct(size)}

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
            ed: A = tx.vmap(fn, in_axes=axis, out_axes=axis)(diagram.diagrams)
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


C = TypeVar("C")


class ShapeVisitor(Generic[C]):
    def visit_path(self, shape: Path) -> C:
        raise NotImplementedError

    def visit_latex(self, shape: Latex) -> C:
        raise NotImplementedError

    def visit_text(self, shape: Text) -> C:
        raise NotImplementedError

    def visit_spacer(self, shape: Spacer) -> C:
        raise NotImplementedError

    def visit_arrowhead(self, shape: ArrowHead) -> C:
        raise NotImplementedError

    def visit_image(self, shape: Image) -> C:
        raise NotImplementedError
