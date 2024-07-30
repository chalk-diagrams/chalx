import chalk.transform as tx

def get_primitives(self) -> List[Primitive]:
    return self.accept(ToListOrder(), tx.ident).ls


def layout(
    self, height: tx.IntLike = 128, width: Optional[tx.IntLike] = None
) -> Tuple[List[Primitive], tx.IntLike, tx.IntLike]:
    envelope = self.get_envelope()
    assert envelope is not None

    pad = 0.05

    # infer width to preserve aspect ratio
    if width is None:
        width = tx.np.round(height * envelope.width / envelope.height).astype(
            int
        )
    else:
        width = width
    assert width is not None
    # determine scale to fit the largest axis in the target frame size
    α = tx.np.where(
        envelope.width - width <= envelope.height - height,
        height / ((1 + pad) * envelope.height),
        width / ((1 + pad) * envelope.width),
    )

    s = self.scale(α).center_xy().pad(1 + pad)
    e = s.get_envelope()
    assert e is not None
    s = s.translate(e(-tx.unit_x), e(-tx.unit_y))

    style = StyleHolder.root(tx.np.maximum(width, height))
    s = s._style(style)
    patches = [Patch.from_prim(prim, style) for prim in s.get_primitives()]
    return patches, height, width


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
            tx.np.roll(tx.np.cumsum(internal.counter, axis=-1), 1, axis=-1),
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
