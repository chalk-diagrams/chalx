from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Iterator, List, Optional, TypeVar

o = TypeVar("o")
A = TypeVar("A")


def reduce_associative(
    fn: Callable[[o, o], o], elems: Iterable[o], empty: o | None = None
) -> o:
    """Tree-reduce an associative binary op (typically ``__add__``).

    Types that used to subclass ``Monoid`` still expose ``empty()`` / ``__add__``;
    multi-element ``concat`` is this helper over ``__add__``.
    """
    ls = list(elems)
    if len(ls) == 0:
        if empty is None:
            raise ValueError("reduce_associative() of empty sequence")
        return empty
    if len(ls) == 1:
        return ls[0]
    off = len(ls) % 2
    v = reduce_associative(
        fn, [fn(ls[i], ls[i + 1]) for i in range(0, len(ls) - off, 2)], empty
    )
    if off:
        v = fn(v, ls[-1])
    return v


@dataclass
class Maybe(Generic[A]):
    data: Optional[A]

    @classmethod
    def empty(cls) -> Maybe[A]:
        return Maybe(None)

    def __add__(self, other: Maybe[A]) -> Maybe[A]:
        if self.data is None:
            return other
        return self


@dataclass
class MList(Generic[A]):
    data: List[A]

    @classmethod
    def empty(cls) -> MList[A]:
        return MList([])

    def __add__(self, other: MList[A]) -> MList[A]:
        return MList(self.data + other.data)

    def __iter__(self) -> Iterator[A]:
        return self.data.__iter__()


__all__ = ["reduce_associative", "Maybe", "MList"]
