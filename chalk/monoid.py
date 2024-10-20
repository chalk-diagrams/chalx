from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
)

from typing_extensions import Self

o = TypeVar("o")


def associative_reduce(fn: Callable[[o, o], o], iter: Iterable[o], initial: o) -> o:
    """Reduce for associative operations."""
    ls = list(iter)
    if len(ls) == 0:
        return initial
    if len(ls) == 1:
        return ls[0]
    off = len(ls) % 2
    v = associative_reduce(
        fn, [fn(ls[i], ls[i + 1]) for i in range(0, len(ls) - off, 2)], initial
    )
    if off:
        v = fn(v, ls[-1])
    return v


class Monoid:
    @classmethod
    def empty(cls) -> Self:
        raise NotImplementedError()

    def __add__(self, other: Self) -> Self:
        raise NotImplementedError()

    @classmethod
    def concat(cls, elems: Iterable[Self]) -> Self:
        elems = list(elems)
        if len(elems) == 1:
            return elems[0]
        return associative_reduce(cls.__add__, elems, cls.empty())

    concat2 = concat

    def reduce(self, axis: int = 0) -> Self:
        # return self.concat([self[i] for i in range(self.shape[0])])
        raise NotImplementedError()


A = TypeVar("A")


@dataclass
class Maybe(Generic[A], Monoid):
    data: Optional[A]

    @classmethod
    def empty(cls) -> Maybe[A]:
        return Maybe(None)

    def __add__(self, other: Maybe[A]) -> Maybe[A]:
        if self.data is None:
            return other
        return self


@dataclass
class MList(Generic[A], Monoid):
    data: List[A]

    @classmethod
    def empty(cls) -> MList[A]:
        return MList([])

    def __add__(self, other: MList[A]) -> MList[A]:
        return MList(self.data + other.data)

    def __iter__(self) -> Iterator[A]:
        return self.data.__iter__()


__all__ = []
