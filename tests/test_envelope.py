import math

import jax.numpy as jnp
import pytest
from hypothesis import given
from hypothesis.strategies import (
    DrawFn,
    composite,
    integers,
    lists,
    one_of,
    sampled_from,
)

import chalk
from chalk import (
    P2,
    V2,
    Diagram,
    Trail,
    circle,
    empty,
    make_path,
    origin,
    rectangle,
    unit_x,
    unit_y,
)


@composite
def vectors(draw: DrawFn) -> V2:
    x = draw(integers(min_value=-2, max_value=2).filter(lambda x: x != 0))
    y = draw(integers(min_value=-2, max_value=2).filter(lambda x: x != 0))
    return V2(x, y)


small_nat = integers(min_value=1, max_value=10)


@composite
def trails(draw: DrawFn) -> Trail:
    vs = draw(lists(vectors(), min_size=1))
    return Trail.from_offsets(vs)


@composite
def paths(draw: DrawFn) -> Diagram:
    return draw(trails()).stroke().center_xy()


@composite
def circles(draw: DrawFn) -> Diagram:
    return circle(draw(small_nat))


@composite
def rects(draw: DrawFn) -> Diagram:
    return rectangle(draw(small_nat), draw(small_nat))


@composite
def shapes(draw: DrawFn) -> Diagram:
    return draw(one_of(paths(), rects(), circles()))


@composite
def diagrams(draw: DrawFn) -> Diagram:
    shape = empty()
    for j in range(3):
        lshape = draw(shapes())
        shape += lshape.apply_transform(draw(transforms()))
    return shape


@composite
def transforms(draw: DrawFn) -> chalk.transform.Affine:
    v2 = draw(vectors())
    return draw(
        sampled_from(
            [
                chalk.transform.scale(v2),
                chalk.transform.translation(v2),
                chalk.transform.rotation_angle(chalk.transform.angle(v2)),
            ]
        )
    )


@given(diagrams(), vectors())
def test_envelope_trail(diagram: Diagram, vec: V2) -> None:
    "Property -> Envelope bounds trace."
    trace = diagram.get_trace()
    env = diagram.get_envelope()
    ts = trace(P2(0, 0), vec)
    e = env(vec)
    for t in ts:
        assert e == pytest.approx(t) or e > t


@given(diagrams(), vectors())
def test_pad(diagram: Diagram, vec: V2) -> None:
    orig = diagram.get_envelope()(vec)
    p = diagram.pad(2)
    assert p.get_envelope()(vec) == pytest.approx(2 * orig)
    vec = chalk.transform.norm(vec)
    orig = diagram.get_envelope()(vec)
    f = diagram.frame(2)
    assert f.get_envelope()(vec) == pytest.approx(2 + orig)


# Some specific tests.
def test_square() -> None:
    square = make_path([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    env = square.get_envelope()
    assert env(unit_x) == pytest.approx(1, abs=0.002)
    assert env(2 * unit_x) == pytest.approx(0.5, abs=0.002)
    assert env(unit_y) == pytest.approx(1, abs=0.002)
    diagonal = chalk.transform.norm(unit_x + unit_y)
    assert env(diagonal) == pytest.approx(math.sqrt(2), abs=0.002)


def test_circle() -> None:
    d = circle(1)
    env = d.get_envelope()
    assert env(unit_x) == 1
    assert env(2 * unit_x) == 0.5
    assert env(unit_y) == 1
    assert env(chalk.transform.norm(unit_x + unit_y)) == pytest.approx(1)


def test_circle_trace() -> None:
    d = circle(1)
    trace = d.get_trace()
    assert {float(x) for x in trace(origin, unit_x)} == {-1.0, 1.0}
    assert {float(x) for x in trace(origin, (2 * unit_x))} == {-0.5, 0.5}
    assert {float(x) for x in trace(origin, unit_y)} == {-1.0, 1.0}
    trace(origin, (unit_x + unit_y))


def test_path_trace() -> None:
    d = make_path([(1, 0), (1, 1)])
    trace = d.get_trace()
    direction = unit_x + unit_y
    hit, mask = trace.trace_v(origin, direction)
    assert bool(mask)
    assert jnp.allclose(chalk.transform.data(hit), chalk.transform.data(V2(1.0, 1.0)))
    hit, mask = trace.trace_v(origin, chalk.transform.norm(direction))
    assert bool(mask)
    assert jnp.allclose(chalk.transform.data(hit), chalk.transform.data(V2(1.0, 1.0)))


def test_transform() -> None:
    square = make_path([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    env = square.scale_x(2).scale_y(3).get_envelope()
    assert env(unit_x) == pytest.approx(2, abs=0.003)
    assert env(2 * unit_x) == pytest.approx(1, abs=0.003)
    assert env(unit_y) == pytest.approx(3, abs=0.004)
    env = square.rotate(45).get_envelope()
    assert env(chalk.transform.norm(unit_x + unit_y)) == pytest.approx(1, abs=0.002)
    env = square.translate(-2, -2).get_envelope()
    assert env(unit_x) == pytest.approx(-1)
