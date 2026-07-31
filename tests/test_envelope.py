import math

import jax.numpy as jnp
import pytest

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
    for direction, expected in (
        (unit_x, jnp.asarray([-1.0, 1.0])),
        (2 * unit_x, jnp.asarray([-0.5, 0.5])),
        (unit_y, jnp.asarray([-1.0, 1.0])),
    ):
        distances, mask = trace(origin, direction)
        assert jnp.allclose(jnp.unique(distances[mask]), expected)
    trace(origin, (unit_x + unit_y))


def test_path_trace() -> None:
    d = make_path([(1, 0), (1, 1)])
    trace = d.get_trace()
    direction = unit_x + unit_y
    hit, mask = trace.trace_v(origin, direction)
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
    assert env(unit_x) == pytest.approx(-1, abs=0.002)
