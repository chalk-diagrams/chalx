import jax
import pytest

from chalk import circle
from chalk.path import Path, concat_paths, make_path, path_located_segments
from chalk.trail import Trail


def test_typeof_path():
    p = Path.from_list_of_tuples([(0, 0), (1, 0), (1, 1)], closed=True)
    assert str(jax.typeof(p)).startswith("cpath[")


def test_concat_paths():
    a = Path.from_list_of_tuples([(0, 0), (1, 0)])
    b = Path.from_list_of_tuples([(0, 1), (1, 1)])
    c = concat_paths(a, b)
    assert jax.typeof(c).n_locs == 2


def test_path_stroke_circle():
    d = circle(1)
    assert d.shape == ()


def test_peek_fails():
    p = Path.from_list_of_tuples([(0, 0), (1, 0)])
    with pytest.raises(AttributeError):
        jax.jit(lambda path: path.loc_trails)(p)
