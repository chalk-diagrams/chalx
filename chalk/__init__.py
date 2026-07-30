"""Chalk Package"""

import sys
from typing import TYPE_CHECKING

import chalk.subdiagram

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

import os

if eval(os.environ.get("CHALK_CHECK", "0")):
    from jaxtyping import install_import_hook

    hook = install_import_hook("chalk", "typeguard.typechecked")
else:
    hook = None


# These will eventually be replaced by optree
import chex
import jax

import chalk.backend.patch
import chalk.core
import chalk.envelope
import chalk.path
import chalk.segment
import chalk.shapes
import chalk.style
import chalk.trace
import chalk.trail
from chalk.arrowheads import *  # noqa: F403
from chalk.core import *  # noqa: F403
from chalk.arrow import *  # noqa: F403
from chalk.combinators import *  # noqa: F403
from chalk.shapes import *  # noqa: F403
from chalk.trail import *  # noqa: F403
from chalk.path import *  # noqa: F403
from chalk.subdiagram import *  # noqa: F403
from chalk.style import *  # noqa: F403
from chalk.types import *  # noqa: F403
from chalk.measure import trace_measure, measure_from_splits  # noqa: F401
from chalk.raster import rasterize  # noqa: F401

if eval(os.environ.get("CHALK_CHECK", "0")):
    assert hook is not None
    hook.uninstall()


jax_type = [
    chalk.shapes.Spacer,
    chalk.backend.patch.Patch,
    chalk.path.Text,
    chalk.subdiagram.Subdiagram,
]
for t in jax_type:
    chex.register_dataclass_type_with_jax_tree_util(t)

import chalk.diag  # noqa: F401  # register opaque diag[] hijax type


if not TYPE_CHECKING:
    # Set library name the same as on PyPI
    # must be the same as setup.py:setup(name=?)
    __libname__: str = "chalk-diagrams"  # custom dunder attribute
    __version__: str = metadata.version(__libname__)
