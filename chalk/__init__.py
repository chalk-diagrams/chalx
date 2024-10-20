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

if eval(os.environ.get("CHALK_CHECK", "0")):
    assert hook is not None
    hook.uninstall()


jax_type = [
    chalk.core.Primitive,
    chalk.core.Compose,
    chalk.envelope.Envelope,
    chalk.core.ApplyTransform,
    chalk.core.Empty,
    chalk.core.ApplyStyle,
    chalk.core.ComposeAxis,
    chalk.style.StyleHolder,
    chalk.path.Path,
    chalk.trail.Located,
    chalk.trail.Trail,
    chalk.shapes.Spacer,
    chalk.backend.patch.Patch,
    chalk.path.Text,
    chalk.subdiagram.Subdiagram,
    chalk.segment.Segment,
]
for t in jax_type:
    chex.register_dataclass_type_with_jax_tree_util(t)


jax.tree_util.register_pytree_node(
    chalk.core.ApplyName,
    lambda tree: ((tree.diagram,), (tree.dname,)),
    lambda extra, args: chalk.core.ApplyName(extra[0], args[0]),  # type: ignore
)


if not TYPE_CHECKING:
    # Set library name the same as on PyPI
    # must be the same as setup.py:setup(name=?)
    __libname__: str = "chalk-diagrams"  # custom dunder attribute
    __version__: str = metadata.version(__libname__)
