import sys
from typing import TYPE_CHECKING

import chalk.subdiagram

if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata


import os
if eval(os.environ.get("CHALK_CHECK", "1")):
    from jaxtyping import install_import_hook
    hook = install_import_hook("chalk", "typeguard.typechecked")

    
import chex
import jax

import chalk.align as align
import chalk.backend.patch
import chalk.core
import chalk.envelope
import chalk.shapes
import chalk.style
import chalk.trace
import chalk.trail
import chalk.path
from chalk.align import *  # noqa: F403
from chalk.arrow import ArrowOpts, arrow_at, arrow_between, arrow_v
from chalk.combinators import *  # noqa: F403
from chalk.core import set_svg_draw_height, set_svg_height
from chalk.envelope import Envelope
from chalk.monoid import Maybe, MList, Monoid
from chalk.shapes import *  # noqa: F403
from chalk.style import Style, to_color
import chalk.segment
from chalk.subdiagram import Name
import chalk.trail as Trail
from chalk.transform import (
    P2,
    V2,
    Affine,
    BoundingBox,
    from_radians,
    to_radians,
    unit_x,
    unit_y,
)
from chalk.types import Diagram


if eval(os.environ.get("CHALK_CHECK", "1")):
    hook.uninstall()


jax_type = [
    chalk.core.Primitive,
    chalk.core.Compose,
    chalk.envelope.Envelope,
    chalk.core.ApplyTransform,
    chalk.core.Empty,
    chalk.core.ApplyStyle,
    chalk.core.ComposeAxis,
    chalk.envelope.EnvDistance,
    chalk.trace.TraceDistances,
    chalk.style.StyleHolder,
    chalk.trail.Trail,
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
    lambda extra, args: chalk.core.ApplyName(extra[0], args[0]),
)



if not TYPE_CHECKING:
    # Set library name the same as on PyPI
    # must be the same as setup.py:setup(name=?)
    __libname__: str = "chalk-diagrams"  # custom dunder attribute
    __version__: str = metadata.version(__libname__)
