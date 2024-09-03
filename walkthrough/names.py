# %% tags=["hide_inp"]
from colour import Color
from chalk.core import BaseDiagram
from chalk import (
    set_svg_height,
    square,
    dart,
    circle,
    hcat,
    vstrut,
    Path,
    Diagram,
    Subdiagram,
    unit_y,
    hstrut,
)
from typing import List

set_svg_height(100)

# %% [markdown]
# Chalk supports basic methods for complex connected layouts and diagrams.
# Individual elements can be assigned names, and then be referenced in their subdiagram locations.
# As we will see, names are particularly useful for connecting different parts of the diagram with arrows.
# Our implementation follows the API of the Haskell [diagrams](https://diagrams.github.io/doc/manual.html#named-subdiagrams) library,
# but named nodes are also common in TikZ.
#
# ### Diagram.named

# %% tags=["hide_inp"]
help(BaseDiagram.named)

# %%
(square(1).show_origin() + dart()).render_svg("tri.svg", 56)

# %%
diagram = circle(0.5).named("x") | square(1)
diagram

# %% [markdown]
# ### Diagram.get_subdiagram

# %% tags=["hide_inp"]
help(BaseDiagram.get_subdiagram)

# %% [markdown]
# A `Subdiagram` is a `Diagram` paired with its enclosing context (a `Transformation` for the moment; but `Style` should also be added at some point).
# It has the following methods:
# - `get_envelope`, which returns the corresponding `Envelope`
# - `get_trace`, which returns the corresponding `Trace`
# - `get_location`, which returns the local origin of the `Subdiagram`
# - `boundary_from`, which return the furthest point on the boundary of the `Subdiagram`, starting from the local origin of the `Subdigram` and going in the direction of a given vector.

# %% [markdown]
#

# %%
diagram = circle(0.5).named("x") | square(1)
sub = diagram.get_subdiagram("x")
assert sub is not None
diagram + circle(0.2).translate_by(sub.get_location())

# %% [markdown]
# ### Diagram.with_names

# %% tags=["hide_inp"]
help(BaseDiagram.with_names)

# %%
root = circle(1).named("root")
leaves = hcat([circle(1).named(c) for c in "abcde"], sep=0.5).center_xy()


# %%
def connect(subs: List[Subdiagram], nodes: Diagram) -> Diagram:
    root, leaf = subs
    pp = root.boundary_from(unit_y)
    pc = leaf.boundary_from(-unit_y)
    d: Diagram = nodes + Path.from_points([pp, pc]).stroke()
    return d


# %%
nodes = root / vstrut(2) / leaves

# %%
for c in "abcde":
    nodes = nodes.with_names(["root", c], connect)
nodes

# %% [markdown]
# ### Diagram.qualify

# %% tags=["hide_inp"]
help(BaseDiagram.qualify)

# %%
red = Color("red")


# %%
def attach(subs: List[Subdiagram], dia: Diagram) -> Diagram:
    sub1, sub2 = subs
    p1 = sub1.get_location()
    p2 = sub2.get_location()
    return dia + Path.from_points([p1, p2]).stroke().line_color(red)


# %%
def squares() -> Diagram:
    s = square(1)
    return (s.named("NW") | s.named("NE")) / (s.named("SW") | s.named("SE"))


# %%
dia = hcat([squares().qualify(str(i)) for i in range(5)], sep=0.5)
pairs = [
    (("0", "NE"), ("2", "SW")),
    (("1", "SE"), ("4", "NE")),
    (("3", "NW"), ("3", "SE")),
    (("0", "SE"), ("1", "NW")),
]

# %%
dia

# %%
for pair in pairs:
    print(pair)
    dia = dia.with_names(list(pair), attach)

# %%
dia

# %% [markdown]
# ### Diagram.show_labels

# %% tags=["hide_inp"]
help(BaseDiagram.show_labels)

# %%
dia.show_labels(font_size=0.2)

# %% [markdown]
# ### Diagram.connect

# %% tags=["hide_inp"]
help(BaseDiagram.connect)

# %%
diagram = circle(0.5).named("x") | hstrut(1) | square(1).named("y")
diagram.connect("y", "x")

# %% [markdown]
# ### Diagram.connect_outside

# %% tags=["hide_inp"]
help(BaseDiagram.connect_outside)

# %%
diagram = circle(0.5).named("x") | hstrut(1) | square(1).named("y")
d = diagram.connect_outside("x", "y")
d.render_svg("connected.svg")
d
