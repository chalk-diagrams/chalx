import sys

from chalk import *
from colour import Color
# pyright: basic

BLACK = Color("black")
STEPS = 7
NODES = 7


def node(i, j):
    name = Name((i, j))
    r = rectangle(1, 0.4, 0.1).named(name)
    t = text(f"Node {i} {j}", 0.2).fill_color(BLACK)
    return r + t


d = hcat([vcat([node(i, j) for j in range(NODES)], sep=1) for i in range(STEPS)], sep=2)

for i in range(NODES - 1):
    for j in range(STEPS):
        for j2 in range(STEPS):
            src = Name((i, j))
            tgt = Name((i + 1, j2))
            d = d.connect_perim(src, tgt, unit_x, -unit_x)


path = "examples/output/lattice.svg"
d.render_svg(path, height=256)

path = "examples/output/lattice.png"
d.render(path, height=256)
