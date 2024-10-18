from chalk import *
from colour import Color
# pyright: basic

grey = Color("#bbbbbb")
papaya = Color("#ff9700")

left_arrow = make_path([(0, 0), (1, 0)]).reflect_x().line_width(0.03).center_xy()


def box(t):
    return rectangle(1.5, 1).line_width(0.05).fill_color(papaya) + text(t, 1).scale(0.7)


def label(tex):
    return text(tex, 1).scale(0.5).pad(0.4)


def arrow(text, d=True):
    return label(text) // left_arrow


# Autograd 1
d = hcat(
    [
        arrow(r"$f'_x(g(x))$"),
        box("$f$"),
        arrow(r"$f'_{g(x)}(g(x))$"),
        box("$g$"),
        arrow("1"),
    ],
    0.2,
)
d.render_svg("examples/output/latex.svg", 100)
