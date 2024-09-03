from chalk import *
from colour import Color
# pyright: basic
# pyright: reportUnusedExpression=none

grey = Color("grey")
blue = Color("blue")
orange = Color("orange")


octagon = regular_polygon(9, 1.5).rotate_by(1 / 16).line_color(grey).line_width(0.5)
dias = (
    octagon.named("first").show_origin()
    | hstrut(3)
    | octagon.named("second").show_origin()
)
ex1 = octagon

ex1 = dias.connect(
    "first",
    "second",
    ArrowOpts(trail=Trail.from_offsets([unit_x, 0.25 * unit_y, unit_x, 0.25 * unit_y])),
)
ex1

output_path = "examples/output/t1.svg"
ex1.render_svg(output_path)

ex1 = dias.connect(
    "first",
    "second",
    ArrowOpts(
        head_style=Style().fill_color(grey),
        arc_height=0.5,
        head_pad=0.1,
        shaft_style=Style().line_color(blue),
    ),
)
ex1 = ex1.connect(
    "second",
    "first",
    ArrowOpts(
        head_style=Style().fill_color(grey),
        arc_height=0.5,
        head_pad=0.1,
        shaft_style=Style().line_color(blue),
    ),
)

ex12 = ex1.connect_perim(
    "first",
    "second",
    unit_x,
    unit_y,
    ArrowOpts(head_pad=0.1, shaft_style=Style().line_color("green")),
)

ex3 = arrow_v(unit_y)
d = ex12 + ex3


output_path = "examples/output/arrows.svg"
d.render_svg(output_path, height=200)


# output_path = "examples/output/arrows.svg"
# d.render_svg(output_path, height=200)

output_path = "examples/output/arrows.png"
d.render(output_path, height=200)

# PILImage.open(output_path)

# output_path = "examples/output/arrows.pdf"
# d.render_pdf(output_path, height=200)
