# Based on the following example from Diagrams
# https://archives.haskell.org/projects.haskell.org/diagrams/gallery/Hilbert.html
# pyright: basic

from PIL import Image as PILImage
from chalk import *
from chalk.transform import *

unit_x_rule, unit_y_rule = Trail.hrule(1), Trail.vrule(1)

# Draw a space filling Hilbert curve


def hilbert(n):
    def hilbert2(m):
        return hilbert(m).rotate_by(0.25)

    if n == 0:
        return Trail.empty()
    h, h2 = hilbert(n - 1), hilbert2(n - 1)
    return (
        h2.reflect_y()
        + unit_y_rule
        + h
        + unit_x_rule
        + h
        + unit_y_rule.reflect_y()
        + h2.reflect_x()
    )


d = hilbert(5).stroke().center_xy().line_width(0.05)
d.render_svg("examples/output/hilbert.svg", 500)
try:
    d.render("examples/output/hilbert.png", 500)
    PILImage.open("examples/output/hilbert.png")
except ModuleNotFoundError:
    print("Need to install Cairo")
