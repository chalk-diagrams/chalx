# Based on the following example from Diagrams
# https://archives.haskell.org/projects.haskell.org/diagrams/gallery/Koch.html
# pyright: basic

from PIL import Image as PILImage
from chalk import *
from chalk.transform import *

base_unit = Trail.hrule(1)


def koch(n):
    if n == 0:
        return base_unit.scale_x(5)
    else:
        return (
            koch(n - 1).scale(1 / 3)
            + koch(n - 1).scale(1 / 3).rotate_by(+1 / 6)
            + koch(n - 1).scale(1 / 3).rotate_by(-1 / 6)
            + koch(n - 1).scale(1 / 3)
        )


d = vcat(koch(i).stroke().line_width(0.01) for i in range(1, 5))


# Render
height = 512
d.render_svg("examples/output/koch.svg", height)
try:
    d.render("examples/output/koch.png", height)
    # d.render_pdf("examples/output/koch.pdf", height)
    PILImage.open("examples/output/koch.png")
except ModuleNotFoundError:
    print("Need to install Cairo")
