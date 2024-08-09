# + tags=["hide_inp"]
from chalk.core import BaseDiagram
from chalk import *
# -


# Each diagram has an origin and an envelope.
# Manipulating the position of the diagram with respect to its origin and envelope allows for precise control of the layout.
# Note that the Chalk API is immutable and always returns a new ``Diagram`` object.

# ### Diagram.show_origin

# + tags=["hide_inp"]
help(BaseDiagram.show_origin)
# -

#

triangle(1).show_origin()


triangle(1).show_origin().render_svg("/tmp/test.svg")

# ### Diagram.show_envelope

# + tags=["hide_inp"]
help(BaseDiagram.show_envelope)
# -

rectangle(1, 1).show_envelope()


triangle(1).show_envelope()


rectangle(1, 1).show_beside(triangle(1), unit_x)


(rectangle(1, 1) | triangle(1)).pad(1.4)


arc(1, 0, 90)

# ### Diagram.align_*

# + tags=["hide_inp"]
help(BaseDiagram.align_t)
# -

#

triangle(1).align_t().show_origin()



triangle(1).align_t().show_beside(rectangle(1, 1).align_b(), unit_x)


# + tags=["hide_inp"]
help(BaseDiagram.align_r)
# -

# ### Diagram.center_xy

# + tags=["hide_inp"]
help(BaseDiagram.center_xy)
# -

#

triangle(1).center_xy().show_envelope().show_origin()


# ### Diagram.pad_*

# + tags=["hide_inp"]
help(BaseDiagram.pad)
# -


#

triangle(1).pad(1.5).show_envelope().show_origin()


# ### Diagram.with_envelope

# + tags=["hide_inp"]
help(BaseDiagram.with_envelope)
# -


#


(rectangle(1, 1) + triangle(0.5)) | rectangle(1, 1)


(rectangle(1, 1) + triangle(0.5)).with_envelope(triangle(0.5)) | rectangle(1, 1).fill_color("red")
