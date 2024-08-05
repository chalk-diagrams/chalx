from chalk import *


d =  square(1).fill_opacity(0) + text("hello", 0.5).fill_color("black").scale(0.5)

#d =  square(1).fill_opacity(0) + text("$\\sum_i^10$", 0.5).fill_color("black").scale(0.5)

d.render_svg("text.svg", 256)