from chalk import *
from IPython.display import Image

d =  square(1) + text("hello", 0.5).fill_color("black").scale(0.5)
d

d =  square(1) + text("$\\sum_i^10$", 0.5).fill_color("black").scale(0.5)
d


d.render("text.png", height=200)
Image("text.png")

d.render_mpl("text.png", height=200)
Image("text.png")