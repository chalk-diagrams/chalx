# from jaxtyping import install_import_hook
# with install_import_hook("chalk", "typeguard.typechecked"):
#     import chalk 
from chalk import *
import numpy as np

d = circle(2).fill_color("black").named(Name("a")) | circle(2).fill_color("blue")
def f(subs, d):
    return d + circle(1).fill_color("red").translate_by(subs[0].get_location())
d.with_names([Name("a")], f)

v = circle(np.arange(1, 5)).get_envelope()

d = circle(np.arange(1, 5)).fill_color("black").named(Name("a")).hcat()

def f(subs, d):
    print(subs[0].get_location())
    return d + circle(1).fill_color("red").translate_by(subs[0].get_location()).concat()

d = d.with_names([Name("a")], f)
d.render_svg("tmp.svg")
