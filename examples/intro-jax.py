import jax
import jax.numpy as np
#import numpy as np
import os
os.environ["CHALK_JAX"] = "1"
from jaxtyping import install_import_hook
with install_import_hook("chalk", "typeguard.typechecked"):
    import chalk 
from chalk import *
from colour import Color
from random import random
import chalk


chalk.tx.set_jax_mode(True)
r, b = to_color("red"), to_color("blue")
from functools import partial
def plot():
    @jax.vmap
    @jax.vmap
    def draw(x, y):      
        rot = np.where(x % 2, 0, 180)
        @partial(jax.vmap, in_axes=[None, 0]) 
        def round(shape, j):
            return shape.rotate(360 * j / 6)

        inner = regular_polygon(4, 1)
        tri = triangle(1).rotate(360 / 12)
        tri = inner.juxtapose(tri, V2(-1, 0))
        layer_1 = round(tri, np.arange(6)).concat()
        sq = square(1)
        sq = inner.juxtapose(sq, V2(0, -1))
        layer_2 = round(sq, np.arange(6)).concat()

        # return (regular_polygon(6, 1) +  around(np.arange(6)).concat() + tri(np.arange(6)).concat()) \
        #     .translate(x / 2, y * np.sqrt(1 - 0.25))#.fill_color(x/50 * r +  y/ 50 * b)
        return (inner + layer_1)#.translate(x, y)
    
    return draw(*np.broadcast_arrays(np.arange(2)[:, None], 
                                     np.arange(2)))
grid = rectangle(1, 1).translate(0, 0.5).fill_color("white") #+ (make_path([(0, 0), (1, 0)]) + make_path([(0, 0), (0, -1)])).line_color("black").line_width(5)

x = plot()
print(x.size())
x = x.concat().concat()

assert x.size() == ()
x = grid.scale(10) + x.translate(5, 5).scale(100)
assert x.size() == ()
prim = x.get_primitives()

# for p in prim:
#     print(p)
jax.tree_util.tree_map_with_path(lambda p,x: print(p, x.shape), prim)
#jax.tree_util.tree_map_with_path(lambda p,x: print(p, x.shape), x)
chalk.backend.svg.prims_to_file(prim, "test.svg", 1000, 1000)
SVG("test.svg")

grid = rectangle(1, 1).translate(0, 0.5).fill_color("white") #+ (make_path([(0, 0), (1, 0)]) + make_path([(0, 0), (0, -1)])).line_color("black").line_width(5)

chalk.tx.set_jax_mode(True)
r, b = to_color("red"), to_color("blue")
def plot():
    @jax.vmap
    @jax.vmap
    def draw(x, y):      
        rot = np.where(x % 2, 0, 180)
        @jax.vmap 
        def tri(j):
            return triangle(1).translate(0, 1).rotate(360 * j / 6).fill_color("green")

        @jax.vmap 
        def around(j):
            return square(1).fill_color("red").line_width(0).translate(0, 1 + 0.5).rotate(360. * j / 6).fill_color("blue")
        # return (regular_polygon(6, 1) +  around(np.arange(6)).concat() + tri(np.arange(6)).concat()) \
        #     .translate(x / 2, y * np.sqrt(1 - 0.25))#.fill_color(x/50 * r +  y/ 50 * b)
        return (around(np.arange(6)).concat() + tri(np.arange(6)).concat()) 
    
    return draw(*np.broadcast_arrays(np.arange(1)[:, None], 
                                     np.arange(1)))
x = plot().concat().concat()
# jax.tree_util.tree_map_with_path(lambda p,x: print(p, x.shape), x)
assert x.size() == ()
x = grid.scale(10) + x.translate(5, 5).scale(100)
assert x.size() == ()
prim = x.get_primitives()
# for p in prim:
#     print(p)
#jax.tree_util.tree_map_with_path(lambda p,x: print(p, x.shape), prim)

chalk.backend.svg.prims_to_file(prim, "test.svg", 1000, 1000)

print("start")

r, b = to_color("red"), to_color("blue")
def plot():
    @jax.vmap
    @jax.vmap
    def draw(x, y):      
        rot = np.where(x % 2, 0, 180) 
        return (regular_polygon(6, 1) + square(1).translate(0, -0.5).translate(0, 1)).translate(x / 2, y * np.sqrt(1 - 0.25)).fill_color(x/50 * r +  y/ 50 * b)
        
    return draw(*np.broadcast_arrays(np.arange(10)[:, None], 
                                     np.arange(10)))
x = plot().concat().concat()

x = grid.scale(10) + x.scale_y(-1)
x.get_primitives()

#x.with_envelope(rectangle(10, 10)).render_svg("tile.svg", 512, 512)

exit()


r, b = to_color("red"), to_color("blue")
def plot():
    @jax.vmap
    @jax.vmap
    def draw(x, y):        
        rot = np.where(x % 2, 0, 180) 
        return triangle(1).rotate(rot).align_t().translate(x / 2, y * np.sqrt(1 - 0.25)).fill_color(x/50 * r +  y/ 50 * b)
        
    return draw(*np.broadcast_arrays(np.arange(20)[:, None], 
                                     np.arange(20)))
x = plot().concat().concat()

x = grid.scale_x(100) + x.with_envelope(empty()).scale_y(-1)
x.render_svg("tile.svg", 512)
exit()

r, b = to_color("red"), to_color("blue")
def plot():
    @jax.vmap
    @jax.vmap
    def draw(x, y):
        
        return square(1).translate(x, y).fill_color( x/50 * r +  y/ 50 * b)
        
    return draw(*np.broadcast_arrays(np.arange(20)[:, None], 
                                     np.arange(20)))
x = plot().concat().concat()

x = grid.scale_x(100) + x.with_envelope(empty()).scale_y(-1)
x.render_svg("tile.svg", 512)
exit()



data = [[random(), random(), random(), random(), random()]
         for _ in range(50)]


def plot(data):
    @jax.vmap
    def draw(pt, pt2, i):
        line = make_path([(i, pt[0]), (i+1, pt2[0])]).line_color("black")
        return line + circle(0.1).translate(i, pt[0]).fill_color("blue")
        
    return draw(data, tx.X.np.roll(data, -1, axis=0), np.arange(50))

x = plot(np.array(data))

x = grid.scale_x(100) + x.concat().with_envelope(empty()).scale_y(-1)
x.render_svg("plot.svg", 512)


exit()
data = [[random(), random(), random(), random(), random()]
         for _ in range(50)]
grid = rectangle(1, 1).align_bl().fill_color("white") + (make_path([(0, 0), (1, 0)]) + make_path([(0, 0), (0, -1)])).line_color("black").line_width(5)

def make_segment(start, end):
    return (seg(V2(1, 0)).rotate(-start) + arc_seg_angle(start, end-start) + seg(V2(-1, 0)).rotate(-end)).close().stroke()

jax.tree_util.tree_map_with_path(lambda p, x: print(p, x.shape), make_segment(0, 1))
#(make_segment(0, 90).fill_color("orange") + make_segment(90, 100).line_width(2).line_color("green").fill_color("blue")).render("/tmp/t2.png")

def pie(data):
    pt = data / data.sum(0)[None]
    pt = np.cumsum(pt, 0)
    print(pt[:, 0])
    T = data.shape[0]
    @jax.vmap
    def draw(pt, pt2, i):
        return make_segment(360 * pt[0], 360 * pt2[0]).fill_color(np.ones(3) * ((10 * i) % T / T))
        
    return draw(pt, tx.X.np.roll(pt, -1, axis=0), np.arange(T))

x = pie(np.array(data))
# jax.tree_util.tree_map_with_path(lambda p, x: print(p, x.shape), x)


x = grid + x.concat().with_envelope(empty())
x.render_svg("pie.svg", 512)

exit()


def bar(data):
    @jax.vmap
    def draw(pt):
        r, b = to_color("red"), to_color("blue")
        bar = rectangle(0.1, pt[0])
        return bar.line_width(1)
        
    return draw(data)

x = bar(np.array(data)).align_t().hcat(0.1)

# print("prims")
x = grid.scale_x(100) + x.with_envelope(empty()).scale_y(-1)

x.render_svg("bar.svg", 512)
exit()


data = [[random(), random(), random(), random(), random()]
         for _ in range(500)]
grid = rectangle(1, 1).align_bl().fill_color("white") + (make_path([(0, 0), (1, 0)]) + make_path([(0, 0), (0, -1)])).line_color("black").line_width(5)


def scatter(data):
    @jax.vmap
    def draw(pt):
        r, b = to_color("red"), to_color("blue")
        mark1 = circle(0.01)
        mark2 = square(2 * 0.01)
        mark = jax.lax.cond(pt[2] > 0.5, 
                     lambda: mark1,
                     lambda: mark2
                     )
        color = r * pt[3] + b * (1-pt[3])
        scale = pt[4]
        return mark.scale(scale).translate(pt[0], pt[1]).fill_color(color).line_width(1)
        
    return draw(data)


# print("scatter")
x = scatter(np.array(data)).concat()

# print("prims")
x = grid + x.with_envelope(empty()).scale_y(-1)

print("Render")
# x.render("scatter.png", 512)
x.render_svg("scatter.svg", 512)
exit()





# Example 1: Vector Arguments
def example(v):
    return circle(1.0).fill_color(np.ones(3) * v[:, None] / 6)

d = example(np.arange(1, 6))
print(d.get_envelope().width)
d = d.hcat()
d.render(f"examples/output/intro-jax-1.png", 64)


# Example 2: Broadcast Arguments
def example(v):
    return circle(v[:, None] / 6) \
        .fill_color(np.ones(3) * v[:, None, None] / 6)

d = example(np.arange(1, 6)).hcat().vcat()
d.render(f"examples/output/intro-jax-2.png", 64)


# Example 3: Broadcast Compose
def example(v):
    return circle(v[:, None] / 6).fill_color("white") + \
        circle(v[:, None, None] / 6).fill_color("blue")
d = example(np.arange(1, 6)).hcat().vcat()
d.render(f"examples/output/intro-jax-3.png", 64)

# Example 4: Vmap
@jax.vmap
def example(i):
    return circle(i / 6).fill_color(i / 6 *  np.ones(3))
d = example(np.arange(1, 6)).hcat()
d.render(f"examples/output/intro-jax-4.png", 64)


# Example 5: Inner Vmap
@jax.vmap
def example(i):
    @jax.vmap
    def inner(j):
        return circle(j / 6 + i / 6)
    return inner(np.arange(1, 6)).fill_color(i / 6 *  np.ones(3)).hcat()
d = example(np.arange(1, 6)).vcat()
d.render(f"examples/output/intro-jax-5.png", 64)







# i =4
# x = (circle(0.3 * i / 6).fill_color(np.ones(3) * i / 6) + 
#             square(0.1).fill_color("white"))
# x = x / circle(0.1).fill_color("blue")
# x.render("/tmp/t3.png")


#@jax.vmap
def inner(i):
    return (circle(0.3 * i / 6).fill_color(np.ones(3) * i[:, None] / 6)
             + 
            square(0.1).fill_color("white"))

# inside = hcat(inner(np.arange(2, 6)))
# inside.render("/tmp/t.png")

# jax.tree_util.tree_map_with_path(lambda p, x: print(p, x.shape), x)
# print(inner(np.arange(4, 6)).transform.shape)
# x = circle(1.0).scale(np.array([1, 2]))
# print(inner(np.arange(4, 6)).size())
# print(inner(np.arange(4, 6)).transform.shape)
x = (inner(np.arange(4, 6)) / circle(0.1).fill_color("blue")).hcat()
x = (inner(np.arange(4, 6)) / circle(0.1).fill_color("blue")).align_t().hcat()
# jax.tree_util.tree_map_with_path(lambda p, x: print(p, x.shape), x)
# exit()

x.render("/tmp/t2.png")
exit()

@jax.vmap
def width(diagram):
    return diagram.get_envelope().width

print(width(inner(np.arange(2, 6))))


exit()

@jax.vmap
def outer(j):
    @jax.vmap
    def inner(i):
        # return (circle(0.3 * i / 6).fill_color(np.ones(3) * i / 6))
        return (circle(0.3 * i / 6).fill_color(np.ones(3) * i / 6) + 
                square(0.1).fill_color("white"))
    inside = inner(np.arange(2, 6))
    return inside
out = outer(np.arange(1, 5))
print("My Size", hcat(out).size())
d = vcat(hcat(out))
print("PRIMS:", [prim.order for prim in  d.accept(chalk.core.ToListOrder(), tx.X.ident).ls])

#arc_seg(V2(0, 1), 1).stroke().render("/tmp/t.png", 64)

#jax.tree.map(lambda x: print(x.shape), d)
d.render("/tmp/t.png")
exit()
# print(out.get_trace()(P2(0, 0), V2(1, 0)))
# d = (rectangle(10, 2).fill_color("white") + out)
# d.render("temp.png", 64)


seed = 1701
size = 50
connects = 1
around = 5
color = np.stack([chalk.style.to_color(c) for c in  Color("red").range_to("blue", size // around)],
                 axis=0)

key = jax.random.PRNGKey(0)
matrix = jax.random.uniform(key, (size, 2)) * 4 - 2
#all = jax.random.categorical(key, np.ones((size, connects, size)), axis=-1)
all = np.stack([np.minimum(size, np.ones((size,), int) + around * (np.arange(size) // around))  for i in range(connects)], axis=1)

@jax.jit
def graph(x):
    #x = np.minimum(np.maximum(x, 0), 1)
    center =  1/10 * np.abs(x).sum()
    repulse = (1/ size) * ((1 / (1e-3 + np.pow(x[:, None, :] - x, 2).sum(-1))) * (1 -np.eye(size))).sum()

    def dots(p, i):
        d = circle(0.1).translate(p[0], p[1]).fill_color(color[i // around] * np.maximum((i % around) / (around - 1), 0.5))
        return d

    def connect(p):
        eps = 1e-4
        d = make_path([(p[0, 0], p[0, 1]), 
                        (p[1, 0]+ eps, p[1, 1] + eps)])
        return d

    out = jax.vmap(dots)(x, np.arange(size)).with_envelope(empty()).line_width(1)    
    a, b = x[:, None, :].repeat(connects, axis=1), x[all]
    v = np.stack([a, b], axis=-2).reshape((-1, 2, 2))
    spring = size * np.pow(np.pow(v[:, 1] - v[:, 0], 2).sum(-1) - 0.04, 2).sum()
    lines = jax.vmap(connect)(v).with_envelope(empty())
    
    out = lines.line_width(1) + out
    out = rectangle(4, 4).fill_color("white")  + out
    out, h, w = chalk.core.layout_primitives(out, 500)
    score = spring + center + repulse
    return score, (out, h, w)

def opt(x, fn):
    res = []
    fn = jax.jit(jax.value_and_grad(fn, has_aux=True))
    solver = optax.adam(learning_rate=0.3)
    opt_state = solver.init(x)
    for j in range(500):
        print(j)
        value, grad = fn(x)
        score, out = value
        updates, opt_state = solver.update(grad, opt_state, x)
        x = optax.apply_updates(x, updates)

        if True:
            out, h, w = out
            out = jax.tree.map(onp.asarray, out)
            import chalk.transform
            chalk.transform.set_jax_mode(False)
            print("RENDER")
            chalk.backend.svg.prims_to_file(out, f"test.{j:03d}.svg", h, w)
            #chalk.backend.cairo.prims_to_file(out, f"test.{j:03d}.png", h, w)
            chalk.transform.set_jax_mode(True)
        print(score)
opt(matrix, graph)

# score, out = graph(matrix)
# out = out.unflatten(no_map=True)[0]
# print(score)
# print("concat")

# print("LINES")
# #out.render("test.png", 200)

# import chalk.backend.cairo
# out, h, w = chalk.core.layout_primitives(out, 200)
# print("OUT")
# out = jax.tree.map(np.asarray, out)
# chalk.backend.cairo.prims_to_file(out, "test.png", h, w)
