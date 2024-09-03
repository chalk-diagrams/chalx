from chalk import (
    P2,
    V2,
    Trail,
    arc_between,
    circle,
    unit_x,
    unit_y,
    arc_seg,
    rectangle,
    arc_seg_angle,
    vcat,
    cat,
)
from colour import Color

# d = Path([ArcSegment.arc_between(P2(0, 0), P2(2, 0), 1), ArcSegment.arc_between(P2(2, 0), P2(0, 0), 1)]).stroke()
# d = Primitive.from_shape(Path([Segment(P2(0, 0), P2(1, 1))]))
# d = Primitive.from_shape(Path([Segment(P2(0, 0), P2(1, 1)),
#                                Segment(P2(1, 1), P2(1, 2))]))

# d = Path([ArcSegment.arc_between(P2(0, 0), P2(2, 0), 1)]).stroke()
r = 0.2
b = 1 - r
rad = 0.2 / 4


a = arc_between(P2(-1, 0), P2(0, 1.0), 1)
# print(a.p, a.q, a.r_x, a.r_y, a.angle, a.tangle)

d = []

d += [circle(1).show_origin()]

# d += [(Trail.square().centered().to_path() + Trail.square().scale(V2(0.5, 0.5)).reverse().centered().to_path()).stroke()]

d += [
    (
        Trail.circle().scale(V2(2, 2)).centered().to_path()
        + Trail.circle(False).centered().to_path()
    ).stroke()
]

d += [arc_seg(2 * unit_x, 1).stroke().show_origin()]

d += [arc_seg(2 * unit_x, -1).stroke().show_origin()]

d += [arc_seg(unit_x + unit_y, 1).stroke().show_origin()]

d += [arc_seg(unit_x + unit_y, 1).scale(-1).stroke().show_origin()]

d += [arc_seg(unit_x + unit_y, -1).stroke().show_origin()]

d += [arc_seg(2 * unit_x, 1).scale_y(0.5).rotate(45).stroke().show_origin()]


d += [rectangle(1, 5, 0.5), rectangle(5, 1, 0.25), rectangle(1, 1, 1)]


d += [Trail.square().stroke().center_xy().show_origin()]

d += [
    cat(
        [
            arc_seg_angle(180, 135).stroke().show_origin(),
            arc_seg_angle(180, 135).scale_x(-1).stroke().show_origin(),
            arc_seg_angle(180, 135).stroke().scale_x(-1).show_origin(),
        ],
        unit_x,
        sep=0.5,
    )
]

d = vcat(d, sep=1.0)

d = d.fill_color(Color("blue"))

d.render_svg("examples/output/path.svg", height=300)
d.render("examples/output/path.png", height=300)
