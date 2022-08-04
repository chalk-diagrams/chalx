from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from planar.py import Ray

import chalk.transform as tx
from chalk.envelope import Envelope
from chalk.trace import Trace
from chalk.transform import P2, V2, unit_x, unit_y, to_radians, from_radians

SignedDistance = float


@dataclass
class Segment(tx.Transformable):
    p: P2
    q: P2

    def get_trace(self) -> Trace:
        def f(point: P2, direction: V2) -> List[float]:
            ray = Ray(point, direction)
            inter = sorted(line_segment(ray, self))
            return inter

        return Trace(f)

    def get_envelope(self) -> Envelope:
        def f(d: V2) -> SignedDistance:
            x: float = max(d.dot(self.q), d.dot(self.p))
            return x

        return Envelope(f)

    def to_ray(self) -> "Ray":
        return Ray(self.p, self.q - self.p)

    @property
    def length(self) -> Any:
        return (self.q - self.p).length

    def apply_transform(self, t: tx.Affine) -> Segment:
        return Segment(tx.apply_affine(t, self.p), tx.apply_affine(t, self.q))

    def render_path(self, ctx):
        ctx.line_to(self.q.x, self.q.y)
    
    def render_svg_path(self) -> str:
        return f"L {self.q.x} {self.q.y}"

    def render_tikz_path(self, pts, pylatex) -> None:
        pts.append("--")
        pts.append(pylatex.TikZCoordinate(self.q.x, self.q.y))



def is_in_mod_360(x: float, a: float, b: float) -> bool:
    """Checks if x ∈ [a, b] mod 360. See the following link for an
    explanation:
    https://fgiesen.wordpress.com/2015/09/24/intervals-in-modular-arithmetic/
    """
    return (x - a) % 360 <= (b - a) % 360


@dataclass
class ArcSegment(tx.Transformable):
    angle0: float
    angle1: float
    t: tx.Transform = tx.Affine.identity()

    @property
    def _start(self) -> P2:
        return P2(math.cos(to_radians(self.angle0)), math.sin(to_radians(self.angle0)))

    @property
    def _end(self) -> P2:
        return P2(math.cos(to_radians(self.angle1)), math.sin(to_radians(self.angle1)))


    @staticmethod
    def arc_between(p: P2, q: P2, height: float):
        h = abs(height)
        d = (q - p).length
        # Determine the arc's angle θ and its radius r
        θ = math.acos((d**2 - 4.0 * h**2) / (d**2 + 4.0 * h**2))
        r = d / (2 * math.sin(θ))

        if height > 0:
            # bend left
            φ = -math.pi / 2
            dy = r - h
        else:
            # bend right
            φ = +math.pi / 2
            dy = h - r

        diff = q - p
        r = (
            ArcSegment(-from_radians(θ), from_radians(θ))
            .scale(r)
            .rotate_rad(math.pi / 2)
            .translate(d/2, dy))
        if height > 0:
            r  = r.reflect_y()
        r = (r.rotate(-diff.angle)
            .translate_by(p)
        )
        return r
    
    
    @staticmethod
    def ellipse_between(p: P2, q: P2, height: float):
        diff = q - p
        r = (
            ArcSegment(180, 360)
            .scale_y(height)
            .translate_by(unit_x)
            .scale_x(diff.length / 2)
            .rotate(-diff.angle)
            .translate_by(p)
        )
        return r

    @property
    def p(self):
        return tx.apply_affine(self.t, self._start)

    @property
    def q(self):
        return tx.apply_affine(self.t, self._end)

    def apply_transform(self, t: tx.Affine) -> ArcSegment:
        return ArcSegment(self.angle0, self.angle1, t * self.t)

    def get_trace(self) -> Trace:
        angle0_deg = self.angle0
        angle1_deg = self.angle1

        def f(p: P2, v: V2) -> List[SignedDistance]:
            ray = Ray(p, v)
            # Same as circle but check that angle is in arc.
            return sorted(
                [
                    d / v.length
                    for d in ray_circle_intersection(ray, 1)
                    if is_in_mod_360(
                        ((d * v) + self._start).angle, angle0_deg, angle1_deg
                    )
                ]
            )

        return Trace(f).apply_transform(self.t)

    def get_envelope(self) -> Envelope:
        angle0_deg = self.angle0
        angle1_deg = self.angle1

        v1 = V2.polar(angle0_deg, 1)
        v2 = V2.polar(angle1_deg, 1)

        def wrapped(d: V2) -> SignedDistance:
            is_circle = abs(angle0_deg - angle1_deg) >= 360
            if is_circle or is_in_mod_360(d.angle, angle0_deg, angle1_deg):
                # Case 1: P2 at arc
                return 1 / d.length  # type: ignore
            else:
                # Case 2: P2 outside of arc
                x: float = max(d.dot(v1), d.dot(v2))
                return x

        return Envelope(wrapped).apply_transform(self.t)

    def render_path(self, ctx):
        end_point = tx.apply_affine(self.t, self._end)
        t2 = tx.remove_translation(self.t)
        r_x = tx.apply_affine(t2, unit_x)
        r_y = tx.apply_affine(t2, unit_y)
        rot = r_x.angle
        start = 180 + rot
        x, y = self.p.x - math.cos(to_radians(start)), self.p.y - math.sin(to_radians(start))
        ctx.new_sub_path()
        ctx.save()
        ctx.translate(x, y)
        ctx.scale(r_x.length, r_y.length)
        ctx.arc(0., 0., 1., start, 180 + rot + self.angle)
        ctx.restore()


    def render_svg_path(self) -> str:
        end_point = tx.apply_affine(self.t, self._end)
        large = 0
        t2 = tx.remove_translation(self.t)
        r_x = tx.apply_affine(t2, unit_x)
        r_y = tx.apply_affine(t2, unit_y)
        rot = r_x.angle
        return f"A {r_x.length} {r_y.length} {rot} {large} 0 {end_point.x} {end_point.y}"

    def render_tikz_path(self, pts, pylatex) -> None:    
        t2 = tx.remove_translation(self.t)
        r_x = tx.apply_affine(t2, unit_x)
        r_y = tx.apply_affine(t2, unit_y)
        rot = r_x.angle
        print(r_x, r_y, self.angle0, self.angle1)
        pts._arg_list.append(pylatex.TikZUserPath(f"{{[rotate={rot}] arc[start angle={self.angle0}, end angle={self.angle1}, x radius={r_x.length}, y radius ={r_y.length}]}}"))



def ray_ray_intersection(
    ray1: Ray, ray2: Ray
) -> Optional[Tuple[float, float]]:
    """Given two rays

    ray₁ = λ t . p₁ + t v₁
    ray₂ = λ t . p₂ + t v₂

    the function returns the parameters t₁ and t₂ at which the two rays meet,
    that is:

    ray₁ t₁ = ray₂ t₂

    """
    u = ray2.anchor - ray1.anchor
    x1 = ray1.direction.cross(ray2.direction)
    x2 = u.cross(ray1.direction)
    x3 = u.cross(ray2.direction)
    if x1 == 0 and x2 != 0:
        # parallel
        return None
    else:
        # intersecting or collinear
        return x3 / x1, x2 / x1


def line_segment(ray: Ray, segment: Segment) -> List[float]:
    """Given a ray and a segment, return the parameter `t` for which the ray
    meets the segment, that is:

    ray t₁ = segment.to_ray t₂, with t₂ ∈ [0, segment.length]

    Note: We need to consider the segment's length separately since `Ray`
    normalizes the direction to unit and hences looses this information. The
    length is important to determine whether the intersection point falls
    within the given segment.

    See also: https://github.com/danoneata/chalk/issues/91

    """
    ray_s = segment.to_ray()
    t = ray_ray_intersection(ray, ray_s)
    if not t:
        return []
    else:
        t1, t2 = t
        # the intersection point is given by any of the two expressions:
        # ray.anchor   + t1 * ray.direction
        # ray_s.anchor + t2 * ray_s.direction
        if 0 <= t2 <= segment.length:
            # intersection point is in segment
            return [t1]
        else:
            # intersection point outside
            return []


def ray_circle_intersection(ray: Ray, circle_radius: float) -> List[float]:
    """Given a ray and a circle centered at the origin, return the parameter t
    where the ray meets the circle, that is:

    ray t = circle θ

    The above equation is solved as follows:

    x + t v_x = r sin θ
    y + t v_y = r cos θ

    By squaring the equations and adding them we get

    (x + t v_x)² + (y + t v_y)² = r²,

    which is equivalent to the following equation:

    (v_x² + v_y²) t² + 2 (x v_x + y v_y) t + (x² + y² - r²) = 0

    This is a quadratic equation, whose solutions are well known.

    """
    p = ray.anchor

    a = ray.direction.length2
    b = 2 * (p.dot(ray.direction))
    c = p.length2 - circle_radius**2

    Δ = b**2 - 4 * a * c
    eps = 1e-6  # rounding error tolerance

    if Δ < -eps:
        # no intersection
        return []
    elif -eps <= Δ < eps:
        # tangent
        return [-b / (2 * a)]
    else:
        # the ray intersects at two points
        return [
            (-b - math.sqrt(Δ)) / (2 * a),
            (-b + math.sqrt(Δ)) / (2 * a),
        ]
