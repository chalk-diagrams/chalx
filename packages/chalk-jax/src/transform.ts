import { np, type JaxArray } from "./jax.js";

export type ScalarLike = number | JaxArray;
export type Vec2Like = [ScalarLike, ScalarLike] | JaxArray;
export type ColorLike =
  | string
  | [ScalarLike, ScalarLike, ScalarLike]
  | [ScalarLike, ScalarLike, ScalarLike, ScalarLike]
  | JaxArray;

export type Affine = JaxArray;
export type Point = JaxArray;
export type Vector = JaxArray;

export const TAU = Math.PI * 2;
export const EPSILON = 1e-6;

const NAMED_COLORS: Record<string, [number, number, number]> = {
  black: [0, 0, 0],
  white: [1, 1, 1],
  red: [1, 0, 0],
  green: [0, 0.5, 0],
  blue: [0, 0, 1],
  orange: [1, 0.647, 0],
  grey: [0.5, 0.5, 0.5],
  gray: [0.5, 0.5, 0.5],
  yellow: [1, 1, 0],
  purple: [0.5, 0, 0.5],
  pink: [1, 0.753, 0.796],
  papaya: [1, 0.592, 0],
};

export function isJaxArray(value: unknown): value is JaxArray {
  return (
    typeof value === "object" &&
    value !== null &&
    "shape" in value &&
    "dispose" in value &&
    "ref" in value
  );
}

export function scalar(value: ScalarLike): JaxArray {
  return typeof value === "number" ? np.array(value) : value;
}

export function matrix3(
  a00: ScalarLike,
  a01: ScalarLike,
  a02: ScalarLike,
  a10: ScalarLike,
  a11: ScalarLike,
  a12: ScalarLike,
  a20: ScalarLike,
  a21: ScalarLike,
  a22: ScalarLike,
): JaxArray {
  return np.stack(
    [
      np.stack([scalar(a00), scalar(a01), scalar(a02)], 0),
      np.stack([scalar(a10), scalar(a11), scalar(a12)], 0),
      np.stack([scalar(a20), scalar(a21), scalar(a22)], 0),
    ],
    0,
  );
}

export function vec2(x: ScalarLike, y: ScalarLike): Vector {
  return np.stack([scalar(x), scalar(y)], 0);
}

export function point(x: ScalarLike, y: ScalarLike): Point {
  return vec2(x, y);
}

export function asVec2(value: Vec2Like): Vector {
  if (isJaxArray(value)) {
    return value;
  }
  return vec2(value[0], value[1]);
}

export function xOf(v: Vec2Like): JaxArray {
  return asVec2(v).ref.slice(0);
}

export function yOf(v: Vec2Like): JaxArray {
  return asVec2(v).ref.slice(1);
}

export function identity(): Affine {
  return np.eye(3);
}

export function translation(x: ScalarLike, y?: ScalarLike): Affine {
  const offset = y === undefined ? asVec2(x as Vec2Like) : vec2(x, y);
  const ox = xOf(offset.ref);
  const oy = yOf(offset.ref);
  return matrix3(1, 0, ox, 0, 1, oy, 0, 0, 1);
}

export function scale(x: ScalarLike, y?: ScalarLike): Affine {
  const sx = scalar(x);
  const sy = y === undefined ? sx.ref : scalar(y);
  return matrix3(sx, 0, 0, 0, sy, 0, 0, 0, 1);
}

export function rotation(thetaRadians: ScalarLike): Affine {
  const theta = scalar(thetaRadians);
  const c = np.cos(theta.ref);
  const s = np.sin(theta);
  return matrix3(c.ref, np.negative(s.ref), 0, s, c, 0, 0, 0, 1);
}

export function rotationTurns(turns: ScalarLike): Affine {
  return rotation(scalar(turns).mul(TAU));
}

export function shearX(lambda: ScalarLike): Affine {
  const l = scalar(lambda);
  return matrix3(1, l, 0, 0, 1, 0, 0, 0, 1);
}

export function shearY(lambda: ScalarLike): Affine {
  const l = scalar(lambda);
  return matrix3(1, 0, 0, l, 1, 0, 0, 0, 1);
}

export function compose(...transforms: Affine[]): Affine {
  if (transforms.length === 0) {
    return identity();
  }
  let out = transforms[0]!.ref;
  for (let i = 1; i < transforms.length; i += 1) {
    out = np.matmul(out, transforms[i]!.ref);
  }
  return out;
}

export function inverse(transform: Affine): Affine {
  return np.linalg.inv(transform);
}

export function homogeneous(v: Vec2Like, w: ScalarLike): JaxArray {
  return np.stack([xOf(v), yOf(v), scalar(w)], 0);
}

export function fromHomogeneous(v: JaxArray): JaxArray {
  const w = v.ref.slice(2);
  const x = v.ref.slice(0);
  const y = v.slice(1);
  return np.stack([x.div(w.ref), y.div(w)], 0);
}

export function applyToPoint(transform: Affine, p: Vec2Like): Point {
  return fromHomogeneous(np.matmul(transform.ref, homogeneous(p, 1)));
}

export function applyToVector(transform: Affine, v: Vec2Like): Vector {
  return fromHomogeneous(np.matmul(transform.ref, homogeneous(v, 0)));
}

export function add(a: Vec2Like, b: Vec2Like): Vector {
  const av = asVec2(a);
  const bv = asVec2(b);
  return av.ref.add(bv.ref);
}

export function sub(a: Vec2Like, b: Vec2Like): Vector {
  const av = asVec2(a);
  const bv = asVec2(b);
  return av.ref.sub(bv.ref);
}

export function mul(a: Vec2Like, amount: ScalarLike): Vector {
  const av = asVec2(a);
  const amt = scalar(amount);
  return av.ref.mul(amt.ref);
}

export function dot(a: Vec2Like, b: Vec2Like): JaxArray {
  const av = asVec2(a);
  const bv = asVec2(b);
  return av.ref.mul(bv.ref).sum();
}

export function cross2d(a: Vec2Like, b: Vec2Like): JaxArray {
  const av = asVec2(a);
  const bv = asVec2(b);
  const avx = av.ref.slice(0);
  const avy = av.slice(1);
  const bvy = bv.ref.slice(1);
  const bvx = bv.slice(0);
  return avx.mul(bvy).sub(avy.mul(bvx));
}

export function lengthSquared(v: Vec2Like): JaxArray {
  return dot(v, v);
}

export function length(v: Vec2Like): JaxArray {
  return np.sqrt(lengthSquared(v).add(EPSILON));
}

export function normalize(v: Vec2Like): Vector {
  const vv = asVec2(v);
  const lv = length(vv.ref);
  return vv.ref.div(lv.ref);
}

export function perpendicular(v: Vec2Like): Vector {
  const vv = asVec2(v);
  return np.stack([np.negative(vv.ref.slice(1)), vv.ref.slice(0)], 0);
}

export function midpoint(a: Vec2Like, b: Vec2Like): Vector {
  return add(a, b).mul(0.5);
}

export function polar(thetaRadians: ScalarLike, radius: ScalarLike = 1): Vector {
  const theta = scalar(thetaRadians);
  const r = scalar(radius);
  return np.stack([np.cos(theta.ref).mul(r.ref), np.sin(theta).mul(r)], 0);
}

export function angleOf(v: Vec2Like): JaxArray {
  return np.atan2(yOf(v), xOf(v));
}

export function wrapAngle(theta: ScalarLike): JaxArray {
  const t = scalar(theta);
  const shifted = t.add(Math.PI);
  const wrapped = shifted.mod(TAU);
  return wrapped.sub(Math.PI);
}

export function clamp(x: ScalarLike, low: ScalarLike, high: ScalarLike): JaxArray {
  return np.clip(scalar(x), scalar(low), scalar(high));
}

export function color(
  r: ScalarLike,
  g: ScalarLike,
  b: ScalarLike,
  a: ScalarLike = 1,
): JaxArray {
  return np.stack([scalar(r), scalar(g), scalar(b), scalar(a)], 0);
}

export function asColor(value: ColorLike): JaxArray {
  if (typeof value === "string") {
    if (value.startsWith("#")) {
      const hex = value.slice(1);
      const normalized =
        hex.length === 3
          ? hex
              .split("")
              .map((ch) => ch + ch)
              .join("")
          : hex;
      if (normalized.length !== 6) {
        throw new Error(`Unsupported hex color: ${value}`);
      }
      const rgb = [
        Number.parseInt(normalized.slice(0, 2), 16) / 255,
        Number.parseInt(normalized.slice(2, 4), 16) / 255,
        Number.parseInt(normalized.slice(4, 6), 16) / 255,
      ] as const;
      return color(rgb[0], rgb[1], rgb[2], 1);
    }
    const named = NAMED_COLORS[value.toLowerCase()];
    if (!named) {
      throw new Error(`Unsupported named color: ${value}`);
    }
    return color(named[0], named[1], named[2], 1);
  }
  if (isJaxArray(value)) {
    return value;
  }
  if (value.length === 3) {
    return color(value[0], value[1], value[2], 1);
  }
  return color(value[0], value[1], value[2], value[3]);
}

export function negate(v: Vec2Like): Vector {
  return np.negative(asVec2(v).ref);
}
