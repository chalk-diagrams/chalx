import { np, type JaxArray } from "./jax.js";
import {
  EPSILON,
  type Affine,
  type ColorLike,
  type Point,
  type ScalarLike,
  type Vec2Like,
  type Vector,
  add,
  angleOf,
  applyToVector,
  asColor,
  asVec2,
  color,
  compose,
  identity,
  length,
  midpoint,
  mul,
  negate,
  normalize,
  perpendicular,
  point,
  polar,
  rotation,
  rotationTurns,
  scale,
  scalar,
  shearX,
  shearY,
  sub,
  translation,
  vec2,
} from "./transform.js";

export interface ArcSegment {
  readonly transform: Affine;
  readonly startAngle: JaxArray;
  readonly deltaAngle: JaxArray;
  readonly endOffset: Vector;
}

export interface Trail {
  readonly segments: readonly ArcSegment[];
  readonly closed: boolean;
}

export interface ShapeStyle {
  readonly fill: JaxArray;
  readonly stroke: JaxArray;
  readonly lineWidth: JaxArray;
  readonly fillOpacity: JaxArray;
  readonly strokeOpacity: JaxArray;
}

export interface Shape {
  readonly trail: Trail;
  readonly location: Point;
  readonly transform: Affine;
  readonly style: ShapeStyle;
}

export interface Scene {
  readonly shapes: readonly Shape[];
}

export interface ResolvedArc {
  readonly transform: Affine;
  readonly startAngle: JaxArray;
  readonly deltaAngle: JaxArray;
}

const DEFAULT_STYLE: ShapeStyle = {
  fill: color(0, 0, 0, 1),
  stroke: color(0, 0, 0, 1),
  lineWidth: scalar(1),
  fillOpacity: scalar(1),
  strokeOpacity: scalar(1),
};

function linearPart(transform: Affine): Affine {
  return np.stack(
    [
      np.stack([transform.slice(0, 0), transform.slice(0, 1), scalar(0)], 0),
      np.stack([transform.slice(1, 0), transform.slice(1, 1), scalar(0)], 0),
      np.stack([scalar(0), scalar(0), scalar(1)], 0),
    ],
    0,
  );
}

export function emptyTrail(): Trail {
  return { segments: [], closed: false };
}

export function emptyScene(): Scene {
  return { shapes: [] };
}

export function trail(...segments: ArcSegment[]): Trail {
  return { segments, closed: false };
}

export function scene(...shapes: Shape[]): Scene {
  return { shapes };
}

export function concatScenes(...scenes: Scene[]): Scene {
  return { shapes: scenes.flatMap((entry) => [...entry.shapes]) };
}

export function concatTrails(...trails: Trail[]): Trail {
  return {
    segments: trails.flatMap((entry) => [...entry.segments]),
    closed: trails.some((entry) => entry.closed),
  };
}

export function closeTrail(input: Trail): Trail {
  return { ...input, closed: true };
}

export function arcSegment(offset: Vec2Like, bendHeight: ScalarLike): ArcSegment {
  const q = asVec2(offset);
  const d = length(q);
  const safeH = scalar(bendHeight);
  const h = np.where(
    np.less(np.absolute(safeH.ref), EPSILON),
    np.copysign(scalar(EPSILON), safeH),
    safeH,
  );
  const d2 = d.ref.mul(d.ref);
  const h2 = h.ref.mul(h.ref);
  const theta = np.arccos(
    np.clip(d2.ref.sub(h2.ref.mul(4)).div(d2.ref.add(h2.mul(4)).add(EPSILON)), -1, 1),
  );
  const radius = d.ref.div(np.sin(theta.ref).mul(2).add(EPSILON));
  const bendsLeft = np.greater(h.ref, 0);
  const phi = np.where(bendsLeft.ref, Math.PI / 2, -Math.PI / 2);
  const dy = np.where(bendsLeft.ref, radius.ref.sub(h.ref), h.sub(radius.ref));
  const flip = np.where(bendsLeft, 1, -1);
  const diffAngle = angleOf(q.ref);
  const transform = compose(
    rotation(np.negative(diffAngle.ref)),
    translation(d.ref.mul(0.5), dy),
    rotation(phi),
    scale(radius.ref, radius),
  );
  return {
    transform,
    startAngle: flip.ref.mul(np.negative(theta.ref)),
    deltaAngle: flip.mul(theta.ref.mul(2)),
    endOffset: q,
  };
}

export function segment(offset: Vec2Like): ArcSegment {
  return arcSegment(offset, EPSILON);
}

export function arcBetween(
  from: Vec2Like,
  to: Vec2Like,
  bendHeight: ScalarLike,
): ArcSegment {
  const diff = sub(to, from);
  const base = arcSegment(diff, bendHeight);
  return {
    ...base,
    transform: compose(translation(asVec2(from)), base.transform),
  };
}

export function trailFromOffsets(offsets: Vec2Like[], closed = false): Trail {
  return {
    segments: offsets.map((offset) => segment(offset)),
    closed,
  };
}

export function trailFromPoints(points: Vec2Like[], closed = false): Trail {
  if (points.length < 2) {
    return emptyTrail();
  }
  const offsets: Vector[] = [];
  for (let i = 0; i < points.length - 1; i += 1) {
    offsets.push(sub(points[i + 1]!, points[i]!));
  }
  if (closed) {
    offsets.push(sub(points[0]!, points[points.length - 1]!));
  }
  return trailFromOffsets(offsets, closed);
}

export function transformTrail(input: Trail, affine: Affine): Trail {
  const linear = linearPart(affine);
  return {
    segments: input.segments.map((segmentEntry) => ({
      ...segmentEntry,
      transform: compose(linear.ref, segmentEntry.transform),
      endOffset: applyToVector(linear.ref, segmentEntry.endOffset),
    })),
    closed: input.closed,
  };
}

export function translateTrail(input: Trail, _x: ScalarLike, _y: ScalarLike): Trail {
  return input;
}

export function scaleTrail(input: Trail, x: ScalarLike, y?: ScalarLike): Trail {
  return transformTrail(input, scale(x, y));
}

export function rotateTrail(input: Trail, thetaRadians: ScalarLike): Trail {
  return transformTrail(input, rotation(thetaRadians));
}

export function rotateTrailBy(input: Trail, turns: ScalarLike): Trail {
  return transformTrail(input, rotationTurns(turns));
}

export function shearTrailX(input: Trail, lambda: ScalarLike): Trail {
  return transformTrail(input, shearX(lambda));
}

export function shearTrailY(input: Trail, lambda: ScalarLike): Trail {
  return transformTrail(input, shearY(lambda));
}

export function stroke(input: Trail, style: Partial<ShapeStyle> = {}): Shape {
  return {
    trail: input,
    location: point(0, 0),
    transform: identity(),
    style: { ...DEFAULT_STYLE, ...style },
  };
}

export function locate(shape: Shape, location: Vec2Like): Shape {
  return { ...shape, location: asVec2(location) };
}

export function transformShape(shape: Shape, affine: Affine): Shape {
  return { ...shape, transform: compose(affine, shape.transform) };
}

export function translateShape(shape: Shape, x: ScalarLike, y: ScalarLike): Shape {
  return transformShape(shape, translation(x, y));
}

export function scaleShape(shape: Shape, x: ScalarLike, y?: ScalarLike): Shape {
  return transformShape(shape, scale(x, y));
}

export function rotateShape(shape: Shape, thetaRadians: ScalarLike): Shape {
  return transformShape(shape, rotation(thetaRadians));
}

export function rotateShapeBy(shape: Shape, turns: ScalarLike): Shape {
  return transformShape(shape, rotationTurns(turns));
}

export function lineWidth(shape: Shape, value: ScalarLike): Shape {
  return { ...shape, style: { ...shape.style, lineWidth: scalar(value) } };
}

export function fillColor(shape: Shape, value: ColorLike): Shape {
  return { ...shape, style: { ...shape.style, fill: asColor(value) } };
}

export function strokeColor(shape: Shape, value: ColorLike): Shape {
  return { ...shape, style: { ...shape.style, stroke: asColor(value) } };
}

export function fillOpacity(shape: Shape, value: ScalarLike): Shape {
  return { ...shape, style: { ...shape.style, fillOpacity: scalar(value) } };
}

export function strokeOpacity(shape: Shape, value: ScalarLike): Shape {
  return { ...shape, style: { ...shape.style, strokeOpacity: scalar(value) } };
}

export function addShape(sceneValue: Scene, shapeValue: Shape): Scene {
  return { shapes: [...sceneValue.shapes, shapeValue] };
}

export function resolvedArcs(shape: Shape): ResolvedArc[] {
  const arcs: ResolvedArc[] = [];
  let cursor = shape.location;
  for (const segmentEntry of shape.trail.segments) {
    arcs.push({
      transform: compose(shape.transform.ref, translation(cursor.ref), segmentEntry.transform),
      startAngle: segmentEntry.startAngle,
      deltaAngle: segmentEntry.deltaAngle,
    });
    cursor = add(cursor.ref, segmentEntry.endOffset);
  }
  return arcs;
}

export function trailEndpoint(input: Trail): Vector {
  let total = vec2(0, 0);
  for (const entry of input.segments) {
    total = add(total.ref, entry.endOffset);
  }
  return total;
}

export function hrule(lengthValue: ScalarLike): Trail {
  return trail(segment(vec2(lengthValue, 0)));
}

export function vrule(lengthValue: ScalarLike): Trail {
  return trail(segment(vec2(0, lengthValue)));
}

export function regularPolygonTrail(sides: number, sideLength: ScalarLike = 1): Trail {
  const edge = hrule(sideLength);
  const pieces: Trail[] = [];
  for (let i = 0; i < sides; i += 1) {
    pieces.push(rotateTrailBy(edge, i / sides));
  }
  return closeTrail(concatTrails(...pieces));
}

export function circleTrail(radius: ScalarLike = 1): Trail {
  const pieces: Trail[] = [];
  for (let i = 0; i < 4; i += 1) {
    pieces.push(
      rotateTrailBy(
        trail({
          transform: compose(translation(-1, 0), scale(radius, radius)),
          startAngle: scalar(0),
          deltaAngle: scalar(Math.PI / 2),
          endOffset: vec2(1, 1),
        }),
        i / 4,
      ),
    );
  }
  return closeTrail(concatTrails(...pieces));
}

export function path(points: Vec2Like[], closed = false): Shape {
  if (points.length === 0) {
    return stroke(emptyTrail());
  }
  return locate(stroke(trailFromPoints(points, closed)), points[0]!);
}

export function rectangle(width: ScalarLike, height: ScalarLike): Shape {
  const w = scalar(width);
  const h = scalar(height);
  const p0 = point(0, 0);
  const p1 = point(w.ref, 0);
  const p2 = point(w.ref, h.ref);
  const p3 = point(0, h);
  return stroke(trailFromPoints([p0, p1, p2, p3], true));
}

export function square(side: ScalarLike): Shape {
  return rectangle(side, side);
}

export function circle(radius: ScalarLike): Shape {
  const r = scalar(radius);
  const q0 = vec2(0, r.ref.mul(2));
  const q1 = vec2(0, np.negative(r.ref.mul(2)));
  const top = arcSegment(q0, r);
  const bottom = arcSegment(q1, r);
  return locate(stroke(closeTrail(trail(top, bottom))), point(r.ref, np.negative(r.ref)));
}

export function circleAt(center: Vec2Like, radius: ScalarLike): Shape {
  return translateShape(circle(radius), asVec2(center).slice(0), asVec2(center).slice(1));
}

export function triangle(width: ScalarLike = 1): Shape {
  return stroke(regularPolygonTrail(3, width));
}

export function centerOfPoints(points: readonly JaxArray[]): Vector {
  if (points.length === 0) {
    return vec2(0, 0);
  }
  let total = vec2(0, 0);
  for (const p of points) {
    total = add(total.ref, p);
  }
  return total.div(points.length);
}

export function centerTrail(input: Trail): Trail {
  const pts: JaxArray[] = [vec2(0, 0)];
  let cursor = vec2(0, 0);
  for (const segEntry of input.segments) {
    cursor = add(cursor.ref, segEntry.endOffset);
    pts.push(cursor);
  }
  const center = centerOfPoints(pts);
  return {
    segments: input.segments.map((segEntry, index) => {
      if (index === 0) {
        return {
          ...segEntry,
          transform: compose(translation(negate(center.ref)), segEntry.transform),
        };
      }
      return segEntry;
    }),
    closed: input.closed,
  };
}

export function circleFromArc(offset: Vec2Like, bendHeight: ScalarLike): Shape {
  return stroke(trail(arcSegment(offset, bendHeight)));
}

export function alignTopLeft(shape: Shape): Shape {
  return shape;
}

export function centered(shape: Shape): Shape {
  return shape;
}

export function makeStyle(style: Partial<ShapeStyle>): ShapeStyle {
  return { ...DEFAULT_STYLE, ...style };
}

export function aroundCircle(radius: ScalarLike, thetaRadians: ScalarLike): Vector {
  return add(midpoint(vec2(0, 0), vec2(0, 0)), polar(thetaRadians, radius));
}
