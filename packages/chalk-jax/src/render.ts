import { np, type JaxArray } from "./jax.js";
import {
  type Affine,
  type Point,
  type ScalarLike,
  TAU,
  inverse,
  scale,
  scalar,
  vec2,
} from "./transform.js";
import {
  type ResolvedArc,
  type Scene,
  type Shape,
  resolvedArcs,
  transformShape,
  translateShape,
} from "./geometry.js";

const HUGE = 1e9;

export interface TraceResult {
  readonly splits: JaxArray;
  readonly mask: JaxArray;
}

export interface RenderOptions {
  readonly width: number;
  readonly height: number;
  readonly background?: JaxArray;
  readonly fillSoftness?: ScalarLike;
  readonly strokeSoftness?: ScalarLike;
  readonly samplesPerArc?: number;
}

export interface Bounds {
  readonly minX: JaxArray;
  readonly minY: JaxArray;
  readonly maxX: JaxArray;
  readonly maxY: JaxArray;
}

function positiveMod(value: JaxArray, modulus: number): JaxArray {
  const wrapped = value.mod(modulus);
  return wrapped.add(modulus).mod(modulus);
}

function sigmoid(value: JaxArray): JaxArray {
  return scalar(1).div(np.exp(np.negative(value)).add(1));
}

function zerosTrace(count: number): TraceResult {
  return {
    splits: np.zeros([count, 0]),
    mask: np.zeros([count, 0], { dtype: np.bool }),
  };
}

function rowPoints(points: Point[] | JaxArray): JaxArray {
  if (Array.isArray(points)) {
    if (points.length === 0) {
      return np.zeros([0, 2]);
    }
    return np.stack(points, 0);
  }
  if (points.ndim === 1) {
    return points.reshape([1, 2]);
  }
  return points;
}

function applyRows(transform: Affine, pointsValue: JaxArray, w: 0 | 1): JaxArray {
  const count = pointsValue.shape[0] ?? 0;
  const tail = np.full([count, 1], w);
  const hom = np.concatenate([pointsValue, tail], 1);
  const out = np.matmul(hom, transform.transpose());
  if (w === 0) {
    return out.slice([], [0, 2]);
  }
  const ww = out.slice([], [2]).add(1e-6);
  return out.slice([], [0, 2]).div(ww);
}

function arcAngleMask(arc: ResolvedArc, localHitPoints: JaxArray): JaxArray {
  const angles = np.atan2(localHitPoints.slice([], 1), localHitPoints.slice([], 0));
  const end = arc.startAngle.add(arc.deltaAngle);
  const low = np.minimum(arc.startAngle, end.ref);
  const high = np.maximum(arc.startAngle.ref, end.ref);
  const span = positiveMod(high.ref.sub(low.ref), TAU);
  const rel = positiveMod(angles.sub(low), TAU);
  return np.lessEqual(rel, span.add(1e-6));
}

function traceArcBatch(arc: ResolvedArc, origins: JaxArray, direction: JaxArray): TraceResult {
  const count = origins.shape[0] ?? 0;
  const inv = inverse(arc.transform);
  const localOrigins = applyRows(inv.ref, origins, 1);
  const dirRows = np.broadcastTo(direction.ref.reshape([1, 2]), [count, 2]);
  const localDirection = applyRows(inv, dirRows, 0);

  const px = localOrigins.slice([], 0);
  const py = localOrigins.slice([], 1);
  const dx = localDirection.slice([], 0);
  const dy = localDirection.slice([], 1);

  const a = dx.ref.mul(dx.ref).add(dy.ref.mul(dy.ref));
  const b = px.ref.mul(dx.ref).add(py.ref.mul(dy.ref)).mul(2);
  const c = px.ref.mul(px.ref).add(py.ref.mul(py.ref)).sub(1);
  const disc = b.ref.mul(b.ref).sub(a.ref.mul(c.ref).mul(4));
  const validDisc = np.greaterEqual(disc.ref, 0);
  const sqrtDisc = np.sqrt(np.maximum(disc, 0));
  const denom = a.ref.mul(2).add(1e-6);

  const t1 = np.negative(b.ref).sub(sqrtDisc.ref).div(denom.ref);
  const t2 = np.negative(b).add(sqrtDisc.ref).div(denom);

  const hit1 = localOrigins.ref.add(localDirection.ref.mul(t1.ref.reshape([count, 1])));
  const hit2 = localOrigins.ref.add(localDirection.ref.mul(t2.ref.reshape([count, 1])));

  const m1 = np.logicalAnd(validDisc.ref, arcAngleMask(arc, hit1));
  const m2 = np.logicalAnd(validDisc, arcAngleMask(arc, hit2));

  return {
    splits: np.stack([t1, t2], 1),
    mask: np.stack([m1, m2], 1),
  };
}

export function getTrace(shape: Shape) {
  const arcs = resolvedArcs(shape);
  return (originsValue: Point[] | JaxArray, directionValue: JaxArray): TraceResult => {
    const origins = rowPoints(originsValue);
    if (arcs.length === 0) {
      return zerosTrace(origins.shape[0] ?? 0);
    }
    const distances: JaxArray[] = [];
    const masks: JaxArray[] = [];
    for (const arc of arcs) {
      const tr = traceArcBatch(arc, origins, directionValue);
      distances.push(tr.splits.slice([], 0), tr.splits.slice([], 1));
      masks.push(tr.mask.slice([], 0), tr.mask.slice([], 1));
    }
    const dist = np.stack(distances, 1);
    const mask = np.stack(masks, 1);
    const padded = np.where(mask.ref, dist.ref, HUGE);
    const sorted = np.sort(padded, 1);
    const sortedMask = np.less(sorted.ref, HUGE / 2);
    return { splits: sorted, mask: sortedMask };
  };
}

export function renderLineSoft(
  splits: JaxArray,
  mask: JaxArray,
  width: number,
  softness: ScalarLike = 0.75,
): JaxArray {
  const rows = splits.shape[0] ?? 0;
  const hits = splits.shape[1] ?? 0;
  if (hits === 0) {
    return np.zeros([rows, width]);
  }
  const xs = np.arange(width).add(0.5).reshape([1, 1, width]);
  const signs = np.array(
    Array.from({ length: hits }, (_, index) => (index % 2 === 0 ? 1 : -1)),
  ).reshape([1, hits, 1]);
  const splitGrid = splits.reshape([rows, hits, 1]);
  const maskGrid = mask.astype(np.float32).reshape([rows, hits, 1]);
  const logits = xs.sub(splitGrid).div(scalar(softness));
  const occupancy = sigmoid(logits).mul(signs).mul(maskGrid).sum(1);
  return np.clip(occupancy, 0, 1);
}

function sampleArc(arc: ResolvedArc, samplesPerArc: number): JaxArray {
  const ts = np.array(
    Array.from({ length: samplesPerArc }, (_, index) =>
      samplesPerArc === 1 ? 0 : index / (samplesPerArc - 1),
    ),
  );
  const angles = arc.startAngle.add(arc.deltaAngle.ref.mul(ts));
  const local = np.stack([np.cos(angles.ref), np.sin(angles)], 1);
  return applyRows(arc.transform, local, 1);
}

function sampledPolyline(shape: Shape, samplesPerArc = 16): JaxArray {
  const arcs = resolvedArcs(shape);
  if (arcs.length === 0) {
    return np.zeros([0, 2]);
  }
  const pieces = arcs.map((arc) => sampleArc(arc, samplesPerArc));
  return np.concatenate(pieces, 0);
}

function gridPoints(width: number, height: number): JaxArray {
  const xs = np.arange(width).add(0.5);
  const ys = np.arange(height).add(0.5);
  const [gridX, gridY] = np.meshgrid([xs, ys], { indexing: "xy" });
  return np.stack([gridX!.ravel(), gridY!.ravel()], 1);
}

function segmentDistance(pointsValue: JaxArray, p0: JaxArray, p1: JaxArray): JaxArray {
  const seg = p1.ref.sub(p0.ref);
  const denom = seg.ref.mul(seg.ref).sum().add(1e-6);
  const rel = pointsValue.ref.sub(p0.ref.reshape([1, 2]));
  const t = np
    .clip(rel.ref.mul(seg.ref.reshape([1, 2])).sum(1).div(denom), 0, 1)
    .reshape([pointsValue.shape[0] ?? 0, 1]);
  const closest = p0.ref.reshape([1, 2]).add(seg.ref.reshape([1, 2]).mul(t));
  return np.sqrt(pointsValue.ref.sub(closest).mul(pointsValue.ref.sub(closest)).sum(1).add(1e-6));
}

function renderStrokeCoverage(
  shape: Shape,
  width: number,
  height: number,
  softness: ScalarLike,
  samplesPerArc: number,
): JaxArray {
  const polyline = sampledPolyline(shape, samplesPerArc);
  const count = polyline.shape[0] ?? 0;
  if (count < 2) {
    return np.zeros([height, width]);
  }
  const pointsValue = gridPoints(width, height);
  const distances: JaxArray[] = [];
  for (let index = 0; index < count - 1; index += 1) {
    const p0 = polyline.slice(index);
    const p1 = polyline.slice(index + 1);
    distances.push(segmentDistance(pointsValue.ref, p0, p1));
  }
  const dist = np.stack(distances, 1).min(1);
  const halfWidth = shape.style.lineWidth.ref.mul(0.5);
  const coverage = sigmoid(halfWidth.sub(dist).div(scalar(softness)));
  return coverage.reshape([height, width]);
}

function renderFillCoverage(
  shape: Shape,
  width: number,
  height: number,
  softness: ScalarLike,
): JaxArray {
  if (!shape.trail.closed) {
    return np.zeros([height, width]);
  }
  const trace = getTrace(shape);
  const rowOrigins = np.stack([np.zeros([height]), np.arange(height).add(0.5)], 1);
  const colOrigins = np.stack([np.arange(width).add(0.5), np.zeros([width])], 1);
  const row = trace(rowOrigins, vec2(1, 0));
  const col = trace(colOrigins, vec2(0, 1));
  const hPass = renderLineSoft(row.splits, row.mask, width, softness);
  const vPass = renderLineSoft(col.splits, col.mask, height, softness).transpose();
  return scalar(1).sub(scalar(1).sub(hPass.ref).mul(scalar(1).sub(vPass)));
}

function compositeLayer(image: JaxArray, coverage: JaxArray, rgba: JaxArray): JaxArray {
  const alpha = coverage.ref.mul(rgba.slice(3)).reshape([coverage.shape[0] ?? 0, coverage.shape[1] ?? 0, 1]);
  const rgb = rgba.slice([0, 3]).reshape([1, 1, 3]);
  return image.ref.mul(scalar(1).sub(alpha.ref)).add(rgb.mul(alpha));
}

function backgroundOf(options: RenderOptions): JaxArray {
  if (options.background) {
    return options.background;
  }
  return np.array([1, 1, 1]);
}

export function renderShape(shape: Shape, options: RenderOptions): JaxArray {
  const fillSoftness = options.fillSoftness ?? 0.75;
  const strokeSoftness = options.strokeSoftness ?? 1.0;
  let image = np.broadcastTo(backgroundOf(options).reshape([1, 1, 3]), [
    options.height,
    options.width,
    3,
  ]);
  const fillCoverage = renderFillCoverage(shape, options.width, options.height, fillSoftness);
  image = compositeLayer(
    image,
    fillCoverage.ref.mul(shape.style.fillOpacity),
    shape.style.fill,
  );
  const strokeCoverage = renderStrokeCoverage(
    shape,
    options.width,
    options.height,
    strokeSoftness,
    options.samplesPerArc ?? 16,
  );
  image = compositeLayer(
    image,
    strokeCoverage.ref.mul(shape.style.strokeOpacity),
    shape.style.stroke,
  );
  return image;
}

export function renderScene(sceneValue: Scene, options: RenderOptions): JaxArray {
  let image = np.broadcastTo(backgroundOf(options).reshape([1, 1, 3]), [
    options.height,
    options.width,
    3,
  ]);
  const fillSoftness = options.fillSoftness ?? 0.75;
  const strokeSoftness = options.strokeSoftness ?? 1.0;
  for (const shape of sceneValue.shapes) {
    const fillCoverage = renderFillCoverage(shape, options.width, options.height, fillSoftness);
    image = compositeLayer(
      image,
      fillCoverage.ref.mul(shape.style.fillOpacity),
      shape.style.fill,
    );
    const strokeCoverage = renderStrokeCoverage(
      shape,
      options.width,
      options.height,
      strokeSoftness,
      options.samplesPerArc ?? 16,
    );
    image = compositeLayer(
      image,
      strokeCoverage.ref.mul(shape.style.strokeOpacity),
      shape.style.stroke,
    );
  }
  return image;
}

export function boundsOfShape(shape: Shape, samplesPerArc = 16): Bounds {
  const pointsValue = sampledPolyline(shape, samplesPerArc);
  if ((pointsValue.shape[0] ?? 0) === 0) {
    return {
      minX: scalar(0),
      minY: scalar(0),
      maxX: scalar(1),
      maxY: scalar(1),
    };
  }
  return {
    minX: pointsValue.slice([], 0).min(),
    minY: pointsValue.slice([], 1).min(),
    maxX: pointsValue.slice([], 0).max(),
    maxY: pointsValue.slice([], 1).max(),
  };
}

export function boundsOfScene(sceneValue: Scene, samplesPerArc = 16): Bounds {
  if (sceneValue.shapes.length === 0) {
    return {
      minX: scalar(0),
      minY: scalar(0),
      maxX: scalar(1),
      maxY: scalar(1),
    };
  }
  const boxes = sceneValue.shapes.map((shape) => boundsOfShape(shape, samplesPerArc));
  let minX = boxes[0]!.minX;
  let minY = boxes[0]!.minY;
  let maxX = boxes[0]!.maxX;
  let maxY = boxes[0]!.maxY;
  for (let index = 1; index < boxes.length; index += 1) {
    const box = boxes[index]!;
    minX = np.minimum(minX.ref, box.minX);
    minY = np.minimum(minY.ref, box.minY);
    maxX = np.maximum(maxX.ref, box.maxX);
    maxY = np.maximum(maxY.ref, box.maxY);
  }
  return { minX, minY, maxX, maxY };
}

export function centerXY(shape: Shape, samplesPerArc = 16): Shape {
  const bounds = boundsOfShape(shape, samplesPerArc);
  const cx = bounds.minX.ref.add(bounds.maxX).mul(0.5);
  const cy = bounds.minY.ref.add(bounds.maxY).mul(0.5);
  return translateShape(shape, np.negative(cx.ref), np.negative(cy));
}

export function centerScene(sceneValue: Scene, samplesPerArc = 16): Scene {
  const bounds = boundsOfScene(sceneValue, samplesPerArc);
  const cx = bounds.minX.ref.add(bounds.maxX).mul(0.5);
  const cy = bounds.minY.ref.add(bounds.maxY).mul(0.5);
  return {
    shapes: sceneValue.shapes.map((shape) =>
      translateShape(shape, np.negative(cx.ref), np.negative(cy.ref)),
    ),
  };
}

export function layoutScene(
  sceneValue: Scene,
  width: number,
  height: number,
  padding = 0.05,
  samplesPerArc = 16,
): Scene {
  const centered = centerScene(sceneValue, samplesPerArc);
  const bounds = boundsOfScene(centered, samplesPerArc);
  const sceneWidth = bounds.maxX.ref.sub(bounds.minX.ref).add(1e-6);
  const sceneHeight = bounds.maxY.ref.sub(bounds.minY.ref).add(1e-6);
  const scaleX = scalar(width).mul(1 - padding).div(sceneWidth);
  const scaleY = scalar(height).mul(1 - padding).div(sceneHeight);
  const uniform = np.minimum(scaleX.ref, scaleY);
  return {
    shapes: centered.shapes.map((shape) =>
      translateShape(transformShape(shape, scale(uniform.ref, uniform)), width / 2, height / 2),
    ),
  };
}
