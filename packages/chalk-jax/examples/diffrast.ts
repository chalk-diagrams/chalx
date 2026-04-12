import {
  blockUntilReady,
  centerXY,
  circle,
  color,
  concatScenes,
  fillColor,
  fillOpacity,
  initJax,
  lineWidth,
  point,
  rectangle,
  renderScene,
  rotateShapeBy,
  scene,
  strokeColor,
  translateShape,
} from "../src/index.js";

async function main(): Promise<void> {
  await initJax();

  const moon = fillColor(
    strokeColor(
      lineWidth(
        translateShape(
          rotateShapeBy(centerXY(circle(18)), 0.1),
          46,
          42,
        ),
        1.5,
      ),
      color(0.2, 0.15, 0.1, 1),
    ),
    color(1, 0.62, 0.15, 1),
  );

  const squareNode = fillColor(
    strokeColor(
      lineWidth(translateShape(centerXY(rectangle(20, 20)), 72, 68), 1),
      color(0.1, 0.2, 0.45, 1),
    ),
    color(0.2, 0.5, 1, 1),
  );

  const canvas = fillOpacity(
    fillColor(rectangle(120, 120), color(1, 1, 1, 1)),
    1,
  );

  const image = renderScene(
    concatScenes(scene(canvas), scene(moon, squareNode)),
    {
      width: 128,
      height: 128,
      samplesPerArc: 24,
      fillSoftness: 0.8,
      strokeSoftness: 1.1,
    },
  );

  await blockUntilReady(image);
  const nested = (await image.jsAsync()) as number[][][];
  const firstPixel = Array.isArray(nested) && Array.isArray(nested[0]) ? nested[0][0] : null;
  console.log(JSON.stringify({ size: image.shape, firstPixel, anchor: point(46, 42).js() }));
}

void main();
