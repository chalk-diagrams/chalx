import { beforeAll, describe, expect, it } from "vitest";

import {
  blockUntilReady,
  type JaxArray,
  color,
  fillColor,
  initJax,
  optimize,
  renderScene,
  scene,
  translateShape,
  circle,
  rectangle,
} from "../src/index.js";
import { np } from "../src/jax.js";

describe("chalk-jax", () => {
  beforeAll(async () => {
    await initJax();
  });

  it("renders overlapping filled shapes to an RGB image", async () => {
    const img = renderScene(
      scene(
        fillColor(rectangle(18, 18), color(1, 0.5, 0.2)),
        fillColor(translateShape(circle(7), 8, 6), color(0.2, 0.4, 1)),
      ),
      { width: 32, height: 32 },
    );

    await blockUntilReady(img.ref);
    expect(img.shape).toEqual([32, 32, 3]);
    const data = (await img.jsAsync()) as number[][][];
    expect(data[16]?.[16]?.[0]).toBeGreaterThan(0);
  });

  it("optimizes a translated circle toward a target image", async () => {
    const targetShape = fillColor(translateShape(circle(5), 18, 15), color(0.9, 0.3, 0.2));
    const target = renderScene(scene(targetShape), { width: 40, height: 40, fillSoftness: 0.9 });

    const objective = (params: { offset: JaxArray }) => {
      const shape = fillColor(
        translateShape(circle(5), params.offset.slice(0), params.offset.slice(1)),
        color(0.9, 0.3, 0.2),
      );
      const image = renderScene(scene(shape), { width: 40, height: 40, fillSoftness: 0.9 });
      const loss = image.ref.sub(target.ref).mul(image.ref.sub(target.ref)).mean();
      return [loss, image] as [JaxArray, JaxArray];
    };

    const start = { offset: np.array([6, 8]) };
    const result = await optimize(start, objective, { steps: 25, learningRate: 0.25 });
    const finalOffset = (await result.params.offset.jsAsync()) as number[];

    expect(finalOffset[0]).toBeGreaterThan(10);
    expect(finalOffset[1]).toBeGreaterThan(10);
  });
});
