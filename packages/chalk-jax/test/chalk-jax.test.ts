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

  it(
    "optimizes fill color toward a target image",
    async () => {
    const targetShape = fillColor(translateShape(circle(5), 18, 15), color(0.9, 0.3, 0.2, 1));
    const target = renderScene(scene(targetShape), { width: 40, height: 40, fillSoftness: 0.9 });

      const objective = (params: { fill: JaxArray }) => {
        const shape = fillColor(
          translateShape(circle(5), 18, 15),
          params.fill.ref,
        );
        const image = renderScene(scene(shape), { width: 40, height: 40, fillSoftness: 0.9 });
        const delta1 = image.ref.sub(target.ref);
        const delta2 = image.ref.sub(target.ref);
        const loss = delta1.mul(delta2).mean();
        return [loss, image] as [JaxArray, JaxArray];
      };

      const start = { fill: np.array([0.1, 0.2, 0.9, 1.0]) };
      const result = await optimize(start, objective, { steps: 6, learningRate: 0.3 });
      const finalColor = (await result.params.fill.jsAsync()) as number[];
      const firstLoss = result.history[0]!.loss;
      const lastLoss = result.history[result.history.length - 1]!.loss;

      expect(lastLoss).toBeLessThan(firstLoss);
      expect(finalColor[0]).toBeGreaterThan(0.1);
      expect(finalColor[2]).toBeLessThan(0.9);
    },
    15000,
  );
});
