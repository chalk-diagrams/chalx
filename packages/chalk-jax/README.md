# chalk-jax

Minimal TypeScript port of the core geometry and rendering ideas used by
[DiffRast](https://srush.github.io/DiffRast/) built on top of
[`@jax-js/jax`](https://github.com/ekzhang/jax-js).

This package is intentionally small. It focuses on the pieces needed for
DiffRast-style examples:

- affine transforms in homogeneous coordinates
- circular-arc path primitives
- simple shapes (`circle`, `rectangle`, `triangle`, polygon/path helpers)
- trace queries (`getTrace`)
- smooth scanline-style raster rendering (`renderScene`, `renderShape`)
- small animation and optimization helpers

## Install

```bash
npm install @jax-js/jax @jax-js/optax
```

## Example

```ts
import { init } from "@jax-js/jax";
import {
  circle,
  fillColor,
  getTrace,
  layoutScene,
  point,
  renderScene,
  scene,
  toColor,
  translateShape,
  vec2,
} from "@chalk-diagrams/chalk-jax";

await init();

const shape = fillColor(
  translateShape(circle(20), 50, 50),
  toColor("orange"),
);

const tr = getTrace(shape);
const hits = tr([point(0, 50)], vec2(1, 0));

const image = renderScene(layoutScene(scene(shape), 100, 100), {
  width: 100,
  height: 100,
});

console.log(hits.splits.js(), image.shape);
```

## Notes

- The original Python DiffRast notebook uses a custom VJP for boundary terms.
  `jax-js` does not currently expose `custom_vjp`, so this port uses a smoother
  fill/stroke coverage model by default. That keeps optimization through
  translation, color, scale, and shape parameters practical in TypeScript.
- The API is data-oriented on purpose so scenes and parameter objects work well
  with `jit`, `vmap`, and `valueAndGrad`.
