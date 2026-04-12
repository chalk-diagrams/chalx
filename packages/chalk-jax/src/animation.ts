import type { JaxArray } from "./jax.js";
import { renderScene, type RenderOptions } from "./render.js";
import type { Scene } from "./geometry.js";

export interface AnimationOptions {
  readonly steps: number;
  readonly includeEndpoint?: boolean;
}

export function frameTimes(
  steps: number,
  options: { includeEndpoint?: boolean } = {},
): number[] {
  if (steps <= 0) {
    return [];
  }
  if (options.includeEndpoint) {
    if (steps === 1) {
      return [0];
    }
    return Array.from({ length: steps }, (_, index) => index / (steps - 1));
  }
  return Array.from({ length: steps }, (_, index) => index / steps);
}

export function sampleAnimation<T>(
  sampler: (t: number, index: number) => T,
  options: AnimationOptions,
): T[] {
  const times = frameTimes(options.steps, { includeEndpoint: options.includeEndpoint });
  return times.map((t, index) => sampler(t, index));
}

export function animateScenes(
  sampler: (t: number, index: number) => Scene,
  options: AnimationOptions,
): Scene[] {
  return sampleAnimation(sampler, options);
}

export function animate(
  sampler: (t: number, index: number) => Scene,
  animation: AnimationOptions,
  render: RenderOptions,
): JaxArray[] {
  return sampleAnimation((t, index) => renderScene(sampler(t, index), render), animation);
}
