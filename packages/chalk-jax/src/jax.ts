import {
  blockUntilReady,
  grad,
  init,
  jit,
  numpy as np,
  tree,
  valueAndGrad,
  vmap,
} from "@jax-js/jax";
import * as optax from "@jax-js/optax";

export { blockUntilReady, grad, init, jit, np, optax, tree, valueAndGrad, vmap };

export type JaxArray = import("@jax-js/jax").Array;
export type JsTree<T> = import("@jax-js/jax").JsTree<T>;

export async function initJax() {
  return init();
}

export async function toNestedArray<T = unknown>(value: JaxArray): Promise<T> {
  return (await value.jsAsync()) as T;
}

export async function toFlatData(value: JaxArray) {
  return value.data();
}
