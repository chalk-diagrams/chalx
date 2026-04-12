import type { GradientTransformation } from "@jax-js/optax";

import { blockUntilReady, grad, optax, type JaxArray, type JsTree } from "./jax.js";

export interface OptimizeOptions<Params extends JsTree<JaxArray>, Aux> {
  readonly steps: number;
  readonly learningRate?: number;
  readonly optimizer?: GradientTransformation;
  readonly onStep?: (step: number, loss: number, params: Params, aux: Aux) => void | Promise<void>;
}

export interface OptimizeResult<Params extends JsTree<JaxArray>, Aux> {
  readonly params: Params;
  readonly history: readonly {
    step: number;
    loss: number;
    aux: Aux;
  }[];
}

export async function optimize<Params extends JsTree<JaxArray>, Aux = undefined>(
  initialParams: Params,
  objective: (params: Params) => [JaxArray, Aux],
  options: OptimizeOptions<Params, Aux>,
): Promise<OptimizeResult<Params, Aux>> {
  const optimizer =
    options.optimizer ?? optax.adam(options.learningRate ?? 0.05);
  const objectiveGrad = grad(
    (params: Params) => objective(params)[0],
  ) as unknown as (params: Params) => Params;

  let params = initialParams;
  let state = optimizer.init(params);
  const history: Array<{ step: number; loss: number; aux: Aux }> = [];

  for (let step = 0; step < options.steps; step += 1) {
    const [lossValue, aux] = objective(params);
    const grads = objectiveGrad(params);
    await blockUntilReady(lossValue);
    const [updates, nextState] = optimizer.update(grads, state, params);
    params = optax.applyUpdates(params, updates);
    state = nextState;
    const loss = lossValue.item();
    history.push({ step, loss, aux });
    if (options.onStep) {
      await options.onStep(step, loss, params, aux);
    }
  }

  return { params, history };
}
