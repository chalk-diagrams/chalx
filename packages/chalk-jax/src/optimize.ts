import type { GradientTransformation } from "@jax-js/optax";

import { blockUntilReady, grad, optax, tree, type JaxArray, type JsTree } from "./jax.js";

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

  let params = tree.ref(initialParams) as Params;
  let state = optimizer.init(tree.ref(initialParams) as Params);
  const history: Array<{ step: number; loss: number; aux: Aux }> = [];

  for (let step = 0; step < options.steps; step += 1) {
    const paramsForLoss = tree.ref(params) as Params;
    const paramsForGrad = tree.ref(params) as Params;
    const paramsForUpdate = tree.ref(params) as Params;
    const paramsForApply = tree.ref(params) as Params;
    const [lossValue, aux] = objective(paramsForLoss);
    const grads = objectiveGrad(paramsForGrad);
    await blockUntilReady(lossValue);
    const [updates, nextState] = optimizer.update(grads, state, paramsForUpdate);
    params = optax.applyUpdates(paramsForApply, updates);
    state = nextState;
    const loss = lossValue.item();
    history.push({ step, loss, aux });
    if (options.onStep) {
      await options.onStep(step, loss, params, aux);
    }
  }

  return { params, history };
}
