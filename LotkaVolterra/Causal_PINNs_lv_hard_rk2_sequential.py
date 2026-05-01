"""
Causal PINNs (hard RK2) for Lotka-Volterra — sequential training.

Window k+1's IC comes from window k's prediction at t=T1, matching the
original CausalPINNs paper protocol.  Single-GPU, no multiprocessing.

Ansatz: u(t) = RK2(IC, t) + tau^3 * NN(tau),  activation = SiLU
where tau = t / T1, RK2 uses 10 substeps.
"""

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

import numpy as onp
from scipy.integrate import odeint as scipy_odeint

# ---------------------------------------------------------------------------
# Lotka-Volterra system
# ---------------------------------------------------------------------------
N_DIM = 2
ALPHA = 1.5
BETA = 1.0
DELTA = 1.0
GAMMA = 3.0

STATE0 = onp.array([10.0, 5.0], dtype=float)


def lv_rhs(state, t):
    x, y = state
    dx = ALPHA * x - BETA * x * y
    dy = DELTA * x * y - GAMMA * y
    return onp.array([dx, dy], dtype=float)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
T = 30.0
T1 = 0.5
DT = 0.01
NUM_WINDOWS = int(T / T1)
TOL_LIST = [1e0, 1e1, 1e2, 1e3, 1e4]
LAYERS = [1, 512, 512, 512, N_DIM]
N_ITER = 300_000

COLLOCATION_EXT_RATIO = 0.0
RK2_SUBSTEPS = 10
CORRECTION_POWER = 3


def main():
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import jax.numpy as np
    from jax import random, jacfwd, vmap, jit, lax, grad
    from jax.example_libraries import optimizers
    from jax.nn import silu
    from jax.flatten_util import ravel_pytree
    import itertools
    from functools import partial
    from tqdm import trange

    def rhs_fn(state):
        x = state[0]
        y = state[1]
        dx = ALPHA * x - BETA * x * y
        dy = DELTA * x * y - GAMMA * y
        return np.array([dx, dy])

    def init_layer(key, d_in, d_out):
        k1, _ = random.split(key)
        glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.0)
        W = glorot_stddev * random.normal(k1, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b

    def MLP(layers, activation=silu):
        def init(rng_key):
            _, *keys = random.split(rng_key, len(layers))
            params = list(map(init_layer, keys, layers[:-1], layers[1:]))
            return params

        def apply(params, inputs):
            for W, b in params[:-1]:
                outputs = np.dot(inputs, W) + b
                inputs = activation(outputs)
            W, b = params[-1]
            outputs = np.dot(inputs, W) + b
            return outputs

        return init, apply

    class PINN:
        def __init__(self, layers, states0, t0, t1, tol):
            self.states0 = states0
            self.t0 = t0
            self.t1 = t1
            self.t_scale = max(float(self.t1), 1e-12)

            n_t = 300
            eps = COLLOCATION_EXT_RATIO * self.t1
            self.t = np.linspace(self.t0, self.t1 + eps, n_t)

            self.M = np.triu(np.ones((n_t, n_t)), k=1).T
            self.tol = tol

            self.init, self.apply = MLP(layers, activation=silu)
            params = self.init(random.PRNGKey(1234))

            self.opt_init, self.opt_update, self.get_params = optimizers.adam(
                optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9)
            )
            self.opt_state = self.opt_init(params)
            _, self.unravel = ravel_pytree(params)

            f0 = rhs_fn(self.states0)
            f_scale = np.sum(f0**2)
            u_scale = np.sum(self.states0**2) / max(self.t1, 1e-12)**2
            self.rate_scale_sq = np.maximum(f_scale, u_scale) + 1e-12

            self.itercount = itertools.count()
            self.loss_log = []
            self.loss_ics_log = []
            self.loss_res_log = []

        def neural_net(self, params, t):
            dt = t
            tau = dt / self.t_scale
            t_in = np.stack([tau])

            n_sub = RK2_SUBSTEPS
            h = dt / n_sub
            state = self.states0
            for _ in range(n_sub):
                k1 = rhs_fn(state)
                k2 = rhs_fn(state + 0.5 * h * k1)
                state = state + h * k2

            outputs = self.apply(params, t_in)
            state = state + (tau**CORRECTION_POWER) * outputs
            return state

        def residual_net(self, params, t):
            state = self.neural_net(params, t)
            state_t = jacfwd(lambda tau: self.neural_net(params, tau))(t)
            return state_t - rhs_fn(state)

        def loss_ics(self, params):
            state_pred = self.neural_net(params, self.t0)
            return np.mean((self.states0 - state_pred) ** 2)

        @partial(jit, static_argnums=(0,))
        def residuals_and_weights(self, params, tol):
            r_pred = vmap(self.residual_net, (None, 0))(params, self.t)
            r_sq = np.sum(r_pred**2, axis=1) / self.rate_scale_sq
            r_sq = np.nan_to_num(r_sq, nan=1e12, posinf=1e12, neginf=1e12)
            log_w = -tol * (self.M @ r_sq)
            log_w = np.clip(log_w, -60.0, 60.0)
            W = lax.stop_gradient(np.exp(log_w))
            return r_pred, W

        @partial(jit, static_argnums=(0,))
        def loss_res(self, params):
            r_pred, W = self.residuals_and_weights(params, self.tol)
            r_sq = np.sum(r_pred**2, axis=1) / self.rate_scale_sq
            r_sq = np.nan_to_num(r_sq, nan=1e12, posinf=1e12, neginf=1e12)
            return np.mean(W * r_sq)

        @partial(jit, static_argnums=(0,))
        def loss(self, params):
            return self.loss_res(params)

        @partial(jit, static_argnums=(0,))
        def step(self, i, opt_state):
            params = self.get_params(opt_state)
            g = grad(self.loss)(params)
            return self.opt_update(i, g, opt_state)

        def train(self, nIter=10000, window_idx=0):
            pbar = trange(nIter, desc=f"Window {window_idx}")
            for it in pbar:
                self.current_count = next(self.itercount)
                self.opt_state = self.step(self.current_count, self.opt_state)

                if it % 1000 == 0:
                    params = self.get_params(self.opt_state)
                    loss_value = self.loss(params)
                    loss_ics_value = self.loss_ics(params)
                    loss_res_value = self.loss_res(params)
                    _, W_value = self.residuals_and_weights(params, self.tol)

                    self.loss_log.append(loss_value)
                    self.loss_ics_log.append(loss_ics_value)
                    self.loss_res_log.append(loss_res_value)

                    loss_scalar = float(loss_value)
                    if not onp.isfinite(loss_scalar):
                        print(f"  [Window {window_idx}] Non-finite loss; stopping tol stage.")
                        break

                    pbar.set_postfix({
                        'Loss': loss_value,
                        'loss_ics': loss_ics_value,
                        'loss_res': loss_res_value,
                        'W_min': W_value.min(),
                    })

                    if W_value.min() > 0.99:
                        break

        @partial(jit, static_argnums=(0,))
        def predict_u(self, params, t_star):
            return vmap(self.neural_net, (None, 0))(params, t_star)

    # ------------------------------------------------------------------
    # Reference solution (for error reporting only, NOT used for ICs)
    # ------------------------------------------------------------------
    t_ref = onp.arange(0.0, T, DT)
    states_ref = scipy_odeint(lv_rhs, STATE0, t_ref)
    print(f"Reference solution computed: {states_ref.shape}")

    # ------------------------------------------------------------------
    # Sequential training loop — chained ICs
    # ------------------------------------------------------------------
    t0 = 0.0
    t1 = T1
    t_eval = np.arange(t0, t1, DT)

    state0 = np.array(STATE0)

    state_pred_list = []
    params_list = []
    losses_list = []

    for k in range(NUM_WINDOWS):
        print(f"\nFinal Time: {(k + 1) * t1:.1f}")

        model = PINN(LAYERS, state0, t0, t1, tol=0.1)

        for tol_val in TOL_LIST:
            model.tol = tol_val
            print(f"  tol: {tol_val}")
            model.train(nIter=N_ITER, window_idx=k)

        params = model.get_params(model.opt_state)
        state_pred = model.predict_u(params, t_eval)

        # Next window's IC = this window's prediction at t = T1
        state0_next = model.neural_net(params, model.t1)
        state0 = state0_next

        state_pred_list.append(onp.array(state_pred))
        flat_params, _ = ravel_pytree(params)
        params_list.append(onp.array(flat_params))
        losses_list.append([
            [float(v) for v in model.loss_ics_log],
            [float(v) for v in model.loss_res_log],
        ])

        # Save after every window
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'hard_rk2_causalpinn_lv_sequential')
        os.makedirs(out_dir, exist_ok=True)
        onp.save(os.path.join(out_dir, 'state_pred_list.npy'), onp.array(state_pred_list))
        onp.save(os.path.join(out_dir, 'params_list.npy'), onp.array(params_list))
        onp.save(os.path.join(out_dir, 'losses_list.npy'),
                 onp.array(losses_list, dtype=object), allow_pickle=True)

        # Running error
        state_preds_so_far = onp.concatenate(state_pred_list, axis=0)
        n = min(len(state_preds_so_far), len(states_ref))
        err = onp.linalg.norm(state_preds_so_far[:n] - states_ref[:n]) / onp.linalg.norm(states_ref[:n])
        labels = ["x_prey", "y_predator"]
        per_dim = onp.linalg.norm(state_preds_so_far[:n] - states_ref[:n], axis=0) / onp.linalg.norm(states_ref[:n], axis=0)
        print(f"  Cumulative relative L2 error: {err:.3e}")
        for i, e in enumerate(per_dim):
            print(f"    {labels[i]}: {e:.3e}")

    print("\nDone. Results saved to", out_dir)


if __name__ == "__main__":
    main()
