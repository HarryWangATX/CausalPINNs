"""
Causal PINNs (hard Euler) for the SEIR model — sequential training.

Window k+1's IC comes from window k's prediction at t=T1. Single process, no
multiprocessing.

Ansatz: Euler base (1 substep) + tau^2 * NN(tau),  activation = tanh
where tau = t / T1. SEIR uses scalar-style (s,e,i,r) decomposition.
"""

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

import numpy as onp
from scipy.integrate import odeint as scipy_odeint

# ---------------------------------------------------------------------------
# SEIR system
# ---------------------------------------------------------------------------
N_DIM = 4
BETA_SEIR = 5.0
SIGMA = 1.0
GAMMA = 0.5

STATE0 = onp.array([0.5, 0.2, 0.2, 0.1], dtype=float)
N_POP = float(onp.sum(STATE0))


def seir_rhs(state, t):
    S, E, I, R = state
    infection = BETA_SEIR * S * I / N_POP
    dS = -infection
    dE = infection - SIGMA * E
    dI = SIGMA * E - GAMMA * I
    dR = GAMMA * I
    return onp.array([dS, dE, dI, dR], dtype=float)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
T = 30.0
T1 = 0.5
DT = 0.01
NUM_WINDOWS = int(T / T1)
TOL_LIST = [1e1, 1e2, 1e3, 1e4, 1e5]
LAYERS = [1, 512, 512, 512, N_DIM]
N_ITER = 300_000
N_SUBSTEPS = 1
COLLOCATION_EXT_RATIO = 0.0
CORRECTION_POWER = 2


def main():
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import jax.numpy as np
    from jax import random, vmap, jit, lax, grad
    from jax.example_libraries import optimizers
    from jax.nn import tanh
    from jax.flatten_util import ravel_pytree
    import itertools
    from functools import partial
    from tqdm import trange

    def rhs_fn(S, E, I, R):
        infection = BETA_SEIR * S * I / N_POP
        dS = -infection
        dE = infection - SIGMA * E
        dI = SIGMA * E - GAMMA * I
        dR = GAMMA * I
        return dS, dE, dI, dR

    def init_layer(key, d_in, d_out):
        k1, _ = random.split(key)
        glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.0)
        W = glorot_stddev * random.normal(k1, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b

    def MLP(layers, activation=tanh):
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

            f1_0, f2_0, f3_0, f4_0 = rhs_fn(states0[0], states0[1], states0[2], states0[3])
            f_scale = f1_0**2 + f2_0**2 + f3_0**2 + f4_0**2
            u_scale = (states0[0]**2 + states0[1]**2 + states0[2]**2 + states0[3]**2) / max(self.t_scale, 1e-12)**2
            self.rate_scale_sq = np.maximum(f_scale, u_scale) + 1e-12

            n_t = 300
            eps = COLLOCATION_EXT_RATIO * self.t1
            self.t = np.linspace(self.t0, self.t1 + eps, n_t)

            self.M = np.triu(np.ones((n_t, n_t)), k=1).T
            self.tol = tol

            self.init, self.apply = MLP(layers, activation=tanh)
            params = self.init(random.PRNGKey(1234))

            self.opt_init, self.opt_update, self.get_params = optimizers.adam(
                optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9)
            )
            self.opt_state = self.opt_init(params)
            _, self.unravel = ravel_pytree(params)

            self.itercount = itertools.count()
            self.loss_log = []
            self.loss_ics_log = []
            self.loss_res_log = []

        def neural_net(self, params, t):
            dt = t
            tau = dt / self.t_scale
            t_in = np.stack([tau])
            h = dt / N_SUBSTEPS

            s, e, i, r = self.states0
            for _ in range(N_SUBSTEPS):
                ds, de, di, dr = rhs_fn(s, e, i, r)
                s = s + h * ds
                e = e + h * de
                i = i + h * di
                r = r + h * dr

            outputs = self.apply(params, t_in)
            s = s + tau ** CORRECTION_POWER * outputs[0]
            e = e + tau ** CORRECTION_POWER * outputs[1]
            i = i + tau ** CORRECTION_POWER * outputs[2]
            r = r + tau ** CORRECTION_POWER * outputs[3]
            return s, e, i, r

        def s_fn(self, params, t):
            s, _, _, _ = self.neural_net(params, t)
            return s

        def e_fn(self, params, t):
            _, e, _, _ = self.neural_net(params, t)
            return e

        def i_fn(self, params, t):
            _, _, i, _ = self.neural_net(params, t)
            return i

        def r_fn(self, params, t):
            _, _, _, r = self.neural_net(params, t)
            return r

        def residual_net(self, params, t):
            s, e, i, r = self.neural_net(params, t)
            s_t = grad(self.s_fn, argnums=1)(params, t)
            e_t = grad(self.e_fn, argnums=1)(params, t)
            i_t = grad(self.i_fn, argnums=1)(params, t)
            r_t = grad(self.r_fn, argnums=1)(params, t)

            f1, f2, f3, f4 = rhs_fn(s, e, i, r)
            return s_t - f1, e_t - f2, i_t - f3, r_t - f4

        def loss_ics(self, params):
            s_pred, e_pred, i_pred, r_pred = self.neural_net(params, self.t0)
            loss_s = (self.states0[0] - s_pred)**2
            loss_e = (self.states0[1] - e_pred)**2
            loss_i = (self.states0[2] - i_pred)**2
            loss_r = (self.states0[3] - r_pred)**2
            return loss_s + loss_e + loss_i + loss_r

        @partial(jit, static_argnums=(0,))
        def residuals_and_weights(self, params, tol):
            r1, r2, r3, r4 = vmap(self.residual_net, (None, 0))(params, self.t)
            r_sq = (r1**2 + r2**2 + r3**2 + r4**2) / self.rate_scale_sq
            W = lax.stop_gradient(np.exp(-tol * self.M @ r_sq))
            return r1, r2, r3, r4, W

        @partial(jit, static_argnums=(0,))
        def loss_res(self, params):
            r1, r2, r3, r4, W = self.residuals_and_weights(params, self.tol)
            r_sq = (r1**2 + r2**2 + r3**2 + r4**2) / self.rate_scale_sq
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
                    _, _, _, _, W_value = self.residuals_and_weights(params, self.tol)

                    self.loss_log.append(loss_value)
                    self.loss_ics_log.append(loss_ics_value)
                    self.loss_res_log.append(loss_res_value)

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
            s, e, i, r = vmap(self.neural_net, (None, 0))(params, t_star)
            return np.stack([s, e, i, r], axis=1)

    # ------------------------------------------------------------------
    # Reference solution (for error reporting only, NOT used for ICs)
    # ------------------------------------------------------------------
    t_ref = onp.arange(0.0, T, DT)
    states_ref = scipy_odeint(seir_rhs, STATE0, t_ref)
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

    labels = ["S", "E", "I", "R"]

    for k in range(NUM_WINDOWS):
        print(f"\nFinal Time: {(k + 1) * t1:.1f}")

        model = PINN(LAYERS, state0, t0, t1, tol=0.1)

        for tol_val in TOL_LIST:
            model.tol = tol_val
            print(f"  tol: {tol_val}")
            model.train(nIter=N_ITER, window_idx=k)

        params = model.get_params(model.opt_state)
        state_pred = model.predict_u(params, t_eval)

        s0, e0, i0, r0 = model.neural_net(params, model.t1)
        state0 = np.array([s0, e0, i0, r0])

        state_pred_list.append(onp.array(state_pred))
        flat_params, _ = ravel_pytree(params)
        params_list.append(onp.array(flat_params))
        losses_list.append([
            [float(v) for v in model.loss_ics_log],
            [float(v) for v in model.loss_res_log],
        ])

        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'hard_euler_causalpinn_seir_sequential',
        )
        os.makedirs(out_dir, exist_ok=True)
        onp.save(os.path.join(out_dir, 'state_pred_list.npy'), onp.array(state_pred_list))
        onp.save(os.path.join(out_dir, 'params_list.npy'), onp.array(params_list))
        onp.save(
            os.path.join(out_dir, 'losses_list.npy'),
            onp.array(losses_list, dtype=object),
            allow_pickle=True,
        )

        state_preds_so_far = onp.concatenate(state_pred_list, axis=0)
        n = min(len(state_preds_so_far), len(states_ref))
        err = onp.linalg.norm(state_preds_so_far[:n] - states_ref[:n]) / onp.linalg.norm(states_ref[:n])
        per_dim = onp.linalg.norm(state_preds_so_far[:n] - states_ref[:n], axis=0) / onp.linalg.norm(states_ref[:n], axis=0)
        print(f"  Cumulative relative L2 error: {err:.3e}")
        for idx, e in enumerate(per_dim):
            print(f"    {labels[idx]}: {e:.3e}")

    print("\nDone. Results saved to", out_dir)


if __name__ == "__main__":
    main()
