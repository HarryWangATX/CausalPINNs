"""
Causal PINNs (baseline) for the Lorenz system — sequential training.

Window k+1's IC comes from window k's prediction at t=T1. Single process, no
multiprocessing.

Ansatz: (x,y,z) from NN with outputs = apply*t then x=outputs[0]+IC_x, etc.
Activation: tanh. Residual uses rho=28, sigma=10, beta=8/3 with per-component
grad time derivatives.
"""

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

import numpy as onp
from scipy.integrate import odeint as scipy_odeint

# ---------------------------------------------------------------------------
# Lorenz system (scipy reference — rho matches PINN)
# ---------------------------------------------------------------------------
RHO = 28.0
SIGMA = 10.0
BETA = 8.0 / 3.0
N_DIM = 3

STATE0 = onp.array([1.0, 1.0, 1.0], dtype=float)


def lorenz_rhs(state, t):
    x, y, z = state
    dx = SIGMA * (y - x)
    dy = x * (RHO - z) - y
    dz = x * y - BETA * z
    return onp.array([dx, dy, dz], dtype=float)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
T = 30.0
T1 = 0.5
DT = 0.01
NUM_WINDOWS = int(T / T1)
TOL_LIST = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
LAYERS = [1, 512, 512, 512, N_DIM]
N_ITER = 300_000


def main():
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    import jax.numpy as np
    from jax import random, vmap, jit, lax, grad
    from jax.example_libraries import optimizers
    from jax.flatten_util import ravel_pytree
    import itertools
    from functools import partial
    from tqdm import trange

    def init_layer(key, d_in, d_out):
        k1, _ = random.split(key)
        glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.0)
        W = glorot_stddev * random.normal(k1, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b

    def MLP(layers, activation=np.tanh):
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

            n_t = 300
            eps = 0.1 * self.t1
            self.t = np.linspace(self.t0, self.t1 + eps, n_t)

            self.M = np.triu(np.ones((n_t, n_t)), k=1).T
            self.tol = tol

            self.rho = 28.0
            self.sigma = 10.0
            self.beta = 8.0 / 3.0

            self.init, self.apply = MLP(layers, activation=np.tanh)
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
            t_in = np.stack([t])
            outputs = self.apply(params, t_in) * t
            x = outputs[0] + self.states0[0]
            y = outputs[1] + self.states0[1]
            z = outputs[2] + self.states0[2]
            return x, y, z

        def x_fn(self, params, t):
            x, _, _ = self.neural_net(params, t)
            return x

        def y_fn(self, params, t):
            _, y, _ = self.neural_net(params, t)
            return y

        def z_fn(self, params, t):
            _, _, z = self.neural_net(params, t)
            return z

        def residual_net(self, params, t):
            x, y, z = self.neural_net(params, t)
            x_t = grad(self.x_fn, argnums=1)(params, t)
            y_t = grad(self.y_fn, argnums=1)(params, t)
            z_t = grad(self.z_fn, argnums=1)(params, t)

            res_1 = x_t - self.sigma * (y - x)
            res_2 = y_t - x * (self.rho - z) + y
            res_3 = z_t - x * y + self.beta * z
            return res_1, res_2, res_3

        def loss_ics(self, params):
            x_pred, y_pred, z_pred = self.neural_net(params, self.t0)
            loss_x_ic = np.mean((self.states0[0] - x_pred) ** 2)
            loss_y_ic = np.mean((self.states0[1] - y_pred) ** 2)
            loss_z_ic = np.mean((self.states0[2] - z_pred) ** 2)
            return loss_x_ic + loss_y_ic + loss_z_ic

        @partial(jit, static_argnums=(0,))
        def residuals_and_weights(self, params, tol):
            r1_pred, r2_pred, r3_pred = vmap(self.residual_net, (None, 0))(params, self.t)
            r_sq = r1_pred**2 + r2_pred**2 + r3_pred**2
            r_sq = np.nan_to_num(r_sq, nan=1e12, posinf=1e12, neginf=1e12)
            log_w = -tol * (self.M @ r_sq)
            log_w = np.clip(log_w, -60.0, 60.0)
            W = lax.stop_gradient(np.exp(log_w))
            return r1_pred, r2_pred, r3_pred, W

        @partial(jit, static_argnums=(0,))
        def loss_res(self, params):
            r1_pred, r2_pred, r3_pred, W = self.residuals_and_weights(params, self.tol)
            r_sq = r1_pred**2 + r2_pred**2 + r3_pred**2
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
                    _, _, _, W_value = self.residuals_and_weights(params, self.tol)

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
            x_pred, y_pred, z_pred = vmap(self.neural_net, (None, 0))(params, t_star)
            return x_pred, y_pred, z_pred

    # ------------------------------------------------------------------
    # Reference solution (for error reporting only, NOT used for ICs)
    # ------------------------------------------------------------------
    t_ref = onp.arange(0.0, T, DT)
    states_ref = scipy_odeint(lorenz_rhs, STATE0, t_ref)
    print(f"Reference solution computed: {states_ref.shape}")

    # ------------------------------------------------------------------
    # Sequential training loop — chained ICs
    # ------------------------------------------------------------------
    t0 = 0.0
    t1 = T1
    t_eval = np.arange(t0, t1, DT)

    state0 = np.array(STATE0)

    x_pred_list = []
    y_pred_list = []
    z_pred_list = []
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
        x_pred, y_pred, z_pred = model.predict_u(params, t_eval)

        x0_pred, y0_pred, z0_pred = model.neural_net(params, model.t1)
        state0 = np.array([x0_pred, y0_pred, z0_pred])

        x_pred_list.append(onp.array(x_pred))
        y_pred_list.append(onp.array(y_pred))
        z_pred_list.append(onp.array(z_pred))
        flat_params, _ = ravel_pytree(params)
        params_list.append(onp.array(flat_params))
        losses_list.append([
            [float(v) for v in model.loss_ics_log],
            [float(v) for v in model.loss_res_log],
        ])

        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'baseline_causalpinn_lorentz_sequential',
        )
        os.makedirs(out_dir, exist_ok=True)
        onp.save(os.path.join(out_dir, 'x_pred_list.npy'), onp.array(x_pred_list))
        onp.save(os.path.join(out_dir, 'y_pred_list.npy'), onp.array(y_pred_list))
        onp.save(os.path.join(out_dir, 'z_pred_list.npy'), onp.array(z_pred_list))
        onp.save(os.path.join(out_dir, 'params_list.npy'), onp.array(params_list))
        onp.save(
            os.path.join(out_dir, 'losses_list.npy'),
            onp.array(losses_list, dtype=object),
            allow_pickle=True,
        )

        x_so_far = onp.concatenate(x_pred_list)
        y_so_far = onp.concatenate(y_pred_list)
        z_so_far = onp.concatenate(z_pred_list)
        n = min(len(x_so_far), len(states_ref))
        err_x = onp.linalg.norm(x_so_far[:n] - states_ref[:n, 0]) / onp.linalg.norm(states_ref[:n, 0])
        err_y = onp.linalg.norm(y_so_far[:n] - states_ref[:n, 1]) / onp.linalg.norm(states_ref[:n, 1])
        err_z = onp.linalg.norm(z_so_far[:n] - states_ref[:n, 2]) / onp.linalg.norm(states_ref[:n, 2])
        err_full = onp.linalg.norm(
            onp.stack([x_so_far[:n], y_so_far[:n], z_so_far[:n]], axis=1) - states_ref[:n]
        ) / onp.linalg.norm(states_ref[:n])
        print(f"  Cumulative relative L2 error (all dims): {err_full:.3e}")
        print(f"    error_x: {err_x:.3e}")
        print(f"    error_y: {err_y:.3e}")
        print(f"    error_z: {err_z:.3e}")

    print("\nDone. Results saved to", out_dir)


if __name__ == "__main__":
    main()
