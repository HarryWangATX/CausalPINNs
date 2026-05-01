"""Physics-informed DeepONet baseline for the autonomous Lotka-Volterra system."""

import os
import sys

import jax.numpy as jnp
import numpy as onp
from dotenv import load_dotenv
from scipy.integrate import odeint as scipy_odeint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, ".env"))

from pi_deeponet_ode import run_pi_deeponet_ode

N_DIM = 2
ALPHA = 1.5
BETA = 1.0
DELTA = 1.0
GAMMA = 3.0

STATE0 = onp.array([10.0, 5.0], dtype=float)
IC_LOW = onp.fromstring(os.environ.get("PI_DEEPONET_IC_LOW", "1.0,0.5"), sep=",")
IC_HIGH = onp.fromstring(os.environ.get("PI_DEEPONET_IC_HIGH", "20.0,10.0"), sep=",")


def lv_rhs_np(state, _t):
    x, y = state
    return onp.array([
        ALPHA * x - BETA * x * y,
        DELTA * x * y - GAMMA * y,
    ], dtype=float)


def lv_rhs_jax(state):
    x, y = state
    return jnp.array([
        ALPHA * x - BETA * x * y,
        DELTA * x * y - GAMMA * y,
    ])


def sample_initial_conditions(n_ic, seed=1234):
    rng = onp.random.RandomState(seed)
    horizon = float(os.environ.get("PI_DEEPONET_IC_HORIZON", "30.0"))
    dt = float(os.environ.get("PI_DEEPONET_IC_DT", "0.01"))
    noise_frac = float(os.environ.get("PI_DEEPONET_IC_NOISE_FRAC", "0.05"))
    t = onp.arange(0.0, horizon, dt)
    traj = scipy_odeint(lv_rhs_np, STATE0, t)
    idx = rng.choice(len(traj), size=n_ic, replace=True)
    noise_scale = noise_frac * (traj.std(axis=0) + 1e-6)
    ics = traj[idx] + rng.randn(n_ic, N_DIM) * noise_scale
    ics = onp.clip(ics, IC_LOW, IC_HIGH)
    ics[0] = STATE0
    return ics.astype(onp.float32)


if __name__ == "__main__":
    run_pi_deeponet_ode(
        system_name="Lotka-Volterra",
        rhs_np=lv_rhs_np,
        rhs_jax=lv_rhs_jax,
        state0=STATE0,
        labels=["x_prey", "y_predator"],
        output_dir_name="pideeponet_lv",
        sample_initial_conditions=sample_initial_conditions,
        metadata={
            "alpha": ALPHA,
            "beta": BETA,
            "delta": DELTA,
            "gamma": GAMMA,
            "ic_low": IC_LOW.tolist(),
            "ic_high": IC_HIGH.tolist(),
            "ic_sampling": "trajectory_aware_clipped_noise",
            "autonomous": True,
        },
        default_t=30.0,
        default_t1=0.5,
    )
