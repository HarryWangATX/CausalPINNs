"""Physics-informed DeepONet baseline for the autonomous Lorenz-96 system."""

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

N_DIM = 5
FORCING = 8.0

STATE0 = onp.full((N_DIM,), FORCING, dtype=float)
STATE0[0] += 0.01
IC_LOW = onp.fromstring(
    os.environ.get("PI_DEEPONET_IC_LOW", ",".join(["0.0"] * N_DIM)), sep=","
)
IC_HIGH = onp.fromstring(
    os.environ.get("PI_DEEPONET_IC_HIGH", ",".join(["12.0"] * N_DIM)), sep=","
)


def lorenz96_rhs_np(state, _t):
    return (onp.roll(state, -1) - onp.roll(state, 2)) * onp.roll(state, 1) - state + FORCING


def lorenz96_rhs_jax(state):
    return (jnp.roll(state, -1) - jnp.roll(state, 2)) * jnp.roll(state, 1) - state + FORCING


def sample_initial_conditions(n_ic, seed=1234):
    rng = onp.random.RandomState(seed)
    horizon = float(os.environ.get("PI_DEEPONET_IC_HORIZON", "5.0"))
    dt = float(os.environ.get("PI_DEEPONET_IC_DT", "0.01"))
    noise_frac = float(os.environ.get("PI_DEEPONET_IC_NOISE_FRAC", "0.05"))
    t = onp.arange(0.0, horizon, dt)
    traj = scipy_odeint(lorenz96_rhs_np, STATE0, t)
    idx = rng.choice(len(traj), size=n_ic, replace=True)
    noise_scale = noise_frac * (traj.std(axis=0) + 1e-6)
    ics = traj[idx] + rng.randn(n_ic, N_DIM) * noise_scale
    ics = onp.clip(ics, IC_LOW, IC_HIGH)
    ics[0] = STATE0
    return ics.astype(onp.float32)


if __name__ == "__main__":
    run_pi_deeponet_ode(
        system_name="Lorenz-96",
        rhs_np=lorenz96_rhs_np,
        rhs_jax=lorenz96_rhs_jax,
        state0=STATE0,
        labels=[f"x{i}" for i in range(1, N_DIM + 1)],
        output_dir_name="pideeponet_lorenz96",
        sample_initial_conditions=sample_initial_conditions,
        metadata={
            "n_dim": N_DIM,
            "forcing": FORCING,
            "ic_low": IC_LOW.tolist(),
            "ic_high": IC_HIGH.tolist(),
            "ic_sampling": "trajectory_aware_clipped_noise",
            "autonomous": True,
        },
        default_t=5.0,
        default_t1=0.5,
    )
