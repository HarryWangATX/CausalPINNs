"""Physics-informed DeepONet baseline for the autonomous Lorenz system."""

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

RHO = 28.0
SIGMA = 10.0
BETA = 8.0 / 3.0
N_DIM = 3

STATE0 = onp.array([1.0, 1.0, 1.0], dtype=float)
IC_LOW = onp.fromstring(os.environ.get("PI_DEEPONET_IC_LOW", "-20,-25,0"), sep=",")
IC_HIGH = onp.fromstring(os.environ.get("PI_DEEPONET_IC_HIGH", "25,30,50"), sep=",")


def lorenz_rhs_np(state, _t):
    x, y, z = state
    return onp.array([
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    ], dtype=float)


def lorenz_rhs_jax(state):
    x, y, z = state
    return jnp.array([
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    ])


def sample_initial_conditions(n_ic, seed=1234):
    """Sample noisy states near the target Lorenz trajectory/attractor."""
    rng = onp.random.RandomState(seed)
    horizon = float(os.environ.get("PI_DEEPONET_IC_HORIZON", "12.0"))
    dt = float(os.environ.get("PI_DEEPONET_IC_DT", "0.01"))
    noise_frac = float(os.environ.get("PI_DEEPONET_IC_NOISE_FRAC", "0.05"))
    t = onp.arange(0.0, horizon, dt)
    traj = scipy_odeint(lorenz_rhs_np, STATE0, t)
    idx = rng.choice(len(traj), size=n_ic, replace=True)
    noise_scale = noise_frac * (traj.std(axis=0) + 1e-6)
    ics = traj[idx] + rng.randn(n_ic, N_DIM) * noise_scale
    ics = onp.clip(ics, IC_LOW, IC_HIGH)
    ics[0] = STATE0
    return ics.astype(onp.float32)


if __name__ == "__main__":
    if os.environ.get("PI_DEEPONET_USE_HARD_RK2", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or os.environ.get("PI_DEEPONET_USE_LEARNABLE_RK2", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or os.environ.get("PI_DEEPONET_USE_HARD_EULER", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        os.environ.setdefault("PI_DEEPONET_CORRECTION_POWER", "1.0")

    run_pi_deeponet_ode(
        system_name="Lorenz",
        rhs_np=lorenz_rhs_np,
        rhs_jax=lorenz_rhs_jax,
        state0=STATE0,
        labels=["x", "y", "z"],
        output_dir_name="pideeponet_lorentz_rho28_T12",
        sample_initial_conditions=sample_initial_conditions,
        metadata={
            "rho": RHO,
            "sigma": SIGMA,
            "beta": BETA,
            "ic_low": IC_LOW.tolist(),
            "ic_high": IC_HIGH.tolist(),
            "ic_sampling": "trajectory_aware_clipped_noise",
            "autonomous": True,
        },
        default_t=12.0,
        default_t1=0.5,
    )
