"""Physics-informed DeepONet baseline for the autonomous SEIR system."""

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

N_DIM = 4
BETA = 5.0
SIGMA = 1.0
GAMMA = 0.5

STATE0 = onp.array([0.5, 0.2, 0.2, 0.1], dtype=float)
N_POP = float(onp.sum(STATE0))


def seir_rhs_np(state, _t):
    S, E, I, R = state
    infection = BETA * S * I / N_POP
    return onp.array([
        -infection,
        infection - SIGMA * E,
        SIGMA * E - GAMMA * I,
        GAMMA * I,
    ], dtype=float)


def seir_rhs_jax(state):
    S, E, I, R = state
    infection = BETA * S * I / N_POP
    return jnp.array([
        -infection,
        infection - SIGMA * E,
        SIGMA * E - GAMMA * I,
        GAMMA * I,
    ])


def sample_initial_conditions(n_ic, seed=1234):
    """Sample near the target trajectory, projected onto the N_POP simplex."""
    rng = onp.random.RandomState(seed)
    horizon = float(os.environ.get("PI_DEEPONET_IC_HORIZON", "30.0"))
    dt = float(os.environ.get("PI_DEEPONET_IC_DT", "0.01"))
    noise_frac = float(os.environ.get("PI_DEEPONET_IC_NOISE_FRAC", "0.05"))
    t = onp.arange(0.0, horizon, dt)
    traj = scipy_odeint(seir_rhs_np, STATE0, t)
    idx = rng.choice(len(traj), size=n_ic, replace=True)
    noise_scale = noise_frac * (traj.std(axis=0) + 1e-6)
    ics = traj[idx] + rng.randn(n_ic, N_DIM) * noise_scale
    ics = onp.maximum(ics, 1e-8)
    ics = ics / ics.sum(axis=1, keepdims=True) * N_POP
    ics[0] = STATE0
    return ics.astype(onp.float32)


def seir_constraint_loss_jax(states):
    conservation = jnp.mean((jnp.sum(states, axis=1) - N_POP) ** 2)
    nonnegative = jnp.mean(jnp.minimum(states, 0.0) ** 2)
    return conservation + nonnegative


if __name__ == "__main__":
    run_pi_deeponet_ode(
        system_name="SEIR",
        rhs_np=seir_rhs_np,
        rhs_jax=seir_rhs_jax,
        state0=STATE0,
        labels=["S", "E", "I", "R"],
        output_dir_name="pideeponet_seir",
        sample_initial_conditions=sample_initial_conditions,
        constraint_loss_jax=seir_constraint_loss_jax,
        metadata={
            "beta": BETA,
            "sigma": SIGMA,
            "gamma": GAMMA,
            "n_pop": N_POP,
            "ic_sampling": "trajectory_aware_simplex_noise",
            "autonomous": True,
        },
        default_t=30.0,
        default_t1=0.5,
    )
