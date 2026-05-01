"""Physics-informed DeepONet baseline for the autonomous FitzHugh-Nagumo system."""

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
A = 0.7
B = 0.8
TAU = 12.5
R_PARAM = 0.1
I_EXT = 0.5

STATE0 = onp.array([0.0, 0.0], dtype=float)
IC_LOW = onp.fromstring(os.environ.get("PI_DEEPONET_IC_LOW", "-2.0,-1.0"), sep=",")
IC_HIGH = onp.fromstring(os.environ.get("PI_DEEPONET_IC_HIGH", "2.0,2.0"), sep=",")


def fhn_rhs_np(state, _t):
    v, w = state
    return onp.array([
        v - (v**3) / 3.0 - w + R_PARAM * I_EXT,
        (v + A - B * w) / TAU,
    ], dtype=float)


def fhn_rhs_jax(state):
    v, w = state
    return jnp.array([
        v - (v**3) / 3.0 - w + R_PARAM * I_EXT,
        (v + A - B * w) / TAU,
    ])


def sample_initial_conditions(n_ic, seed=1234):
    rng = onp.random.RandomState(seed)
    horizon = float(os.environ.get("PI_DEEPONET_IC_HORIZON", "30.0"))
    dt = float(os.environ.get("PI_DEEPONET_IC_DT", "0.01"))
    noise_frac = float(os.environ.get("PI_DEEPONET_IC_NOISE_FRAC", "0.05"))
    t = onp.arange(0.0, horizon, dt)
    traj = scipy_odeint(fhn_rhs_np, STATE0, t)
    idx = rng.choice(len(traj), size=n_ic, replace=True)
    noise_scale = noise_frac * (traj.std(axis=0) + 1e-6)
    ics = traj[idx] + rng.randn(n_ic, N_DIM) * noise_scale
    ics = onp.clip(ics, IC_LOW, IC_HIGH)
    ics[0] = STATE0
    return ics.astype(onp.float32)


if __name__ == "__main__":
    run_pi_deeponet_ode(
        system_name="FitzHugh-Nagumo",
        rhs_np=fhn_rhs_np,
        rhs_jax=fhn_rhs_jax,
        state0=STATE0,
        labels=["v", "w"],
        output_dir_name="pideeponet_fhn",
        sample_initial_conditions=sample_initial_conditions,
        metadata={
            "a": A,
            "b": B,
            "tau": TAU,
            "r_param": R_PARAM,
            "i_ext": I_EXT,
            "ic_low": IC_LOW.tolist(),
            "ic_high": IC_HIGH.tolist(),
            "ic_sampling": "trajectory_aware_clipped_noise",
            "autonomous": True,
        },
        default_t=30.0,
        default_t1=0.5,
    )
