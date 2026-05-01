"""Lorenz hard RK2 sequential training with learnable RK2 c2."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as onp

from learnable_rk2_sequential import run_sequential_learnable_rk2


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


def make_rhs_jax(np):
    def rhs_fn(state):
        x = state[0]
        y = state[1]
        z = state[2]
        dx = SIGMA * (y - x)
        dy = x * (RHO - z) - y
        dz = x * y - BETA * z
        return np.array([dx, dy, dz])

    return rhs_fn


def main():
    run_sequential_learnable_rk2(
        {
            "T": 12.0,
            "T1": 0.5,
            "DT": 0.01,
            "TOL_LIST": [1e-3, 1e-2, 1e-1, 1e0, 1e1],
            "LAYERS": [1, 512, 512, 512, N_DIM],
            "N_ITER": 300_000,
            "COLLOCATION_EXT_RATIO": 0.1,
            "RK2_SUBSTEPS": 10,
            "CORRECTION_POWER": 1,
            "ACTIVATION": "tanh",
            "INPUT_MODE": "raw",
            "USE_RATE_SCALE": False,
            "C2_ALPHA": 0.0001,
            "STATE0": STATE0,
            "rhs_ref": lorenz_rhs,
            "make_rhs_jax": make_rhs_jax,
            "LABELS": ["x", "y", "z"],
            "OUT_DIR": os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "hard_rk2_learnable_causalpinn_lorentz_sequential",
            ),
        }
    )


if __name__ == "__main__":
    main()
