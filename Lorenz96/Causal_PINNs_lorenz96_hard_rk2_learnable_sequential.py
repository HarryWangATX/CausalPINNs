"""Lorenz-96 hard RK2 sequential training with learnable RK2 c2."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as onp

from learnable_rk2_sequential import run_sequential_learnable_rk2


N_DIM = 5
FORCING = 8.0

STATE0 = onp.full((N_DIM,), FORCING)
STATE0[0] += 0.01


def lorenz96_rhs(state, t):
    return (onp.roll(state, -1) - onp.roll(state, 2)) * onp.roll(state, 1) - state + FORCING


def make_rhs_jax(np):
    def rhs_fn(state):
        return (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1) - state + FORCING

    return rhs_fn


def main():
    run_sequential_learnable_rk2(
        {
            "T": 5.0,
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
            "rhs_ref": lorenz96_rhs,
            "make_rhs_jax": make_rhs_jax,
            "LABELS": [f"x{i}" for i in range(1, N_DIM + 1)],
            "OUT_DIR": os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "hard_rk2_learnable_causalpinn_lorenz96_sequential",
            ),
        }
    )


if __name__ == "__main__":
    main()
