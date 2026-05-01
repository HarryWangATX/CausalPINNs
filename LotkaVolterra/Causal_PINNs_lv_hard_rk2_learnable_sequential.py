"""Lotka-Volterra hard RK2 sequential training with learnable RK2 c2."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as onp

from learnable_rk2_sequential import run_sequential_learnable_rk2


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


def make_rhs_jax(np):
    def rhs_fn(state):
        x = state[0]
        y = state[1]
        dx = ALPHA * x - BETA * x * y
        dy = DELTA * x * y - GAMMA * y
        return np.array([dx, dy])

    return rhs_fn


def main():
    run_sequential_learnable_rk2(
        {
            "T": 30.0,
            "T1": 0.5,
            "DT": 0.01,
            "TOL_LIST": [1e0, 1e1, 1e2, 1e3, 1e4],
            "LAYERS": [1, 512, 512, 512, N_DIM],
            "N_ITER": 300_000,
            "COLLOCATION_EXT_RATIO": 0.0,
            "RK2_SUBSTEPS": 10,
            "CORRECTION_POWER": 3,
            "ACTIVATION": "tanh",
            "INPUT_MODE": "tau",
            "USE_RATE_SCALE": True,
            "C2_ALPHA": 0.001,
            "STATE0": STATE0,
            "rhs_ref": lv_rhs,
            "make_rhs_jax": make_rhs_jax,
            "LABELS": ["x_prey", "y_predator"],
            "OUT_DIR": os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "hard_rk2_learnable_causalpinn_lv_sequential",
            ),
            "PARAMS_MESSAGE": (
                f"LV params: alpha={ALPHA}, beta={BETA}, delta={DELTA}, "
                f"gamma={GAMMA}, equilibrium=({GAMMA / DELTA:.3f}, {ALPHA / BETA:.3f})"
            ),
        }
    )


if __name__ == "__main__":
    main()
