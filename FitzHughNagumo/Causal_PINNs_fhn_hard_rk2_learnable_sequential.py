"""FitzHugh-Nagumo hard RK2 sequential training with learnable RK2 c2."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as onp

from learnable_rk2_sequential import run_sequential_learnable_rk2


N_DIM = 2
A = 0.7
B = 0.8
TAU = 12.5
R_PARAM = 0.1
I_EXT = 0.5

STATE0 = onp.array([0.0, 0.0], dtype=float)


def fhn_rhs(state, t):
    v, w = state
    dv = v - (v**3) / 3.0 - w + R_PARAM * I_EXT
    dw = (v + A - B * w) / TAU
    return onp.array([dv, dw], dtype=float)


def make_rhs_jax(np):
    def rhs_fn(state):
        v = state[0]
        w = state[1]
        dv = v - (v**3) / 3.0 - w + R_PARAM * I_EXT
        dw = (v + A - B * w) / TAU
        return np.array([dv, dw])

    return rhs_fn


def main():
    run_sequential_learnable_rk2(
        {
            "T": 30.0,
            "T1": 0.5,
            "DT": 0.01,
            "TOL_LIST": [1e1, 1e2, 1e3, 1e4, 1e5],
            "LAYERS": [1, 512, 512, 512, N_DIM],
            "N_ITER": 300_000,
            "COLLOCATION_EXT_RATIO": 0.0,
            "RK2_SUBSTEPS": 1,
            "CORRECTION_POWER": 3,
            "ACTIVATION": "tanh",
            "INPUT_MODE": "tau",
            "USE_RATE_SCALE": True,
            "C2_ALPHA": 0.001,
            "STATE0": STATE0,
            "rhs_ref": fhn_rhs,
            "make_rhs_jax": make_rhs_jax,
            "LABELS": ["v", "w"],
            "OUT_DIR": os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "hard_rk2_learnable_causalpinn_fhn_sequential",
            ),
        }
    )


if __name__ == "__main__":
    main()
