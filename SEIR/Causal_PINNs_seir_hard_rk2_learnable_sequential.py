"""SEIR hard RK2 sequential training with learnable RK2 c2."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as onp

from learnable_rk2_sequential import run_sequential_learnable_rk2


N_DIM = 4
BETA = 5.0
SIGMA = 1.0
GAMMA = 0.5

STATE0 = onp.array([0.5, 0.2, 0.2, 0.1], dtype=float)
N_POP = float(onp.sum(STATE0))


def seir_rhs(state, t):
    S, E, I, R = state
    infection = BETA * S * I / N_POP
    dS = -infection
    dE = infection - SIGMA * E
    dI = SIGMA * E - GAMMA * I
    dR = GAMMA * I
    return onp.array([dS, dE, dI, dR], dtype=float)


def make_rhs_jax(np):
    def rhs_fn(state):
        S, E, I, R = state
        infection = BETA * S * I / N_POP
        dS = -infection
        dE = infection - SIGMA * E
        dI = SIGMA * E - GAMMA * I
        dR = GAMMA * I
        return np.array([dS, dE, dI, dR])

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
            "rhs_ref": seir_rhs,
            "make_rhs_jax": make_rhs_jax,
            "LABELS": ["S", "E", "I", "R"],
            "OUT_DIR": os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "hard_rk2_learnable_causalpinn_seir_sequential",
            ),
            "PARAMS_MESSAGE": (
                f"SEIR params: beta={BETA}, sigma={SIGMA}, gamma={GAMMA}, "
                f"N_POP={N_POP}, R0={BETA / GAMMA:.3f}"
            ),
        }
    )


if __name__ == "__main__":
    main()
"""
Sequential Causal PINNs runner for the SEIR hard RK2 learnable-c2 variant.

Window k+1's IC comes from window k's prediction at t=T1. This reuses the
learnable RK2 window trainer and only changes the scheduling/IC protocol.
"""

import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

import numpy as onp
from scipy.integrate import odeint as scipy_odeint

import Causal_PINNs_seir_hard_rk2_learnable as base


T = 30.0
DT = base.DT
NUM_WINDOWS = int(T / base.T1)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "hard_rk2_learnable_causalpinn_seir_sequential",
)
LABELS = ["S", "E", "I", "R"]


def _detect_gpu_id():
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible.strip():
        return int(cuda_visible.split(",")[0])
    return 0


def _configure_base():
    base.T = T
    base.NUM_WINDOWS = NUM_WINDOWS
    base.CORRECTION_POWER = 1


def _save_results(state_pred_list, params_list, losses_list, rk_final_list, c2_history_list):
    os.makedirs(OUT_DIR, exist_ok=True)
    onp.save(os.path.join(OUT_DIR, "state_pred_list.npy"), onp.array(state_pred_list))
    onp.save(os.path.join(OUT_DIR, "params_list.npy"), onp.array(params_list))
    onp.save(
        os.path.join(OUT_DIR, "losses_list.npy"),
        onp.array(losses_list, dtype=object),
        allow_pickle=True,
    )
    onp.save(
        os.path.join(OUT_DIR, "rk_final_list.npy"),
        onp.array(rk_final_list, dtype=object),
        allow_pickle=True,
    )
    onp.save(
        os.path.join(OUT_DIR, "c2_history_list.npy"),
        onp.array(c2_history_list, dtype=object),
        allow_pickle=True,
    )


def _print_running_error(state_pred_list, states_ref):
    state_preds_so_far = onp.concatenate(state_pred_list, axis=0)
    n = min(len(state_preds_so_far), len(states_ref))
    err = onp.linalg.norm(state_preds_so_far[:n] - states_ref[:n]) / onp.linalg.norm(states_ref[:n])
    per_dim = onp.linalg.norm(state_preds_so_far[:n] - states_ref[:n], axis=0) / onp.linalg.norm(
        states_ref[:n], axis=0
    )
    print(f"  Cumulative relative L2 error: {err:.3e}")
    for label, value in zip(LABELS, per_dim):
        print(f"    {label}: {value:.3e}")


def main():
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault("PINN_ASSIGNED_GPU", str(_detect_gpu_id()))
    _configure_base()

    print(
        "Config: "
        f"T={T}, T1={base.T1}, DT={DT}, NUM_WINDOWS={NUM_WINDOWS}, "
        f"N_SUBSTEPS={base.N_SUBSTEPS}, N_ITER={base.N_ITER}, "
        f"CORRECTION_POWER={base.CORRECTION_POWER}"
    )
    print(f"TOL_LIST={base.TOL_LIST}")
    print(f"RK2 learnable: c2 in [{base.C2_MIN}, {base.C2_MAX}], alpha={base.C2_ALPHA}")
    print(
        f"SEIR params: beta={base.BETA}, sigma={base.SIGMA}, gamma={base.GAMMA}, "
        f"N_POP={base.N_POP}, R0={base.BETA / base.GAMMA:.3f}"
    )

    t_ref = onp.arange(0.0, T, DT)
    states_ref = scipy_odeint(base.seir_rhs, base.STATE0, t_ref)
    print(f"Reference solution computed: {states_ref.shape}")

    state0 = onp.array(base.STATE0, dtype=float)
    state_pred_list = []
    params_list = []
    losses_list = []
    rk_final_list = []
    c2_history_list = []

    for k in range(NUM_WINDOWS):
        print(f"\nFinal Time: {(k + 1) * base.T1:.1f}")
        result = base.train_window((k, state0.tolist(), 0.0, base.T1))

        state_pred_list.append(result["state_pred"])
        params_list.append(result["flat_params"])
        losses_list.append(result["losses"])
        rk_final_list.append(result["rk_final"])
        c2_history_list.append(result["c2_history"])
        state0 = onp.array(result["next_state"], dtype=float)

        _save_results(state_pred_list, params_list, losses_list, rk_final_list, c2_history_list)
        _print_running_error(state_pred_list, states_ref)

    c2_vals = onp.array([item["c2"] for item in rk_final_list], dtype=float)
    print(
        f"\nFinal RK2 c2: mean={c2_vals.mean():.6f}, std={c2_vals.std():.3e}, "
        f"min={c2_vals.min():.6f}, max={c2_vals.max():.6f}"
    )
    print("Done. Results saved to", OUT_DIR)


if __name__ == "__main__":
    main()
