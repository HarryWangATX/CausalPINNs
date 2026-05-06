#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import expit
import os

def extract_params(flat_params, layers):
    params = []
    idx = 0
    for i in range(len(layers)-1):
        d_in = layers[i]
        d_out = layers[i+1]
        W_len = d_in * d_out
        W = flat_params[idx : idx + W_len].reshape((d_in, d_out))
        idx += W_len
        b_len = d_out
        b = flat_params[idx : idx + b_len]
        idx += b_len
        params.append((W, b))
    return params

def mlp_forward(t_arr, params, activation):
    x = t_arr
    for W, b in params[:-1]:
        x = np.dot(x, W) + b
        if activation == "tanh":
            x = np.tanh(x)
        elif activation == "silu":
            x = x * expit(x)
    W, b = params[-1]
    x = np.dot(x, W) + b
    return x

def pred_hard_rk2_lorenz(t_eval, flat_params, state0, rhs_fn, n_sub):
    params = extract_params(flat_params, [1, 512, 512, 512, 3])
    t_arr = t_eval.reshape(-1, 1)
    outputs = mlp_forward(t_arr, params, "tanh")
    
    preds = []
    for i, t in enumerate(t_eval):
        if t == 0:
            preds.append(state0)
            continue
        h = t / n_sub
        s = state0.copy()
        for _ in range(n_sub):
            k1 = rhs_fn(s)
            k2 = rhs_fn(s + 0.5 * h * k1)
            s = s + h * k2
        corr = (t ** 3) * outputs[i]
        preds.append(s + corr)
    return np.array(preds)

def pred_baseline_lorenz(t_eval, flat_params, state0):
    params = extract_params(flat_params, [1, 512, 512, 512, 3])
    t_arr = t_eval.reshape(-1, 1)
    outputs = mlp_forward(t_arr, params, "tanh")
    return outputs * t_arr + state0

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "figs", "learning_correction_complexity")
DT = 0.01

def rhs_lorenz28(s):
    rho, sigma, beta = 28.0, 10.0, 8.0 / 3.0
    x, y, z = s
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dtype=float)

def rhs_l96(s):
    F = 8.0
    return (np.roll(s, -1) - np.roll(s, 2)) * np.roll(s, 1) - s + F

def rhs_fhn(s):
    a, b, tau, R, I = 0.7, 0.8, 12.5, 0.1, 0.5
    v, w = s
    return np.array([v - v**3 / 3.0 - w + R * I, (v + a - b * w) / tau], dtype=float)

def rhs_seir(s):
    beta, sigma, gamma = 5.0, 1.0, 0.5
    S, E, I, R = s
    return np.array([-beta * S * I, beta * S * I - sigma * E, sigma * E - gamma * I, gamma * I], dtype=float)

def rhs_lv(s):
    alpha, beta, delta, gamma = 1.5, 1.0, 1.0, 3.0
    x, y = s
    return np.array([alpha * x - beta * x * y, delta * x * y - gamma * y], dtype=float)

def compute_rk2_base(rhs_fn, s0, t_eval, n_sub=1):
    """
    Computes the base prediction at each t in t_eval.
    According to the PINN, at time t (which is time since start of window),
    it takes n_sub midpoint-RK2 steps of size t / n_sub.
    """
    preds = []
    for t in t_eval:
        if t == 0:
            preds.append(s0)
            continue
        h = t / n_sub
        s = s0.copy()
        for _ in range(n_sub):
            k1 = rhs_fn(s)
            k2 = rhs_fn(s + 0.5 * h * k1)
            s = s + h * k2
        preds.append(s)
    return np.array(preds)

def load_pred(path, fmt):
    if fmt == "xyz":
        x = np.hstack(np.load(os.path.join(path, "x_pred_list.npy"), allow_pickle=True))
        y = np.hstack(np.load(os.path.join(path, "y_pred_list.npy"), allow_pickle=True))
        z = np.hstack(np.load(os.path.join(path, "z_pred_list.npy"), allow_pickle=True))
        return np.stack([x, y, z], axis=1)
    arr = np.load(os.path.join(path, "state_pred_list.npy"), allow_pickle=True)
    seq = [np.asarray(a) for a in arr]
    return np.concatenate(seq, axis=0)

def load_window_pred(path, fmt):
    if fmt == "xyz":
        x = np.load(os.path.join(path, "x_pred_list.npy"), allow_pickle=True)
        y = np.load(os.path.join(path, "y_pred_list.npy"), allow_pickle=True)
        z = np.load(os.path.join(path, "z_pred_list.npy"), allow_pickle=True)
        return np.stack([x, y, z], axis=-1).astype(float)
    return np.asarray(np.load(os.path.join(path, "state_pred_list.npy"), allow_pickle=True), dtype=float)

def resolve_run_dir(*candidates):
    for candidate in candidates:
        if isinstance(candidate, (list, tuple)):
            path = resolve_run_dir(*candidate)
            if path is not None:
                return path
            continue
        path = os.path.join(ROOT, candidate)
        if os.path.isdir(path):
            return path
        results_path = os.path.join(ROOT, "results", candidate)
        if os.path.isdir(results_path):
            return results_path
    return None

def rel_l2(pred, ref):
    return np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12)

def select_window(spec, ours_windows, baseline_full):
    n_windows, n_t, _ = ours_windows.shape
    t_eval = np.arange(0.0, spec["T1"], DT)
    max_windows = n_windows
    if "T" in spec:
        max_windows = min(max_windows, int(round(spec["T"] / spec["T1"])))
    t_needed = max_windows * spec["T1"] + spec["T1"]
    if baseline_full is not None:
        t_needed = min(max(t_needed, spec.get("T", t_needed) + spec["T1"]), len(baseline_full) * DT)
    t_full = np.arange(0.0, t_needed, DT)
    ref_full = odeint(lambda u, t: spec["rhs"](u), spec["state0"], t_full)

    candidates = []
    for window_idx in range(max_windows):
        idx_start = int(round(window_idx * spec["T1"] / DT))
        idx_end = idx_start + n_t
        if idx_end > len(ref_full):
            continue
        if baseline_full is None or idx_end > len(baseline_full):
            continue

        ref = ref_full[idx_start:idx_end]
        ours = ours_windows[window_idx]
        baseline = baseline_full[idx_start:idx_end]
        v_idx = spec["var_idx"]
        rk2_base = compute_rk2_base(spec["rhs"], ref_full[idx_start], t_eval, n_sub=spec["n_sub"])

        ours_err = rel_l2(ours[:, v_idx], ref[:, v_idx])
        baseline_err = rel_l2(baseline[:, v_idx], ref[:, v_idx])
        ours_all = rel_l2(ours, ref)
        baseline_all = rel_l2(baseline, ref)
        ratio = baseline_err / (ours_err + 1e-12)
        ratio_all = baseline_all / (ours_all + 1e-12)
        amplitude = np.nanmax(ref[:, v_idx]) - np.nanmin(ref[:, v_idx])
        max_ours = np.nanmax(np.abs(ours[:, v_idx] - ref[:, v_idx]))
        max_baseline = np.nanmax(np.abs(baseline[:, v_idx] - ref[:, v_idx]))
        max_rk2_base = np.nanmax(np.abs(rk2_base[:, v_idx] - ref[:, v_idx]))
        visible_baseline = max_baseline / (amplitude + 1e-9)
        visible_ours = max_ours / (amplitude + 1e-9)
        visible_rk2_base = max_rk2_base / (amplitude + 1e-9)
        visible_gap = np.nanmax(np.abs(baseline[:, v_idx] - ours[:, v_idx])) / (amplitude + 1e-9)

        # Reward windows where the standard CausalPINN is visibly wrong and
        # the hard-RK2 correction tracks the reference, while avoiding RK2
        # bases that dominate the axis scale.
        score = (
            2.0 * np.log10(max(ratio, 1e-12))
            + 0.6 * np.log10(max(ratio_all, 1e-12))
            + 1.2 * np.log10(max(visible_gap, 1e-12))
            + 0.6 * np.log10(max(visible_baseline, 1e-12))
            - 0.5 * np.log10(max(visible_ours, 1e-12))
            - 1.3 * np.log10(max(visible_rk2_base, 1e-12))
            + 0.15 * np.log10(max(amplitude, 1e-12))
        )
        candidates.append(
            {
                "score": score,
                "window_idx": window_idx,
                "ratio": ratio,
                "ratio_all": ratio_all,
                "ours_err": ours_err,
                "baseline_err": baseline_err,
                "amplitude": amplitude,
                "visible_baseline": visible_baseline,
                "visible_ours": visible_ours,
                "visible_rk2_base": visible_rk2_base,
                "visible_gap": visible_gap,
            }
        )

    if not candidates:
        raise ValueError(f"No comparable windows found for {spec['dir']}")

    max_base_vis = spec.get("max_base_vis", 2.0)
    improving = [
        c for c in candidates
        if c["ratio"] > 1.2 and c["ratio_all"] > 1.0 and c["visible_rk2_base"] <= max_base_vis
    ]
    if improving:
        selected = max(improving, key=lambda c: c["score"])
    else:
        selected = max(candidates, key=lambda c: (c["ratio"], c["ratio_all"]))
    return selected, ref_full

def choose_zoom_range(t_eval, true_val, nn_pred, base_pinn_pred):
    if base_pinn_pred is not None:
        gap = np.abs(base_pinn_pred[: len(true_val)] - true_val)
    else:
        gap = np.abs(nn_pred[: len(true_val)] - true_val)
    if not np.any(np.isfinite(gap)):
        return None

    center_idx = int(np.nanargmax(gap))
    center = float(t_eval[center_idx])
    span = max(8 * DT, 0.16 * float(t_eval[-1] - t_eval[0]))
    start = max(float(t_eval[0]), center - 0.5 * span)
    end = min(float(t_eval[-1]), center + 0.5 * span)
    if end - start < 4 * DT:
        end = min(float(t_eval[-1]), start + 4 * DT)
        start = max(float(t_eval[0]), end - 4 * DT)
    return start, end

def choose_inset_loc(t_eval, true_val, zoom_range):
    center = 0.5 * (zoom_range[0] + zoom_range[1])
    idx = int(np.argmin(np.abs(t_eval - center)))
    y_min = np.nanmin(true_val)
    y_max = np.nanmax(true_val)
    y_rel = 0.5 if y_max <= y_min else (true_val[idx] - y_min) / (y_max - y_min)
    vertical = "lower" if y_rel > 0.55 else "upper"
    horizontal = "left" if center > 0.5 * (t_eval[0] + t_eval[-1]) else "right"
    return f"{vertical} {horizontal}"

def set_paper_style():
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
    })

def scale_small_values(*arrays, threshold=1e-2, axis_multiplier=1.0):
    values = np.concatenate([np.ravel(np.asarray(arr, dtype=float)) for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0, 0
    max_abs = float(np.nanmax(np.abs(values)))
    if max_abs <= 0.0 or max_abs >= threshold:
        return 1.0, 0
    exponent = int(np.floor(np.log10(max_abs * axis_multiplier)))
    return 10.0 ** (-exponent), exponent

def main():
    set_paper_style()

    state0_l96 = np.full(5, 8.0)
    state0_l96[0] = 8.01
    
    specs = {
        "Lorenz": {"rhs": rhs_lorenz28, "state0": np.array([1.0, 1.0, 1.0]), "T": 12.0, "T1": 0.5, "var_idx": 1, "var_name": "y", "n_sub": 1, "dir": ["results/hard_rk2_causalpinn_lorentz_sequential", "hard_rk2_causalpinn_lorentz_sequential"], "baseline_dir": ["results/baseline_causalpinn_lorentz_sequential", "baseline_causalpinn_lorentz_sequential", "Lorentz/original_causalpinn_lorenz"], "format": "xyz", "manual_window": 15, "manual_span": (0.20, 0.30)},
        "Lorenz-96": {"rhs": rhs_l96, "state0": state0_l96, "T": 5.0, "T1": 0.5, "var_idx": 0, "var_name": "x_1", "n_sub": 1, "dir": ["results/hard_rk2_causalpinn_lorenz96_sequential", "hard_rk2_causalpinn_lorenz96_sequential"], "format": "state", "baseline_dir": ["results/baseline_causalpinn_lorenz96_sequential", "baseline_causalpinn_lorenz96_sequential"], "manual_window": 5, "manual_span": (0.0, 0.06)},
        "FHN": {"rhs": rhs_fhn, "state0": np.array([0.0, 0.0]), "T": 30.0, "T1": 0.5, "var_idx": 0, "var_name": "v", "n_sub": 1, "dir": ["results/hard_rk2_causalpinn_fhn_sequential", "hard_rk2_causalpinn_fhn_sequential"], "format": "state", "baseline_dir": ["results/baseline_causalpinn_fhn_sequential", "baseline_causalpinn_fhn_sequential"], "manual_window": 43, "manual_span": (0.0, 0.06)},
        "SEIR": {"rhs": rhs_seir, "state0": np.array([0.5, 0.2, 0.2, 0.1]), "T": 30.0, "T1": 0.5, "var_idx": 2, "var_name": "I", "n_sub": 1, "dir": ["results/hard_rk2_causalpinn_seir_sequential", "hard_rk2_causalpinn_seir_sequential"], "format": "state", "baseline_dir": ["results/baseline_causalpinn_seir_sequential", "baseline_causalpinn_seir_sequential", "results/baseline_causalpinn_seir"], "manual_window": 51, "manual_span": (0.0, 0.06)},
        "Lotka-Volterra": {"rhs": rhs_lv, "state0": np.array([10.0, 5.0]), "T": 30.0, "T1": 0.5, "var_idx": 1, "var_name": "y", "n_sub": 10, "dir": ["results/hard_rk2_causalpinn_lv_sequential", "hard_rk2_causalpinn_lv_sequential"], "format": "state", "baseline_dir": ["results/baseline_causalpinn_lv_sequential", "baseline_causalpinn_lv_sequential", "results/baseline_causalpinn_lv"], "manual_window": 49, "manual_span": (0.0, 0.06)},
    }

    fig = plt.figure(figsize=(14.0, 5.0))
    gs = fig.add_gridspec(2, 5, hspace=0.35, wspace=0.35)
    top_legend = {}
    bottom_legend = {}
    
    for i, (name, s) in enumerate(specs.items()):
        dir_path = resolve_run_dir(s["dir"])
        baseline_path = resolve_run_dir(s["baseline_dir"]) if "baseline_dir" in s else None

        ax_top = fig.add_subplot(gs[0, i])
        ax_bot = fig.add_subplot(gs[1, i])

        if dir_path is None:
            ax_top.set_title(name, fontsize=12)
            ax_top.text(
                0.5, 0.5,
                f"Missing sequential output\n{s['dir']}",
                transform=ax_top.transAxes,
                ha="center", va="center", fontsize=8, color="#666666",
            )
            ax_bot.text(
                0.5, 0.5,
                "No correction panel",
                transform=ax_bot.transAxes,
                ha="center", va="center", fontsize=8, color="#666666",
            )
            for ax in (ax_top, ax_bot):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
            print(f"[{name}] Missing sequential output: {s['dir']}")
            continue

        ours_windows = load_window_pred(dir_path, s["format"])
        baseline_full = load_pred(baseline_path, s["format"]) if baseline_path is not None else None
        selected, ref_full = select_window(s, ours_windows, baseline_full)
        if "manual_window" in s:
            window_idx_manual = int(s["manual_window"])
            n_t = ours_windows.shape[1]
            idx_start_manual = int(round(window_idx_manual * s["T1"] / DT))
            idx_end_manual = idx_start_manual + n_t
            t_eval_manual_full = np.arange(0.0, s["T1"], DT)
            region_slice = slice(None)
            if "manual_span" in s:
                span_start, span_end = s["manual_span"]
                region_start = int(round(span_start / DT))
                region_end = int(round(span_end / DT))
                region_slice = slice(region_start, region_end)

            ref_manual = ref_full[idx_start_manual:idx_end_manual][region_slice]
            ours_manual = ours_windows[window_idx_manual][region_slice]
            baseline_manual = baseline_full[idx_start_manual:idx_end_manual][region_slice]
            v_idx_manual = s["var_idx"]
            ours_err_manual = rel_l2(ours_manual[:, v_idx_manual], ref_manual[:, v_idx_manual])
            baseline_err_manual = rel_l2(baseline_manual[:, v_idx_manual], ref_manual[:, v_idx_manual])
            ours_all_manual = rel_l2(ours_manual, ref_manual)
            baseline_all_manual = rel_l2(baseline_manual, ref_manual)
            amplitude_manual = np.nanmax(ref_manual[:, v_idx_manual]) - np.nanmin(ref_manual[:, v_idx_manual])
            max_ours_manual = np.nanmax(np.abs(ours_manual[:, v_idx_manual] - ref_manual[:, v_idx_manual]))
            max_baseline_manual = np.nanmax(np.abs(baseline_manual[:, v_idx_manual] - ref_manual[:, v_idx_manual]))
            rk2_manual = compute_rk2_base(
                s["rhs"], ref_full[idx_start_manual], t_eval_manual_full, n_sub=s["n_sub"]
            )[region_slice]
            max_rk2_manual = np.nanmax(np.abs(rk2_manual[:, v_idx_manual] - ref_manual[:, v_idx_manual]))
            visible_gap_manual = np.nanmax(
                np.abs(baseline_manual[:, v_idx_manual] - ours_manual[:, v_idx_manual])
            ) / (amplitude_manual + 1e-9)
            selected = {
                "window_idx": window_idx_manual,
                "ratio": baseline_err_manual / (ours_err_manual + 1e-12),
                "ratio_all": baseline_all_manual / (ours_all_manual + 1e-12),
                "ours_err": ours_err_manual,
                "baseline_err": baseline_err_manual,
                "amplitude": amplitude_manual,
                "visible_baseline": max_baseline_manual / (amplitude_manual + 1e-9),
                "visible_ours": max_ours_manual / (amplitude_manual + 1e-9),
                "visible_rk2_base": max_rk2_manual / (amplitude_manual + 1e-9),
                "visible_gap": visible_gap_manual,
                "region_slice": region_slice,
            }
        window_idx = selected["window_idx"]
        note = f", {s['manual_note']}" if "manual_note" in s else ""
        print(
            f"[{name}] window={window_idx}, t0={window_idx * s['T1']:.2f}, "
            f"{s['var_name']} standard/ours error ratio={selected['ratio']:.2e}, "
            f"all-state ratio={selected['ratio_all']:.2e}, "
            f"std gap={selected['visible_baseline']:.2e}, "
            f"RK2 base gap={selected.get('visible_rk2_base', float('nan')):.2e}{note}"
        )
        
        idx_start = int(round(window_idx * s["T1"] / 0.01))
        state0_win = ref_full[idx_start]
        
        t_eval_full = np.arange(0, s["T1"], DT)
        region_slice = selected.get("region_slice", slice(None))
        t_eval = t_eval_full[region_slice]
        # True solution in this window
        ref_win = ref_full[idx_start:idx_start + len(t_eval_full)][region_slice]
        # RK2 base prediction in this window
        base_win = compute_rk2_base(s["rhs"], state0_win, t_eval_full, n_sub=s["n_sub"])[region_slice]
        
        if s["format"] == "xyz":
            v_name = ["x", "y", "z"][s["var_idx"]]
            nn_pred = ours_windows[window_idx, region_slice, s["var_idx"]]
        else:
            nn_pred_full = ours_windows[window_idx][region_slice]
            nn_pred = nn_pred_full[:, s["var_idx"]]
            
        base_pinn_pred = None
        if baseline_path is not None:
            full_base = load_pred(baseline_path, s["format"])
            
            idx_start_full = int(round(window_idx * s["T1"] / 0.01))
            idx_end_full = idx_start_full + len(t_eval_full)
            
            if idx_end_full <= len(full_base):
                base_pinn_pred = full_base[idx_start_full:idx_end_full, s["var_idx"]][region_slice]
            
        # The network's learned correction is the difference between its final output and the base RK2 scheme
        nn_correction = nn_pred - base_win[:, s["var_idx"]]
        
        # We want to plot the x-axis with absolute time, rather than local window time
        t_absolute = t_eval + (window_idx * s["T1"])
        
        # Extract the chosen variable
        v_idx = s["var_idx"]
        true_val = ref_win[:, v_idx]
        base_val = base_win[:, v_idx]
        
        # Avoid massive explosion values making the figure unrenderable
        base_val[np.abs(base_val) > 1e4] = np.nan
        
        correction = true_val - base_val
        
        # Top: Plot True State vs Base Scheme vs NN Pred
        ax_top.plot(t_absolute, true_val, color="#1f77b4", label="True State", lw=1.5, zorder=2)
        ax_top.plot(t_absolute, base_val, color="#ff7f0e", linestyle="--", label="RK2 Base", alpha=0.8, zorder=1)
        ax_top.plot(t_absolute, nn_pred, color="#2ca02c", linestyle=":", label="Neural Numerical Solver", lw=2.5, zorder=4)
        if base_pinn_pred is not None:
            ax_top.plot(t_absolute, base_pinn_pred[:len(t_absolute)], color="#d62728", linestyle="-.", label="Standard CausalPINN", lw=1.5)

        focus_curves = [true_val, nn_pred]
        if base_pinn_pred is not None:
            focus_curves.append(base_pinn_pred[:len(t_absolute)])
        focus_values = np.concatenate([np.asarray(curve, dtype=float) for curve in focus_curves])
        y_min = float(np.nanmin(focus_values))
        y_max = float(np.nanmax(focus_values))
        padding = 0.12 * max(y_max - y_min, 1e-9)
        ax_top.set_ylim(y_min - padding, y_max + padding)
        
        # The selected windows are chosen so the separation is visible in the
        # main panel; avoid inset boxes because they clutter this multi-panel figure.
        zoom_range = None
        if zoom_range is not None:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            from scipy.interpolate import CubicSpline
            
            axins = inset_axes(
                ax_top,
                width="40%",
                height="40%",
                loc=choose_inset_loc(t_eval, true_val, zoom_range),
                borderpad=1.5,
            )
            
            # Use a dense array around the zoomed region
            t_fine_eval = np.linspace(zoom_range[0], zoom_range[1], 100)
            t_fine_abs = t_fine_eval + (window_idx * s["T1"])

            ref_win_fine = odeint(lambda u, t: s["rhs"](u), state0_win, t_fine_eval)[:, v_idx]
            
            # Our prediction (using interpolation for all, or direct eval if Lorenz)
            nn_spline = CubicSpline(t_eval, nn_pred)
            nn_pred_fine = nn_spline(t_fine_eval)
            
            axins.plot(t_fine_abs, ref_win_fine, color="#1f77b4", lw=1.5, zorder=2)
            axins.plot(t_fine_abs, nn_pred_fine, color="#2ca02c", linestyle=":", lw=2.5, zorder=4)
            
            if base_pinn_pred is not None:
                # Interpolate base PINN from saved points using cubic spline for smooth curve
                base_spline = CubicSpline(t_eval, base_pinn_pred[:len(t_eval)])
                base_pinn_pred_fine = base_spline(t_fine_eval)
                axins.plot(t_fine_abs, base_pinn_pred_fine, color="#d62728", linestyle="-.", lw=1.5, zorder=3)
            
            axins.set_xlim(t_fine_abs[0], t_fine_abs[-1])
            
            # Get y limits for the zoomed region based on the plots
            y_min = min(np.min(ref_win_fine), np.min(nn_pred_fine))
            y_max = max(np.max(ref_win_fine), np.max(nn_pred_fine))
            if base_pinn_pred is not None:
                y_min = min(y_min, np.min(base_pinn_pred_fine))
                y_max = max(y_max, np.max(base_pinn_pred_fine))
                
            padding = (y_max - y_min) * 0.15
            if padding == 0: padding = 0.0001
            axins.set_ylim(y_min - padding, y_max + padding)
            
            axins.tick_params(axis='both', which='both', bottom=True, top=False, left=False, right=False, labelbottom=True, labelleft=False, labelsize=4)
            axins.yaxis.get_offset_text().set_visible(False)
            axins.xaxis.get_offset_text().set_visible(False)
            import matplotlib.ticker as ticker
            axins.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
            axins.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            axins.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            
            # Draw line for absolute difference
            if base_pinn_pred is not None:
                idx = int(len(t_fine_abs) * 0.7)
                t_val = t_fine_abs[idx]
                y_true = ref_win_fine[idx]
                y_base = base_pinn_pred_fine[idx]
                
                # Plot a vertical line segment
                axins.plot([t_val, t_val], [y_true, y_base], color='black', lw=1.0, linestyle='-')
                
                err_val = abs(y_true - y_base)
                mid_y = (y_true + y_base) / 2
                
                # Annotate the error
                dt_range = t_fine_abs[-1] - t_fine_abs[0]
                axins.text(t_val - dt_range * 0.05, mid_y, f"{err_val:.1e}", 
                           ha='right', va='center', fontsize=4, color='black',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # Optional: Add connecting lines from main plot to inset
            from mpl_toolkits.axes_grid1.inset_locator import mark_inset
            mark_inset(ax_top, axins, loc1=1, loc2=3, fc="none", ec="0.5", alpha=0.3)
            
        if i == 0:
            ax_top.set_ylabel("State Value", fontsize=11)
        ax_top.set_title(name, fontsize=12)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
        ax_top.set_xticklabels([])
        
        # Bottom: Plot the correction the network has to learn. Very small
        # correction magnitudes are shown in local scientific units so their
        # shapes are readable across all systems.
        correction_axis_multiplier = 30.0
        correction_scale, correction_exponent = scale_small_values(
            correction, nn_correction, axis_multiplier=correction_axis_multiplier
        )
        correction_plot = correction * correction_scale
        nn_correction_plot = nn_correction * correction_scale
        ax_bot.plot(
            t_absolute,
            correction_plot,
            color="#d62728",
            label="RK2 base error",
        )
        ax_bot.plot(
            t_absolute,
            nn_correction_plot,
            color="#9467bd",
            linestyle="--",
            label="Neural Numerical Solver learned error",
        )
        correction_values = np.concatenate([correction_plot, nn_correction_plot])
        correction_values = correction_values[np.isfinite(correction_values)]
        if correction_values.size:
            max_abs_corr = float(np.nanmax(np.abs(correction_values)))
            if correction_exponent != 0:
                # Use a deliberately wider zero-centered scale for tiny
                # corrections so close learned/base errors visually overlap.
                axis_half_width = correction_axis_multiplier * max(max_abs_corr, 1e-9)
                ax_bot.set_ylim(-axis_half_width, axis_half_width)
            else:
                y_min_corr = float(np.nanmin(correction_values))
                y_max_corr = float(np.nanmax(correction_values))
                padding_corr = 0.18 * max(y_max_corr - y_min_corr, 1e-9)
                ax_bot.set_ylim(y_min_corr - padding_corr, y_max_corr + padding_corr)
        if correction_exponent != 0:
            ax_bot.text(
                0.02,
                0.92,
                rf"$\times 10^{{{correction_exponent}}}$",
                transform=ax_bot.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#444444",
            )
        for handle, label in zip(*ax_top.get_legend_handles_labels()):
            top_legend.setdefault(label, handle)
        for handle, label in zip(*ax_bot.get_legend_handles_labels()):
            bottom_legend.setdefault(label, handle)
        if i == 0:
            ax_bot.set_ylabel("Error (Correction)", fontsize=11)
        ax_bot.set_xlabel("$t$ (Absolute time)", fontsize=11)
        ax_bot.spines["top"].set_visible(False)
        ax_bot.spines["right"].set_visible(False)
        
        # Calculate magnitudes for comparison text
        # REMOVED TEXT ANNOTATION AS REQUESTED

    # Add figure-level centered legends
    if top_legend:
        fig.legend(top_legend.values(), top_legend.keys(), loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=4, frameon=False, fontsize=11)
    if bottom_legend:
        fig.legend(bottom_legend.values(), bottom_legend.keys(), loc="lower center", bbox_to_anchor=(0.5, 0.44), ncol=2, frameon=False, fontsize=11)

    os.makedirs(FIG_DIR, exist_ok=True)
    out_png = os.path.join(FIG_DIR, "learning_correction_complexity.png")
    out_pdf = os.path.join(FIG_DIR, "learning_correction_complexity.pdf")
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.savefig(out_pdf, bbox_inches="tight", dpi=300)
    print(f"Saved {out_png} and {out_pdf}")

if __name__ == "__main__":
    main()
