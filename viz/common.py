"""Shared helpers for paper figures. Reads the CSV(s) written by main.py."""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["random", "coreset", "lsh", "hybrid", "el2n", "ccs", "d2", "uncertainty", "knn_shapley", "valuation"]
LABELS = {"random": "Random", "coreset": "k-Center Greedy", "lsh": "LSH", "hybrid": "Hybrid", "el2n": "EL2N",
          "ccs": "CCS (AUM)", "d2": "D2 Pruning", "uncertainty": "Uncertainty", "knn_shapley": "KNN-Shapley", "valuation": "KNN-Shapley"}
COLORS = {"random": "#7f7f7f", "coreset": "#d62728", "lsh": "#1f77b4", "hybrid": "#2ca02c", "el2n": "#ff7f0e",
          "ccs": "#9467bd", "d2": "#8c564b", "uncertainty": "#ff7f0e", "knn_shapley": "#17becf", "valuation": "#17becf"}
MARKERS = {"random": "o", "coreset": "s", "lsh": "^", "hybrid": "D", "el2n": "v", "ccs": "P", "d2": "X",
           "uncertainty": "v", "knn_shapley": "*", "valuation": "*"}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 100, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

NICE = {
    "accuracy": "Test accuracy", "ece": "ECE", "ood_auroc": "OOD AUROC", "ood_fpr95": "OOD FPR95",
    "w_effective_rank": "Effective rank", "w_stable_rank": "Stable rank",
    "w_spectral_entropy": "Spectral entropy", "w_decay_rate": "Spectral decay rate",
    "w_spectral_norm": "Spectral norm", "w_frobenius_norm": "Frobenius norm",
    "w_nuclear_norm": "Nuclear norm", "w_condition_number": "Condition number",
    "w_top1_energy": "Top-1 singular energy", "w_value_entropy": "Weight-value entropy",
    "w_kurtosis": "Weight excess kurtosis", "w_mean": "Weight mean", "w_std": "Weight std",
    "w_rownorm_mean": "Class-vector norm (mean)", "w_rownorm_std": "Class-vector norm (std)",
    "w_mean_cos_sim": "Mean class cosine sim.", "w_bias_norm": "Bias norm",
}


def setup(desc, clean_only=True, need_training=True):
    """clean_only=True keeps just the label-noise-free runs, so the original figures 1-10 are not
    contaminated by the noisy-label runs that share the same CSV. The robustness figures pass False."""
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--csv", default="./*results*.csv", help="glob of result CSVs (multiple files = seeds/replicates)")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    return args, load_results(args.csv, clean_only=clean_only, need_training=need_training)


def load_results(pattern, clean_only=True, need_training=True):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No CSV matched {pattern!r}. Run main.py first or pass --csv.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # columns added by the robustness studies may be missing in older CSVs
    if "label_noise_type" not in df:
        df["label_noise_type"] = "none"
    if "label_noise_rate" not in df:
        df["label_noise_rate"] = 0.0
    df["label_noise_type"] = df["label_noise_type"].fillna("none")
    df["label_noise_rate"] = df["label_noise_rate"].fillna(0.0)
    if clean_only:
        df = df[df.label_noise_type == "none"]
    if need_training:
        df = df[df["accuracy"].notna()]               # drops selection-only rows
    df = df.reset_index(drop=True)
    if "total_energy_j" in df and "idle_power_w" in df:
        # energy with the measured idle floor subtracted (idle_power_w * duration), floored at 0
        idle = df["idle_power_w"].fillna(0)
        df["net_prune_energy_j"] = (df["prune_energy_j"] - idle * df["prune_time_s"]).clip(lower=0)
        df["net_train_energy_j"] = (df["train_energy_j"] - idle * df["train_time_s"]).clip(lower=0)
        df["net_total_energy_j"] = df["net_prune_energy_j"] + df["net_train_energy_j"]
    if need_training and len(df):
        df["sv"] = df["singular_values"].apply(lambda s: np.array(json.loads(s)))
        noise = df["noise_robustness"].apply(json.loads)
        for k in noise.iloc[0]:
            df[k] = noise.apply(lambda d, k=k: d[k])      # e.g. acc_eps_0.05
    return df


def eps_cols(df):
    return sorted([c for c in df.columns if c.startswith("acc_eps_")], key=lambda c: float(c.split("_")[-1]))


def methods_in(df):
    present = list(df["method"].unique())
    return [m for m in METHODS if m in present] + [m for m in present if m not in METHODS]


def ratios_in(df):
    return sorted(df["ratio"].unique())


def agg(df, col):
    return df.groupby(["method", "ratio"])[col].agg(["mean", "std"]).fillna(0).reset_index()


def curve(ax, df, col, title=None, ylabel=None, logy=False):
    """Metric vs retention ratio, one line per method (mean +/- std over replicates)."""
    a = agg(df, col)
    for m in methods_in(df):
        g = a[a.method == m].sort_values("ratio")
        ax.plot(g.ratio, g["mean"], marker=MARKERS.get(m, "o"), color=COLORS.get(m), label=LABELS.get(m, m))
        ax.fill_between(g.ratio, g["mean"] - g["std"], g["mean"] + g["std"], color=COLORS.get(m), alpha=0.15)
    ax.set_xticks(ratios_in(df))
    ax.set_xlabel("Retention ratio")
    ax.set_ylabel(ylabel or NICE.get(col, col))
    ax.set_title(title or NICE.get(col, col))
    if logy:
        ax.set_yscale("log")


def grouped_bars(ax, df, col, title=None):
    a = agg(df, col)
    ms, rs = methods_in(df), ratios_in(df)
    w = 0.8 / len(ms)
    for i, m in enumerate(ms):
        g = a[a.method == m].set_index("ratio").reindex(rs)
        ax.bar(np.arange(len(rs)) + i * w - 0.4 + w / 2, g["mean"], w, yerr=g["std"], capsize=2,
               color=COLORS.get(m), label=LABELS.get(m, m))
    ax.set_xticks(range(len(rs)))
    ax.set_xticklabels([f"{r:g}" for r in rs])
    ax.set_xlabel("Retention ratio")
    ax.set_ylabel(NICE.get(col, col))
    ax.set_title(title or NICE.get(col, col))


def mean_sv(df, method, ratio):
    g = df[(df.method == method) & (df.ratio == ratio)]
    return np.mean(np.stack(g.sv.values), axis=0) if len(g) else None


def panel_grid(df, cols, ncols=3, logy=()):
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    for ax, c in zip(axes.flat, cols):
        curve(ax, df, c, logy=c in logy)
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    axes.flat[0].legend()
    fig.tight_layout()
    return fig


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"saved {out}/{name}.[png|pdf]")


def heatmap(ax, M, xt, yt, vmax=None, cmap="RdBu_r", fmt="{:.2f}"):
    vmax = vmax if vmax is not None else np.nanmax(np.abs(M)) or 1
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt, rotation=45, ha="right")
    ax.set_yticks(range(len(yt))); ax.set_yticklabels(yt)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=7)
    return im


# ------------------------------------------------------------------ robustness-study helpers
NOISE_TYPE_LABELS = {"symmetric": "Symmetric noise", "pair": "Pair-flip noise",
                     "instance": "Instance-dependent noise", "none": "Clean"}


def noisy_types_in(df):
    return [t for t in ["symmetric", "pair", "instance"] if t in set(df.label_noise_type)]


def mean_std(df, by, col):
    g = df.groupby(by)[col].agg(["mean", "std", "count"]).reset_index()
    g["std"] = g["std"].fillna(0)
    return g


def shift_long(df):
    """One row per (run, condition@severity): columns condition, severity, shift_acc_cond, plus run metadata."""
    d = df[df["shift_acc"].notna()] if "shift_acc" in df else df.iloc[0:0]
    rows = []
    for _, r in d.iterrows():
        for key, a in json.loads(r["shift_acc"]).items():
            cond, _, sev = key.partition("@")
            rows.append({"seed": r.get("seed", 0), "method": r["method"], "ratio": r["ratio"], "accuracy": r["accuracy"],
                         "condition": cond, "severity": int(sev) if sev else 0, "acc_shift": a})
    L = pd.DataFrame(rows)
    if len(L):
        L["rel_drop"] = (L.accuracy - L.acc_shift) / L.accuracy.clip(lower=1e-12)
        L["abs_drop"] = L.accuracy - L.acc_shift
    return L


def need_cols(df, cols, what):
    miss = [c for c in cols if c not in df.columns or df[c].notna().sum() == 0]
    if miss:
        raise SystemExit(f"No data for {what} (missing/empty columns: {miss}). "
                         f"Enable it in config.yaml and re-run main.py.")


def pareto_mask(x, y, higher_is_better=True):
    """Non-dominated points when minimising x (energy) and optimising y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    yy = y if higher_is_better else -y
    keep = np.ones(len(x), bool)
    for i in range(len(x)):
        keep[i] = not np.any((x <= x[i]) & (yy >= yy[i]) & ((x < x[i]) | (yy > yy[i])))
    return keep


def energy_args(desc):
    """Common CLI for the energy figures. --net subtracts the measured idle power."""
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--csv", default="./*results*.csv")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--net", action="store_true", help="use energy net of the idle floor")
    ap.add_argument("--noise_type", default=None)
    ap.add_argument("--noise_rate", type=float, default=None, help="label-noise rate for the noisy-accuracy panel (default: max)")
    return ap.parse_args()
