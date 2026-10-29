"""Fig 18a: energy vs quality. Every point = (method, retention ratio), x = energy to PRODUCE the model
(standalone pruning + training, kJ, log scale), labels = retention ratio. Black dashed line = Pareto frontier
(least energy for the best metric). Panels: accuracy, accuracy under shift, relative drop under shift, ECE, OOD AUROC,
accuracy with noisy labels. Use --net to subtract the measured idle power."""
from common import *

a = energy_args(__doc__)
df = load_results(a.csv, clean_only=False)
ecol = "net_total_energy_j" if a.net else "total_energy_j"
need_cols(df, [ecol], "energy")
df = df[df[ecol].notna()].copy()
df["E"] = df[ecol] / 1e3
clean = df[df.label_noise_type == "none"]
noisy = df[df.label_noise_type == (a.noise_type or next(iter(noisy_types_in(df)), "none"))]
if len(noisy) and a.noise_rate is None:
    noisy = noisy[noisy.label_noise_rate == noisy.label_noise_rate.max()]
elif len(noisy):
    noisy = noisy[np.isclose(noisy.label_noise_rate, a.noise_rate)]
nr_txt = f"{noisy.label_noise_type.iloc[0]} noise {noisy.label_noise_rate.iloc[0]:g}" if len(noisy) else ""
panels = [("accuracy", "Test accuracy", True, clean), ("shift_acc_mean", "Accuracy under shift", True, clean),
          ("shift_rel_drop_mean", "Relative drop under shift", False, clean), ("ece", "ECE", False, clean),
          ("ood_auroc", "OOD AUROC", True, clean), ("accuracy", f"Accuracy, {nr_txt}", True, noisy)]
panels = [p for p in panels if len(p[3]) and p[0] in p[3] and p[3][p[0]].notna().any()]
nc = 3
nr = int(np.ceil(len(panels) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(5.0 * nc, 3.9 * nr), squeeze=False)
for ax, (col, title, hib, src) in zip(axes.flat, panels):
    g = src.groupby(["method", "ratio"])[["E", col]].mean().reset_index()
    for m in methods_in(g):
        h = g[g.method == m].sort_values("ratio")
        ax.plot(h.E, h[col], marker=MARKERS.get(m, "o"), color=COLORS.get(m), label=LABELS.get(m, m), alpha=0.9)
        for _, r in h.iterrows():
            ax.annotate(f"{r.ratio:g}", (r.E, r[col]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    pm = pareto_mask(g.E, g[col], hib)
    P = g[pm].sort_values("E")
    ax.step(P.E, P[col], where="post", color="k", ls="--", lw=1, label="Pareto frontier")
    ax.set_xscale("log")
    ax.set_xlabel("Energy to produce model (kJ" + (", net of idle)" if a.net else ")"))
    ax.set_ylabel(title)
    ax.set_title(title + ("  (higher better)" if hib else "  (lower better)"), fontsize=9)
for ax in axes.flat[len(panels):]:
    ax.axis("off")
axes[0, 0].legend(fontsize=6)
fig.tight_layout()
save(fig, a.out, "fig18a_energy_vs_metrics")
