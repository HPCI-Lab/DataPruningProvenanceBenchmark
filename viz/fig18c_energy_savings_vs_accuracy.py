"""Fig 18c: is pruning worth it? x = energy saved vs training on the full data (%, negative = the method costs MORE
than just training on everything), y = accuracy change vs full-data training (percentage points). Top-right is the goal.
Baseline = mean energy / accuracy at retention ratio 1.0 (training only, no pruning cost)."""
from common import *

a = energy_args(__doc__)
df = load_results(a.csv, clean_only=True)
ecol = "net_total_energy_j" if a.net else "total_energy_j"
tcol = "net_train_energy_j" if a.net else "train_energy_j"
need_cols(df, [ecol], "energy")
full = df[df.ratio == 1.0]
if full.empty:
    raise SystemExit("Need runs at retention ratio 1.0 as the full-data baseline.")
E0, A0 = full[tcol].mean(), full.accuracy.mean()
g = df[df.ratio < 1.0].groupby(["method", "ratio"])[[ecol, "accuracy"]].mean().reset_index()
g["saved"] = 100 * (E0 - g[ecol]) / E0
g["dacc"] = 100 * (g.accuracy - A0)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for m in methods_in(g):
    h = g[g.method == m].sort_values("ratio")
    for k in range(2):
        ax[k].plot(h.saved, h.dacc, marker=MARKERS.get(m, "o"), color=COLORS.get(m), label=LABELS.get(m, m))
    for _, r in h.iterrows():
        ax[0].annotate(f"{r.ratio:g}", (r.saved, r.dacc), fontsize=6, xytext=(3, 3), textcoords="offset points")
for k in range(2):
    ax[k].axhline(0, color="k", lw=0.8)
    ax[k].axvline(0, color="k", lw=0.8)
    ax[k].set_xlabel("Energy saved vs full-data training (%)")
    ax[k].set_ylabel("Accuracy change vs full data (pp)")
ax[0].set_title("All methods (labels = retention ratio)")
pos = g[g.saved > -20]
if len(pos):
    ax[1].set_xlim(min(-20, pos.saved.min() - 5), 100)
ax[1].set_title("Zoom: methods with net savings")
ax[0].legend(fontsize=7)
fig.tight_layout()
save(fig, a.out, "fig18c_energy_savings_vs_accuracy")
