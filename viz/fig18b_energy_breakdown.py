"""Fig 18b: where does the energy go? Top: stacked bars of standalone PRUNING vs TRAINING energy per method and retention
ratio (kJ). Bottom: pruning share of the total. A method whose selection costs more than the training it saves is a net loss."""
from common import *

a = energy_args(__doc__)
df = load_results(a.csv, clean_only=True)
pc, tc = ("net_prune_energy_j", "net_train_energy_j") if a.net else ("prune_energy_j", "train_energy_j")
need_cols(df, [pc, tc], "energy")
g = df.groupby(["method", "ratio"])[[pc, tc]].mean().reset_index()
g[[pc, tc]] /= 1e3
ms, rs = methods_in(g), ratios_in(g)
fig, ax = plt.subplots(2, 1, figsize=(max(8, 1.0 * len(ms) * len(rs) * 0.55 + 3), 7.5), sharex=True)
w, x0 = 0.8 / len(rs), np.arange(len(ms))
cmap = plt.cm.Greys(np.linspace(0.35, 0.8, len(rs)))
for k, r in enumerate(rs):
    h = g[g.ratio == r].set_index("method").reindex(ms)
    xs = x0 - 0.4 + w * (k + 0.5)
    ax[0].bar(xs, h[tc], w * 0.92, color=[COLORS.get(m) for m in ms], alpha=0.55, edgecolor="k", linewidth=0.4,
              label="training" if k == 0 else None)
    ax[0].bar(xs, h[pc], w * 0.92, bottom=h[tc], color=[COLORS.get(m) for m in ms], hatch="///", edgecolor="k",
              linewidth=0.4, label="pruning (standalone)" if k == 0 else None)
    share = 100 * h[pc] / (h[pc] + h[tc])
    ax[1].bar(xs, share, w * 0.92, color=[COLORS.get(m) for m in ms], edgecolor="k", linewidth=0.4)
    for xi, rr in zip(xs, [r] * len(ms)):
        ax[1].text(xi, 0.5, f"{rr:g}", ha="center", va="bottom", fontsize=6, rotation=90)
ax[0].set_ylabel("Energy (kJ)")
ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="lightgrey", ec="k"), plt.Rectangle((0, 0), 1, 1, fc="lightgrey", ec="k", hatch="///")],
             labels=["training", "pruning (standalone)"], fontsize=8)
ax[1].set_ylabel("Pruning share of total (%)")
ax[1].set_xticks(x0)
ax[1].set_xticklabels([LABELS.get(m, m) for m in ms])
ax[0].set_title("Energy breakdown per method (bars within a group = retention ratios, labelled at the bottom)")
fig.tight_layout()
save(fig, a.out, "fig18b_energy_breakdown")
