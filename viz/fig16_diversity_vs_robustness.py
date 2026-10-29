"""Fig 16: does subset diversity explain shift robustness? Each point = (method, ratio). x = diversity descriptor of the
selected subset, y = relative accuracy drop under shift (minus Random at the same ratio). Left: class balance;
right: feature spread (variance of the subset / variance of the full set)."""
from common import *

args, df = setup(__doc__, clean_only=True)
need_cols(df, ["shift_rel_drop_mean", "sel_class_entropy", "sel_spread_ratio"], "diversity vs shift")
d = df[df.shift_rel_drop_mean.notna() & (df.ratio < 1.0)]
a = d.groupby(["method", "ratio"])[["shift_rel_drop_mean", "sel_class_entropy", "sel_spread_ratio"]].mean().reset_index()
if "random" in a.method.values:
    base = a[a.method == "random"].set_index("ratio").shift_rel_drop_mean
    a["excess_drop"] = a.shift_rel_drop_mean - a.ratio.map(base)
    ycol, ylab = "excess_drop", "Relative drop minus Random (same ratio)"
else:
    ycol, ylab = "shift_rel_drop_mean", "Relative drop under shift"
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, (xc, xl) in enumerate([("sel_class_entropy", "Class-balance entropy of subset (1 = balanced)"),
                              ("sel_spread_ratio", "Feature spread: subset var / full var")]):
    for m in methods_in(a):
        g = a[a.method == m]
        ax[k].scatter(g[xc], g[ycol], s=30 + 250 * g.ratio, color=COLORS.get(m), marker=MARKERS.get(m, "o"),
                      label=LABELS.get(m, m), edgecolor="k", linewidth=0.5)
        for _, r in g.iterrows():
            ax[k].annotate(f"{r.ratio:g}", (r[xc], r[ycol]), fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax[k].axhline(0, color="k", ls="--", lw=0.8)
    ax[k].set(xlabel=xl, ylabel=ylab)
ax[0].legend()
fig.suptitle("Marker size ~ retention ratio", y=1.0)
fig.tight_layout()
save(fig, args.out, "fig16_diversity_vs_robustness")
