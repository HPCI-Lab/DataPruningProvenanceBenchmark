"""Fig 2: accuracy-calibration trade-off; each point is a (method, ratio), labelled by ratio."""
from common import *

args, df = setup(__doc__)
a = df.groupby(["method", "ratio"])[["accuracy", "ece"]].mean().reset_index()
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for m in methods_in(df):
    g = a[a.method == m].sort_values("ratio")
    ax.plot(g.accuracy, g.ece, color=COLORS.get(m), alpha=0.4)
    ax.scatter(g.accuracy, g.ece, s=30 + 150 * g.ratio, color=COLORS.get(m), marker=MARKERS.get(m, "o"),
               label=LABELS.get(m, m), edgecolor="k", linewidth=0.5)
    for _, row in g.iterrows():
        ax.annotate(f"{row.ratio:g}", (row.accuracy, row.ece), fontsize=6, xytext=(4, 4), textcoords="offset points")
ax.set(xlabel="Test accuracy", ylabel="ECE (lower is better)", title="Accuracy vs calibration (marker size ~ ratio)")
ax.legend()
save(fig, args.out, "fig2_accuracy_vs_calibration")
