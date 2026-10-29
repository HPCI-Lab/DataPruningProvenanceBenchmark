"""Fig 1: composite 2x2 - accuracy, ECE, OOD AUROC vs ratio + classifier singular-value decay."""
from common import *

args, df = setup(__doc__)
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
curve(ax[0, 0], df, "accuracy", "Retention ratio vs test accuracy")
curve(ax[0, 1], df, "ece", "Retention ratio vs ECE")
curve(ax[1, 0], df, "ood_auroc", "Retention ratio vs OOD AUROC")
r = max(ratios_in(df))
for m in methods_in(df):
    s = mean_sv(df, m, r)
    ax[1, 1].plot(range(1, len(s) + 1), s, marker=MARKERS.get(m, "o"), color=COLORS.get(m), label=LABELS.get(m, m))
ax[1, 1].set(title=f"Classifier singular-value decay (ratio={r:g})", xlabel="Index", ylabel="Singular value")
ax[0, 0].legend()
fig.tight_layout()
save(fig, args.out, "fig1_summary_2x2")
