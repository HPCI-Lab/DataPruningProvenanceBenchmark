"""Fig 13: do coreset models lock in the noisy labels? memorization_rate = P(model predicts the WRONG noisy
label | sample is mislabeled), measured on the training subset. Higher = noise memorised. Rows = noise type, cols = retention ratio."""
from common import *

args, df = setup(__doc__, clean_only=False)
df = df[df.label_noise_rate > 0]
need_cols(df, ["memorization_rate"], "memorization")
types, rs = noisy_types_in(df), ratios_in(df)
fig, axes = plt.subplots(len(types), len(rs), figsize=(3.4 * len(rs), 3.0 * len(types) + 0.6), sharey=True, squeeze=False)
for i, t in enumerate(types):
    for j, r in enumerate(rs):
        ax = axes[i, j]
        g = df[(df.label_noise_type == t) & (df.ratio == r)]
        a = mean_std(g, ["method", "label_noise_rate"], "memorization_rate")
        for m in methods_in(g):
            h = a[a.method == m].sort_values("label_noise_rate")
            ax.errorbar(h.label_noise_rate, h["mean"], yerr=h["std"], marker=MARKERS.get(m, "o"),
                        color=COLORS.get(m), label=LABELS.get(m, m), capsize=2)
        ax.set_xticks(sorted(g.label_noise_rate.unique()))
        if i == 0:
            ax.set_title(f"Retention {r:g}")
        if i == len(types) - 1:
            ax.set_xlabel("Label-noise rate")
        if j == 0:
            ax.set_ylabel(f"{NOISE_TYPE_LABELS[t]}\nnoisy labels memorised")
axes[0, 0].legend()
fig.tight_layout()
save(fig, args.out, "fig13_noise_memorization")
