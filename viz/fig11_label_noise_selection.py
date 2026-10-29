"""Fig 11 (Key question 1): do geometry-based selectors over-select mislabeled samples?
noise_enrichment = (noisy fraction inside the subset) / (noisy fraction in the full train set).
1 = neutral (dashed line); >1 = the selector pulls in mislabeled samples. Rows = noise type, columns = noise rate.
Ratio 1.0 is omitted (it keeps everything, enrichment is 1 by construction). Works with --csv on selection-only runs too."""
from common import *

args, df = setup(__doc__, clean_only=False, need_training=False)
df = df[(df.label_noise_rate > 0) & (df.ratio < 1.0)]
need_cols(df, ["noise_enrichment"], "label-noise selection")
types, rates, rs = noisy_types_in(df), sorted(df.label_noise_rate.unique()), ratios_in(df)
fig, axes = plt.subplots(len(types), len(rates), figsize=(3.6 * len(rates), 3.0 * len(types) + 0.6),
                         sharey=True, squeeze=False)
for i, t in enumerate(types):
    for j, nr in enumerate(rates):
        ax = axes[i, j]
        g = df[(df.label_noise_type == t) & (df.label_noise_rate == nr)]
        a = mean_std(g, ["method", "ratio"], "noise_enrichment")
        for m in methods_in(g):
            h = a[a.method == m].sort_values("ratio")
            ax.errorbar(h.ratio, h["mean"], yerr=h["std"], marker=MARKERS.get(m, "o"), color=COLORS.get(m),
                        label=LABELS.get(m, m), capsize=2)
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_xticks(rs)
        if i == 0:
            ax.set_title(f"noise rate {nr:g}")
        if i == len(types) - 1:
            ax.set_xlabel("Retention ratio")
        if j == 0:
            ax.set_ylabel(f"{NOISE_TYPE_LABELS[t]}\nenrichment of noisy samples")
axes[0, 0].legend()
fig.suptitle("Share of mislabeled samples in the coreset relative to the full set (1 = neutral)", y=1.0)
fig.tight_layout()
save(fig, args.out, "fig11_label_noise_selection")
