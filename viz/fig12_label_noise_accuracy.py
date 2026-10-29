"""Fig 12: test accuracy vs injected label-noise rate. Rows = noise type, columns = retention ratio.
Rate 0 is the shared clean baseline."""
from common import *

args, df = setup(__doc__, clean_only=False)
need_cols(df, ["label_noise_rate"], "label noise")
types, rs = noisy_types_in(df), ratios_in(df)
if not types:
    raise SystemExit("No noisy-label runs found.")
base = df[df.label_noise_type == "none"]
fig, axes = plt.subplots(len(types), len(rs), figsize=(3.4 * len(rs), 3.0 * len(types) + 0.6), sharey="row", squeeze=False)
for i, t in enumerate(types):
    sub = pd.concat([base, df[df.label_noise_type == t]])
    for j, r in enumerate(rs):
        ax = axes[i, j]
        a = mean_std(sub[sub.ratio == r], ["method", "label_noise_rate"], "accuracy")
        for m in methods_in(sub):
            h = a[a.method == m].sort_values("label_noise_rate")
            ax.errorbar(h.label_noise_rate, h["mean"], yerr=h["std"], marker=MARKERS.get(m, "o"),
                        color=COLORS.get(m), label=LABELS.get(m, m), capsize=2)
        ax.set_xticks(sorted(sub.label_noise_rate.unique()))
        if i == 0:
            ax.set_title(f"Retention {r:g}")
        if i == len(types) - 1:
            ax.set_xlabel("Label-noise rate")
        if j == 0:
            ax.set_ylabel(f"{NOISE_TYPE_LABELS[t]}\nclean test accuracy")
axes[0, 0].legend()
fig.tight_layout()
save(fig, args.out, "fig12_label_noise_accuracy")

# Companion: accuracy lost relative to the same method's noise-free run, as a heatmap (rows method, cols rate)
fig, axes = plt.subplots(len(types), len(rs), figsize=(3.0 * len(rs), 2.4 * len(types) + 0.6), squeeze=False)
mean_acc = df.groupby(["label_noise_type", "label_noise_rate", "method", "ratio"]).accuracy.mean()
ms = methods_in(df)
last = None
for i, t in enumerate(types):
    rates = sorted(df[df.label_noise_type == t].label_noise_rate.unique())
    for j, r in enumerate(rs):
        M = np.array([[100 * (mean_acc[(t, nr, m, r)] - mean_acc[("none", 0.0, m, r)]) / mean_acc[("none", 0.0, m, r)]
                       if (t, nr, m, r) in mean_acc.index and ("none", 0.0, m, r) in mean_acc.index else np.nan
                       for nr in rates] for m in ms])
        last = heatmap(axes[i, j], M, [f"{x:g}" for x in rates], [LABELS.get(m, m) for m in ms], vmax=40, fmt="{:.0f}")
        axes[i, j].set_title(f"{NOISE_TYPE_LABELS[t].split()[0]} | ratio {r:g}", fontsize=9)
fig.colorbar(last, ax=axes.ravel().tolist(), label="% accuracy change vs noise-free")
save(fig, args.out, "fig12b_label_noise_relative_loss")
