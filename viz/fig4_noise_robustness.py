"""Fig 4: accuracy under additive Gaussian noise; one panel per retention ratio (eps=0 is clean accuracy)."""
from common import *

args, df = setup(__doc__)
ec = eps_cols(df)
eps = [0.0] + [float(c.split("_")[-1]) for c in ec]
rs = ratios_in(df)
fig, axes = plt.subplots(1, len(rs), figsize=(3.6 * len(rs), 3.4), sharey=True, squeeze=False)
for ax, r in zip(axes[0], rs):
    for m in methods_in(df):
        g = df[(df.method == m) & (df.ratio == r)]
        if g.empty:
            continue
        y = g[["accuracy"] + ec].mean().values
        e = g[["accuracy"] + ec].std().fillna(0).values
        ax.errorbar(eps, y, yerr=e, marker=MARKERS.get(m, "o"), color=COLORS.get(m), label=LABELS.get(m, m), capsize=2)
    ax.set(title=f"Retention {r:g}", xlabel="Noise std $\\epsilon$")
    ax.set_xticks(eps)
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].legend()
fig.tight_layout()
save(fig, args.out, "fig4_noise_robustness")
