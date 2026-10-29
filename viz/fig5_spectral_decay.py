"""Fig 5: normalised singular-value spectrum S_i/S_1 of the classifier weights, per retention ratio."""
from common import *

args, df = setup(__doc__)
rs = ratios_in(df)
fig, axes = plt.subplots(1, len(rs), figsize=(3.6 * len(rs), 3.4), sharey=True, squeeze=False)
for ax, r in zip(axes[0], rs):
    for m in methods_in(df):
        s = mean_sv(df, m, r)
        if s is None:
            continue
        ax.plot(range(1, len(s) + 1), s / s[0], marker=MARKERS.get(m, "o"), color=COLORS.get(m), label=LABELS.get(m, m))
    ax.set(title=f"Retention {r:g}", xlabel="Singular value index", yscale="log")
axes[0, 0].set_ylabel("$S_i / S_1$ (log)")
axes[0, 0].legend()
fig.tight_layout()
save(fig, args.out, "fig5_spectral_decay")
