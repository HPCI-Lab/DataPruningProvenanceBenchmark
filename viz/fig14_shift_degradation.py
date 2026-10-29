"""Fig 14 (Key question 2): clean vs shifted accuracy and relative degradation per method / retention ratio.
(a) clean accuracy, (b) mean shifted accuracy, (c) relative drop (clean-shifted)/clean, (d) shifted accuracy as a
difference to Random at the same ratio (negative = worse than random = 'collapse')."""
from common import *

args, df = setup(__doc__, clean_only=True)
need_cols(df, ["shift_acc_mean"], "distribution shift")
d = df[df.shift_acc_mean.notna()]
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
curve(ax[0], d, "accuracy", "(a) Clean test accuracy")
curve(ax[1], d, "shift_acc_mean", "(b) Mean accuracy on shifted data", "Shifted accuracy")
curve(ax[2], d, "shift_rel_drop_mean", "(c) Relative drop (clean - shifted) / clean", "Relative drop")
if "random" in d.method.values:
    rnd = d[d.method == "random"].groupby(["ratio"]).shift_acc_mean.mean()
    for m in methods_in(d):
        if m == "random":
            continue
        g = d[d.method == m].groupby("ratio").shift_acc_mean.agg(["mean", "std"]).fillna(0)
        diff = g["mean"] - rnd.reindex(g.index)
        ax[3].errorbar(g.index, 100 * diff, yerr=100 * g["std"], marker=MARKERS.get(m), color=COLORS.get(m),
                       label=LABELS.get(m, m), capsize=2)
    ax[3].axhline(0, color="k", ls="--", lw=0.8)
    ax[3].set(title="(d) Shifted acc. minus Random", xlabel="Retention ratio", ylabel="pp vs Random")
    ax[3].set_xticks(ratios_in(d))
ax[0].legend()
fig.tight_layout()
save(fig, args.out, "fig14_shift_degradation")
