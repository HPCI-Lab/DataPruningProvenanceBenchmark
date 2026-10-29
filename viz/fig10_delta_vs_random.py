"""Fig 10: relative change (%) of each metric vs the Random baseline, per method and ratio (signed; not 'good/bad')."""
from common import *

args, df = setup(__doc__)
ec = eps_cols(df)
cols = ["accuracy", "ece", "ood_auroc", "ood_fpr95", ec[-1], "w_effective_rank", "w_frobenius_norm", "w_mean_cos_sim"]
names = [NICE.get(c, c) if not c.startswith("acc_eps") else f"Acc @ $\\epsilon$={c.split('_')[-1]}" for c in cols]
mean = df.groupby(["method", "ratio"])[cols].mean()
rs = ratios_in(df)
ms = [m for m in methods_in(df) if m != "random"]
if "random" not in df.method.values:
    raise SystemExit("Need 'random' runs as baseline.")
fig, axes = plt.subplots(1, len(ms), figsize=(3.4 * len(ms) + 1, 4), sharey=True, squeeze=False)
D = {m: np.array([[100 * (mean.loc[(m, r), c] - mean.loc[("random", r), c]) / (abs(mean.loc[("random", r), c]) + 1e-12)
                   for r in rs] for c in cols]) for m in ms}
vmax = min(np.nanmax([np.abs(d).max() for d in D.values()]), 100)
for ax, m in zip(axes[0], ms):
    im = heatmap(ax, np.clip(D[m], -100, 100), [f"{r:g}" for r in rs], names, vmax=vmax, fmt="{:.0f}")
    ax.set(title=LABELS.get(m, m), xlabel="Retention ratio")
fig.colorbar(im, ax=axes[0].tolist(), label="% change vs Random")
save(fig, args.out, "fig10_delta_vs_random")
