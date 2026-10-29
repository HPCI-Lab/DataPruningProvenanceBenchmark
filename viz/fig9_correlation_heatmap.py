"""Fig 9: Spearman correlation between weight statistics (rows) and behavioural metrics (cols), across all runs."""
from common import *

args, df = setup(__doc__)
ec = eps_cols(df)
beh = ["accuracy", "ece", "ood_auroc", "ood_fpr95", ec[-1]]
wcols = [c for c in df.columns if c.startswith("w_")]
C = df[wcols + beh].corr(method="spearman").loc[wcols, beh].values
fig, ax = plt.subplots(figsize=(6, 0.4 * len(wcols) + 1.5))
im = heatmap(ax, C, [NICE.get(b, b) for b in beh[:-1]] + [f"Acc @ $\\epsilon$={ec[-1].split('_')[-1]}"],
             [NICE.get(w, w) for w in wcols], vmax=1)
fig.colorbar(im, ax=ax, label="Spearman $\\rho$")
ax.set_title(f"Weight statistics vs behaviour (n={len(df)} runs)")
save(fig, args.out, "fig9_correlation_heatmap")
