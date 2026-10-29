"""Fig 3: OOD detection (MSP) - AUROC (higher better) and FPR95 (lower better), grouped by ratio."""
from common import *

args, df = setup(__doc__)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
grouped_bars(ax[0], df, "ood_auroc")
grouped_bars(ax[1], df, "ood_fpr95")
lo = df.ood_auroc.min()
ax[0].set_ylim(max(0, lo - 0.05), 1.0)
ax[0].legend()
fig.tight_layout()
save(fig, args.out, "fig3_ood_detection")
