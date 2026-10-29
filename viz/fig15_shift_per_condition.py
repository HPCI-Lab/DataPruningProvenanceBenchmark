"""Fig 15: where does each pruning method break under shift? Heatmaps of (method accuracy - Random accuracy) in
percentage points per shift condition (columns) and retention ratio (rows), plus relative-drop-vs-severity curves
for CIFAR-style corruptions."""
from common import *

args, df = setup(__doc__, clean_only=True)
L = shift_long(df)
if L.empty:
    raise SystemExit("No shift results in CSV. Enable shift in config.yaml and re-run main.py.")
rs, ms = ratios_in(L), [m for m in methods_in(L) if m != "random"]
L["key"] = L.apply(lambda r: f"{r.condition}@{r.severity}" if r.severity else r.condition, axis=1)
keys = list(dict.fromkeys(L.key))
M = L.groupby(["method", "ratio", "key"]).acc_shift.mean()
if "random" in L.method.values and ms:
    fig, axes = plt.subplots(1, len(ms), figsize=(0.55 * len(keys) * len(ms) + 2, 0.55 * len(rs) + 2.2), squeeze=False)
    for ax, m in zip(axes[0], ms):
        D = np.array([[100 * (M.get((m, r, k), np.nan) - M.get(("random", r, k), np.nan)) for k in keys] for r in rs])
        im = heatmap(ax, D, keys, [f"{r:g}" for r in rs], vmax=max(np.nanmax(np.abs(D)), 1), fmt="{:.0f}")
        ax.set(title=LABELS.get(m, m), ylabel="Retention ratio")
    fig.colorbar(im, ax=axes[0].tolist(), label="accuracy - Random (pp)")
    save(fig, args.out, "fig15a_shift_vs_random_heatmap")

S = L[L.severity > 0]
if len(S):
    conds = list(dict.fromkeys(S.condition))
    sevs = sorted(S.severity.unique())
    fig, axes = plt.subplots(len(rs), len(conds), figsize=(2.9 * len(conds), 2.4 * len(rs) + 0.6), sharey=True, squeeze=False)
    for i, r in enumerate(rs):
        for j, c in enumerate(conds):
            ax = axes[i, j]
            for m in methods_in(S):
                h = S[(S.condition == c) & (S.ratio == r) & (S.method == m)].groupby("severity").rel_drop.mean()
                ax.plot(h.index, h.values, marker=MARKERS.get(m), color=COLORS.get(m), label=LABELS.get(m, m))
            ax.set_xticks(sevs)
            if i == 0:
                ax.set_title(c.replace("_", " "))
            if i == len(rs) - 1:
                ax.set_xlabel("Severity")
            if j == 0:
                ax.set_ylabel(f"ratio {r:g}\nrelative drop")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    save(fig, args.out, "fig15b_shift_severity_curves")
