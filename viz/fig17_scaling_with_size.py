"""Fig 17: how every metric scales with model size. Pass one results CSV per model, e.g.
  python viz/fig17_scaling_with_size.py --files 0_results_ResNet18.csv 0_results_ViTSmall.csv 0_results_ViTMedium.csv
Model name = text after 'results_' in the filename. Override param counts (millions) with --params ViTMedium=38.7."""
import re, argparse, os
from common import *

PARAMS = {"SimpleCNN": 0.3, "ResNet18": 11.2, "ResNet34": 21.3, "ResNet50": 23.5, "ViTTiny": 5.7,
          "ViTSmall": 22.0, "ViTMedium": 38.7, "ViTBase": 86.0, "ViTLarge": 304.0}
ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--params", nargs="*", default=[])
ap.add_argument("--ratios", nargs="*", type=float, default=[0.1, 0.5])
ap.add_argument("--noise_type", default=None, help="label-noise type for the noise panels (default: first found)")
ap.add_argument("--out", default="figures")
a = ap.parse_args()
PARAMS.update({k: float(v) for k, v in (p.split("=") for p in a.params)})

dfs = []
for f in a.files:
    d = load_results(f, clean_only=False)
    name = re.sub(r"^\d+_", "", os.path.basename(f))
    name = re.sub(r"^results_|\.csv$", "", name).replace("results_", "")
    d["model"] = name
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
models = sorted(df.model.unique(), key=lambda m: PARAMS.get(m, np.inf))
missing = [m for m in models if m not in PARAMS]
if missing:
    raise SystemExit(f"Give param counts via --params for: {missing}")
x = {m: PARAMS[m] for m in models}

clean = df[df.label_noise_type == "none"]
nt = a.noise_type or next(iter(noisy_types_in(df)), None)
noisy = df[(df.label_noise_type == nt)] if nt else df.iloc[0:0]
if len(noisy):
    nr = noisy.label_noise_rate.max()
    noisy = noisy[noisy.label_noise_rate == nr]
    base = clean.groupby(["model", "method", "ratio"]).accuracy.mean()
    noisy = noisy.assign(rel_loss=noisy.apply(lambda r: (base.get((r.model, r.method, r.ratio), np.nan) - r.accuracy)
                                              / base.get((r.model, r.method, r.ratio), np.nan), axis=1))
panels = [("accuracy", "Clean accuracy", clean), ("total_energy_j", "Energy: prune+train (J)", clean), ("ece", "ECE", clean), ("ood_auroc", "OOD AUROC", clean),
          ("shift_rel_drop_mean", "Relative drop under shift", clean)]
if len(noisy):
    panels += [("rel_loss", f"Rel. acc. loss @ {nt} noise {nr:g}", noisy),
               ("memorization_rate", f"Noise memorization @ {nr:g}", noisy)]
panels = [p for p in panels if p[0] in p[2] and p[2][p[0]].notna().any()]

fig, axes = plt.subplots(len(a.ratios), len(panels), figsize=(3.3 * len(panels), 2.9 * len(a.ratios) + 0.5), squeeze=False)
for i, r in enumerate(a.ratios):
    for j, (col, title, src) in enumerate(panels):
        ax = axes[i, j]
        g = mean_std(src[np.isclose(src.ratio, r)], ["method", "model"], col)
        for m in methods_in(src):
            h = g[g.method == m].assign(p=lambda t: t.model.map(x)).sort_values("p")
            ax.errorbar(h.p, h["mean"], yerr=h["std"], marker=MARKERS.get(m, "o"), color=COLORS.get(m),
                        label=LABELS.get(m, m), capsize=2)
        ax.set_xscale("log")
        ax.set_xticks([x[m] for m in models]); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
        ax.minorticks_off()
        if i == 0:
            ax.set_title(title, fontsize=9)
        if j == 0:
            ax.set_ylabel(f"retention {r:g}")
axes[0, 0].legend(fontsize=6)
fig.tight_layout()
save(fig, a.out, "fig17_scaling_with_size")
