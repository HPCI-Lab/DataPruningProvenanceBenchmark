"""Table: energy per method/ratio, savings vs full-data training, accuracy cost, and accuracy per kJ.
Writes <out>/summary_energy.csv. Usage: python viz/summarize_energy.py --csv "./*results*.csv" [--net]"""
from common import *

a = energy_args(__doc__)
df = load_results(a.csv, clean_only=True)
ecol = "net_total_energy_j" if a.net else "total_energy_j"
pc, tc = ("net_prune_energy_j", "net_train_energy_j") if a.net else ("prune_energy_j", "train_energy_j")
need_cols(df, [ecol], "energy")
full = df[df.ratio == 1.0]
E0 = full[tc].mean() if len(full) else np.nan
A0 = full.accuracy.mean() if len(full) else np.nan
t = df.groupby(["method", "ratio"]).agg(prune_kJ=(pc, "mean"), train_kJ=(tc, "mean"), total_kJ=(ecol, "mean"),
                                       acc=("accuracy", "mean")).reset_index()
for c in ("prune_kJ", "train_kJ", "total_kJ"):
    t[c] /= 1e3
t["prune_share_%"] = 100 * t.prune_kJ / t.total_kJ
t["energy_saved_%"] = 100 * (E0 / 1e3 - t.total_kJ) / (E0 / 1e3)
t["acc_change_pp"] = 100 * (t.acc - A0)
t["acc_per_kJ"] = 100 * t.acc / t.total_kJ
os.makedirs(a.out, exist_ok=True)
t.to_csv(os.path.join(a.out, "summary_energy.csv"), index=False)
pd.set_option("display.width", 200)
print(t.round(3).to_string(index=False))
src = df.get("energy_source")
if src is not None:
    print("\nenergy source(s):", ", ".join(sorted(set(src.dropna()))))
    if any("estimate" in s or "none" in s for s in set(src.dropna())):
        print("WARNING: some energy values are ESTIMATES or missing a device (see utils/energy.py); do not read them as measurements.")
