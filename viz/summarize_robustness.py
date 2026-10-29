"""Prints + saves tables that answer the two robustness questions directly.
Q1 (label noise):  mean noise_enrichment per method / noise type (+ fraction of runs with |z|>2).
Q2 (shift):        relative drop under shift per method / ratio, and the gap to Random.
Usage: python viz/summarize_robustness.py --csv "./*results*.csv" --out figures"""
from common import *

args, df = setup(__doc__, clean_only=False, need_training=False)
os.makedirs(args.out, exist_ok=True)
pd.set_option("display.width", 200)

q1 = df[(df.label_noise_rate > 0) & (df.ratio < 1.0)]
if len(q1) and q1.noise_enrichment.notna().any():
    t = q1.groupby(["label_noise_type", "method", "ratio"]).agg(
        enrichment=("noise_enrichment", "mean"), z=("noise_enrich_z", "mean"),
        sel_noisy_frac=("sel_noisy_frac", "mean")).reset_index()
    t.to_csv(os.path.join(args.out, "summary_q1_label_noise_selection.csv"), index=False)
    print("\n=== Q1: noise enrichment in selected subsets (1 = neutral; |z|>2 = unlikely by chance) ===")
    print(t.round(3).to_string(index=False))
    for typ, g in t.groupby("label_noise_type"):
        w = g.groupby("method").enrichment.mean().sort_values(ascending=False)
        print(f"[{typ}] highest mean enrichment: {w.index[0]} ({w.iloc[0]:.2f}x); lowest: {w.index[-1]} ({w.iloc[-1]:.2f}x)")

tr = df[(df.label_noise_type == "none") & df.accuracy.notna()]
if "shift_rel_drop_mean" in tr and tr.shift_rel_drop_mean.notna().any():
    t = tr.groupby(["method", "ratio"]).agg(clean_acc=("accuracy", "mean"), shifted_acc=("shift_acc_mean", "mean"),
                                           rel_drop=("shift_rel_drop_mean", "mean")).reset_index()
    if "random" in t.method.values:
        rnd = t[t.method == "random"].set_index("ratio")
        t["rel_drop_minus_random"] = t.rel_drop - t.ratio.map(rnd.rel_drop)
        t["shifted_acc_minus_random"] = t.shifted_acc - t.ratio.map(rnd.shifted_acc)
    t.to_csv(os.path.join(args.out, "summary_q2_shift_degradation.csv"), index=False)
    print("\n=== Q2: degradation under distribution shift ===")
    print(t.round(3).to_string(index=False))
    print("(positive rel_drop_minus_random = loses MORE accuracy under shift than random sampling)")
