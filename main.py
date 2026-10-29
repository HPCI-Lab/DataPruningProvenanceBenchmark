import os
import json
import random
import argparse
import yaml
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from pruners import build_pruner
from trainer import build_model, train_model, get_logits, evaluate_accuracy
from metrics import compute_ece, compute_ood_metrics, evaluate_noise_robustness, analyze_classifier_spectral_decay, classifier_weight_stats, WEIGHT_STAT_KEYS
from utils.data_loader import get_dataset, get_pruned_loader, get_loader, get_norm
from utils.logger import Logger
from utils.label_noise import (get_targets, pca_features, make_noisy_labels, NoisyLabelDataset,
                               selection_noise_stats, selection_diversity_stats, memorization_stats)
from utils.shift import evaluate_shift, summarize_shift, condition_keys, official_available
from pruners.coreset_pruner import _extract_features
from utils import energy
from torch.utils.data import Subset

# from utils.torch_distributed_helper import setup, cleanup, to_distributed_dataloader
# from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
# import torch.distributed as dist

import yprov4ml

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Columns added by the label-noise / distribution-shift studies (NaN when not applicable).
NOISE_FIELDS = [
    "label_noise_type", "label_noise_rate", "actual_noise_rate",
    "sel_n_noisy", "sel_noisy_frac", "noise_enrichment", "noise_enrich_z",
    "memorization_rate", "clean_fit_rate",
]
DIVERSITY_FIELDS = ["sel_class_entropy", "sel_min_class_frac", "sel_spread_ratio"]
ENERGY_FIELDS = energy.ENERGY_FIELDS
SHIFT_FIELDS = ["shift_acc", "shift_acc_mean", "shift_acc_worst", "shift_abs_drop_mean", "shift_rel_drop_mean"]
BASE_FIELDS = ["seed", "method", "ratio", "n_samples", "accuracy", "ece",
               "ood_auroc", "ood_fpr95", "noise_robustness", "singular_values"]
FIELDNAMES = BASE_FIELDS + WEIGHT_STAT_KEYS + NOISE_FIELDS + DIVERSITY_FIELDS + SHIFT_FIELDS + ENERGY_FIELDS


def _open_csv(csv_path: str):
    """Create the CSV (with header) or verify an existing file has the same header."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        if header != FIELDNAMES:
            raise SystemExit(f"{csv_path} has an older/different column layout. Move or rename it "
                             f"(or change output.results_csv) so a fresh file is created.")
        return
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def _append(csv_path, row):
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def _noise_settings(cfg):
    """List of (type, rate). Rate 0 is run once as ('none', 0.0)."""
    ln = cfg.get("label_noise", {})
    if not ln.get("enabled", False):
        return [("none", 0.0)]
    rates = sorted(set(float(r) for r in ln.get("rates", [0.0])))
    out = [("none", 0.0)] if 0.0 in rates or not rates else []
    for t in ln.get("types", ["symmetric"]):
        out += [(t, r) for r in rates if r > 0]
    return out or [("none", 0.0)]


def run_experiment(cfg: dict, csv_path: str, selection_only: bool = False) -> dict:
    

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dname = cfg["dataset"]
    C = cfg["num_classes"]

    yprov4ml.start_run(
        prov_user_namespace="www.example.org",
        experiment_name="dataprune-bench",
        provenance_save_dir="prov",
        save_after_n_logs=100,
        collect_all_processes=False,
        metrics_file_type="csv",
        use_compressor=False,
    )
    yprov4ml.log_param("seed", cfg["seed"], is_input_to="Setup")
    yprov4ml.log_param("dataset", dname, is_input_to="Setup")
    yprov4ml.log_param("model", cfg["model"], is_input_to="Setup")
    yprov4ml.log_param("epochs", cfg["train"]["epochs"], is_input_to="Setup")
    yprov4ml.log_param("lr", cfg["train"]["lr"], is_input_to="Setup")
    yprov4ml.log_param("optimizer", cfg["train"]["optimizer"], is_input_to="Setup")
    yprov4ml.log_param("batch_size", cfg["train"]["batch_size"], is_input_to="Setup")

    en_cfg = cfg.get("energy", {})
    if en_cfg.get("enabled", True):
        tracker = energy.EnergyTracker(device=device, cpu_tdp_w=en_cfg.get("cpu_tdp_w", 65.0),
                                       poll_s=en_cfg.get("gpu_poll_interval_s", 0.1),
                                       rapl_root=en_cfg.get("rapl_root", "/sys/class/powercap"))
        if en_cfg.get("idle_baseline_seconds", 5) > 0:
            w = tracker.measure_idle(en_cfg.get("idle_baseline_seconds", 5))
            print(f"[energy] source={tracker.source}  idle power={w:.1f} W")
        else:
            print(f"[energy] source={tracker.source}")
    else:
        tracker = energy.NullTracker()
    energy.set_tracker(tracker)
    run_meter = tracker.measure()
    run_total = run_meter.__enter__()

    train_set = get_dataset(dname, train=True, in_channels=cfg["in_channels"])
    train_set = Subset(train_set, range(128))
    test_set = get_dataset(dname, train=False, in_channels=cfg["in_channels"])
    test_set = Subset(test_set, range(128))
    ood_set = get_dataset(cfg["ood_dataset"], train=False, in_channels=cfg["in_channels"])
    ood_set = Subset(ood_set, range(128))

    test_loader = get_loader(test_set, batch_size=8)
    ood_loader = get_loader(ood_set, batch_size=8)
    full_loader = get_loader(train_set, batch_size=8)

    # ---- label-noise study setup: fixed corrupted label sets, shared by every method -------------
    ln_cfg = cfg.get("label_noise", {})
    true_targets = get_targets(train_set)
    settings = _noise_settings(cfg)
    pix = _extract_features(train_set, device=device)              # same pixel features the pruners use
    geo = pca_features(pix, ln_cfg.get("feature_dim", 64), device=device, seed=cfg["seed"])
    del pix
    noisy_sets = {}                                                # (type, rate) -> (noisy_targets, is_noisy)
    for t, r in settings:
        noisy_sets[(t, r)] = make_noisy_labels(
            true_targets, r, t if t != "none" else "symmetric", C, seed=cfg["seed"] + 1,
            features=geo, dataset_name=dname, instance_beta=ln_cfg.get("instance_beta", 2.0))
        print(f"label noise {t} {r:.2f}: corrupted {int(noisy_sets[(t, r)][1].sum())}/{len(true_targets)}")

    # one dataset object per setting (stable identity => proxy-score caches in label-aware pruners work)
    noisy_ds = {k: (train_set if k[0] == "none" else NoisyLabelDataset(train_set, *v)) for k, v in noisy_sets.items()}

    # ---- shift study setup -----------------------------------------------------------------------
    sh_cfg = cfg.get("shift", {})
    shift_on = sh_cfg.get("enabled", False) and not selection_only
    if shift_on:
        names = sh_cfg.get("conditions", {}).get(dname, [])
        sevs = sh_cfg.get("severities", [1, 3, 5])
        source = sh_cfg.get("source", "synthetic")
        if source == "official" and not official_available(sh_cfg.get("official_root", ""), names):
            print("[shift] official CIFAR-10-C files not found -> falling back to synthetic corruptions")
            source = "synthetic"
        mean, std = get_norm(dname)
        shift_on = bool(names)
        if not shift_on:
            print(f"[shift] no conditions configured for dataset {dname}; skipping shift eval")

    _open_csv(csv_path)
    results = {}

    for method in cfg["methods"]:
        results[method] = {}
        pruner = build_pruner(method, seed=cfg["seed"], device=device, num_classes=C,
                              in_channels=cfg["in_channels"], **cfg.get("pruner_kwargs", {}).get(method, {}))

        for ratio in cfg["ratios"]:
            # Label-agnostic pruners: select ONCE and reuse for every noise setting.
            # Label-aware pruners (el2n/ccs/d2/knn_shapley) read the labels, so they are re-run on the
            # noisy-label dataset of every setting (their proxy training is cached across ratios).
            if not pruner.label_aware:
                with energy.prune_scope() as prune_res:
                    indices = pruner.select_subset(train_set, ratio)
                div = selection_diversity_stats(indices, true_targets, geo, C)

            for (ntype, nrate) in settings:
                noisy_targets, is_noisy = noisy_sets[(ntype, nrate)]
                if pruner.label_aware:
                    with energy.prune_scope() as prune_res:
                        indices = pruner.select_subset(noisy_ds[(ntype, nrate)], ratio)
                    div = selection_diversity_stats(indices, true_targets, geo, C)
                print(f"[{method}] ratio={ratio} label_noise={ntype}:{nrate}")
                nstats = selection_noise_stats(indices, is_noisy)
                noise_row = {
                    "label_noise_type": ntype, "label_noise_rate": nrate,
                    "actual_noise_rate": float(is_noisy.mean()), **nstats,
                    "memorization_rate": float("nan"), "clean_fit_rate": float("nan"),
                }
                base = {"seed": cfg["seed"], "method": method, "ratio": ratio, "n_samples": int(len(indices))}
                tag = f"{method}_{ratio}_{ntype}{nrate}"
                yprov4ml.log_dataset("train_full", full_loader, is_input_to=f"Pruning_{tag}")

                if selection_only:      # fast path: only selection diagnostics, no training
                    _append(csv_path, {**base, **noise_row, **div, **energy.energy_row(prune=prune_res)})
                    continue

                train_data = noisy_ds[(ntype, nrate)]
                train_loader = get_pruned_loader(train_data, indices, batch_size=cfg["train"]["batch_size"])
                yprov4ml.log_dataset(f"pruned_train_{tag}", train_loader,
                                     is_input_to=f"Training_{tag}", is_output_to=f"Pruning_{tag}")

                set_seed(cfg["seed"])
                model = build_model(cfg["model"], cfg["in_channels"], cfg["num_classes"])

                # mp_policy = MixedPrecisionPolicy(
                #     param_dtype=torch.bfloat16, 
                #     reduce_dtype=torch.float32
                # )

                # for layer in model.blocks:
                #     fully_shard(layer) #mp_policy=mp_policy
                # fully_shard(model) #mp_policy=mp_policy
                yprov4ml.log_model(f"init_{tag}", model, is_input_to=f"Training_{tag}")

                logger = Logger(cfg["train"]["logger_backend"], run_name=f"{method}_{ratio}_{ntype}{nrate}")
                with tracker.measure() as train_m:
                  train_model(model, train_loader, device,
                              epochs=cfg["train"]["epochs"], lr=cfg["train"]["lr"],
                              optimizer_name=cfg["train"]["optimizer"], logger=logger,
                              grad_clip=cfg["train"].get("grad_clip", 5.0),
                              prov_context=f"Training_{tag}")
                logger.close()
                yprov4ml.log_model(f"{yprov4ml.get_artifacts_dir()}/model_{tag}", model,
                                   is_input_to=f"Evaluation_{tag}", is_output_to=f"Training_{tag}")
                yprov4ml.log_dataset("test_data", test_loader, is_input_to=f"Evaluation_{tag}")
                yprov4ml.log_dataset("ood_data", ood_loader, is_input_to=f"Evaluation_{tag}")
                eval_meter = tracker.measure()
                eval_m = eval_meter.__enter__()

                test_logits, test_labels = get_logits(model, test_loader, device)
                ood_logits, _ = get_logits(model, ood_loader, device)

                assert torch.isfinite(test_logits).all() and torch.isfinite(ood_logits).all(), "NaN/Inf logits: training diverged"
                acc = (test_logits.argmax(dim=1) == test_labels).float().mean().item()
                ece = compute_ece(test_logits, test_labels, n_bins=cfg["eval"]["ece_bins"])
                ood_metrics = compute_ood_metrics(test_logits, ood_logits)
                noise_acc = evaluate_noise_robustness(model, test_loader, device,
                                                    epsilons=cfg["eval"]["noise_epsilons"])
                singular_values = analyze_classifier_spectral_decay(model)
                weight_stats = classifier_weight_stats(model)

                if ntype != "none":
                    noise_row.update(memorization_stats(model, train_set, indices, noisy_targets, is_noisy, device))

                shift_row = {k: float("nan") for k in SHIFT_FIELDS}
                if shift_on and (ntype == "none" or sh_cfg.get("eval_with_label_noise", False)):
                    sacc = evaluate_shift(model, test_loader, device, names, sevs, mean, std,
                                        seed=cfg["seed"], source=source,
                                        official_root=sh_cfg.get("official_root", "./data/CIFAR-10-C"))
                    shift_row = {"shift_acc": json.dumps(sacc), **summarize_shift(acc, sacc)}

                metrics_dict = {
                    "n_samples": int(len(indices)),
                    "accuracy": acc,
                    "ece": ece,
                    "ood_auroc": ood_metrics["auroc"],
                    "ood_fpr95": ood_metrics["fpr95"],
                    "noise_robustness": json.dumps(noise_acc),
                    "singular_values": json.dumps(singular_values.tolist()),
                    **weight_stats,
                }
                results[method][f"{ratio}|{ntype}:{nrate}"] = metrics_dict

                eval_meter.__exit__(None, None, None)
                en_row = energy.energy_row(prune=prune_res, train=train_m.cost, eval_=eval_m.cost)
                for k, v in {**metrics_dict, **en_row}.items():
                    if isinstance(v, (int, float)) and np.isfinite(v):
                        yprov4ml.log_metric(f"{k}_{tag}", v, is_output_to=f"Evaluation_{tag}", step=0)

                # Write row immediately so partial runs are never lost
                _append(csv_path, {**base, **metrics_dict, **noise_row, **div, **shift_row, **en_row})

    run_meter.__exit__(None, None, None)
    _write_energy_summary(csv_path, run_total.cost, tracker)

    yprov4ml.end_run(create_graph=True, create_svg=True, crate_ro_crate=False, create_card=True)

    return results


def _write_energy_summary(csv_path, total_cost, tracker):
    """Whole-run energy vs. what the CSV rows account for (rest = benchmark overhead: data loading, noise generation,
    diagnostics, shift/OOD/ECE evaluation of repeated settings, label-agnostic selections reused across settings...)."""
    try:
        import pandas as pd
        d = pd.read_csv(csv_path)
        d = d.drop_duplicates(subset=["seed", "method", "ratio", "label_noise_type", "label_noise_rate"])
        acc = float(d.get("total_energy_j", 0).sum(skipna=True)) + float(d.get("eval_energy_j", 0).sum(skipna=True))
    except Exception:
        acc = float("nan")
    out = {"source": tracker.source, "idle_power_w": tracker.idle_power_w, "run_energy_j": total_cost.j,
           "run_gpu_j": total_cost.gpu_j, "run_cpu_j": total_cost.cpu_j, "run_seconds": total_cost.s,
           "sum_of_row_standalone_plus_eval_j": acc,
           "note": "rows charge shared cached work (features, proxy training, greedy prefix) to EVERY row that uses it (standalone cost), so the sum can exceed the measured run energy"}
    with open(csv_path.replace(".csv", "_energy_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[energy] run total %.1f kJ (%.0f s); sum of per-row standalone costs + eval: %.1f kJ (can exceed the run total: shared work is charged to each user)" % (total_cost.j / 1e3, total_cost.s, acc / 1e3))


def plot_results(results: dict, plot_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    methods = list(results.keys())

    for method in methods:
        # only the clean runs (key suffix '|none:0.0')
        keys = [k for k in results[method] if k.endswith("|none:0.0")]
        ratios = sorted(float(k.split("|")[0]) for k in keys)
        get = lambda r: results[method][f"{r}|none:0.0"]
        accs = [get(r)["accuracy"] for r in ratios]
        eces = [get(r)["ece"] for r in ratios]
        aurocs = [get(r)["ood_auroc"] for r in ratios]

        axes[0, 0].plot(ratios, accs, marker="o", label=method)
        axes[0, 1].plot(ratios, eces, marker="o", label=method)
        axes[1, 0].plot(ratios, aurocs, marker="o", label=method)

        sv = np.array(json.loads(get(max(ratios))["singular_values"]))
        axes[1, 1].plot(np.arange(len(sv)), sv, marker="o", label=method)

    axes[0, 0].set(title="Retention Ratio vs Test Accuracy", xlabel="Retention Ratio", ylabel="Accuracy")
    axes[0, 1].set(title="Retention Ratio vs ECE", xlabel="Retention Ratio", ylabel="ECE")
    axes[1, 0].set(title="Retention Ratio vs OOD AUROC", xlabel="Retention Ratio", ylabel="AUROC")
    axes[1, 1].set(title="Classifier Weight Singular Value Decay", xlabel="Index", ylabel="Singular Value")

    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main(config_path: str = "./dataprune_bench/config.yaml", selection_only: bool = False):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rank = int(os.environ.get("RANK", 0))

    csv_filename = cfg["output"].get("results_csv", cfg["output"].get("results_json", "results.json").replace(".json", ".csv"))
    if selection_only:
        csv_filename = "selonly_" + csv_filename.replace("results", "sel")
    out_csv = f"./{rank}_{csv_filename}"

    run_experiment(cfg, out_csv, selection_only=selection_only)
    print(f"Results are being continuously updated and saved to {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configCNNprovExample.yaml")
    ap.add_argument("--selection-only", action="store_true",
                    help="skip training; only record which corrupted labels each pruner selects (fast, answers the label-noise question)")
    a = ap.parse_args()

    # setup()

    main(a.config, a.selection_only)

    # cleanup()