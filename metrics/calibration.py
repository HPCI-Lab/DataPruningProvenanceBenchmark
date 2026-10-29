import torch


def compute_ece(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error using n_bins equal-width bins over confidence."""
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()

    bin_edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=logits.device)
    n = conf.size(0)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc_bin = correct[in_bin].mean()
            conf_bin = conf[in_bin].mean()
            ece += torch.abs(conf_bin - acc_bin) * prop

    return ece.item()


def compute_mce(logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Maximum Calibration Error: worst per-bin |confidence - accuracy| gap."""
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()

    bin_edges = torch.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.float().mean().item() > 0:
            gap = torch.abs(conf[in_bin].mean() - correct[in_bin].mean()).item()
            mce = max(mce, gap)

    return mce
