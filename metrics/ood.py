import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def _msp(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1).max(dim=1).values.detach().cpu().numpy()


def compute_ood_metrics(in_logits: torch.Tensor, ood_logits: torch.Tensor) -> dict:
    """MSP-based OOD detection: in-distribution scores are treated as the
    positive class (label 1); higher MSP => more confident => more ID-like."""
    id_scores = _msp(in_logits)
    ood_scores = _msp(ood_logits)

    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])

    auroc = float(roc_auc_score(labels, scores))

    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, 0.95)
    idx = min(idx, len(fpr) - 1)
    fpr95 = float(fpr[idx])

    return {"auroc": auroc, "fpr95": fpr95}
