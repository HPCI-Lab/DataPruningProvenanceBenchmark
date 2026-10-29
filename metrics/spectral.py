import numpy as np
import torch
import torch.nn as nn


def get_classifier_layer(model: nn.Module) -> nn.Linear:
    """Final linear layer: `model.classifier` (SimpleCNN/ResNet18) or timm's `get_classifier()` / `.head`."""
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        return model.classifier
    if hasattr(model, "get_classifier"):
        layer = model.get_classifier()
        if isinstance(layer, nn.Linear):
            return layer
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        return model.head
    raise AttributeError("Could not locate the final nn.Linear classifier layer of the model.")


def analyze_classifier_spectral_decay(model: nn.Module) -> np.ndarray:
    """SVD singular values of the final linear classifier layer's weight
    matrix W (shape [C, D])."""
    W = get_classifier_layer(model).weight.detach().cpu().numpy()
    S = np.linalg.svd(W, compute_uv=False)
    return S


def effective_rank(S: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon-entropy-based effective rank (Roy & Vetterli, 2007)."""
    p = S / (S.sum() + eps)
    p = p[p > eps]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def spectral_decay_rate(S: np.ndarray) -> float:
    """Fits log(S_i) ~ -alpha * i, returns alpha (decay exponent)."""
    S = S[S > 1e-12]
    if len(S) < 2:
        return 0.0
    x = np.arange(len(S))
    y = np.log(S)
    alpha, _ = np.polyfit(x, y, 1)
    return float(-alpha)


WEIGHT_STAT_KEYS = [
    "w_effective_rank", "w_stable_rank", "w_spectral_entropy", "w_decay_rate",
    "w_spectral_norm", "w_frobenius_norm", "w_nuclear_norm", "w_condition_number",
    "w_top1_energy", "w_value_entropy", "w_kurtosis", "w_mean", "w_std",
    "w_rownorm_mean", "w_rownorm_std", "w_mean_cos_sim", "w_bias_norm",
]


def classifier_weight_stats(model: nn.Module, n_bins: int = 50) -> dict:
    """Scalar descriptors of the final linear layer W (C x D) for tracking weight behaviour."""
    layer = get_classifier_layer(model)
    W = layer.weight.detach().cpu().double().numpy()
    S = np.linalg.svd(W, compute_uv=False)
    p = S / (S.sum() + 1e-12)
    p_nz = p[p > 1e-12]
    spec_entropy = float(-(p_nz * np.log(p_nz)).sum())          # Shannon entropy of normalized spectrum

    # entropy of the weight-value histogram (spread of weight magnitudes)
    hist, _ = np.histogram(W.ravel(), bins=n_bins)
    q = hist / hist.sum()
    q = q[q > 0]
    val_entropy = float(-(q * np.log(q)).sum())

    w = W.ravel()
    std = w.std()
    kurt = float(((w - w.mean()) ** 4).mean() / (std ** 4 + 1e-12) - 3.0)  # excess kurtosis

    row_norms = np.linalg.norm(W, axis=1)
    Wn = W / (row_norms[:, None] + 1e-12)
    cos = Wn @ Wn.T
    iu = np.triu_indices(len(W), k=1)

    bias = layer.bias.detach().cpu().double().numpy() if layer.bias is not None else np.zeros(1)

    return {
        "w_effective_rank": float(np.exp(spec_entropy)),
        "w_stable_rank": float((S ** 2).sum() / (S[0] ** 2 + 1e-12)),   # ||W||_F^2 / ||W||_2^2
        "w_spectral_entropy": spec_entropy,
        "w_decay_rate": spectral_decay_rate(S),
        "w_spectral_norm": float(S[0]),
        "w_frobenius_norm": float(np.sqrt((S ** 2).sum())),
        "w_nuclear_norm": float(S.sum()),
        "w_condition_number": float(S[0] / (S[-1] + 1e-12)),
        "w_top1_energy": float(S[0] ** 2 / ((S ** 2).sum() + 1e-12)),
        "w_value_entropy": val_entropy,
        "w_kurtosis": kurt,
        "w_mean": float(w.mean()),
        "w_std": float(std),
        "w_rownorm_mean": float(row_norms.mean()),
        "w_rownorm_std": float(row_norms.std()),
        "w_mean_cos_sim": float(cos[iu].mean()),   # class-vector alignment (-> -1/(C-1) when well separated)
        "w_bias_norm": float(np.linalg.norm(bias)),
    }
