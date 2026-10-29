"""Synthetic label-noise injection + diagnostics of what a pruner selected.

Noise types (all corrupt an EXACT fraction `rate` of the training labels, and every corrupted
label is guaranteed to differ from the true one):

  symmetric : victims chosen uniformly at random, new label uniform over the other classes.
              Independent of x, so ANY label-agnostic selector should pick noisy samples at the
              base rate (enrichment ~ 1). This is the null hypothesis.
  pair      : victims chosen uniformly, label flipped to a fixed "confusable" partner class
              (e.g. CIFAR-10 cat<->dog, truck->automobile). Still independent of x.
  instance  : feature-dependent noise. Samples that are atypical for their class (far from the
              class centroid in the same PCA pixel space the pruners use) are more likely to be
              corrupted, and flip to the nearest *other* class centroid. This mimics real
              annotation errors (ambiguous / outlying images) and is the setting where
              geometry-based selectors (k-center) can over-select noisy samples.

Because the pruners in this repo only look at features (never labels), selection is identical
whether it runs before or after noise injection; the interesting quantity is therefore how the
selected set overlaps the corrupted set (`selection_noise_stats`).
"""
import numpy as np
import torch

NOISE_TYPES = ("symmetric", "pair", "instance")

# Class-dependent flips that mimic human confusions (Patrini et al. style).
PAIR_MAPS = {
    "CIFAR10": {9: 1, 2: 0, 4: 7, 3: 5, 5: 3},          # truck->auto, bird->plane, deer->horse, cat<->dog
    "MNIST": {7: 1, 2: 7, 5: 6, 6: 5, 3: 8},
    "FashionMNIST": {6: 0, 2: 4, 4: 2, 7: 9, 5: 7},     # shirt->tshirt, pullover<->coat, sneaker->boot, sandal->sneaker
}


def get_targets(dataset) -> np.ndarray:
    """Integer labels of a dataset (without decoding images). A NoisyLabelDataset yields its corrupted labels."""
    if hasattr(dataset, "noisy_targets"):
        return dataset.noisy_targets.numpy().astype(np.int64)
    if hasattr(dataset, "targets"):
        t = dataset.targets
    elif hasattr(dataset, "labels"):          # SVHN
        t = dataset.labels
    else:
        t = [y for _, y in dataset]
    return np.asarray(t, dtype=np.int64)


def pca_features(X: np.ndarray, dim: int = 64, device="cpu", seed: int = 0) -> np.ndarray:
    """Project to `dim` principal components (same trick the k-center pruner uses)."""
    Xt = torch.from_numpy(np.ascontiguousarray(X)).float().to(device)
    if dim and Xt.shape[1] > dim:
        torch.manual_seed(seed)
        Xt = Xt - Xt.mean(0, keepdim=True)
        _, _, V = torch.pca_lowrank(Xt, q=dim, center=False, niter=2)
        Xt = Xt @ V
    return Xt.cpu().numpy()


def _pick_exact(rng, weights, k):
    """Sample k distinct indices with probability ~ weights (Gumbel-top-k)."""
    keys = np.log(weights + 1e-12) + rng.gumbel(size=len(weights))
    return np.argpartition(-keys, k - 1)[:k]


def make_noisy_labels(targets, rate: float, kind: str, num_classes: int, seed: int,
                      features=None, dataset_name: str = "", instance_beta: float = 2.0):
    """Return (noisy_targets, is_noisy mask). `features` (N, D) is only needed for kind='instance'."""
    targets = np.asarray(targets, dtype=np.int64)
    N = len(targets)
    noisy = targets.copy()
    is_noisy = np.zeros(N, dtype=bool)
    k = int(round(rate * N))
    if k == 0:
        return noisy, is_noisy
    if kind not in NOISE_TYPES:
        raise ValueError(f"Unknown label-noise type {kind!r}; choose from {NOISE_TYPES}")
    rng = np.random.default_rng(seed)

    if kind == "symmetric":
        victims = rng.choice(N, size=k, replace=False)
        offs = rng.integers(1, num_classes, size=k)             # 1..C-1  => never the true class
        noisy[victims] = (targets[victims] + offs) % num_classes

    elif kind == "pair":
        victims = rng.choice(N, size=k, replace=False)
        pm = PAIR_MAPS.get(dataset_name, {})
        for i in victims:
            y = int(targets[i])
            noisy[i] = pm.get(y, (y + 1) % num_classes)

    else:  # instance
        if features is None:
            raise ValueError("kind='instance' needs `features`")
        F = np.asarray(features, dtype=np.float64)
        cent = np.stack([F[targets == c].mean(0) if (targets == c).any() else np.full(F.shape[1], np.inf)
                         for c in range(num_classes)])
        d_own = np.linalg.norm(F - cent[targets], axis=1)
        z = np.zeros(N)
        for c in range(num_classes):                              # within-class standardised distance
            m = targets == c
            if m.any():
                z[m] = (d_own[m] - d_own[m].mean()) / (d_own[m].std() + 1e-12)
        victims = _pick_exact(rng, np.exp(instance_beta * z), k)
        # flip to nearest other class centroid
        Dv = np.linalg.norm(F[victims][:, None, :] - cent[None, :, :], axis=2)
        Dv[np.arange(k), targets[victims]] = np.inf
        noisy[victims] = Dv.argmin(1)

    is_noisy[victims] = True
    assert (noisy[is_noisy] != targets[is_noisy]).all()
    return noisy, is_noisy


class NoisyLabelDataset(torch.utils.data.Dataset):
    """Wraps a dataset and serves corrupted labels (images untouched, no copies)."""

    def __init__(self, base, noisy_targets, is_noisy):
        self.base = base
        self.noisy_targets = torch.as_tensor(noisy_targets, dtype=torch.long)
        self.is_noisy = np.asarray(is_noisy, dtype=bool)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, int(self.noisy_targets[i])


# ----------------------------------------------------------------------------- diagnostics
def selection_noise_stats(indices, is_noisy) -> dict:
    """How much of the corrupted set did the pruner pull in?

    enrichment = (noisy fraction inside subset) / (noisy fraction overall). 1 = neutral,
    >1 = the selector over-selects mislabeled samples, <1 = it avoids them.
    z = hypergeometric z-score of the observed count (|z|>~2 => unlikely to be chance)."""
    idx = np.asarray(indices, dtype=np.int64)
    N, K, n = len(is_noisy), int(is_noisy.sum()), len(idx)
    obs = int(is_noisy[idx].sum())
    out = {"sel_n_noisy": obs, "sel_noisy_frac": obs / max(n, 1)}
    if K == 0:
        out.update(noise_enrichment=float("nan"), noise_enrich_z=float("nan"))
        return out
    p = K / N
    out["noise_enrichment"] = (obs / max(n, 1)) / p
    var = n * p * (1 - p) * (N - n) / max(N - 1, 1)
    out["noise_enrich_z"] = (obs - n * p) / np.sqrt(var) if var > 0 else float("nan")
    return out


def selection_diversity_stats(indices, true_targets, pca_feats, num_classes) -> dict:
    """Cheap diversity descriptors of a subset (use TRUE labels / clean geometry):
    sel_class_entropy : normalised entropy of the class histogram (1 = perfectly balanced)
    sel_min_class_frac: smallest class share relative to uniform (0 = a class vanished)
    sel_spread_ratio  : trace of the subset's feature covariance / trace for the full set
                        (<1 = subset is concentrated, >1 = pushed toward extremes)"""
    idx = np.asarray(indices, dtype=np.int64)
    h = np.bincount(true_targets[idx], minlength=num_classes).astype(float)
    p = h / h.sum()
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(num_classes))
    full_tr = pca_feats.var(0).sum()
    sub_tr = pca_feats[idx].var(0).sum() if len(idx) > 1 else 0.0
    return {"sel_class_entropy": ent,
            "sel_min_class_frac": float(p.min() * num_classes),
            "sel_spread_ratio": float(sub_tr / (full_tr + 1e-12))}


@torch.no_grad()
def memorization_stats(model, train_set, indices, noisy_targets, is_noisy, device, batch_size=512) -> dict:
    """On the selected subset: how often does the trained model reproduce the (wrong) noisy label?
    memorization_rate = P(pred == noisy label | sample is mislabeled)   (lock-in of noise)
    clean_fit_rate    = P(pred == label        | sample is clean)"""
    idx = np.asarray(indices, dtype=np.int64)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_set, idx), batch_size=batch_size, shuffle=False)
    model.eval()
    preds = torch.cat([model(x.to(device)).argmax(1).cpu() for x, _ in loader]).numpy()
    lab = noisy_targets[idx]
    noisy = is_noisy[idx]
    hit = preds == lab
    return {"memorization_rate": float(hit[noisy].mean()) if noisy.any() else float("nan"),
            "clean_fit_rate": float(hit[~noisy].mean()) if (~noisy).any() else float("nan")}
