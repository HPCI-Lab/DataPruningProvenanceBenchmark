import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .base import BasePruner
from utils import energy

# Module-level caches (shared by CoresetGreedyPruner and HybridPruner, across all ratios).
_FEAT_CACHE = {}    # key -> (dataset_ref, features)
_GREEDY_CACHE = {}  # key -> (dataset_ref, _GreedyState)


def _extract_features(dataset, feature_extractor=None, batch_size=1024, device="cpu"):
    """(N, D) float32 numpy features; defaults to flattened normalized pixels. Cached per dataset."""
    if isinstance(dataset, Subset):  # slice from the parent's cache instead of re-reading data
        return _extract_features(dataset.dataset, feature_extractor, batch_size, device)[np.asarray(dataset.indices)]
    key = (id(dataset), len(dataset), id(feature_extractor))
    if key not in _FEAT_CACHE:
        with energy.setup_scope(("feat", key)):
            feats = []
            with torch.no_grad():
                for x, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                    f = feature_extractor(x.to(device)).cpu() if feature_extractor is not None else x.flatten(1)
                    feats.append(f.float().numpy())
            _FEAT_CACHE[key] = (dataset, np.concatenate(feats))  # keep dataset ref so id() can't be reused
    else:
        energy.touch(("feat", key))
    return _FEAT_CACHE[key][1]


class _GreedyState:
    """Incremental k-center greedy. The order is a prefix sequence, so the 10% / 20% / 50% subsets
    are just prefixes of one run; asking for a larger n continues where the last call stopped."""

    def __init__(self, X: np.ndarray, seed: int, device, proj_dim):
        X = torch.from_numpy(X).to(device)
        if proj_dim and X.shape[1] > proj_dim:  # PCA keeps most of the pairwise-distance structure
            X = X - X.mean(0, keepdim=True)
            _, _, V = torch.pca_lowrank(X, q=proj_dim, center=False, niter=2)
            X = X @ V
        self.X = X.contiguous()
        self.sq = (self.X ** 2).sum(1)
        N = self.X.shape[0]
        first = torch.tensor(int(np.random.default_rng(seed).integers(0, N)), device=self.X.device)
        self.order = torch.zeros(N, dtype=torch.long, device=self.X.device)
        self.order[0] = first
        self.min_d2 = self._d2(first)
        self.min_d2[first] = -1.0
        self.n = 1

    def _d2(self, idx):  # squared distance of every point to point idx, via ||x||^2 + ||c||^2 - 2 x.c
        return self.sq + self.sq[idx] - 2.0 * (self.X @ self.X[idx])

    @torch.no_grad()
    def extend(self, n: int):
        for t in range(self.n, n):
            idx = torch.argmax(self.min_d2)          # stays on device: no per-step host sync
            self.order[t] = idx
            self.min_d2 = torch.minimum(self.min_d2, self._d2(idx))
            self.min_d2[idx] = -1.0                  # never reselect
        self.n = max(self.n, n)


class CoresetGreedyPruner(BasePruner):
    """k-center greedy: repeatedly picks the point farthest from the selected set
    (extreme/boundary points). Fast version: incremental/prefix-shared, GPU, PCA-reduced features.

    proj_dim: PCA dimension used for distances (None = exact, full-dimensional)."""

    def __init__(self, seed: int = 42, feature_extractor=None, device="cpu", proj_dim=128):
        super().__init__(seed, device)
        self.feature_extractor = feature_extractor
        self.proj_dim = proj_dim

    def select_subset(self, dataset, ratio: float) -> np.ndarray:
        N = len(dataset)
        n = self._n_select(dataset, ratio)
        if n >= N:
            return np.arange(N, dtype=np.int64)
        device = self.device or "cpu"
        key = (id(dataset), N, self.seed, id(self.feature_extractor), self.proj_dim, str(device))
        gk = ("greedy", key)
        if key not in _GREEDY_CACHE:
            with energy.setup_scope(gk):
                X = _extract_features(dataset, self.feature_extractor, device=device)
                _GREEDY_CACHE[key] = (dataset, _GreedyState(X, self.seed, device, self.proj_dim))
        state = _GREEDY_CACHE[key][1]
        with energy.setup_scope(gk):          # incremental: cost of every extension accumulates on the same ledger entry
            state.extend(n)
        return state.order[:n].cpu().numpy().astype(np.int64)
