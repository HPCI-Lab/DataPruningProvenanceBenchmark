import numpy as np
from collections import defaultdict
from .base import BasePruner
from .coreset_pruner import _extract_features


class LSHPruner(BasePruner):
    """Random-hyperplane LSH bucketing; samples proportionally within each
    bucket to preserve the original density distribution."""

    def __init__(self, seed: int = 42, n_hyperplanes: int = 8, feature_extractor=None, device="cpu"):
        super().__init__(seed)
        self.n_hyperplanes = n_hyperplanes
        self.feature_extractor = feature_extractor
        self.device = device

    def _hash(self, X: np.ndarray) -> np.ndarray:
        planes = self.rng.normal(size=(X.shape[1], self.n_hyperplanes))
        proj = X @ planes
        bits = (proj > 0).astype(np.int64)
        weights = (1 << np.arange(self.n_hyperplanes))
        return bits @ weights

    def select_subset(self, dataset, ratio: float) -> np.ndarray:
        n = self._n_select(dataset, ratio)
        X = _extract_features(dataset, self.feature_extractor, device=self.device)
        buckets = defaultdict(list)
        for i, h in enumerate(self._hash(X)):
            buckets[int(h)].append(i)

        selected = []
        remaining = n
        bucket_items = list(buckets.values())
        total = sum(len(b) for b in bucket_items)

        for i, items in enumerate(bucket_items):
            frac = len(items) / total
            k = int(round(frac * n)) if i < len(bucket_items) - 1 else remaining
            k = min(k, len(items), remaining)
            k = max(k, 0)
            picked = self.rng.choice(items, size=k, replace=False) if k > 0 else np.array([], dtype=np.int64)
            selected.extend(picked.tolist())
            remaining -= k
            if remaining <= 0:
                break

        # top up if under-selected due to rounding
        if remaining > 0:
            all_idx = np.arange(X.shape[0])
            unpicked = np.setdiff1d(all_idx, selected)
            extra = self.rng.choice(unpicked, size=min(remaining, len(unpicked)), replace=False)
            selected.extend(extra.tolist())

        return np.array(selected[:n], dtype=np.int64)
