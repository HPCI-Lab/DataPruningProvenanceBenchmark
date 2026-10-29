import numpy as np
import torch
from .base import BasePruner
from .coreset_pruner import CoresetGreedyPruner
from .lsh_pruner import LSHPruner


class HybridPruner(BasePruner):
    """50/50 split: half via k-center greedy (extreme/boundary points),
    half via LSH density-preserving sampling (typical/interior points)."""

    def __init__(self, seed: int = 42, feature_extractor=None, device="cpu"):
        super().__init__(seed)
        self.coreset = CoresetGreedyPruner(seed, feature_extractor, device)
        self.lsh = LSHPruner(seed, feature_extractor=feature_extractor, device=device)

    def select_subset(self, dataset, ratio: float) -> np.ndarray:
        n = self._n_select(dataset, ratio)
        n_core = n // 2
        n_lsh = n - n_core

        core_idx = self.coreset.select_subset(dataset, n_core / len(dataset)) if n_core > 0 else np.array([], dtype=np.int64)

        # sample LSH half from remaining pool to avoid duplicate indices
        remaining_mask = np.ones(len(dataset), dtype=bool)
        remaining_mask[core_idx] = False
        remaining_pool = np.where(remaining_mask)[0]
        sub_dataset = torch.utils.data.Subset(dataset, remaining_pool) if n_lsh > 0 else None

        if n_lsh > 0:
            local_idx = self.lsh.select_subset(sub_dataset, n_lsh / len(sub_dataset))
            lsh_idx = remaining_pool[local_idx]
        else:
            lsh_idx = np.array([], dtype=np.int64)

        return np.concatenate([core_idx, lsh_idx]).astype(np.int64)
