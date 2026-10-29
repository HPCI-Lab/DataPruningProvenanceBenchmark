import numpy as np
from .base import BasePruner

class RandomPruner(BasePruner):
    def select_subset(self, dataset, ratio: float) -> np.ndarray:
        n = self._n_select(dataset, ratio)
        idx = self.rng.choice(len(dataset), size=n, replace=False)
        return idx
