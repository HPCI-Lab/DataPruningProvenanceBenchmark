from abc import ABC, abstractmethod
import numpy as np

class BasePruner(ABC):
    # True if selection reads the training labels (then it must be re-run for every label-noise setting)
    label_aware = False

    def __init__(self, seed: int = 42, device=None):
        self.seed = seed
        self.device = device
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select_subset(self, dataset, ratio: float) -> np.ndarray:
        """Return array of selected indices, len = ratio * len(dataset)."""
        raise NotImplementedError

    @staticmethod
    def _n_select(dataset, ratio: float) -> int:
        return max(1, int(round(len(dataset) * ratio)))
