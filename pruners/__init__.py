import inspect
from .base import BasePruner
from .random_pruner import RandomPruner
from .coreset_pruner import CoresetGreedyPruner
from .lsh_pruner import LSHPruner
from .hybrid_pruner import HybridPruner
from .uncertainty_pruner import UncertaintyPruner, El2nPruner, CcsPruner, D2Pruner
from .valuation_pruner import DataValuationPruner

REGISTRY = {
    "random": RandomPruner,
    "coreset": CoresetGreedyPruner,
    "lsh": LSHPruner,
    "hybrid": HybridPruner,
    # label-aware pruners (train a small proxy model; see pruners/proxy.py)
    "el2n": El2nPruner,                    # Paul et al. 2021, classic hardest-first
    "ccs": CcsPruner,                      # Zheng et al. 2023, stratified sampling over AUM difficulty
    "d2": D2Pruner,                        # Maharana et al. 2024, difficulty + diversity message passing
    "uncertainty": UncertaintyPruner,      # generic: pass score= / selection= via pruner_kwargs
    "knn_shapley": DataValuationPruner,    # Jia et al. 2019, exact KNN-Shapley
    "valuation": DataValuationPruner,
}


def build_pruner(name: str, **kwargs) -> BasePruner:
    """Extra kwargs (num_classes, in_channels, per-method options) are passed only if the pruner accepts them."""
    cls = REGISTRY[name]
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        accepted = kwargs
    else:
        accepted = {k: v for k, v in kwargs.items() if k in params}
    return cls(**accepted)
