"""Data-valuation pruning with exact KNN-Shapley (Jia et al., VLDB 2019, "Efficient task-specific data valuation
for nearest neighbor algorithms").

Your previous version scored a sample by how often its neighbours share its label. That is not Shapley and used random
features when none were attached. Here:
  * features = penultimate layer of a short proxy training run (pruners/proxy.py), L2-normalised;
  * value of training point i = exact Shapley value of the K-NN utility, averaged over `n_val` validation points,
    via the closed-form recursion  s_{a_N} = 1[y_{a_N}=y]/N,
    s_{a_i} = s_{a_{i+1}} + (1[y_{a_i}=y] - 1[y_{a_{i+1}}=y])/K * min(K,i)/i   (a_i = i-th nearest neighbour);
  * validation points are drawn from the training set with the labels the model sees and EXCLUDE THEMSELVES (exact:
    the point is removed from its own neighbour list). With noisy labels the valuation therefore sees noisy labels too.
Keep the highest-valued samples (class-balanced by default). Low / negative Shapley values flag mislabeled data, which
makes this method a natural candidate for the label-noise study.
"""
import numpy as np
import torch
from .base import BasePruner
from utils import energy
from .proxy import proxy_scores, dataset_labels, class_budgets


def knn_shapley(feats: torch.Tensor, labels: torch.Tensor, val_idx: torch.Tensor, K: int, chunk: int = 128):
    """Mean exact KNN-Shapley value of every row of `feats`, w.r.t. validation points feats[val_idx] (self excluded)."""
    N = feats.shape[0]
    Np = N - 1
    sv = torch.zeros(N, dtype=torch.float64, device=feats.device)
    i_range = torch.arange(1, Np, device=feats.device, dtype=torch.float64)           # i = 1..N'-1
    coef = torch.clamp(i_range, max=K) / (K * i_range)
    for st in range(0, len(val_idx), chunk):
        v = val_idx[st:st + chunk]
        d = torch.cdist(feats[v], feats)
        d[torch.arange(len(v)), v] = float("inf")                                      # self goes last
        order = d.argsort(dim=1)[:, :Np]                                               # drop self
        m = (labels[order] == labels[v][:, None]).to(torch.float64)                   # (B, N')
        t = (m[:, :-1] - m[:, 1:]) * coef                                              # (B, N'-1)
        tail = torch.flip(torch.cumsum(torch.flip(t, [1]), 1), [1])                    # sum_{j>=i} t_j
        s = torch.cat([tail + (m[:, -1:] / Np), m[:, -1:] / Np], dim=1)               # (B, N')
        sv.index_add_(0, order.reshape(-1), s.reshape(-1))
    return (sv / len(val_idx)).cpu().numpy()


class DataValuationPruner(BasePruner):
    label_aware = True

    def __init__(self, seed: int = 42, device=None, num_classes: int = 10, in_channels: int = 3,
                 k_neighbors: int = 10, n_val: int = 5000, class_balanced: bool = True,
                 proxy_model: str = "SimpleCNN", proxy_epochs: int = 10, proxy_runs: int = 1, proxy_lr: float = 0.05):
        super().__init__(seed, device)
        self.num_classes, self.in_channels = num_classes, in_channels
        self.K, self.n_val, self.class_balanced = k_neighbors, n_val, class_balanced
        self.proxy = dict(model_name=proxy_model, epochs=proxy_epochs, n_runs=proxy_runs, lr=proxy_lr)
        self._sv = {}

    def _values(self, dataset):
        key = id(dataset)
        if key in self._sv:
            energy.touch(("shapley", id(self), key))
        else:
          with energy.setup_scope(("shapley", id(self), key)):
            P = proxy_scores(dataset, self.num_classes, self.in_channels, device=self.device or "cpu",
                             seed=self.seed, **self.proxy)
            dev = torch.device(self.device or "cpu")
            f = torch.nn.functional.normalize(torch.from_numpy(P["features"]).float(), dim=1).to(dev)
            y = torch.from_numpy(dataset_labels(dataset)).to(dev)
            nv = min(self.n_val, len(y))
            val = torch.from_numpy(self.rng.choice(len(y), nv, replace=False)).to(dev)
            self._sv[key] = (dataset, knn_shapley(f, y, val, self.K))
        return self._sv[key][1]

    def select_subset(self, dataset, ratio: float) -> np.ndarray:
        N = len(dataset)
        n_sel = self._n_select(dataset, ratio)
        if n_sel >= N:
            return np.arange(N)
        sv = self._values(dataset)
        if not self.class_balanced:
            return np.sort(np.argsort(sv)[-n_sel:])
        labels = dataset_labels(dataset)
        budgets = class_budgets(labels, n_sel, self.num_classes)
        sel = []
        for c in range(self.num_classes):
            idx = np.where(labels == c)[0]
            if budgets[c]:
                sel.append(idx[np.argsort(sv[idx])[-budgets[c]:]])
        sel = np.concatenate(sel)
        if len(sel) < n_sel:
            rest = np.setdiff1d(np.arange(N), sel)
            sel = np.concatenate([sel, rest[np.argsort(sv[rest])[-(n_sel - len(sel)):]]])
        return np.sort(sel)
