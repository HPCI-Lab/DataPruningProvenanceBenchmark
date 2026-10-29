"""Uncertainty / difficulty-based pruning with proper training-dynamics scores.

score     : el2n | aum | forgetting | entropy | margin      (see pruners/proxy.py; higher = harder)
selection : top  -> keep the hardest (classic EL2N / Forgetting / Entropy baselines)
            ccs  -> Coverage-centric Coreset Selection (Zheng et al., ICLR 2023): drop the hardest `hard_cut`
                    fraction (likely mislabeled/ambiguous), split the remaining score range into `strata`
                    equal-width bins and sample (almost) uniformly across bins so easy, dense regions stay covered.
            d2   -> D2-Pruning-style (Maharana et al., ICLR 2024): kNN graph in proxy-feature space, forward message
                    passing (a sample's score absorbs its neighbours' distance-weighted scores), then greedy selection
                    with reverse message passing (neighbours of a picked sample are down-weighted -> diversity).
                    Re-implemented from the paper's description and the authors' published settings
                    (k=5, gamma=0.1, label-balanced, hard cut 0.4 at high pruning); NOT a line-by-line port.
All variants are label-balanced w.r.t. the labels the model sees (noisy ones in the label-noise study).
`hard_cut='auto'` is my heuristic, min(0.4, 0.5*(1-ratio)); pass a float to fix it.
"""
import numpy as np
import torch
from .base import BasePruner
from .proxy import proxy_scores, dataset_labels, class_budgets


class UncertaintyPruner(BasePruner):
    label_aware = True

    def __init__(self, seed: int = 42, device=None, num_classes: int = 10, in_channels: int = 3,
                 score: str = "el2n", selection: str = "top", hard_cut="auto", strata: int = 50,
                 n_neighbors: int = 5, gamma: float = 0.1, proxy_model: str = "SimpleCNN",
                 proxy_epochs: int = 10, proxy_runs: int = 2, proxy_lr: float = 0.05):
        super().__init__(seed, device)
        assert score in ("el2n", "aum", "forgetting", "entropy", "margin")
        assert selection in ("top", "ccs", "d2")
        self.__dict__.update(num_classes=num_classes, in_channels=in_channels, score=score, selection=selection,
                             hard_cut=hard_cut, strata=strata, n_neighbors=n_neighbors, gamma=gamma,
                             proxy=dict(model_name=proxy_model, epochs=proxy_epochs, n_runs=proxy_runs, lr=proxy_lr))

    # ------------------------------------------------------------------
    def select_subset(self, dataset, ratio: float) -> np.ndarray:
        N = len(dataset)
        n_sel = self._n_select(dataset, ratio)
        if n_sel >= N:
            return np.arange(N)
        P = proxy_scores(dataset, self.num_classes, self.in_channels, device=self.device or "cpu",
                         seed=self.seed, **self.proxy)
        scores, feats = P[self.score], P["features"]
        labels = dataset_labels(dataset)
        beta = min(0.4, 0.5 * (1 - ratio)) if self.hard_cut == "auto" else float(self.hard_cut)
        if self.selection == "top":
            beta = 0.0 if self.hard_cut == "auto" else beta
        budgets = class_budgets(labels, n_sel, self.num_classes)
        chosen = []
        for c in range(self.num_classes):
            idx_c = np.where(labels == c)[0]
            if budgets[c] == 0 or len(idx_c) == 0:
                continue
            s_c = scores[idx_c]
            keep = np.argsort(s_c)[: max(int(len(idx_c) * (1 - beta)), budgets[c])]   # drop the hardest beta
            loc = self._pick(s_c[keep], feats[idx_c[keep]], budgets[c])
            chosen.append(idx_c[keep[loc]])
        sel = np.concatenate(chosen) if chosen else np.array([], dtype=np.int64)
        if len(sel) < n_sel:                                       # top up (class had too few samples)
            rest = np.setdiff1d(np.arange(N), sel)
            sel = np.concatenate([sel, self.rng.choice(rest, n_sel - len(sel), replace=False)])
        return np.sort(sel[:n_sel])

    # ------------------------------------------------------------------
    def _pick(self, s, f, m):
        if m >= len(s):
            return np.arange(len(s))
        if self.selection == "top":
            return np.argsort(s)[-m:]
        if self.selection == "ccs":
            return self._ccs(s, m)
        return self._d2(s, f, m)

    def _ccs(self, s, m):
        lo, hi = s.min(), s.max()
        b = np.minimum(((s - lo) / (hi - lo + 1e-12) * self.strata).astype(int), self.strata - 1)
        pools = sorted((np.where(b == k)[0] for k in range(self.strata) if (b == k).any()), key=len)
        out, left = [], m
        for i, pool in enumerate(pools):                           # smallest stratum first (CCS Alg. 1)
            take = min(len(pool), left // (len(pools) - i))
            if take:
                out.append(self.rng.choice(pool, take, replace=False))
            left -= take
        out = np.concatenate(out) if out else np.array([], dtype=np.int64)
        if left > 0:                                               # rounding leftovers
            rest = np.setdiff1d(np.arange(len(s)), out)
            out = np.concatenate([out, self.rng.choice(rest, left, replace=False)])
        return out

    def _d2(self, s, f, m):
        n = len(s)
        k = min(self.n_neighbors, n - 1)
        X = torch.from_numpy(f).float()
        X = torch.nn.functional.normalize(X, dim=1)
        d = torch.cdist(X, X)
        d.fill_diagonal_(float("inf"))
        dist, nbr = d.topk(k, dim=1, largest=False)
        dist, nbr = dist.numpy(), nbr.numpy()
        w = np.exp(-(dist ** 2) / (2 * dist.mean() ** 2 + 1e-12))                # edge weights in (0,1]
        x = (s - s.min()) / (s.max() - s.min() + 1e-12)                          # difficulty in [0,1]
        dens = x + (w * x[nbr]).sum(1)                                           # forward message passing
        dens = dens.copy()
        taken = np.zeros(n, dtype=bool)
        order = []
        for _ in range(m):
            i = int(np.argmax(np.where(taken, -np.inf, dens)))
            taken[i] = True
            order.append(i)
            dens[nbr[i]] -= self.gamma * w[i] * dens[i]                          # reverse message passing
        return np.array(order)


# Presets with the settings recommended in the literature -------------------------------------------
class El2nPruner(UncertaintyPruner):
    """Paul et al. 2021: keep the highest-EL2N samples."""
    def __init__(self, **kw):
        super().__init__(**{**dict(score="el2n", selection="top"), **kw})


class CcsPruner(UncertaintyPruner):
    """Zheng et al. 2023: CCS stratified sampling on AUM scores (their headline configuration)."""
    def __init__(self, **kw):
        super().__init__(**{**dict(score="aum", selection="ccs"), **kw})


class D2Pruner(UncertaintyPruner):
    """Maharana et al. 2024: D2 message passing on forgetting scores (their CIFAR configuration)."""
    def __init__(self, **kw):
        super().__init__(**{**dict(score="forgetting", selection="d2", hard_cut="auto"), **kw})
