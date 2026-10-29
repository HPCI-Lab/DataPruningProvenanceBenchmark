"""Proxy-model training used by the label-aware pruners (uncertainty / data valuation).

One short training run on the (possibly label-noisy) dataset yields, per sample:
  el2n       ||softmax - onehot||_2 at the end of training (Paul et al. 2021), mean over runs
  entropy    predictive entropy at the end of training (Coleman et al. 2020)
  margin     1 - (p_top1 - p_top2) at the end of training (uncertainty form of margin sampling)
  aum        -(mean over training of [logit_assigned - max other logit]); Pleiss et al. 2020 (higher = harder)
  forgetting number of correct->incorrect flips during training; never-learned = #epochs (Toneva et al. 2019)
All scores are oriented 'higher = harder / more uncertain'. Penultimate-layer features of run 0 are returned too.
Results are cached per (dataset object, settings), so several pruners / ratios share one proxy training.
"""
import numpy as np
import torch
import torch.nn.functional as F
from utils import energy
from torch.utils.data import DataLoader, Dataset

_CACHE = {}


class _Indexed(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(y), i


def _classifier_layer(model):
    from metrics.spectral import get_classifier_layer
    return get_classifier_layer(model)


def _build(name, in_channels, num_classes):
    from trainer import build_model
    return build_model(name, in_channels, num_classes)


def proxy_scores(dataset, num_classes, in_channels, device="cpu", model_name="SimpleCNN", epochs=10,
                 n_runs=2, lr=0.05, batch_size=256, seed=0):
    key = (id(dataset), model_name, epochs, n_runs, lr, seed)
    if key in _CACHE:
        energy.touch(("proxy", key))
        return _CACHE[key][1]
    with energy.setup_scope(("proxy", key)):
        return _proxy_scores(key, dataset, num_classes, in_channels, device, model_name, epochs, n_runs, lr,
                             batch_size, seed)


def _proxy_scores(key, dataset, num_classes, in_channels, device, model_name, epochs, n_runs, lr, batch_size, seed):
    N = len(dataset)
    dev = torch.device(device)
    acc = {k: np.zeros(N) for k in ("el2n", "entropy", "margin", "aum", "forgetting")}
    feats = None
    for run in range(n_runs):
        torch.manual_seed(seed + run)
        model = _build(model_name, in_channels, num_classes).to(dev)
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        loader = DataLoader(_Indexed(dataset), batch_size=batch_size, shuffle=True)
        margin_sum, margin_cnt = np.zeros(N), np.zeros(N)
        prev = -np.ones(N, dtype=np.int8)
        forg, learned = np.zeros(N), np.zeros(N, dtype=bool)
        for _ in range(epochs):
            model.train()
            for x, y, idx in loader:
                x, y = x.to(dev), y.to(dev)
                z = model(x)
                loss = F.cross_entropy(z, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                with torch.no_grad():
                    zd = z.detach()
                    assigned = zd.gather(1, y[:, None])[:, 0]
                    other = zd.scatter(1, y[:, None], float("-inf")).max(1).values
                    m = (assigned - other).cpu().numpy()
                    ok = (zd.argmax(1) == y).cpu().numpy()
                ii = idx.numpy()
                margin_sum[ii] += m
                margin_cnt[ii] += 1
                forg[ii] += (prev[ii] == 1) & (~ok)
                prev[ii] = ok
                learned[ii] |= ok
        forg[~learned] = epochs
        # end-of-training evaluation pass: probabilities + features
        hook_out = []
        h = _classifier_layer(model).register_forward_pre_hook(lambda mod, inp: hook_out.append(inp[0].detach().cpu()))
        model.eval()
        P, Y = [], []
        with torch.no_grad():
            for x, y, idx in DataLoader(_Indexed(dataset), batch_size=512, shuffle=False):
                P.append(F.softmax(model(x.to(dev)), 1).cpu())
                Y.append(y)
        h.remove()
        P, Y = torch.cat(P), torch.cat(Y)
        if run == 0:
            feats = torch.cat(hook_out).numpy()
        oh = F.one_hot(Y, num_classes).float()
        top2 = P.topk(2, dim=1).values
        acc["el2n"] += (P - oh).norm(dim=1).numpy()
        acc["entropy"] += (-(P * (P + 1e-12).log()).sum(1)).numpy()
        acc["margin"] += (1 - (top2[:, 0] - top2[:, 1])).numpy()
        acc["aum"] += -(margin_sum / np.maximum(margin_cnt, 1))
        acc["forgetting"] += forg
    out = {k: v / n_runs for k, v in acc.items()}
    out["features"] = feats
    _CACHE[key] = (dataset, out)          # keep a reference so id(dataset) cannot be recycled
    return out


def dataset_labels(dataset):
    from utils.label_noise import get_targets
    return get_targets(dataset)


def class_budgets(labels, n_sel, num_classes):
    """Class-proportional budgets (largest remainder) summing exactly to n_sel."""
    cnt = np.bincount(labels, minlength=num_classes).astype(float)
    raw = cnt / cnt.sum() * n_sel
    b = np.floor(raw).astype(int)
    for c in np.argsort(-(raw - b))[: n_sel - b.sum()]:
        b[c] += 1
    return np.minimum(b, cnt.astype(int))
