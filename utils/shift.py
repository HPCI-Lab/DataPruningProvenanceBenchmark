"""Covariate-shift test sets built on the fly from the clean test set.

CIFAR-10-C style corruptions (gaussian_noise, shot_noise, impulse_noise, gaussian_blur, contrast,
brightness, fog, snow, pixelate; severities 1-5) and MNIST shifts (rotate_<deg>, patch_permute,
pixel_permute).

IMPORTANT: the CIFAR corruptions here are lightweight torch re-implementations *inspired by*
Hendrycks & Dietterich (2019): same parameter tables where practical, but fog/snow/blur are
approximations, so numbers are not directly comparable with published CIFAR-10-C results. To use
the official files, download CIFAR-10-C (the .npy release), put it in ./data/CIFAR-10-C/ and set
`shift.source: official` in config.yaml.

Every condition is generated with a fixed seed, so all models see exactly the same shifted data.
"""
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

SEVERITY_CONDITIONS = ("gaussian_noise", "shot_noise", "impulse_noise", "gaussian_blur",
                       "contrast", "brightness", "fog", "snow", "pixelate")


def _randn(shape, gen):
    return torch.randn(shape, generator=gen)


def _rand(shape, gen):
    return torch.rand(shape, generator=gen)


# ---- CIFAR-style corruptions: x is (B,C,H,W) float in [0,1] on CPU ---------------------------
def gaussian_noise(x, s, gen):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][s - 1]
    return x + c * _randn(x.shape, gen)


def shot_noise(x, s, gen):
    c = [500, 250, 100, 75, 50][s - 1]
    return torch.poisson(x * c, generator=gen) / c


def impulse_noise(x, s, gen):
    c = [0.01, 0.02, 0.03, 0.05, 0.07][s - 1]
    hit = _rand(x.shape, gen) < c
    salt = _rand(x.shape, gen) < 0.5
    return torch.where(hit, salt.float(), x)


def gaussian_blur(x, s, gen):
    sigma = [0.5, 0.75, 1.0, 1.5, 2.0][s - 1]
    k = 2 * math.ceil(3 * sigma) + 1
    return TF.gaussian_blur(x, [k, k], [sigma, sigma])


def contrast(x, s, gen):
    c = [0.75, 0.5, 0.4, 0.3, 0.15][s - 1]
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - m) * c + m


def brightness(x, s, gen):
    c = [0.05, 0.1, 0.15, 0.2, 0.3][s - 1]
    return x + c


def _plasma(B, H, W, decay, gen):
    """Cheap fractal-ish noise in [0,1]: sum of upsampled random grids with geometric amplitude decay."""
    out = torch.zeros(B, 1, H, W)
    amp, tot = 1.0, 0.0
    res = 2
    while res <= max(H, W):
        out += amp * F.interpolate(_rand((B, 1, res, res), gen), size=(H, W), mode="bicubic", align_corners=False)
        tot += amp
        amp /= decay
        res *= 2
    out = out / tot
    mn = out.amin(dim=(1, 2, 3), keepdim=True)
    mx = out.amax(dim=(1, 2, 3), keepdim=True)
    return (out - mn) / (mx - mn + 1e-8)


def fog(x, s, gen):
    c0, decay = [(0.2, 3), (0.5, 3), (0.75, 2.5), (1.0, 2), (1.5, 1.75)][s - 1]
    B, _, H, W = x.shape
    mx = x.amax(dim=(1, 2, 3), keepdim=True)
    x = x + c0 * _plasma(B, H, W, decay, gen)
    return x * mx / (mx + c0)


def snow(x, s, gen):
    dens = [0.01, 0.03, 0.06, 0.10, 0.15][s - 1]
    blend = [0.8, 0.7, 0.6, 0.5, 0.4][s - 1]
    B, C, H, W = x.shape
    gray = x.mean(1, keepdim=True)
    base = blend * x + (1 - blend) * torch.maximum(x, gray * 1.5 + 0.5)
    flakes = (_rand((B, 1, H, W), gen) < dens).float() * (0.5 + 0.5 * _rand((B, 1, H, W), gen))
    k = torch.eye(5).unsqueeze(0).unsqueeze(0) / 5.0           # diagonal "motion" streak
    flakes = F.conv2d(F.pad(flakes, (2, 2, 2, 2)), k) * 3.0
    return base + flakes


def pixelate(x, s, gen):
    c = [0.95, 0.9, 0.85, 0.75, 0.65][s - 1]
    H, W = x.shape[-2:]
    small = F.interpolate(x, size=(max(1, round(H * c)), max(1, round(W * c))), mode="area")
    return F.interpolate(small, size=(H, W), mode="nearest")


_SEV_FUNCS = {f.__name__: f for f in (gaussian_noise, shot_noise, impulse_noise, gaussian_blur,
                                      contrast, brightness, fog, snow, pixelate)}


# ---- MNIST-style shifts ---------------------------------------------------------------------
def _patch_permute(x, gen, patch):
    B, C, H, W = x.shape
    ph, pw = H // patch, W // patch
    Hc, Wc = ph * patch, pw * patch
    core = x[..., :Hc, :Wc]
    tiles = core.reshape(B, C, ph, patch, pw, patch).permute(0, 1, 2, 4, 3, 5).reshape(B, C, ph * pw, patch, patch)
    perm = torch.randperm(ph * pw, generator=gen)           # one fixed permutation for the whole test set
    tiles = tiles[:, :, perm]
    core = tiles.reshape(B, C, ph, pw, patch, patch).permute(0, 1, 2, 4, 3, 5).reshape(B, C, Hc, Wc)
    out = x.clone()
    out[..., :Hc, :Wc] = core
    return out


def apply_condition(x01, name, severity, gen):
    """x01: (B,C,H,W) in [0,1] -> shifted, clamped to [0,1]."""
    if name in _SEV_FUNCS:
        out = _SEV_FUNCS[name](x01, int(severity), gen)
    elif name.startswith("rotate_"):
        out = TF.rotate(x01, float(name.split("_")[1]), interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
    elif name == "patch_permute":
        out = _patch_permute(x01, gen, patch=max(1, x01.shape[-1] // 4))   # 4x4 grid of patches
    elif name == "pixel_permute":
        B, C, H, W = x01.shape
        perm = torch.randperm(H * W, generator=gen)
        out = x01.reshape(B, C, H * W)[:, :, perm].reshape(B, C, H, W)
    else:
        raise ValueError(f"Unknown shift condition {name!r}")
    return out.clamp(0.0, 1.0)


def condition_keys(names, severities):
    """Flat list of (key, name, severity); key like 'fog@3' or 'rotate_45'."""
    keys = []
    for n in names:
        if n in _SEV_FUNCS:
            keys += [(f"{n}@{s}", n, s) for s in severities]
        else:
            keys.append((n, n, None))
    return keys


# ---- evaluation ------------------------------------------------------------------------------
@torch.no_grad()
def _acc_on_batches(model, batches, device):
    model.eval()
    correct = total = 0
    for x, y in batches:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def _synthetic_batches(loader, name, severity, mean, std, seed):
    gen = torch.Generator().manual_seed(seed)          # same shifted data for every model
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t = torch.tensor(std).view(1, -1, 1, 1)
    for x, y in loader:
        x01 = (x * std_t + mean_t).clamp(0, 1)         # undo Normalize -> pixel space
        xs = apply_condition(x01, name, severity, gen)
        yield (xs - mean_t) / std_t, y


def _official_batches(root, name, severity, mean, std, batch_size=256):
    path = os.path.join(root, f"{name}.npy")
    labels = np.load(os.path.join(root, "labels.npy"))
    imgs = np.load(path)
    sl = slice((severity - 1) * 10000, severity * 10000)
    imgs, labels = imgs[sl], labels[sl]
    mean_t = torch.tensor(mean).view(1, -1, 1, 1)
    std_t = torch.tensor(std).view(1, -1, 1, 1)
    for i in range(0, len(imgs), batch_size):
        x = torch.from_numpy(imgs[i:i + batch_size]).permute(0, 3, 1, 2).float() / 255.0
        yield (x - mean_t) / std_t, torch.from_numpy(labels[i:i + batch_size]).long()


def official_available(root, names):
    return os.path.exists(os.path.join(root, "labels.npy")) and all(
        os.path.exists(os.path.join(root, f"{n}.npy")) for n in names if n in _SEV_FUNCS)


def evaluate_shift(model, test_loader, device, names, severities, mean, std, seed=0,
                   source="synthetic", official_root="./data/CIFAR-10-C") -> dict:
    """Accuracy under every (condition, severity). Returns {'fog@3': acc, 'rotate_45': acc, ...}."""
    res = {}
    for key, n, s in condition_keys(names, severities):
        if source == "official" and n in _SEV_FUNCS:
            batches = _official_batches(official_root, n, s, mean, std)
        else:
            batches = _synthetic_batches(test_loader, n, s, mean, std, seed)
        res[key] = _acc_on_batches(model, batches, device)
    return res


def summarize_shift(clean_acc: float, shift_acc: dict) -> dict:
    """Scalar degradation summaries (absolute and relative-to-clean drop)."""
    v = np.array(list(shift_acc.values()), dtype=float)
    return {
        "shift_acc_mean": float(v.mean()),
        "shift_acc_worst": float(v.min()),
        "shift_abs_drop_mean": float(clean_acc - v.mean()),
        "shift_rel_drop_mean": float((clean_acc - v.mean()) / max(clean_acc, 1e-12)),
    }
