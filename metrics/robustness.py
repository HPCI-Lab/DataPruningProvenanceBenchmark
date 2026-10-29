import torch


@torch.no_grad()
def evaluate_noise_robustness(model, dataloader, device, epsilons=(0.05, 0.1, 0.2)) -> dict:
    """Top-1 accuracy under additive Gaussian noise of std=eps, for each eps."""
    model.eval()
    results = {}

    for eps in epsilons:
        correct, total = 0, 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x_noisy = torch.clamp(x + eps * torch.randn_like(x), min=x.min(), max=x.max())
            logits = model(x_noisy)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        results[f"acc_eps_{eps}"] = correct / total

    return results
