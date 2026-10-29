import torch
import torch.nn as nn
from tqdm import tqdm
import yprov4ml
from models.simple_cnn import SimpleCNN
from models.resnet import ResNet18
from models.vits import get_vit_medium, get_vit_small, get_vit_large, get_vit_huge
from models.mae import mae_vit_tiny_patch16, mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch16
from models.swinv2 import swinvit_small_cls, swinvit_medium_cls, swinvit_large_cls, swinvit_huge_cls
from models.deit3 import deit3_small_cls, deit3_medium_cls, deit3_large_cls

MODEL_REGISTRY = {
    "SimpleCNN": SimpleCNN, "ResNet18": ResNet18, 
    "ViTSmall": get_vit_small, "ViTMedium": get_vit_medium, "ViTLarge": get_vit_large, "ViTExtraLarge": get_vit_huge, 
    "MAESmall": mae_vit_tiny_patch16, "MAEMedium": mae_vit_base_patch16, "MAELarge": mae_vit_large_patch16, "MAEExtraLarge": mae_vit_huge_patch16, 
    "SwinViTSmall": swinvit_small_cls, "SwinViTMedium": swinvit_medium_cls, "SwinViTLarge": swinvit_large_cls, "SwinViTHuge": swinvit_huge_cls, 
    "DeitSmall": deit3_small_cls, "DeitMedium": deit3_medium_cls, "DeitLarge": deit3_large_cls,  
    }


def build_model(name: str, in_channels: int = 3, num_classes: int = 10) -> nn.Module:
    return MODEL_REGISTRY[name](in_channels=in_channels, num_classes=num_classes)


def build_optimizer(model, name: str = "sgd", lr: float = 0.1, weight_decay: float = 5e-4):
    if name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def train_model(model, train_loader, device, epochs: int = 20, lr: float = 0.1, optimizer_name: str = "sgd", logger=None, grad_clip: float = 5.0, prov_context: str = None):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optimizer_name, lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}; lower the learning rate.")
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        scheduler.step()

        train_loss, train_acc = total_loss / total, correct / total
        if logger is not None:
            logger.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})
        if prov_context is not None:
            yprov4ml.log_metric("train_loss", train_loss, is_output_to=prov_context, step=epoch)
            yprov4ml.log_metric("train_acc", train_acc, is_output_to=prov_context, step=epoch)
            yprov4ml.log_metric("lr", scheduler.get_last_lr()[0], is_output_to=prov_context, step=epoch)

    return model


@torch.no_grad()
def get_logits(model, dataloader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y in dataloader:
        x = x.to(device)
        out = model(x)
        all_logits.append(out.cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device) -> float:
    logits, labels = get_logits(model, dataloader, device)
    return (logits.argmax(dim=1) == labels).float().mean().item()


# import torch

# # Elenco delle tue funzioni di creazione modello
# models = [
#     get_vit_medium, get_vit_small, get_vit_large, get_vit_huge, 
#     mae_vit_tiny_patch16, mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch16, 
#     swinvit_small_cls, swinvit_medium_cls, swinvit_large_cls, swinvit_huge_cls, 
#     deit3_small_cls, deit3_medium_cls, deit3_large_cls
# ]

# def count_parameters(model):
#     """Conta il numero totale di parametri allenabili di un modello PyTorch."""
#     return sum(p.numel() for p in model.parameters())

# if __name__ == "__main__":
#     print(f"{'Nome Funzione':<30} | {'Numero di Parametri':<20} | {'Milioni (M)':<10}")
#     print("-" * 68)
    
#     for model_fn in models:
#         name = model_fn.__name__
#         try:
#             model = model_fn()
#             num_params = count_parameters(model)
#             millions = num_params / 1e6
#             print(f"{name:<30} | {num_params:>20,d} | {millions:>9.2f}M")
#         except Exception as e:
#             print(f"{name:<30} | Errore: {type(e).__name__}: {e}")