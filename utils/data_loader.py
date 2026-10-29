import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# from utils.torch_distributed_helper import setup, cleanup, to_distributed_dataloader


NORM = {
    "MNIST": ((0.1307,), (0.3081,)),
    "FashionMNIST": ((0.2860,), (0.3530,)),
    "CIFAR10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "CIFAR10C": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "SVHN": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
}


def _get_norm_key(name: str) -> str:
    if name in ["fashion_mnist", "Fashion-MNIST", "FashionMNIST"]:
        return "FashionMNIST"
    return name


def get_norm(name: str):
    """(mean, std) tuples used by the dataset's Normalize transform."""
    return NORM[_get_norm_key(name)]


def _transform(name: str, in_channels: int):
    norm_key = _get_norm_key(name)
    mean, std = NORM[norm_key]
    tfs = [T.ToTensor()]
    
    # Handle channel adjustments if necessary
    if in_channels == 3 and norm_key in ["MNIST", "FashionMNIST"]:
        tfs.append(T.Grayscale(num_output_channels=3))
    elif in_channels == 1 and norm_key not in ["MNIST", "FashionMNIST"]:
        tfs.append(T.Grayscale(num_output_channels=1))
        
    tfs.append(T.Normalize(mean, std))
    return T.Compose(tfs)


def get_dataset(name: str, train: bool, root: str = "./data", in_channels: int = 3):
    tf = _transform(name, in_channels)
    norm_key = _get_norm_key(name)
    
    if norm_key == "MNIST":
        return torchvision.datasets.MNIST(root, train=train, download=True, transform=tf)
    if norm_key == "FashionMNIST":
        return torchvision.datasets.FashionMNIST(root, train=train, download=True, transform=tf)
    if norm_key == "CIFAR10":
        return torchvision.datasets.CIFAR10(root, train=train, download=True, transform=tf)
    if norm_key == "SVHN":
        split = "train" if train else "test"
        return torchvision.datasets.SVHN(root, split=split, download=True, transform=tf)
    if norm_key == "CIFAR10C":
        from stable_datasets.images.cifar10_c import CIFAR10C   # optional dependency, imported lazily
        split = "train" if train else "test"
        return CIFAR10C(root, split=split, download=True, transform=tf)

    raise ValueError(f"Unknown dataset: {name}")


def get_pruned_loader(dataset, indices, batch_size: int = 128, shuffle: bool = True) -> DataLoader:
    subset = Subset(dataset, indices)
    # return to_distributed_dataloader(subset, batch_size=batch_size)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def get_loader(dataset, batch_size: int = 256, shuffle: bool = False) -> DataLoader:
    # return to_distributed_dataloader(dataset, batch_size=batch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)