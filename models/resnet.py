import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNet18(nn.Module):
    """torchvision ResNet18 adapted for small (32x32) images, with a
    named `classifier` attribute for consistent spectral analysis access."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        backbone = tvm.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        f = self.backbone(x)
        out = self.classifier(f)
        return (out, f) if return_features else out
