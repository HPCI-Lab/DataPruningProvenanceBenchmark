import torch
import torch.nn as nn
import types
from functools import partial
from swin_transformer_v2 import (
    swin_transformer_v2_t, 
    swin_transformer_v2_s, 
    swin_transformer_v2_b, 
    swin_transformer_v2_l, 
    swin_transformer_v2_h
)

def _wrap_swin_classification(model, embed_dim, num_classes=10):
    model.head = nn.Linear(embed_dim, num_classes)
    
    def classification_forward(self, x):
        # Extract features using forward_features if available, 
        # otherwise use standard forward pass
        features = self.forward_features(x) if hasattr(self, 'forward_features') else self(x)
        
        # Global average pooling if output is 3D (B, N, C) or 4D (B, C, H, W)
        if features.ndim == 4:
            features = features.mean(dim=[2, 3])
        elif features.ndim == 3:
            features = features.mean(dim=1)
            
        return self.head(features)
        
    model.forward = types.MethodType(classification_forward, model)
    return model

def swinvit_tiny_cls(num_classes=10, **kwargs):
    model = swin_transformer_v2_t(
        in_channels=3, window_size=8, input_resolution=(32, 32),
        sequential_self_attention=False, use_checkpoint=False, **kwargs
    )
    # Swin-T final feature dimension is typically 768
    return _wrap_swin_classification(model, embed_dim=768, num_classes=num_classes)

def swinvit_small_cls(num_classes=10, **kwargs):
    model = swin_transformer_v2_s(
        in_channels=3, window_size=8, input_resolution=(32, 32),
        sequential_self_attention=False, use_checkpoint=False, **kwargs
    )
    # Swin-S final feature dimension is typically 768
    return _wrap_swin_classification(model, embed_dim=768, num_classes=num_classes)

def swinvit_medium_cls(num_classes=10, **kwargs):
    model = swin_transformer_v2_b(
        in_channels=3, window_size=8, input_resolution=(32, 32),
        sequential_self_attention=False, use_checkpoint=False, **kwargs
    )
    # Swin-B/Medium final feature dimension is typically 1024
    return _wrap_swin_classification(model, embed_dim=1024, num_classes=num_classes)

def swinvit_large_cls(num_classes=10, **kwargs):
    model = swin_transformer_v2_l(
        in_channels=3, window_size=8, input_resolution=(32, 32),
        sequential_self_attention=False, use_checkpoint=False, **kwargs
    )
    # Swin-L final feature dimension is typically 1536
    return _wrap_swin_classification(model, embed_dim=1536, num_classes=num_classes)

def swinvit_huge_cls(num_classes=10, **kwargs):
    model = swin_transformer_v2_h(
        in_channels=3, window_size=8, input_resolution=(32, 32),
        sequential_self_attention=False, use_checkpoint=False, **kwargs
    )
    # Swin-H final feature dimension is typically 2048 (adjust if your implementation differs)
    return _wrap_swin_classification(model, embed_dim=2048, num_classes=num_classes)