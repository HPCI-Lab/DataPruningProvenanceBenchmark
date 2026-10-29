import torch
import torch.nn as nn
import types
import timm

def _wrap_timm_classification(model, num_classes=10):
    # timm models usually have a 'head' or 'fc' attribute for classification
    # Let's dynamically find the input features to the classifier head
    if hasattr(model, 'head'):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    def classification_forward(self, x):
        # timm's forward_features gives token sequence or feature maps
        x_feat = self.forward_features(x)
        
        # If it returns a sequence (B, N, C), pull the [CLS] token (index 0)
        if x_feat.ndim == 3:
            x_cls = x_feat[:, 0]
        else:
            x_cls = x_feat
            
        head = self.head if hasattr(self, 'head') else self.fc
        return head(x_cls)
        
    model.forward = types.MethodType(classification_forward, model)
    return model

def deit3_small_cls(num_classes=10, pretrained=False, **kwargs):
    # Load DeiT III Small via timm
    model = timm.create_model('deit3_small_patch16_224', img_size=32, pretrained=pretrained, **kwargs)
    return _wrap_timm_classification(model, num_classes=num_classes)

def deit3_medium_cls(num_classes=10, pretrained=False, **kwargs):
    model = timm.create_model('deit3_base_patch16_224', img_size=32, pretrained=pretrained, **kwargs)
    return _wrap_timm_classification(model, num_classes=num_classes)

def deit3_large_cls(num_classes=10, pretrained=False, **kwargs):
    model = timm.create_model('deit3_large_patch16_224', img_size=32, pretrained=pretrained, **kwargs)
    return _wrap_timm_classification(model, num_classes=num_classes)