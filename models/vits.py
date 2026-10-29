import timm

def get_vit_medium(in_channels: int = 3, num_classes: int = 10): 
    # Overriding standard ViT-Base for 32x32 images with patch size 4
    return timm.create_model(
        'vit_base_patch16_224', 
        pretrained=False, 
        img_size=32,
        patch_size=4,
        in_chans=in_channels, 
        num_classes=num_classes
    )
     
def get_vit_small(in_channels: int = 3, num_classes: int = 10): 
    # Using DeiT/ViT-Tiny with overridden dimensions for 32x32
    return timm.create_model(
        'deit_tiny_patch16_224', 
        pretrained=False, 
        img_size=32,
        patch_size=4,
        in_chans=in_channels, 
        num_classes=num_classes
    )

def get_vit_large(in_channels: int = 3, num_classes: int = 10): 
    # Swin Transformer V2 configured for small inputs
    return timm.create_model(
        'vit_large_patch16_224', 
        pretrained=False, 
        img_size=32,
        in_chans=in_channels, 
        num_classes=num_classes
    )

def get_vit_huge(in_channels: int = 3, num_classes: int = 10): 
    # Standard Vision Transformer Huge, adapted for smaller inputs if needed
    return timm.create_model(
        'vit_huge_patch14_224', 
        pretrained=False, 
        img_size=32,
        # patch_size=4,
        in_chans=in_channels, 
        num_classes=num_classes
    )

# model = get_vit_small() # Or whatever model function you are calling
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total Parameters: {total_params:,}")
# print(f"Trainable Parameters: {trainable_params:,}")
# model = get_vit_medium() # Or whatever model function you are calling
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total Parameters: {total_params:,}")
# print(f"Trainable Parameters: {trainable_params:,}")
# model = get_vit_huge() # Or whatever model function you are calling
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total Parameters: {total_params:,}")
# print(f"Trainable Parameters: {trainable_params:,}")
