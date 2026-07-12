"""
Unified data loading interface for ImageNet-1K and ImageNet-100
"""

from .pretrain.dataloader import create_dataloaders as create_imagenet1k_dataloaders
from .ablation.dataloader import create_imagenet100_dataloaders

def get_dataloader(dataset_type, **kwargs):
    """
    Unified dataloader factory function
    
    Args:
        dataset_type: 'imagenet1k' or 'imagenet100'
        **kwargs: Arguments passed to specific dataloader functions
    
    Returns:
        train_loader, val_loader, mixup_fn (for imagenet1k)
        train_loader, val_loader, dataset_info (for imagenet100)
    """
    if dataset_type == 'imagenet1k':
        return create_imagenet1k_dataloaders(**kwargs)
    elif dataset_type == 'imagenet100':
        return create_imagenet100_dataloaders(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}. Use 'imagenet1k' or 'imagenet100'")

__all__ = [
    'get_dataloader',
    'create_imagenet1k_dataloaders',
    'create_imagenet100_dataloaders'
]