#!/usr/bin/env python3
"""
ImageNet-100 DataLoader
Complete data loading implementation for ImageNet-100 dataset
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


DEFAULT_IMAGENET100_ROOT = Path(__file__).resolve().parents[3] / 'datasets' / 'imagenet-100'


class ImageNet100ValDataset(Dataset):
    """
    Custom dataset for ImageNet-100 validation set
    Handles flat validation directory structure with separate ground truth file
    """
    
    def __init__(self, val_dir: str, annotation_file: str, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        
        # Read validation labels (1-based in file, convert to 0-based)
        with open(annotation_file, 'r') as f:
            self.labels = [int(line.strip()) - 1 for line in f.readlines()]
        
        # Get sorted list of validation images
        self.images = sorted([f for f in os.listdir(val_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg'))])
        
        assert len(self.images) == len(self.labels), \
            f"Mismatch: {len(self.images)} images vs {len(self.labels)} labels"
        
        print(f"Loaded {len(self.images)} validation images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.val_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_imagenet100_transforms():
    """Get standard ImageNet transforms for training and validation"""
    
    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_imagenet100_dataloaders(
    dataset_root: str = str(DEFAULT_IMAGENET100_ROOT),
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Create ImageNet-100 train and validation dataloaders
    
    Args:
        dataset_root: Root directory of ImageNet-100 dataset
        batch_size: Batch size for both train and val
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        train_loader, val_loader, dataset_info
    """
    
    print(f"Creating ImageNet-100 dataloaders from: {dataset_root}")
    
    # Define paths
    train_dir = os.path.join(dataset_root, 'train')
    val_dir = os.path.join(dataset_root, 'val')
    info_file = os.path.join(dataset_root, 'dataset_info.json')
    val_gt_file = os.path.join(dataset_root, 'ILSVRC2012_devkit_t12', 'data', 'ILSVRC2012_validation_ground_truth.txt')
    
    # Verify paths exist
    assert os.path.exists(train_dir), f"Training directory not found: {train_dir}"
    assert os.path.exists(val_dir), f"Validation directory not found: {val_dir}"
    assert os.path.exists(val_gt_file), f"Validation ground truth file not found: {val_gt_file}"
    
    # Load dataset info
    dataset_info = {}
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            dataset_info = json.load(f)
        print(f"Dataset info loaded")
    
    # Get transforms
    train_transform, val_transform = get_imagenet100_transforms()
    
    # Load dataset info and build class_to_idx for ImageFolder
    with open(info_file, 'r') as f:
        meta = json.load(f)
    wnid_to_new = {k: int(v) for k, v in meta['selected_classes']['wnid_to_new_idx'].items()}
    
    # Keep only classes that actually exist in train/ directory
    train_folders = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    class_to_idx = {wnid: wnid_to_new[wnid] for wnid in train_folders}
    
    # Create training dataset, then manually set class_to_idx
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform
    )
    
    # Manually override class_to_idx mapping
    train_dataset.class_to_idx = class_to_idx
    train_dataset.classes = list(class_to_idx.keys())
    
    # Rebuild targets list to match new indices
    train_dataset.targets = []
    for path, _ in train_dataset.samples:
        class_name = os.path.basename(os.path.dirname(path))
        train_dataset.targets.append(class_to_idx[class_name])
    
    # Rebuild samples list
    new_samples = []
    for path, _ in train_dataset.samples:
        class_name = os.path.basename(os.path.dirname(path))
        new_idx = class_to_idx[class_name]
        new_samples.append((path, new_idx))
    train_dataset.samples = new_samples
    
    # Validation checks
    assert set(train_dataset.class_to_idx.keys()) == set(wnid_to_new.keys()), \
        "Train classes don't match dataset info"
    assert set(train_dataset.class_to_idx.values()) == set(range(100)), \
        "Train indices not in range [0, 99]"
    
    # Create validation dataset (using custom class for flat structure)
    val_dataset = ImageNet100ValDataset(
        val_dir=val_dir,
        annotation_file=val_gt_file,
        transform=val_transform
    )
    
    print(f"Training dataset: {len(train_dataset)} images, {len(train_dataset.classes)} classes")
    print(f"Validation dataset: {len(val_dataset)} images")
    
    # Verify class count matches
    expected_classes = 100
    assert len(train_dataset.classes) == expected_classes, \
        f"Expected {expected_classes} classes, got {len(train_dataset.classes)}"
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Keep all validation samples
        persistent_workers=num_workers > 0
    )
    
    print(f"Created dataloaders: Train={len(train_loader)} batches, Val={len(val_loader)} batches")
    
    return train_loader, val_loader, dataset_info


def get_class_mapping(dataset_root: str) -> dict:
    """
    Get class mapping information from dataset_info.json
    
    Returns:
        Dictionary with class mappings
    """
    info_file = os.path.join(dataset_root, 'dataset_info.json')
    
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            dataset_info = json.load(f)
        return dataset_info['selected_classes']
    else:
        print("Warning: dataset_info.json not found")
        return {}


def verify_dataloaders(train_loader: DataLoader, val_loader: DataLoader):
    """Verify that dataloaders work correctly"""
    
    print("Verifying dataloaders...")
    
    # Test training loader
    train_batch = next(iter(train_loader))
    train_images, train_labels = train_batch
    print(f"Train batch shape: {train_images.shape}")
    print(f"Train label range: {train_labels.min().item()} - {train_labels.max().item()}")
    
    # Test validation loader
    val_batch = next(iter(val_loader))
    val_images, val_labels = val_batch
    print(f"Val batch shape: {val_images.shape}")
    print(f"Val label range: {val_labels.min().item()} - {val_labels.max().item()}")
    
    # Check that all labels are in valid range [0, 99]
    assert train_labels.min() >= 0 and train_labels.max() < 100, "Invalid train labels"
    assert val_labels.min() >= 0 and val_labels.max() < 100, "Invalid val labels"
    
    print("Dataloader verification passed")


# Mixup/CutMix related functions
def mixup_criterion(criterion, outputs, targets_a, targets_b, lam):
    """Mixup loss function"""
    return lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)


class Mixup:
    """Mixup data augmentation"""
    def __init__(self, mixup_alpha=1.0, cutmix_alpha=0.0, prob=1.0, switch_prob=0.5, 
                 mode='batch', label_smoothing=0.1, num_classes=100):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha  
        self.prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        
    def __call__(self, x, target):
        if not self.mixup_alpha > 0 and not self.cutmix_alpha > 0:
            return x, target
            
        # Simple mixup implementation
        if torch.rand(1) < self.prob:
            lam = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha).sample()
            batch_size = x.size(0)
            index = torch.randperm(batch_size).to(x.device)
            
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = target, target[index]
            return mixed_x, (y_a, y_b, lam)
        
        return x, target


def create_dataloaders_with_mixup(config):
    """Create dataloaders with mixup - reads from config"""
    # Extract config sections
    data_config = config['data']
    aug_config = config.get('aug', {})
    
    # Create base dataloaders using config
    train_loader, val_loader, dataset_info = create_imagenet100_dataloaders(
        dataset_root=data_config['data_dir'],
        batch_size=data_config['batch_size'],
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True)
    )
    
    # Create mixup function from config
    mixup_fn = None
    if aug_config.get('mixup', 0) > 0 or aug_config.get('cutmix', 0) > 0:
        mixup_fn = Mixup(
            mixup_alpha=aug_config.get('mixup', 0.8),
            cutmix_alpha=aug_config.get('cutmix', 1.0),
            prob=aug_config.get('mixup_prob', 1.0),
            switch_prob=aug_config.get('mixup_switch_prob', 0.5),
            mode=aug_config.get('mixup_mode', 'batch'),
            num_classes=config['model']['num_classes']
        )
        print(f"Mixup enabled: mixup_alpha={aug_config.get('mixup', 0.8)}, cutmix_alpha={aug_config.get('cutmix', 1.0)}")
    
    return train_loader, val_loader, mixup_fn


def verify_dataset(data_dir):
    """Verify dataset exists and has correct structure"""
    assert os.path.exists(data_dir), f"Dataset directory not found: {data_dir}"
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    info_file = os.path.join(data_dir, 'dataset_info.json')
    val_gt_file = os.path.join(data_dir, 'ILSVRC2012_devkit_t12', 'data', 'ILSVRC2012_validation_ground_truth.txt')
    
    assert os.path.exists(train_dir), f"Training directory not found: {train_dir}"
    assert os.path.exists(val_dir), f"Validation directory not found: {val_dir}"
    assert os.path.exists(info_file), f"Dataset info file not found: {info_file}"
    assert os.path.exists(val_gt_file), f"Validation ground truth file not found: {val_gt_file}"
    
    # Count classes and images
    train_classes = [d for d in os.listdir(train_dir) 
                    if os.path.isdir(os.path.join(train_dir, d))]
    val_images = [f for f in os.listdir(val_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg'))]
    
    print(f"Dataset verification:")
    print(f"  Training classes: {len(train_classes)}")
    print(f"  Validation images: {len(val_images)}")
    
    return True


# Alternative: Simple function for quick setup
def get_imagenet100_loaders(batch_size=32, num_workers=4):
    """Quick setup function"""
    return create_imagenet100_dataloaders(
        dataset_root=str(DEFAULT_IMAGENET100_ROOT),
        batch_size=batch_size,
        num_workers=num_workers
    )


# Example usage
if __name__ == '__main__':
    # Create dataloaders
    train_loader, val_loader, dataset_info = create_imagenet100_dataloaders(
        dataset_root=str(DEFAULT_IMAGENET100_ROOT),
        batch_size=64,
        num_workers=8
    )
    
    # Verify everything works
    verify_dataloaders(train_loader, val_loader)
    
    # Print class mapping info
    class_mapping = get_class_mapping(str(DEFAULT_IMAGENET100_ROOT))
    if class_mapping:
        print(f"First 5 WNID to index mappings:")
        wnid_to_idx = class_mapping['wnid_to_new_idx']
        for i, (wnid, idx) in enumerate(list(wnid_to_idx.items())[:5]):
            print(f"  {wnid} -> {idx}")
    
    print("Dataloader setup complete")
