import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from typing import Dict, Any, Tuple
import numpy as np
from .imagenet_official_wnids import IMAGENET_OFFICIAL_WNIDS

from timm.data import create_transform
from timm.data.mixup import Mixup
from timm.data.auto_augment import rand_augment_transform
from timm.data.random_erasing import RandomErasing
TIMM_AVAILABLE = True


class ImageNet1KDataset(Dataset):
    """Custom ImageNet-1K dataset that handles original format."""
    
    def __init__(self, root: str, split: str = 'train', transform=None):
        """
        Args:
            root: Path to ImageNet-1K dataset
            split: 'train' or 'val'
            transform: Transform to apply to images
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        if split == 'train':
            self._load_train_data()
        elif split == 'val':
            self._load_val_data()
        else:
            raise ValueError(f"Split {split} not supported. Use 'train' or 'val'.")
    
    def _load_train_data(self):
        """Load training data from ILSVRC2012_img_train."""
        train_dir = os.path.join(self.root, 'ILSVRC2012_img_train')
        if not os.path.exists(train_dir):
            train_dir = os.path.join(self.root, 'train')
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        self.images = []
        self.labels = []
        self.classes = []
        self.class_to_idx = {}
        
        available_dirs = set(os.listdir(train_dir))
        
        # Use official WNID order extracted from meta.mat
        class_dirs = [wnid for wnid in IMAGENET_OFFICIAL_WNIDS 
                     if wnid in available_dirs and os.path.isdir(os.path.join(train_dir, wnid))]
        
        if len(class_dirs) != 1000:
            print(f"Warning: Found {len(class_dirs)} classes, expected 1000")
            missing_wnids = [wnid for wnid in IMAGENET_OFFICIAL_WNIDS 
                           if wnid not in available_dirs or not os.path.isdir(os.path.join(train_dir, wnid))]
            if missing_wnids:
                print(f"Missing WNIDs: {len(missing_wnids)} classes")
        
        for idx, class_name in enumerate(class_dirs):
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx
            
            class_dir = os.path.join(train_dir, class_name)
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpeg', '.jpg'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                self.images.append(img_path)
                self.labels.append(idx)
        
        print(f"Loaded {len(self.images)} training images from {len(self.classes)} classes")
    
    def _load_val_data(self):
        """Load validation data: ILSVRC2012_img_val + ground truth."""
        val_dir = os.path.join(self.root, 'ILSVRC2012_img_val')
        if not os.path.exists(val_dir):
            val_dir = os.path.join(self.root, 'val')
        
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        
        train_dir = os.path.join(self.root, 'ILSVRC2012_img_train')
        if not os.path.exists(train_dir):
            train_dir = os.path.join(self.root, 'train')
        
        available_dirs = set(os.listdir(train_dir))
        class_dirs = [wnid for wnid in IMAGENET_OFFICIAL_WNIDS 
                     if wnid in available_dirs and os.path.isdir(os.path.join(train_dir, wnid))]
        
        self.classes = class_dirs
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}
        
        # Read validation ground truth
        gt_file = os.path.join(self.root, 'ILSVRC2012_devkit_t12', 'data', 'ILSVRC2012_validation_ground_truth.txt')
        if not os.path.exists(gt_file):
            raise FileNotFoundError(f"Validation ground truth not found: {gt_file}")
        
        with open(gt_file, 'r') as f:
            val_labels = [int(line.strip()) - 1 for line in f.readlines()]
        
        # Check if labels are within valid range
        invalid_labels = [label for label in val_labels if label < 0 or label >= len(self.classes)]
        if invalid_labels:
            print(f"Found {len(invalid_labels)} invalid labels")
        
        # Load validation images
        self.images = []
        self.labels = []
        
        valid_count = 0
        for i in range(1, len(val_labels) + 1):
            img_name = f"ILSVRC2012_val_{i:08d}.JPEG"
            img_path = os.path.join(val_dir, img_name)
            
            if os.path.exists(img_path):
                label = val_labels[i-1]
                if 0 <= label < len(self.classes):
                    self.images.append(img_path)
                    self.labels.append(label)
                    valid_count += 1
        
        print(f"Loaded {len(self.images)} validation images")
        if len(val_labels) - valid_count > 0:
            print(f"Skipped {len(val_labels) - valid_count} images with invalid labels")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224, augment_train=True, config=None):
    """
    Get transforms for training and validation with Swin-style augmentations.
    
    Args:
        img_size: Target image size
        augment_train: Whether to apply data augmentation to training set
        config: Configuration dict with augmentation parameters
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # ImageNet normalization values
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    # Validation transform (standard ImageNet preprocessing)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    
    # Training transform
    if augment_train and TIMM_AVAILABLE:
        try:
            train_transform = create_transform(
                input_size=img_size,
                is_training=True,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
                interpolation='bicubic',
                mean=imagenet_mean,
                std=imagenet_std,
            )
            print("Using timm augmentations with RandAugment and RandomErasing")
        except (TypeError, ImportError) as e:
            print(f"timm create_transform failed, using fallback augmentations")
            train_transform = _get_fallback_transform(img_size, imagenet_mean, imagenet_std)
    elif augment_train:
        train_transform = _get_fallback_transform(img_size, imagenet_mean, imagenet_std)
        if not TIMM_AVAILABLE:
            print("Using basic augmentations (install timm for full Swin-style augmentations)")
    else:
        train_transform = val_transform
    
    return train_transform, val_transform


def _get_fallback_transform(img_size, imagenet_mean, imagenet_std):
    """Get fallback training transforms when timm is not available or fails."""
    from torchvision.transforms import InterpolationMode
    
    transform_list = [
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), 
                                   interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                             saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ]
    
    return transforms.Compose(transform_list)


def create_mixup_fn(config):
    """
    Create Mixup/CutMix augmentation function.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Mixup function or None
    """
    if not TIMM_AVAILABLE:
        return None
    
    # Default Swin-style mixup parameters
    aug_config = config.get('aug', {}) if config else {}
    mixup_alpha = aug_config.get('mixup', 0.8)
    cutmix_alpha = aug_config.get('cutmix', 1.0)
    
    if mixup_alpha > 0 or cutmix_alpha > 0:
        mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            cutmix_minmax=None,
            prob=aug_config.get('mixup_prob', 1.0),
            switch_prob=aug_config.get('mixup_switch_prob', 0.5),
            mode=aug_config.get('mixup_mode', 'batch'),
            label_smoothing=config.get('model', {}).get('label_smoothing', 0.1),
            num_classes=config.get('model', {}).get('num_classes', 1000)
        )
        print(f"Mixup/CutMix enabled: mixup_alpha={mixup_alpha}, cutmix_alpha={cutmix_alpha}")
        return mixup_fn
    
    return None


def create_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    img_size: int = 224,
    pin_memory: bool = True,
    augment_train: bool = True,
    config: Dict[str, Any] = None,
    distributed: bool = False
) -> Tuple[DataLoader, DataLoader, Any]:
    """
    Create train and validation dataloaders for ImageNet-1K with optional Mixup.
    
    Args:
        data_dir: Path to ImageNet-1K dataset directory
        batch_size: Batch size per GPU (not total batch size)
        num_workers: Number of worker processes per GPU
        img_size: Target image size
        pin_memory: Whether to pin memory for faster GPU transfer
        augment_train: Whether to apply data augmentation
        distributed: Whether to use distributed training (DistributedSampler)
        config: Full configuration dictionary
    
    Returns:
        Tuple of (train_dataloader, val_dataloader, mixup_fn)
    """
    # Get transforms
    train_transform, val_transform = get_transforms(img_size, augment_train, config)
    
    # Create datasets
    train_dataset = ImageNet1KDataset(
        root=data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = ImageNet1KDataset(
        root=data_dir,
        split='val',
        transform=val_transform
    )
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        print("Using DistributedSampler for multi-GPU training")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0
    )
    
    # Create mixup function
    mixup_fn = create_mixup_fn(config) if config else None
    
    print(f"Created dataloaders: Train={len(train_loader)} batches, Val={len(val_loader)} batches")
    
    return train_loader, val_loader, mixup_fn


def validate_dataset(data_dir: str) -> bool:
    """
    Validate ImageNet-1K dataset structure.
    
    Args:
        data_dir: Path to dataset directory
    
    Returns:
        True if dataset structure is valid
    """
    print(f"Validating ImageNet-1K dataset at: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory does not exist: {data_dir}")
        return False
    
    # Check train directory
    train_dir = os.path.join(data_dir, 'ILSVRC2012_img_train')
    if not os.path.exists(train_dir):
        train_dir = os.path.join(data_dir, 'train')
    
    if not os.path.exists(train_dir):
        print("Training directory not found")
        return False
    
    # Check val directory  
    val_dir = os.path.join(data_dir, 'ILSVRC2012_img_val')
    if not os.path.exists(val_dir):
        val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(val_dir):
        print("Validation directory not found")
        return False
    
    # Check validation ground truth
    gt_file = os.path.join(data_dir, 'ILSVRC2012_devkit_t12', 'data', 'ILSVRC2012_validation_ground_truth.txt')
    if not os.path.exists(gt_file):
        print("Validation ground truth file not found")
        return False
    
    # Count classes in training set
    train_classes = [d for d in os.listdir(train_dir) 
                    if os.path.isdir(os.path.join(train_dir, d)) and d.startswith('n')]
    
    if len(train_classes) == 0:
        print("No class directories found in training set")
        return False
    
    print("Dataset validation passed")
    print(f"Found {len(train_classes)} classes in training set")
    
    return True


def get_dataset_info(data_dir: str) -> Dict[str, Any]:
    """
    Get information about the dataset.
    
    Args:
        data_dir: Path to dataset directory
    
    Returns:
        Dictionary with dataset information
    """
    try:
        train_dataset = ImageNet1KDataset(root=data_dir, split='train', transform=None)
        val_dataset = ImageNet1KDataset(root=data_dir, split='val', transform=None)
        
        return {
            'dataset_name': 'ImageNet-1K',
            'num_classes': len(train_dataset.classes),
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'classes': train_dataset.classes,
            'class_to_idx': train_dataset.class_to_idx
        }
    except Exception as e:
        print(f"Warning: Could not get dataset info: {e}")
        return {
            'dataset_name': 'ImageNet-1K',
            'num_classes': 1000,
            'train_size': 0,
            'val_size': 0,
            'classes': [],
            'class_to_idx': {}
        }


# Export main functions and classes
__all__ = [
    'ImageNet1KDataset',
    'get_transforms',
    'create_dataloaders',
    'validate_dataset',
    'get_dataset_info',
    'create_mixup_fn'
]