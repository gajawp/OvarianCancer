import os
import random
import yaml
import logging
import torch
import numpy as np
from typing import Dict, Any, Tuple, List
import math
from torch.optim.lr_scheduler import _LRScheduler

# =============================================================================
# Basic utilities
# =============================================================================
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def get_device(gpu_id: int = 0) -> torch.device:
    """Get device for training."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def print_gpu_info() -> None:
    """Print GPU information."""
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Current GPU: {torch.cuda.current_device()}")
    else:
        print("CUDA available: False")


# =============================================================================
# Configuration management
# =============================================================================
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


# =============================================================================
# Model utilities
# =============================================================================
def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """Get model size information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'params_millions': total_params / 1e6,
        'model_size_mb': total_params * 4 / 1024 / 1024  # Assuming float32
    }

def print_model_info(model: torch.nn.Module) -> None:
    """Print model information."""
    model_info = get_model_size(model)
    print(f"Model Information:")
    print(f"  Total parameters: {model_info['total_params']:,}")
    print(f"  Trainable parameters: {model_info['trainable_params']:,}")
    print(f"  Parameters (millions): {model_info['params_millions']:.2f}M")
    print(f"  Model size: {model_info['model_size_mb']:.2f} MB")

# =============================================================================
# Checkpoint management
# =============================================================================
def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = 'checkpoint.pth',
    is_best: bool = False
) -> None:
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save regular checkpoint
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    # Save best model
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_filepath)
        print(f"✅ Best model saved: {best_filepath}")

def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint loaded: {checkpoint_path}")
    
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'best_acc' in checkpoint:
        print(f"  Best accuracy: {checkpoint['best_acc']:.3f}%")
    
    return checkpoint


# =============================================================================
# Training metrics
# =============================================================================
class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# =============================================================================
# log management
# =============================================================================
def setup_logging(log_file: str = None, level: str = 'INFO') -> logging.Logger:
    """Setup basic logging."""
    # Create handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    logger = logging.getLogger(__name__)
    
    if log_file:
        logger.info(f"Logging to file: {log_file}")
    
    return logger


# =============================================================================
# Output directory management
# =============================================================================
def create_output_dirs(base_dir: str, model_name: str, experiment_type: str = None) -> Dict[str, str]:
    """Create output directories for training with timestamp-based organization."""
    from datetime import datetime
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Define paths with experiment type organization
    exp_name = f"{model_name}_{timestamp}"
    
    # Use experiment_type as the folder name instead of model_name
    folder_name = experiment_type if experiment_type else model_name
    timestamp_dir = exp_name
    
    log_dir = os.path.join(base_dir, 'logs', folder_name, timestamp_dir)
    checkpoint_dir = os.path.join(base_dir, 'checkpoints', folder_name, timestamp_dir)
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    paths = {
        'exp_name': exp_name,
        'timestamp_dir': timestamp_dir,
        'log_dir': log_dir,
        'checkpoint_dir': checkpoint_dir,
        'tensorboard_dir': tensorboard_dir,
        'log_file': os.path.join(log_dir, f'{exp_name}.log')
    }
    
    print(f"Output directories created:")
    print(f"  Logs: {log_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  TensorBoard: {tensorboard_dir}")
    
    return paths


# =============================================================================
# Early Stopping Mechanism
# =============================================================================
class EarlyStopping:
    """
    Early stopping implementation to prevent overfitting.
    Supports both 'min' mode (for loss) and 'max' mode (for accuracy).
    """
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.001,
        mode: str = 'max',
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
            restore_best_weights: Whether to restore best weights when stopping
            verbose: Whether to print stopping information
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_weights = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best - self.min_delta
            self.best_score = float('inf')
        elif mode == 'max':
            self.monitor_op = lambda current, best: current > best + self.min_delta
            self.best_score = float('-inf')
        else:
            raise ValueError(f"Mode {mode} is unknown! Use 'min' or 'max'")
    
    def __call__(self, current_score: float, model_weights: dict = None, epoch: int = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current validation metric value
            model_weights: Current model state dict for restoration
            epoch: Current epoch number
        
        Returns:
            True if should stop, False otherwise
        """
        if self.monitor_op(current_score, self.best_score):
            # Improvement detected
            self.best_score = current_score
            self.best_epoch = epoch if epoch is not None else self.best_epoch
            self.counter = 0
            
            if self.restore_best_weights and model_weights is not None:
                self.best_weights = model_weights.copy()
            
            if self.verbose:
                direction = "decreased" if self.mode == 'min' else "increased"
                print(f"EarlyStopping: Validation metric {direction} to {current_score:.6f}")
        
        else:
            # No improvement
            self.counter += 1
            
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training after {self.patience} epochs without improvement")
                    print(f"Best {self.mode} score: {self.best_score:.6f} at epoch {self.best_epoch}")
                return True
        
        return False
    
    def get_best_weights(self) -> dict:
        """Get the best model weights."""
        return self.best_weights
    
    def get_best_score(self) -> float:
        """Get the best validation score."""
        return self.best_score
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.best_epoch = 0
        self.best_weights = None
        self.early_stop = False


# =============================================================================
# Learning Rate Utilities
# =============================================================================
def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0

def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate for optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_lr(
    optimizer: torch.optim.Optimizer, 
    current_step: int, 
    warmup_steps: int, 
    base_lr: float
) -> None:
    """
    Apply linear learning rate warmup.
    
    Args:
        optimizer: Optimizer to update
        current_step: Current training step
        warmup_steps: Total warmup steps
        base_lr: Target learning rate after warmup
    """
    if current_step < warmup_steps:
        lr = base_lr * (current_step + 1) / warmup_steps
        set_lr(optimizer, lr)


# =============================================================================
# Training State Management
# =============================================================================
class TrainingState:
    """Track training state and best metrics."""
    
    def __init__(self):
        self.epoch = 0
        self.best_acc = 0.0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.learning_rates = []
    
    def update(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: float, 
        val_acc: float,
        lr: float = None
    ) -> bool:
        """
        Update training state.
        
        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            val_acc: Validation accuracy
            lr: Current learning rate
        
        Returns:
            True if this is a new best model (by accuracy)
        """
        self.epoch = epoch
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        
        if lr is not None:
            self.learning_rates.append(lr)
        
        # Check if best accuracy
        is_best_acc = val_acc > self.best_acc
        if is_best_acc:
            self.best_acc = val_acc
            self.best_epoch = epoch
        
        # Update best loss
        if val_loss < self.best_loss:
            self.best_loss = val_loss
        
        return is_best_acc
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'current_epoch': self.epoch,
            'best_acc': self.best_acc,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.train_losses),
            'current_train_loss': self.train_losses[-1] if self.train_losses else 0,
            'current_val_loss': self.val_losses[-1] if self.val_losses else 0,
            'current_val_acc': self.val_accs[-1] if self.val_accs else 0
        }
    
    def save_history(self, save_path: str) -> None:
        """Save training history to file."""
        import json
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'learning_rates': self.learning_rates,
            'best_acc': self.best_acc,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to {save_path}")


# =============================================================================
# Training Time Estimation
# =============================================================================
class TimeEstimator:
    """Estimate remaining training time and track training speed."""
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_start_time = None
        self.epoch_times = []
        self.batch_times = []
    
    def start_training(self):
        """Start timing the entire training process."""
        import time
        self.start_time = time.time()
        print(f"Training started for {self.total_epochs} epochs")
    
    def start_epoch(self):
        """Start timing current epoch."""
        import time
        self.epoch_start_time = time.time()
    
    def end_epoch(self, current_epoch: int):
        """End timing current epoch and estimate remaining time."""
        if self.epoch_start_time is None:
            return
        
        import time
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # Calculate estimates
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - current_epoch - 1
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        # Total elapsed time
        total_elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Format and print
        epoch_str = self._format_time(epoch_time)
        remaining_str = self._format_time(estimated_remaining)
        elapsed_str = self._format_time(total_elapsed)
        
        print(f"Timing - Epoch: {epoch_str}, Elapsed: {elapsed_str}, ETA: {remaining_str}")
    
    def update_batch_time(self, batch_time: float):
        """Update batch processing time."""
        self.batch_times.append(batch_time)
        
        # Keep only recent batch times to avoid memory issues
        if len(self.batch_times) > 1000:
            self.batch_times = self.batch_times[-1000:]
    
    def get_speed_stats(self) -> Dict[str, float]:
        """Get training speed statistics."""
        stats = {}
        
        if self.epoch_times:
            stats['avg_epoch_time'] = np.mean(self.epoch_times)
            stats['min_epoch_time'] = np.min(self.epoch_times)
            stats['max_epoch_time'] = np.max(self.epoch_times)
        
        if self.batch_times:
            stats['avg_batch_time'] = np.mean(self.batch_times)
        
        return stats
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

# =============================================================================
# Warmup Learning Rate Scheduler
# =============================================================================
class WarmupCosineScheduler(_LRScheduler):
    """Linear Warmup + Cosine Annealing Learning Rate Scheduler."""
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-6, eta_min=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            return [
                self.warmup_start_lr + (base_lr - self.warmup_start_lr) * self.last_epoch / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing phase
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr in self.base_lrs
            ]


def create_optimizer_and_scheduler(model, config, steps_per_epoch=None):
    """Create optimizer with official Swin parameter grouping logic and warmup scheduler."""
    # Get model's no_weight_decay settings
    skip = set()
    skip_keywords = set()
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    
    # Parameter grouping: 1D parameters (LayerNorm, etc.) and bias parameters skip weight decay
    has_decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        should_skip = (
            len(param.shape) == 1 or 
            name.endswith(".bias") or 
            name in skip or
            any(keyword in name for keyword in skip_keywords)
        )
        
        if should_skip:
            no_decay.append(param)
        else:
            has_decay.append(param)
    
    # Create optimizer
    optimizer_config = config['optimizer']
    optimizer = torch.optim.AdamW([
        {'params': has_decay, 'weight_decay': optimizer_config.get('weight_decay', 0.05)},
        {'params': no_decay, 'weight_decay': 0.0}
    ],
    lr=optimizer_config['lr'],
    betas=optimizer_config.get('betas', [0.9, 0.999]),
    eps=optimizer_config.get('eps', 1e-8))
    
    # Create scheduler with warmup support
    scheduler_config = config['scheduler']
    warmup_epochs = scheduler_config.get('warmup_epochs', 0)
    
    if scheduler_config['name'] == 'cosine' and steps_per_epoch is not None:
        # Calculate total steps and warmup steps
        total_steps = config['epochs'] * steps_per_epoch
        warmup_steps = scheduler_config.get('warmup_epochs', 20) * steps_per_epoch
        
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            warmup_lr_init=scheduler_config.get('warmup_lr_init', 1e-6),
            min_lr=scheduler_config.get('min_lr', 1e-5)
        )
        print(f"Scheduler: CosineWarmupScheduler with {warmup_epochs} warmup epochs")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        return optimizer, scheduler, True  
    else:
        scheduler = None
    print(f"Optimizer: AdamW with parameter grouping ({len(has_decay)} with decay, {len(no_decay)} without)")
    return optimizer, scheduler, False

class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine Annealing with Warmup Learning Rate Scheduler based on step.
    Fixed version based on Swin official implementation, with backward compatibility.
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, warmup_lr_init=1e-6, min_lr=1e-5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_lr_init = warmup_lr_init
        self.min_lr = min_lr
        
        # Important: cosine steps should be total_steps - warmup_steps
        self.cosine_steps = max(1, total_steps - warmup_steps)
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        Calculate learning rate for current step.
        Warmup phase: linear increase
        Cosine phase: cosine annealing (similar to timm's warmup_prefix=True behavior)
        """
        if self.last_epoch < self.warmup_steps:
            # Warmup phase - linear increase
            lrs = []
            for base_lr in self.base_lrs:
                lr = self.warmup_lr_init + (base_lr - self.warmup_lr_init) * self.last_epoch / self.warmup_steps
                lrs.append(lr)
            return lrs
        else:
            # Cosine Annealing phase
            # Important: calculate cosine schedule progress starting from warmup end
            t = self.last_epoch - self.warmup_steps  # current step in cosine phase
            
            lrs = []
            for base_lr in self.base_lrs:
                # Cosine annealing: from base_lr to min_lr
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * t / self.cosine_steps))
                lrs.append(lr)
            
            return lrs
    
    def step(self, epoch=None):
        """Override step method with boundary checking"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        # Get new learning rates
        new_lrs = self.get_lr()
        
        # Update learning rates in optimizer
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr
        
        # Update _last_lr for PyTorch API compatibility
        self._last_lr = new_lrs
    
    def state_dict(self):
        """Save scheduler state"""
        state = super().state_dict()
        state['warmup_steps'] = self.warmup_steps
        state['total_steps'] = self.total_steps
        state['cosine_steps'] = self.cosine_steps
        state['warmup_lr_init'] = self.warmup_lr_init
        state['min_lr'] = self.min_lr
        return state
    
    def load_state_dict(self, state_dict):
        """
        Load scheduler state - supports loading from old version checkpoints
        """
        # Save current values as defaults
        default_warmup_steps = self.warmup_steps
        default_total_steps = self.total_steps
        default_warmup_lr_init = self.warmup_lr_init
        default_min_lr = self.min_lr
        
        # Restore parameters from state_dict
        self.warmup_steps = state_dict.pop('warmup_steps', default_warmup_steps)
        self.total_steps = state_dict.pop('total_steps', default_total_steps)
        self.warmup_lr_init = state_dict.pop('warmup_lr_init', default_warmup_lr_init)
        self.min_lr = state_dict.pop('min_lr', default_min_lr)
        
        # cosine_steps might not exist in old checkpoints
        if 'cosine_steps' in state_dict:
            self.cosine_steps = state_dict.pop('cosine_steps')
        else:
            # Calculate cosine_steps from other parameters
            self.cosine_steps = max(1, self.total_steps - self.warmup_steps)
            print(f"[Scheduler] cosine_steps not found in checkpoint, calculated as {self.cosine_steps}")
        
        # Call parent's load_state_dict to restore last_epoch etc.
        super().load_state_dict(state_dict)
        
        # Verify restored state
        if self.last_epoch >= self.warmup_steps:
            current_lr = self.get_lr()[0]
            print(f"[Scheduler] Resumed at step {self.last_epoch}, current LR: {current_lr:.6f}")